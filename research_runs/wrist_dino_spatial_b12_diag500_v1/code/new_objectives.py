"""Three distinct DINO-only objectives; readout hooks NEVER alter activations."""
from __future__ import annotations
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from common import require


@contextmanager
def capture_layers(dino, layers, qkv=False):
    captures, handles = {}, []
    require(len(set(layers)) == len(layers) and all(0 <= i < len(dino.blocks) for i in layers),
            "Invalid actual DINO layers")
    def hook(layer):
        def capture(_module, _args, output):
            require(layer not in captures, "Expected one Wrist encoder pass per layer")
            captures[layer] = output
            # Explicitly no return value: leave native forward untouched.
        return capture
    try:
        for layer in layers:
            module = dino.blocks[layer].attn.qkv if qkv else dino.blocks[layer]
            handles.append(module.register_forward_hook(hook(layer)))
        yield captures
        require(set(captures) == set(layers), "Requested layers were not executed")
    finally:
        for handle in handles:
            handle.remove()


def features_forward(engine, pixels, method, cfg):
    layers = cfg["attention_layers"] if method == "dino_attention" else cfg["multilayer_layers"] if method == "dino_multilayer" else []
    with capture_layers(engine.dino, layers, qkv=method == "dino_attention") as captures:
        dense = engine.direct_dino(pixels)
    if method == "dino_multilayer":
        prefix = int(engine.dino.num_prefix_tokens)
        captures = {layer: value[:, prefix:] for layer, value in captures.items()}
        require(all(value.shape[:2] == (1, 256) for value in captures.values()), "Block spatial layout changed")
    return dense, captures


def unpatched_tokens(footprint, minimum=1):
    require(footprint.ndim == 1 and len(footprint) == 256 and torch.isfinite(footprint).all(), "Footprint changed")
    indices = torch.nonzero(footprint.detach() <= 1e-6, as_tuple=False).flatten()
    require(len(indices) >= minimum, "EOT left too few unpatched tokens")
    return indices


def split_native_qk(value, attention, prefix=5):
    require(value.ndim == 3 and value.shape[:2] == (1, prefix + 256), "Native QKV layout changed")
    heads = int(attention.num_heads)
    require(value.shape[-1] % (3 * heads) == 0, "Native QKV/head shape changed")
    q, k, _ = value.reshape(1, prefix + 256, 3, heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    return attention.q_norm(q).float(), attention.k_norm(k).float()


def attention_objective(captures, modules, footprint, prefix=5):
    queries = unpatched_tokens(footprint)
    keys = torch.nonzero(footprint.detach() > 1e-6, as_tuple=False).flatten()
    require(bool(captures) and set(captures) == set(modules), "Attention layers/captures differ")
    if len(keys) == 0:
        # Keep the author's EOT and optimizer clock, never silently resample.
        zero = sum(value.float().sum() * 0 for value in captures.values())
        return zero, {"patch_attention_mass": 0., "query_tokens": float(len(queries)),
                      "patch_key_tokens": 0., "attention_objective_active": 0.}
    losses, masses = [], []
    for layer, value in captures.items():
        q, k = split_native_qk(value, modules[layer], prefix)
        logits = (q.index_select(2, queries + prefix) * float(modules[layer].scale)) @ k.transpose(-2, -1)
        log_mass = torch.logsumexp(logits.index_select(-1, keys + prefix), -1) - torch.logsumexp(logits, -1)
        losses.append(-log_mass.mean())
        masses.append(log_mass.detach().exp().mean())
    require(bool(losses), "No attention layers captured")
    loss = torch.stack(losses).mean()
    return loss, {"patch_attention_mass": float(torch.stack(masses).mean()),
                  "query_tokens": float(len(queries)), "patch_key_tokens": float(len(keys)),
                  "attention_objective_active": 1.}


def multilayer_reference(captures, eps=1e-6):
    references = {}
    for layer, value in captures.items():
        clean = value.detach().float()
        channels = clean.shape[-1]
        require(clean.shape[:2] == (1, 256) and channels >= 2, "Invalid clean block features")
        indices = clean.abs().topk(channels // 2, dim=-1, sorted=False).indices
        strong = torch.zeros_like(clean, dtype=torch.bool).scatter_(-1, indices, True)
        references[layer] = {"strong": strong, "scale": clean.square().mean().sqrt().clamp_min(eps)}
    return references


def multilayer_objective(captures, references, footprint, eps=1e-6):
    indices, losses, details = unpatched_tokens(footprint), [], {}
    require(set(captures) == set(references), "Layer references/captures differ")
    for layer, value in captures.items():
        current = value.float().index_select(1, indices) / references[layer]["scale"]
        strong = references[layer]["strong"].index_select(1, indices)
        high = current[strong].square().mean().add(eps * eps).sqrt()
        low = current[~strong].square().mean().add(eps * eps).sqrt()
        losses.append(high.log() - low.log())
        details.update({f"layer{layer}_strong_rms": float(high.detach()), f"layer{layer}_weak_rms": float(low.detach()),
                        f"layer{layer}_strong_fraction": float((high.square() / (high.square() + low.square())).detach())})
    require(bool(losses), "No block layers captured")
    return torch.stack(losses).mean(), details


def affinity_objective(clean, current, footprint, topk=8):
    require(clean.shape == current.shape and clean.shape[:2] == (1, 256), "Dense feature pair changed")
    indices = unpatched_tokens(footprint, minimum=2 * topk + 1)
    clean = F.normalize(clean.detach().float().index_select(1, indices), dim=-1)[0]
    current = F.normalize(current.float().index_select(1, indices), dim=-1)[0]
    target = clean @ clean.T
    diagonal = torch.eye(len(indices), device=target.device, dtype=torch.bool)
    positive = target.masked_fill(diagonal, -torch.inf).topk(topk, dim=-1).indices
    negative_scores = target.masked_fill(diagonal, torch.inf).scatter(1, positive, torch.inf)
    negative = negative_scores.topk(topk, dim=-1, largest=False).indices
    similarity = current @ current.T
    high, low = similarity.gather(1, positive), similarity.gather(1, negative)
    loss = high.mean() - low.mean()
    return loss, {"affinity_high_mean": float(high.detach().mean()), "affinity_low_mean": float(low.detach().mean()),
                  "affinity_reversed_queries": float((high.detach().mean(-1) < low.detach().mean(-1)).float().mean()),
                  "affinity_query_tokens": float(len(indices))}


def objective(method, dense, captures, reference, engine, footprint, cfg):
    if method == "dino_attention":
        modules = {i: engine.dino.blocks[i].attn for i in cfg["attention_layers"]}
        return attention_objective(captures, modules, footprint, int(engine.dino.num_prefix_tokens))
    if method == "dino_multilayer":
        return multilayer_objective(captures, reference, footprint, cfg["objective_epsilon"])
    require(method == "dino_affinity", "Unknown DINO objective")
    return affinity_objective(reference, dense, footprint, cfg["affinity_topk"])
