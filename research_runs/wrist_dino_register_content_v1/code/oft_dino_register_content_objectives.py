"""Training-only native Wrist K/V intervention; no evaluation dependency."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from white_patch.oft_dino_register_delayed_objectives import auxiliary_active, combine_gradients


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def content_indices(target: str, coverage: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Exactly four register tokens or four deterministic non-patch dense tokens."""
    _check(coverage.shape == (256,) and bool(torch.isfinite(coverage).all()),
           "Invalid actual patch footprint")
    count, prefix = int(cfg["content_token_count"]), int(cfg["dino_prefix_tokens"])
    _check(count == 4 and prefix == 5, "Content token layout changed")
    if target == "register_content":
        indices = torch.tensor(cfg["register_query_tokens"], device=coverage.device, dtype=torch.long)
    else:
        _check(target == "dense_content", "Unknown content target")
        available = torch.nonzero(coverage <= 0, as_tuple=True)[0]
        _check(available.numel() >= count, "Too few non-patch dense tokens")
        offsets = torch.linspace(0, available.numel() - 1, count, device=coverage.device).round().long()
        indices = available.index_select(0, offsets) + prefix
    _check(indices.numel() == count and indices.unique().numel() == count,
           "Content tokens must be four distinct indices")
    return indices


def extract_kv(qkv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _check(qkv.ndim == 3 and qkv.shape[0] == 1 and qkv.shape[1] == 261
           and qkv.shape[2] % 3 == 0, "Native DINO qkv layout changed")
    return qkv.reshape(1, 261, 3, qkv.shape[-1] // 3).index_select(1, indices)[:, :, 1:, :]


def replace_kv(qkv: torch.Tensor, indices: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
    """All Q and every unselected K/V remain EXACTLY the native clean values."""
    expected = extract_kv(qkv, indices)
    _check(donor.shape == expected.shape and donor.device == qkv.device
           and bool(torch.isfinite(donor).all()), "Invalid donor K/V")
    clean = qkv.reshape(1, 261, 3, qkv.shape[-1] // 3)
    selected = torch.cat((clean.index_select(1, indices)[:, :, :1, :], donor.to(qkv.dtype)), dim=2)
    return clean.index_copy(1, indices, selected).reshape_as(qkv)


@contextmanager
def replace_wrist_kv(qkv_module, indices: torch.Tensor, donor: torch.Tensor):
    """Intercept only the second (Wrist) native qkv call; clean up on all exits."""
    audit = {"calls": 0, "replaced_calls": 0, "selected_tokens": indices.detach().cpu().tolist(),
             "clean_kv": None}

    def hook(_module, _inputs, output):
        index = audit["calls"]
        audit["calls"] += 1
        if index == 1:
            audit["replaced_calls"] += 1
            audit["clean_kv"] = extract_kv(output, indices).detach()
            return replace_kv(output, indices, donor)
        return output

    handle = qkv_module.register_forward_hook(hook)
    try:
        yield audit
        _check(audit["calls"] == 2 and audit["replaced_calls"] == 1,
               "Native Front/Wrist qkv call order/count changed")
    finally:
        handle.remove()


def cpu_preflight(cfg: dict) -> dict:
    """CPU native-timm identity and equivalent segmented RGB gradient checks."""
    from timm.models.vision_transformer import Attention
    from white_patch.oft_dino_register_delayed_objectives import cpu_preflight as attention_preflight

    attention = attention_preflight(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        native = Attention(32, num_heads=4, qk_norm=True).eval()
        for parameter in native.parameters():
            parameter.requires_grad_(False)
        front, clean = torch.randn(1, 261, 32), torch.randn(1, 261, 32)
        coverage = torch.zeros(256)
        coverage[80:96] = 1
        baseline = native(clean)
        reports = []
        for target in ("register_content", "dense_content"):
            indices = content_indices(target, coverage, cfg)
            donor_clean = extract_kv(native.qkv(clean), indices)
            with replace_wrist_kv(native.qkv, indices, donor_clean) as audit:
                actual_front, identity = native(front), native(clean)
            _check(torch.equal(identity, baseline) and torch.equal(actual_front, native(front)),
                   "Native content identity/Front isolation failed")
            patch = torch.randn(1, 1, 32, requires_grad=True)
            donor = extract_kv(native.qkv(clean + .1 * patch), indices)
            with replace_wrist_kv(native.qkv, indices, donor):
                native(front)
                direct_loss = native(clean)[:, 5:, :].square().mean()
            direct_grad = torch.autograd.grad(direct_loss, patch)[0]
            donor = extract_kv(native.qkv(clean + .1 * patch), indices)
            leaf = donor.detach().clone().requires_grad_(True)
            with replace_wrist_kv(native.qkv, indices, leaf):
                native(front)
                split_loss = native(clean)[:, 5:, :].square().mean()
            leaf_grad = torch.autograd.grad(split_loss, leaf)[0]
            split_grad = torch.autograd.grad(donor, patch, grad_outputs=leaf_grad)[0]
            _check(bool(torch.isfinite(split_grad).all()) and float(split_grad.norm()) > 0
                   and torch.allclose(direct_grad, split_grad, atol=1e-9, rtol=1e-5),
                   "Content branch did not propagate the exact chain-rule RGB gradient")
            _check(not native.qkv._forward_hooks, "Content hook leaked after training branch")
            reports.append({"target": target, "identity_exact": True, "front_exact": True,
                            "chain_rule_close": True, "rgb_gradient_norm": float(split_grad.norm()),
                            "token_indices": indices.tolist(), "native_attention_unchanged": True})
    return {"status": "PASS", "attention": attention, "content": reports,
            "cuda_initialized": False, "model_loaded": False}
