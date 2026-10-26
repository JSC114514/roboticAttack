"""Expert-derived action targets, geometry projection, native attention scores."""
from __future__ import annotations
import math
import random
import torch
import torch.nn.functional as F
from common import require

def target_action(expert, family, cfg):
    require(expert.shape == (1, 8, 7) and torch.isfinite(expert).all(), "Expected normalized expert[1,8,7]")
    target = expert.detach().clone()
    if family == "uada":
        # Official normalized action range; never empirical policy q01/q99.
        target = torch.where(expert >= 0, -torch.ones_like(expert), torch.ones_like(expert)).detach()
    else:
        require(family == "tma", "Unknown action family")
        target[..., cfg["tma_maskidx"]] = target.new_tensor(cfg["tma_normalized_target"])
    return target

def action_loss(action, target, family, cfg):
    idx = cfg["uada_maskidx"] if family == "uada" else cfg["tma_maskidx"]
    return F.mse_loss(action[..., idx], target[..., idx])

def sampled_index(scores, seed):
    require(len(scores) == 4 and all(math.isfinite(s) and s >= 0 for s in scores), "Invalid DINO contribution scores")
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    weights = [1.] * 4
    for i in order[:2]:
        if scores[i] > 0:
            weights[i] = 2.
    return random.Random(seed).choices(range(4), weights=weights, k=1)[0], weights

def total_variation(patch):
    return (patch[..., :, 1:] - patch[..., :, :-1]).abs().mean() + (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean()

def rgb_metrics(patch, initial):
    value = patch.detach().float()
    fy = torch.fft.fftfreq(value.shape[-2], device=value.device)[:, None]
    fx = torch.fft.rfftfreq(value.shape[-1], device=value.device)[None, :]
    spectrum = torch.fft.rfft2(value - value.mean(dim=(-2, -1), keepdim=True)).abs().square()
    high = fx.square() + fy.square() > .25 ** 2
    return {"tv": float(total_variation(value)), "high_frequency_fraction": float(spectrum[:, high].sum() / spectrum.sum().clamp_min(1e-12)),
            "initial_rgb_mae": float((value - initial.to(value.device)).abs().mean()),
            "pixel_min": float(value.min()), "pixel_max": float(value.max()),
            "saturated_pixel_fraction": float(((value <= 0) | (value >= 1)).float().mean())}

def frequency_filter(delta, strength=8., power=.75):
    h, w = delta.shape[-2:]
    fy = torch.fft.fftfreq(h, device=delta.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=delta.device)[None, :]
    gain = (1 + strength * (fx.square() + fy.square())).pow(-power)
    require(bool((gain > 0).all()), "Frequency filter must remain full rank")
    return torch.fft.irfft2(torch.fft.rfft2(delta.float()) * gain, s=(h, w))

def precondition_step(before, proposal, scale, cfg):
    delta = frequency_filter(proposal - before, cfg["frequency_strength"], cfg["frequency_power"])
    target = (proposal.clamp(0, 1) - before).norm()
    if scale is None and float(target) > 0 and float(delta.norm()) > 0:
        lo, hi = 0., 1.
        for _ in range(30):
            if float(((before + hi * delta).clamp(0, 1) - before).norm()) >= float(target):
                break
            hi *= 2
        require(float(((before + hi * delta).clamp(0, 1) - before).norm()) >= float(target), "Initial RGB step matching failed")
        for _ in range(40):
            mid = (lo + hi) / 2
            if float(((before + mid * delta).clamp(0, 1) - before).norm()) < float(target):
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
    return (before + (1. if scale is None else scale) * delta).clamp(0, 1), scale

def token_footprint(mask, grid=16):
    require(mask.shape == (224, 224), "Actual RGB paste support shape changed")
    # Conservative support: JPEG block effects and crop interpolation included.
    value = F.max_pool2d(mask.detach().float()[None, None], 17, stride=1, padding=8)
    axis = torch.linspace(-math.sqrt(.9), math.sqrt(.9), 224, device=mask.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    value = F.grid_sample(value, torch.stack((xx, yy), -1)[None], mode="bilinear", padding_mode="border", align_corners=True)
    return F.max_pool2d(value, 224 // grid).flatten().clamp(0, 1)

def action_token_weights(gradient):
    require(gradient.ndim == 3 and gradient.shape[:2] == (1, 256), "Action-DINO gradient shape changed")
    require(torch.isfinite(gradient).all(), "Action-DINO gradient nonfinite")
    norms = gradient.detach().float().norm(dim=-1)[0]
    return norms / norms.sum().clamp_min(1e-12)

def geometry_reference(clean, references, gradient):
    require(len(references) == 4, "Four source-only geometry references required")
    scores = []
    for value in references:
        delta = value.detach().float() - clean.detach().float()
        scores.append(float(-(gradient.detach().float() * delta).sum() / delta.square().mean().sqrt().clamp_min(1e-6)))
    selected = max(range(4), key=lambda i: scores[i])
    if scores[selected] <= 0:
        return None, scores, selected
    direction = F.normalize(references[selected].detach().float(), dim=-1) - F.normalize(clean.detach().float(), dim=-1)
    if float(direction.norm()) <= 1e-8:
        return None, scores, selected
    return direction.detach(), scores, selected

def geometry_loss(current, clean, direction, weights, footprint):
    empty = {"geometry_projection": 0., "geometry_cosine": 0., "aux_active": 0.}
    if direction is None:
        return None, empty
    w = weights.detach() * (footprint <= 1e-6)
    reference = (direction.square().sum(-1)[0] * w).sum()
    if float(reference) <= 1e-12 or float(w.sum()) <= 1e-12:
        return None, empty
    delta = F.normalize(current.float(), dim=-1) - F.normalize(clean.detach().float(), dim=-1)
    projection = ((delta * direction).sum(-1)[0] * w).sum() / reference
    cosine = F.cosine_similarity((delta.detach()[0] * w[:, None].sqrt()).flatten(),
                                 (direction[0] * w[:, None].sqrt()).flatten(), dim=0)
    return -projection, {"geometry_projection": float(projection.detach()), "geometry_cosine": float(cosine), "aux_active": 1.}

def split_qkv(value, attention):
    require(value.ndim == 3 and value.shape[0] == 1 and value.shape[1] == 261, "Expected CLS+4 registers+256 dense QKV")
    heads = int(attention.num_heads)
    q, k, _ = value.reshape(1, 261, 3, heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    return attention.q_norm(q), attention.k_norm(k)

def attention_loss(captures, modules, weights, footprint, scales):
    valid = footprint <= 1e-6
    w = weights.detach().float() * valid
    patch = footprint.detach().float()
    if float(w.sum()) <= 1e-12 or float(patch.sum()) <= 1e-12:
        return None, {"attention_score": 0., "aux_active": 0.}
    w = w / w.sum()
    key_weight = patch / patch.sum()
    scores = []
    for layer, values in captures.items():
        q, k = split_qkv(values, modules[layer])
        # Dense queries/keys only. Native q_norm/k_norm and scaling preserved.
        logits = (q[:, :, 5:].float() * modules[layer].scale) @ k[:, :, 5:].float().transpose(-2, -1)
        score = (logits * w[None, None, :, None] * key_weight[None, None, None, :]).sum((-1, -2)).mean()
        scores.append(score / max(float(scales[layer]), 1e-6))
    result = torch.stack(scores).mean()
    return -result, {"attention_score": float(result.detach()), "aux_active": 1.}

def combine_gradients(main, auxiliary, ratio=.1):
    require(torch.isfinite(main).all(), "Nonfinite main RGB gradient")
    if auxiliary is None:
        return main, {"aux_gradient_cosine": 0., "effective_aux_ratio": 0.}
    require(auxiliary.shape == main.shape and torch.isfinite(auxiliary).all(), "Invalid auxiliary RGB gradient")
    mn, an = main.float().norm(), auxiliary.float().norm()
    if float(mn) <= 1e-12 or float(an) <= 1e-12:
        return main, {"aux_gradient_cosine": 0., "effective_aux_ratio": 0.}
    contribution = ratio * mn / an * auxiliary
    cosine = F.cosine_similarity(main.float().flatten(), auxiliary.float().flatten(), dim=0)
    return main + contribution, {"aux_gradient_cosine": float(cosine), "effective_aux_ratio": float(contribution.norm() / mn)}
