"""Actual continuous-action objectives and full-rank RGB-step preconditioning."""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from common import require


def target_action(clean, family, bounds, cfg):
    if family == "uada":
        low = torch.as_tensor(bounds["q01"], device=clean.device).reshape(1, 1, 7)
        high = torch.as_tensor(bounds["q99"], device=clean.device).reshape(1, 1, 7)
        return torch.where(clean >= (low + high) / 2, low, high).detach()
    require(family == "tma", "Unknown action family")
    target = clean.detach().clone()
    target[..., cfg["tma_maskidx"]] = torch.as_tensor(cfg["tma_normalized_target"], device=clean.device,
                                                   dtype=clean.dtype)
    return target


def action_loss(action, target, family, cfg):
    idx = cfg["uada_maskidx"] if family == "uada" else cfg["tma_maskidx"]
    return F.mse_loss(action[..., idx], target[..., idx])


def total_variation(patch):
    return (patch[:, :, 1:] - patch[:, :, :-1]).abs().mean() + (patch[:, 1:, :] - patch[:, :-1, :]).abs().mean()


def appearance_loss(patch, cfg):
    return F.relu(total_variation(patch) - cfg["tv_threshold"]) * cfg["tv_weight"]


def distillation_loss(student, teacher, clean, floor):
    # Compare sample-specific dense CHANGES, never a global averaged direction.
    reference = (teacher.detach().float() - clean.detach().float())
    scale = reference.square().mean().sqrt().clamp_min(floor)
    return F.mse_loss((student.float() - clean.detach().float()) / scale, reference / scale)


def rgb_metrics(patch, initial):
    value = patch.detach().float()
    return {"tv": float(total_variation(value)), "initial_rgb_mae": float((value - initial.to(value.device)).abs().mean()),
            "pixel_min": float(value.min()), "pixel_max": float(value.max())}


def frequency_filter(delta, strength=8., power=.75):
    h, w = delta.shape[-2:]
    fy = torch.fft.fftfreq(h, device=delta.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=delta.device)[None, :]
    gain = (1 + strength * (fx.square() + fy.square())).pow(-power)
    require(bool((gain > 0).all()), "No frequency may be removed")
    return torch.fft.irfft2(torch.fft.rfft2(delta.float()) * gain, s=(h, w))


def precondition_step(before, proposal, scale, cfg):
    delta = frequency_filter(proposal - before, cfg["frequency_strength"], cfg["frequency_power"])
    target = (proposal.clamp(0, 1) - before).norm()
    if scale is None and float(target) > 0 and float(delta.norm()) > 0:
        # Calibrate ONCE against actual clipped RGB step, not gradient magnitude.
        lo, hi = 0., 1.
        for _ in range(30):
            if float(((before + hi * delta).clamp(0, 1) - before).norm()) >= float(target):
                break
            hi *= 2
        require(float(((before + hi * delta).clamp(0, 1) - before).norm()) >= float(target),
                "Could not align initial RGB proposal")
        for _ in range(40):
            mid = (lo + hi) / 2
            if float(((before + mid * delta).clamp(0, 1) - before).norm()) < float(target):
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
    value = (before + (1. if scale is None else scale) * delta).clamp(0, 1)
    return value, scale


def clean_index(outer, size, seed):
    epoch, offset = divmod(outer - 1, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def scored_weights(clips, scores):
    weights = [1.] * len(clips)
    for task in range(10):
        for phase in ("early", "middle", "late"):
            ids = [i for i, r in enumerate(clips) if r["task_id"] == task and r["phase_proxy"] == phase]
            order = sorted(ids, key=lambda i: (-scores[i], i))
            for i in order[:(len(order) + 1) // 2]:
                if scores[i] > 0:
                    weights[i] = 2.
    return weights


def draw_record(bank, method, outer, weights, cfg):
    from common import method_flags
    flags = method_flags(method)
    base = bank["clean"][clean_index(outer, len(bank["clean"]), cfg["data_seed"] + cfg["seed"])]
    local = outer - cfg["bootstrap_outer"]
    if not flags["replay"] or local <= 0 or local % 2:
        return {"uids": [base], "origin": "clean"}
    task = bank["observations"][base]["task_id"]
    phase = ("early", "middle", "late")[(local // 2 - 1) % 3]
    ids = [i for i, r in enumerate(bank["clips"]) if r["task_id"] == task and r["phase_proxy"] == phase]
    # Very short successful replay may not contain three phases. Keep the task.
    if not ids:
        ids = [i for i, r in enumerate(bank["clips"]) if r["task_id"] == task]
    require(ids, "Missing task replay; never fall back to test data")
    p = [weights[i] for i in ids] if flags["weighted"] else None
    index = random.Random((cfg["data_seed"] + cfg["seed"]) * 1000000 + local).choices(ids, weights=p, k=1)[0]
    return {**bank["clips"][index], "origin": "replay", "clip_index": index}
