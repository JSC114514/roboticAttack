"""Model-free, fixed contracts for Wrist-DINO Sampling + Clear RGB V1.

The hard operation restores feasibility; it is not a Euclidean projection.
All RGB pixels remain independent optimizer variables. DINO scores only
produce replay weights and never enter the training loss in this module.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
import random
from typing import Mapping

import torch
import torch.nn.functional as F


TV_THRESHOLD = .3
TV_INTERNAL_BUDGET = TV_THRESHOLD - 2 / 255
HF_THRESHOLD = 16 / 255
HF_INTERNAL_BUDGET = HF_THRESHOLD - 1 / 255
GAUSSIAN_SIGMA = .5
MAX_CORRECTION_PASSES = 3
TV_SCALE = .4
# A fixed numerical margin on an already-required shrink, not an extra budget.
# 1e-6 is smaller than accumulated float32 blur/blend roundoff near 15/255.
SHRINK_FLOAT_SAFETY = 1e-5
PHASES = ("early", "middle", "late")


def _fixed_config(cfg: Mapping | None) -> None:
    """Reject scientific drift while permitting unrelated trainer settings."""
    if cfg is None:
        return
    fixed = {
        "tv_threshold": TV_THRESHOLD,
        "tv_internal_budget": TV_INTERNAL_BUDGET,
        "hf_threshold": HF_THRESHOLD,
        "hf_internal_budget": HF_INTERNAL_BUDGET,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "gaussian_kernel_size": 3,
        "hard_correction_passes": MAX_CORRECTION_PASSES,
        "feasibility_passes": MAX_CORRECTION_PASSES,
        "hf_epsilon": HF_THRESHOLD,
        "tv_quantization_margin": 2 / 255,
        "hf_quantization_margin": 1 / 255,
        "tv_scale": TV_SCALE,
        "tv_weight": 1,
    }
    for name, expected in fixed.items():
        if name in cfg:
            value = float(cfg[name])
            if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Frozen appearance contract changed: {name}={value}")


def _validate_patch(patch: torch.Tensor) -> None:
    if not isinstance(patch, torch.Tensor) or patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError("Expected one RGB patch with shape [3,H,W]")
    if min(patch.shape[-2:]) < 2:
        raise ValueError("Reflect 3x3 blur and TV require H,W >= 2")
    if patch.dtype not in (torch.float32, torch.float64):
        raise ValueError("RGB constraints require float32 or float64 tensors")
    if not bool(torch.isfinite(patch).all()):
        raise ValueError("Nonfinite RGB patch")


@lru_cache(maxsize=32)
def _gaussian_kernel(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    values = [math.exp(-x * x / (2 * sigma * sigma)) for x in (-1, 0, 1)]
    normalizer = sum(values)
    values = [x / normalizer for x in values]
    # A first call during inference must not poison the cached kernel for a
    # later differentiable blur, which needs to save this tensor for backward.
    with torch.inference_mode(False), torch.no_grad():
        kernel = torch.tensor([[y * x for x in values] for y in values], dtype=dtype, device="cpu")
        # Normalize on CPU before copying the same weights to any device.
        kernel = kernel / kernel.sum()
        return kernel.to(device=device)


def gaussian_blur(patch: torch.Tensor, sigma: float = GAUSSIAN_SIGMA) -> torch.Tensor:
    """Differentiable reflect 3x3 Gaussian with fixed row-major arithmetic.

    Explicit weighted shifts avoid convolution backend/TF32 choices during
    the tight HF feasibility check. CPU and GPU receive the same normalized
    weights and the same sequence of nine separate multiply/add operations.
    """
    _validate_patch(patch)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    kernel = _gaussian_kernel(sigma, patch.dtype, patch.device)
    height, width = patch.shape[-2:]
    padded = F.pad(patch[None], (1, 1, 1, 1), mode="reflect")[0]
    result = torch.zeros_like(patch)
    for row in range(3):
        for column in range(3):
            weighted = padded[:, row:row + height, column:column + width] * kernel[row, column]
            result = result + weighted
    return result


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    """Legacy TV: horizontal and vertical mean absolute differences, summed."""
    return ((patch[..., :, 1:] - patch[..., :, :-1]).abs().mean()
            + (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean())


def high_frequency_max(patch: torch.Tensor, sigma: float = GAUSSIAN_SIGMA) -> torch.Tensor:
    """Maximum absolute per-channel pixel residual against the actual blur."""
    return (patch - gaussian_blur(patch, sigma)).abs().max()


@torch.no_grad()
def constraint_metrics(patch: torch.Tensor, cfg: Mapping | None = None) -> dict:
    """Python metrics for continuous RGB or the actual rounded PNG tensor."""
    _fixed_config(cfg)
    _validate_patch(patch)
    tv = float(total_variation(patch))
    hf = float(high_frequency_max(patch))
    low, high = float(patch.min()), float(patch.max())
    rgb = low >= 0 and high <= 1
    return {
        "tv": tv, "hf_max": hf, "rgb_min": low, "rgb_max": high, "finite": True,
        "feasible_internal": rgb and tv <= TV_INTERNAL_BUDGET and hf <= HF_INTERNAL_BUDGET,
        "feasible_scientific": rgb and tv <= TV_THRESHOLD and hf <= HF_THRESHOLD,
    }


@torch.no_grad()
def enforce_constraints(patch: torch.Tensor, cfg: Mapping | None = None) -> tuple[torch.Tensor, dict]:
    """Restore fixed joint RGB/TV/HF feasibility without modifying the input.

    A feasible input is bitwise unchanged. After RGB clipping, at most three
    local residual corrections recompute G; any remaining violation is removed
    by one necessary mean-preserving per-channel global shrink. The final
    actual values are checked, including the updated blur after each correction.
    """
    _fixed_config(cfg)
    _validate_patch(patch)
    original = patch.detach()
    before = constraint_metrics(original)
    result = original.clone()
    passes, scale = 0, 1.0
    if not before["feasible_internal"]:
        result = result.clamp(0, 1)
        current = constraint_metrics(result)
        while passes < MAX_CORRECTION_PASSES and current["hf_max"] > HF_INTERNAL_BUDGET:
            blurred = gaussian_blur(result)
            result = (blurred + (result - blurred).clamp(
                -HF_INTERNAL_BUDGET, HF_INTERNAL_BUDGET)).clamp(0, 1)
            passes += 1
            current = constraint_metrics(result)
        scale = min(1.0,
                    TV_INTERNAL_BUDGET / current["tv"] if current["tv"] > 0 else 1.0,
                    HF_INTERNAL_BUDGET / current["hf_max"] if current["hf_max"] > 0 else 1.0)
        if scale < 1:
            scale *= 1 - SHRINK_FLOAT_SAFETY
            mean = result.mean(dim=(-2, -1), keepdim=True)
            result = (mean + scale * (result - mean)).clamp(0, 1)
    final = constraint_metrics(result)
    if not final["feasible_internal"]:
        raise AssertionError(f"Joint RGB/TV/HF feasibility restoration failed: {final}")
    difference = result - original
    info = {
        "correction_applied": not torch.equal(result, original),
        "correction_passes": passes,
        "correction_distance": float(difference.norm()),
        "correction_mean_abs": float(difference.abs().mean()),
        "correction_rms": float(difference.square().mean().sqrt()),
        "correction_linf": float(difference.abs().max()),
        "correction_changed_fraction": float((difference != 0).float().mean()),
        "shrink_scale": scale,
        "before_metrics": before,
        "final_metrics": final,
    }
    return result.detach(), info


def soft_tv_loss(patch: torch.Tensor, cfg: Mapping | None = None) -> torch.Tensor:
    """Arm A's fixed-scale TV hinge; no HF loss or hard-constraint promise."""
    _fixed_config(cfg)
    return F.relu((total_variation(patch) - TV_THRESHOLD) / TV_SCALE).square()


def stable_score(delta1: float, delta2: float) -> float:
    """Clamp only after taking the minimum of both RAW action-score deltas."""
    values = [float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
              for x in (delta1, delta2)]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("DINO score deltas must be finite")
    return max(0.0, min(values))


def apply_score_weights(records: list[dict], seed: int, refresh_outer: int,
                        stable: bool = False) -> list[dict]:
    """Apply the legacy exact 2:1 rule and a local task/phase permutation.

    The trainer supplies each current-patch ``dino_score`` (already combined
    with ``stable_score`` for C/D). The upper ceil(group_size/2) ranks are
    intersected with positive scores, exactly as in dino_window_common.
    Every shuffled group uses its own current weights and a private RNG.
    """
    if not records:
        raise ValueError("Empty replay bank")
    if not isinstance(seed, int) or not isinstance(refresh_outer, int) or refresh_outer < 0:
        raise ValueError("Seed and nonnegative refresh outer must be integers")
    result = [dict(record) for record in records]
    groups = defaultdict(list)
    for index, record in enumerate(result):
        task, phase, score = int(record["task_id"]), record["phase_proxy"], float(record["dino_score"])
        if not 0 <= task < 10 or phase not in PHASES or not math.isfinite(score):
            raise ValueError("Invalid task, phase, or current DINO score")
        if stable and score < 0:
            raise ValueError("Stable DINO scores must be clamped after both raw views")
        groups[(task, phase)].append(index)
    for (task, phase), ids in sorted(groups.items()):
        order = sorted(ids, key=lambda i: (-float(result[i]["dino_score"]),
                                           int(result[i]["state_index"]),
                                           int(result[i]["decision_index"])))
        strong = {i for i in order[:(len(order) + 1) // 2]
                  if float(result[i]["dino_score"]) > 0}
        weights = [2.0 if i in strong else 1.0 for i in ids]
        shuffled = weights.copy()
        random.Random(seed + refresh_outer * 1_000_000 + 100 * task + PHASES.index(phase)).shuffle(shuffled)
        for index, weight, control in zip(ids, weights, shuffled, strict=True):
            result[index]["dino_weight"] = weight
            result[index]["shuffled_weight"] = control
    return result
