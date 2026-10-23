"""Patch Attention Dominance helpers."""

from __future__ import annotations

from typing import Any


def downsample_mask_to_vis_tokens(mask: Any, num_tokens: int) -> Any:
    """Downsample an image-space patch mask to a visual-token mask."""
    import math
    import torch.nn.functional as F

    side = int(math.sqrt(int(num_tokens)))
    if side * side != int(num_tokens):
        raise ValueError(f"num_tokens must be square for mask downsample, got {num_tokens}")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    token_mask = F.interpolate(mask.float(), size=(side, side), mode="area")
    return token_mask.flatten(1).clamp(0, 1)


def patch_attention_dominance_objective(
    clean_attn: Any,
    adv_attn: Any,
    patch_token_mask: Any,
    *,
    lambda_nonpatch: float = 0.8,
    margin: float = 0.1,
) -> Any:
    """A compact PAD objective based on patch vs non-patch attention deltas."""
    import torch
    import torch.nn.functional as F

    delta = adv_attn - clean_attn.detach()
    mask = patch_token_mask.float()
    while mask.ndim < delta.ndim:
        mask = mask.unsqueeze(1)
    patch_delta = (delta * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    non_mask = 1.0 - mask
    non_delta = (delta * non_mask).sum(dim=-1) / non_mask.sum(dim=-1).clamp_min(1.0)
    rank = F.relu(float(margin) - (patch_delta - non_delta)).mean()
    return patch_delta.mean() - float(lambda_nonpatch) * F.relu(non_delta).mean() - rank
