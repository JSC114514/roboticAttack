"""Patch Semantic Misalignment helpers."""

from __future__ import annotations

from typing import Any


def pool_patch_tokens(visual_tokens: Any, patch_token_mask: Any, *, eps: float = 1e-6) -> Any:
    """Pool and L2-normalize patch-covered visual tokens."""
    import torch.nn.functional as F

    mask = patch_token_mask.float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    pooled = (visual_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(float(eps))
    return F.normalize(pooled, dim=-1)


def patch_semantic_misalignment_objective(
    patch_tokens: Any,
    patch_token_mask: Any,
    probe_prototypes: Any,
    instruction_embedding: Any,
    *,
    tau: float = 0.3,
    alpha: float = 1.0,
    beta_neg: float = 0.5,
) -> Any:
    """PSM objective: pull patch features to probes and push away from instruction."""
    import torch
    import torch.nn.functional as F

    patch_vec = pool_patch_tokens(patch_tokens, patch_token_mask)
    probes = F.normalize(probe_prototypes, dim=-1)
    instruction = F.normalize(instruction_embedding, dim=-1)
    pull = torch.logsumexp((patch_vec @ probes.T) / float(tau), dim=-1).mean()
    push = (patch_vec * instruction).sum(dim=-1).mean()
    return float(alpha) * pull - float(beta_neg) * push
