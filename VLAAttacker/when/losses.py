"""Feature-space losses for WHEN / UPA-RFAS."""

from __future__ import annotations

from typing import Any


def feature_deviation_objective(adv_features: Any, clean_features: Any, *, use_l2: bool = False) -> Any:
    """Return a maximization objective that increases feature deviation."""
    import torch.nn.functional as F

    if use_l2:
        return F.mse_loss(adv_features, clean_features.detach(), reduction="mean")
    return F.l1_loss(adv_features, clean_features.detach(), reduction="mean")


def repulsive_infonce_objective(adv_features: Any, clean_features: Any, *, temperature: float = 0.07) -> Any:
    """Return the negative InfoNCE loss used as a repulsive maximization objective."""
    import torch
    import torch.nn.functional as F

    bsz = adv_features.shape[0]
    adv = F.normalize(adv_features.reshape(bsz, -1, adv_features.shape[-1]), dim=-1)
    clean = F.normalize(clean_features.detach().reshape(bsz, -1, clean_features.shape[-1]), dim=-1)
    logits = torch.bmm(adv, clean.transpose(1, 2)) / float(temperature)
    labels = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
    return -loss


def when_outer_objective(
    adv_features: Any,
    clean_features: Any,
    *,
    lambda_con: float = 1.0,
    lambda_1: float = 0.1,
    use_l2: bool = False,
) -> Any:
    """Feature part of the UPA-RFAS outer maximization objective."""
    return (
        float(lambda_con) * repulsive_infonce_objective(adv_features, clean_features)
        + float(lambda_1) * feature_deviation_objective(adv_features, clean_features, use_l2=use_l2)
    )
