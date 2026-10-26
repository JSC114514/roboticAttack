"""WHEN's actual tokenwise contrastive core and an eight-state mean gradient."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from when_core import BiContrastiveLoss

def feature_objective(clean, perturbed, coverage, temperature=.07, lambda_con=10.):
    if clean.shape != perturbed.shape or clean.ndim != 3 or clean.shape[0] != 1:
        raise ValueError("Expected one image's actual clean/perturbed dense tokens")
    if coverage.shape != (clean.shape[1],):
        raise ValueError("Footprint/token alignment changed")
    valid = coverage <= 0
    if int(valid.sum()) < 2:
        raise ValueError("Insufficient non-patch tokens")
    clean = clean.detach().float()[:, valid]
    perturbed = perturbed.float()[:, valid]
    l1 = F.l1_loss(perturbed, clean)
    criterion = BiContrastiveLoss(initial_temp=temperature, final_temp=temperature, total_batches=50)
    contrast = criterion(perturbed, clean, bicos=1, single=True)
    loss = -(l1 + lambda_con * contrast)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite WHEN-DINO loss")
    return loss, {"feature_l1":float(l1.detach()), "feature_contrast":float(contrast.detach()),
                  "feature_valid_tokens":int(valid.sum())}

def mean_gradient(patch, losses, expected=8):
    """Consume independent loss graphs; average, then the caller takes ONE step."""
    total = torch.zeros_like(patch)
    values = []
    for loss in losses:
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite state loss")
        grad = torch.autograd.grad(loss, patch)[0]
        if not bool(torch.isfinite(grad).all()):
            raise ValueError("Nonfinite state gradient")
        total.add_(grad / expected)
        values.append(float(loss.detach()))
    if len(values) != expected:
        raise ValueError("Effective batch count changed")
    return total, sum(values) / expected
