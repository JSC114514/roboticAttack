"""Pure tensor objectives; no model loading, source mutation, or submission."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def token_coverage(mask: torch.Tensor, grid: int = 16) -> torch.Tensor:
    """Area proxy through the same 224 center-crop grid as the RGB adapter."""
    if mask.ndim != 2:
        raise ValueError("Expected one 2D image mask")
    x = mask.detach().float()[None, None]
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    axis = torch.linspace(-math.sqrt(0.9), math.sqrt(0.9), 224, device=x.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = F.grid_sample(x, torch.stack((xx, yy), -1)[None], mode="bilinear",
                      padding_mode="border", align_corners=True)
    if 224 % grid:
        raise ValueError("Token grid must divide processor image size")
    return F.avg_pool2d(x, 224 // grid).flatten().clamp(0, 1)


def shift_right(values: torch.Tensor, grid: int, dx: int = 1) -> torch.Tensor:
    """Translate a distribution, dropping boundary mass; never torch.roll."""
    if not 0 < dx < grid or values.shape[-1] != grid * grid:
        raise ValueError("Invalid spatial shift")
    image = values.reshape(*values.shape[:-1], grid, grid)
    out = torch.zeros_like(image)
    out[..., :, dx:] = image[..., :, :-dx]
    return out.reshape_as(values)


@dataclass
class CorrespondenceTeacher:
    queries: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    support: torch.Tensor
    target: torch.Tensor
    clean_probability: torch.Tensor
    temperature: float


def make_teacher(clean: torch.Tensor, object_mask: torch.Tensor, *, grid: int = 16,
                 max_queries: int = 16, min_coverage: float = 0.02,
                 radius: int = 2, dx: int = 1, temperature: float = 0.07) -> CorrespondenceTeacher:
    if clean.ndim != 3 or clean.shape[:2] != (1, grid * grid):
        raise ValueError("Expected one dense DINO feature grid without CLS token")
    if not temperature > 0 or object_mask.numel() != grid * grid:
        raise ValueError("Invalid teacher mask/temperature")
    clean = F.normalize(clean.detach().float()[0], dim=-1)
    weights = object_mask.detach().float().to(clean.device).flatten()
    # Stable sorting makes tied area fractions deterministic, unlike topk ties.
    indices = torch.argsort(weights, descending=True, stable=True)
    indices = indices[(weights[indices] >= min_coverage) & (indices % grid < grid - dx)][:max_queries]
    if indices.numel() == 0:
        raise ValueError("No visible object query for correspondence teacher")
    support = F.max_pool2d((weights >= min_coverage).float().reshape(1, 1, grid, grid),
                           2 * radius + 1, stride=1, padding=radius).flatten().bool()
    queries = clean[indices]
    logits = (queries @ clean.T) / temperature
    clean_p = logits.masked_fill(~support[None], -1e9).softmax(-1)
    target = shift_right(clean_p, grid, dx) * support[None]
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    return CorrespondenceTeacher(queries, indices, weights[indices], support,
                                 target.detach(), clean_p.detach(), temperature)


def visible_teacher(teacher: CorrespondenceTeacher, footprint: torch.Tensor,
                    threshold: float = 0.01):
    visible = footprint.detach().flatten().to(teacher.support.device) <= threshold
    if visible.numel() != teacher.support.numel():
        raise ValueError("Footprint/teacher shape mismatch")
    candidates = teacher.support & visible
    target = teacher.target * candidates[None]
    mass = target.sum(-1)
    keep = visible[teacher.indices] & (mass > 1e-8)
    if int(candidates.sum()) < 2 or not bool(keep.any()):
        return None
    return candidates, keep, target[keep] / mass[keep, None]


def correspondence_loss(attacked: torch.Tensor, teacher: CorrespondenceTeacher,
                        footprint: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
    visible = visible_teacher(teacher, footprint)
    if visible is None:
        return None, {"visible_queries": 0, "visible_candidates": 0}
    candidates, keep, target = visible
    features = F.normalize(attacked.float()[0], dim=-1)
    logits = teacher.queries[keep] @ features.T / teacher.temperature
    log_p = logits.masked_fill(~candidates[None], -1e9).log_softmax(-1)
    weights = teacher.weights[keep]
    loss = (-(target * log_p).sum(-1) * weights).sum() / weights.sum()
    grid = math.isqrt(features.shape[0])
    xs = (torch.arange(features.shape[0], device=features.device) % grid).float()
    clean_p = teacher.clean_probability[keep] * candidates[None]
    clean_p = clean_p / clean_p.sum(-1, keepdim=True).clamp_min(1e-12)
    displacement = ((log_p.exp() - clean_p) * xs).sum(-1)
    desired = ((target - clean_p) * xs).sum(-1)
    return loss, {"visible_queries": int(keep.sum()), "visible_candidates": int(candidates.sum()),
                  "matching_dx": float((displacement.detach() * weights).sum() / weights.sum()),
                  "target_dx": float((desired.detach() * weights).sum() / weights.sum())}


def feature_loss(attacked: torch.Tensor, clean: torch.Tensor, teacher: CorrespondenceTeacher,
                 footprint: torch.Tensor) -> torch.Tensor | None:
    # Same support and validity rule as the correspondence arm; no whole-image shortcut.
    visible = visible_teacher(teacher, footprint)
    if visible is None:
        return None
    candidates = visible[0]
    cosine = F.cosine_similarity(attacked.float(), clean.detach().float(), dim=-1)[0]
    return cosine[candidates].mean() - 1.0


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    return ((patch[..., :, 1:] - patch[..., :, :-1]).abs().mean()
            + (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean())


def appearance_loss(patch: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0:
        raise ValueError("TV threshold must be positive")
    return F.relu(total_variation(patch) / threshold - 1).square()


def calibrated_gradient(main: torch.Tensor, auxiliary: torch.Tensor, weight: float | None,
                        ratio: float) -> tuple[torch.Tensor, float | None, dict]:
    mn, an = main.float().norm(), auxiliary.float().norm()
    if not bool(torch.isfinite(mn) & torch.isfinite(an)):
        raise ValueError("Nonfinite objective gradient")
    if float(an) <= 1e-12 or float(mn) <= 1e-12:
        return main, weight, {"aux_ratio": 0.0, "aux_cosine": 0.0, "aux_active": 0.0}
    if weight is None:
        weight = ratio * float(mn / an)
    return main + weight * auxiliary, weight, {
        "aux_ratio": weight * float(an / mn),
        "aux_cosine": float(F.cosine_similarity(main.flatten().float(), auxiliary.flatten().float(), dim=0)),
        "aux_active": 1.0}
