"""ILA-inspired directional and matched direction-free dense-feature objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dense_objectives(current, clean, reference, current_coverage, reference_coverage,
                     coverage_threshold=0.01, min_reference_rms=1e-6, eps=1e-12):
    """Return differentiable losses, detached diagnostics; teachers never get grads.

    Unit-normalize tokens before flattening: increasing a token's norm cannot
    improve either loss. Exclude the union of both actual RGB paste supports.
    A direction too small to define safely disables BOTH matched objectives.
    """
    if current.ndim != 3 or current.shape[0] != 1 or clean.shape != current.shape or reference.shape != current.shape:
        raise ValueError("Expected matching batch-one dense features")
    if not all(bool(torch.isfinite(x).all()) for x in (current, clean, reference)):
        raise ValueError("Nonfinite dense features")
    tokens = current.shape[1]
    if current_coverage.shape != (tokens,) or reference_coverage.shape != (tokens,):
        raise ValueError("Dense coverage shape drift")
    if not all(bool(torch.isfinite(x).all() and (x >= 0).all() and (x <= 1).all())
               for x in (current_coverage, reference_coverage)):
        raise ValueError("Invalid paste coverage")
    visible = ((current_coverage.detach() <= coverage_threshold)
               & (reference_coverage.detach() <= coverage_threshold))
    count = int(visible.sum())
    stats = {"visible_tokens": count, "reference_rms": 0., "projection_score": 0.,
             "direction_cosine": 0., "delta_rms": 0., "feature_active": 0.,
             "support_disagreement_tokens": int(((current_coverage > coverage_threshold)
                                                 != (reference_coverage > coverage_threshold)).sum())}
    if not count:
        return None, None, stats
    z = F.normalize(current.float()[0], dim=-1)[visible]
    z0 = F.normalize(clean.detach().float()[0], dim=-1)[visible]
    zr = F.normalize(reference.detach().float()[0], dim=-1)[visible]
    d, delta = (zr - z0).detach(), z - z0
    rms = float(d.square().mean().sqrt())
    stats["reference_rms"] = rms
    if rms <= min_reference_rms:
        return None, None, stats
    projection = (delta * d).sum() / d.square().sum().clamp_min(eps)
    disruption = (z * z0).sum(-1).mean() - 1.
    stats.update(projection_score=float(projection.detach()), feature_active=1.,
                 direction_cosine=float(F.cosine_similarity(delta.detach().flatten(), d.flatten(), dim=0)),
                 delta_rms=float(delta.detach().square().mean().sqrt()))
    return -projection, disruption, stats
