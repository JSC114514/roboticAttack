"""Shared fixed/local projection with exclusion of every pasted support."""
from __future__ import annotations

import torch

from white_patch.oft_ila_objectives import dense_objectives


def alternating_projection(current, anchor, endpoint, current_coverage, anchor_coverage,
                           endpoint_coverage, coverage_threshold=0.01, min_reference_rms=1e-6, eps=1e-12):
    # dense_objectives excludes current and reference support. Union the frozen
    # local-origin support into the reference, so all THREE inputs are covered.
    if anchor_coverage is not None:
        if anchor_coverage.shape != endpoint_coverage.shape:
            raise ValueError("Anchor coverage shape drift")
        if not bool(torch.isfinite(anchor_coverage).all()
                    and (anchor_coverage >= 0).all() and (anchor_coverage <= 1).all()):
            raise ValueError("Invalid anchor coverage")
        endpoint_coverage = torch.maximum(anchor_coverage.detach(), endpoint_coverage.detach())
    loss, _, stats = dense_objectives(current, anchor, endpoint, current_coverage,
                                      endpoint_coverage, coverage_threshold, min_reference_rms, eps)
    return loss, stats
