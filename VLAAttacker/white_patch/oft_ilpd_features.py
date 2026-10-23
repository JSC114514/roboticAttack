"""Temporary Wrist-only feature transforms and full-RGB appearance projection."""
from __future__ import annotations

from contextlib import contextmanager

import torch

METHODS = ("uada_bounded", "dino_gradscale_bounded", "dino_ilpd_bounded", "siglip_ilpd_bounded")


def transform_dense(value, anchor, alpha, backward_only=False):
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[:2] != (1, 256):
        raise RuntimeError("Expected consumed Wrist dense features [1,256,C]")
    if not 0 <= alpha <= 1 or not bool(torch.isfinite(value).all()):
        raise RuntimeError("Invalid feature coefficient/value")
    if alpha == 1:
        return value
    if backward_only:
        # Bitwise-identical forward; only the local derivative changes.
        return value.detach() + alpha * (value - value.detach())
    if anchor is None or anchor.shape != value.shape or not bool(torch.isfinite(anchor).all()):
        raise RuntimeError("Clean feature anchor missing/invalid")
    clean = anchor.detach().to(device=value.device, dtype=value.dtype)
    # Arithmetic in float32, returning exactly the dtype consumed by OFT.
    return (clean.float() + alpha * (value.float() - clean.float())).to(value.dtype)


@contextmanager
def wrist_feature_transform(backbone, method, anchors, alpha):
    """Capture both towers, but modify only call 1 of the selected tower.

    Each full OFT forward must encode Front then Wrist exactly once. Handles
    are removed even if the model or a downstream action head raises.
    """
    if method not in METHODS:
        raise RuntimeError("Unknown ILPD method")
    selected = "siglip" if method == METHODS[3] else "dino"
    capture, handles = {"dino": [], "siglip": []}, []
    def make_hook(branch):
        def hook(module, args, value):
            index = len(capture[branch])
            if index >= 2:
                raise RuntimeError("Unexpected encoder call count/order")
            output = value
            if index == 1 and branch == selected and method != METHODS[0]:
                output = transform_dense(value, anchors.get(branch), alpha, method == METHODS[1])
            capture[branch].append({"raw": value, "forward": output})
            return output
        return hook
    try:
        for branch, module in (("dino", backbone.featurizer), ("siglip", backbone.fused_featurizer)):
            handles.append(module.register_forward_hook(make_hook(branch)))
        yield capture
        if any(len(capture[branch]) != 2 for branch in capture):
            raise RuntimeError("Expected exactly two Front/Wrist calls in each encoder")
    finally:
        for handle in handles:
            handle.remove()


def project_rgb_(patch, initial, radius):
    """Projected AdamW update; all 7500 pixels remain optimization variables."""
    if not 0 < radius <= 1 or patch.shape != initial.shape:
        raise RuntimeError("Invalid RGB projection box")
    with torch.no_grad():
        patch.copy_(torch.maximum(torch.minimum(patch, (initial + radius).clamp(0, 1)),
                                  (initial - radius).clamp(0, 1)))


def box_metrics(patch, initial, radius):
    delta = (patch.detach() - initial).abs()
    if not bool(torch.isfinite(patch).all()) or float(delta.max()) > radius + 1e-7:
        raise RuntimeError("RGB patch left the frozen appearance box")
    low, high = (initial - radius).clamp(0, 1), (initial + radius).clamp(0, 1)
    on_boundary = (patch.detach() - low).abs() < 1e-7
    on_boundary |= (patch.detach() - high).abs() < 1e-7
    return {"rgb_max_abs_delta": float(delta.max()), "box_boundary_fraction": float(on_boundary.float().mean())}
