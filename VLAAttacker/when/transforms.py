"""Digital patch transforms and protocol assertions for WHEN evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DIGITAL_PROTOCOLS


PROTOCOL_TO_VIEWS = {
    "digital_front": ("front",),
    "digital_wrist": ("wrist",),
    "digital_both": ("front", "wrist"),
}

VIEW_TO_OBS_KEY = {
    "front": "full_image",
    "wrist": "wrist_image",
}

UPA_RFAS_OFT_POSITIONS = {
    "libero_10": (5, 160),
    "libero_object": (0, 0),
    "libero_goal": (15, 158),
    "libero_spatial": (120, 160),
}


def views_for_protocol(protocol: str) -> tuple[str, ...]:
    """Return affected views for a digital protocol."""
    if protocol not in PROTOCOL_TO_VIEWS:
        raise ValueError(f"Unknown attack_protocol={protocol!r}; expected one of {DIGITAL_PROTOCOLS}")
    return PROTOCOL_TO_VIEWS[protocol]


def _patch_to_hwc_uint8(patch: Any) -> np.ndarray:
    """Convert a tensor/array patch to HWC uint8 RGB."""
    try:
        import torch

        if torch.is_tensor(patch):
            patch = patch.detach().cpu().float().clamp(0, 1).numpy()
    except Exception:
        pass
    arr = np.asarray(patch)
    if arr.ndim != 3:
        raise ValueError(f"patch must be CHW or HWC, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"patch must have 3 RGB channels, got shape={arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _select_xy(
    image_h: int,
    image_w: int,
    patch_h: int,
    patch_w: int,
    *,
    position: str,
    suite: str | None,
) -> tuple[int, int]:
    max_x = max(0, image_w - patch_w)
    max_y = max(0, image_h - patch_h)
    if position == "upa_rfas_default" and suite in UPA_RFAS_OFT_POSITIONS:
        x, y = UPA_RFAS_OFT_POSITIONS[str(suite)]
        return min(max(0, x), max_x), min(max(0, y), max_y)
    if position == "center":
        return max_x // 2, max_y // 2
    if position == "bottom_right":
        return max_x, max_y
    return 0, 0


def apply_patch_to_rgb(
    image: Any,
    patch: Any,
    *,
    position: str = "upa_rfas_default",
    suite: str | None = None,
) -> np.ndarray:
    """Paste an RGB patch onto a single HWC image."""
    out = np.asarray(image, dtype=np.uint8).copy()
    patch_np = _patch_to_hwc_uint8(patch)
    image_h, image_w = out.shape[:2]
    patch_h, patch_w = patch_np.shape[:2]
    if patch_h > image_h or patch_w > image_w:
        from PIL import Image

        scale = min(image_h / patch_h, image_w / patch_w)
        patch_h = max(1, int(round(patch_h * scale)))
        patch_w = max(1, int(round(patch_w * scale)))
        patch_np = np.asarray(Image.fromarray(patch_np).resize((patch_w, patch_h)), dtype=np.uint8)
    x, y = _select_xy(image_h, image_w, patch_h, patch_w, position=position, suite=suite)
    out[y : y + patch_h, x : x + patch_w] = patch_np
    return out


def apply_digital_patch_to_observation(
    observation: dict[str, Any],
    patch: Any,
    *,
    attack_protocol: str,
    suite: str | None = None,
    position: str = "upa_rfas_default",
) -> dict[str, Any]:
    """Apply a digital WHEN patch to front/wrist observation streams."""
    views = views_for_protocol(attack_protocol)
    patched = dict(observation)
    for view in views:
        key = VIEW_TO_OBS_KEY[view]
        if key not in patched:
            raise KeyError(f"{attack_protocol} requires observation key {key!r}")
        patched[key] = apply_patch_to_rgb(patched[key], patch, position=position, suite=suite)
    return patched


def assert_digital_protocol_effect(
    clean_observation: dict[str, Any],
    attacked_observation: dict[str, Any],
    *,
    attack_protocol: str,
) -> None:
    """Assert front/wrist streams changed exactly as the digital protocol says."""
    expected = set(views_for_protocol(attack_protocol))
    for view, key in VIEW_TO_OBS_KEY.items():
        if key not in clean_observation or key not in attacked_observation:
            continue
        same = np.array_equal(np.asarray(clean_observation[key]), np.asarray(attacked_observation[key]))
        if view in expected and same:
            raise AssertionError(f"{attack_protocol} should modify {key}, but it is unchanged")
        if view not in expected and not same:
            raise AssertionError(f"{attack_protocol} should leave {key} unchanged")
