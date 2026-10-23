from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import require, sha256_file


def render_latent(latent: torch.Tensor) -> torch.Tensor:
    require(tuple(latent.shape) == (3, 8, 8), f"Latent shape drift: {tuple(latent.shape)}")
    patch = F.interpolate(latent.unsqueeze(0), size=(50, 50), mode="bilinear", align_corners=False).squeeze(0)
    require(tuple(patch.shape) == (3, 50, 50), "Rendered patch shape drift")
    return patch.clamp(0.0, 1.0)


def load_learned(path: str | Path, device: torch.device) -> tuple[torch.Tensor, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    latent = payload["latent"].to(device=device, dtype=torch.float32)
    return render_latent(latent), sha256_file(path)


def load_official(path: str | Path, device: torch.device) -> tuple[torch.Tensor, str]:
    target = Path(path)
    if target.suffix.lower() == ".png":
        array = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32) / 255.0
        patch = torch.as_tensor(array, device=device).permute(2, 0, 1)
    else:
        value: Any = torch.load(target, map_location=device, weights_only=True)
        if isinstance(value, dict):
            value = value.get("patch", value.get("rgb_patch", value))
        patch = torch.as_tensor(value, device=device, dtype=torch.float32)
        if patch.ndim == 3 and patch.shape[-1] == 3:
            patch = patch.permute(2, 0, 1)
    require(tuple(patch.shape) == (3, 50, 50), f"Official patch shape drift: {tuple(patch.shape)}")
    return patch.clamp(0, 1), sha256_file(target)


def action_metrics(clean: np.ndarray, attacked: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    raw_delta = np.asarray(attacked, dtype=np.float64) - np.asarray(clean, dtype=np.float64)
    scale_array = np.asarray(scale, dtype=np.float64)
    if scale_array.shape == (7,):
        scale_array = scale_array.reshape(1, 7)
    require(scale_array.shape in {(1, 7), raw_delta.shape}, f"Action scale shape mismatch: {scale_array.shape} vs {raw_delta.shape}")
    delta = raw_delta / scale_array
    require(delta.shape[1] == 7 and np.isfinite(delta).all(), "Invalid action delta")
    return {
        "action_l2": float(np.linalg.norm(delta)),
        "translation_l2": float(np.linalg.norm(delta[:, :3])),
        "dx_abs": float(np.mean(np.abs(delta[:, 0]))),
        "dy_abs": float(np.mean(np.abs(delta[:, 1]))),
        "dz_abs": float(np.mean(np.abs(delta[:, 2]))),
        "rotation_l2": float(np.linalg.norm(delta[:, 3:6])),
        "gripper_abs": float(np.mean(np.abs(delta[:, 6]))),
    }
