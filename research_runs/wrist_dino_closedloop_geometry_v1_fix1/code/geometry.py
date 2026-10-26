"""Direct uint8 wrist-stream transformations; edge padding prevents black occlusion."""
from __future__ import annotations
import numpy as np
from PIL import Image


def transform(image, candidate):
    image = np.asarray(image)
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError("Expected the native 224x224 uint8 wrist image")
    scale, dx, dy = (float(candidate[k]) for k in ("scale", "dx", "dy"))
    if not (0.9 <= scale <= 1.1 and abs(dx) <= 4 and abs(dy) <= 4):
        raise ValueError("Transformation outside the frozen geometry budget")
    if (scale, dx, dy) == (1., 0., 0.):
        return image.copy()
    pad, center = 32, 111.5
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    affine = (1 / scale, 0, center - (center + dx) / scale + pad,
              0, 1 / scale, center - (center + dy) / scale + pad)
    return np.asarray(Image.fromarray(padded).transform((224, 224), Image.Transform.AFFINE, affine,
                      resample=Image.Resampling.BILINEAR)).copy()


def observation_with_geometry(obs, candidate):
    # The official helper normalizes obs['state'] in place. Never share arrays.
    out = {key: np.asarray(value).copy() for key, value in obs.items()}
    out["wrist_image"] = transform(out["wrist_image"], candidate)
    return out


def pixel_metrics(clean, attacked):
    difference = np.asarray(attacked, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
    return {"rgb_mae_255": float(np.abs(difference).mean()), "rgb_max_abs_255": float(np.abs(difference).max()),
            "changed_pixel_fraction": float(np.any(difference != 0, axis=-1).mean())}
