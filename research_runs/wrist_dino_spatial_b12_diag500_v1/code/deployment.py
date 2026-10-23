"""Normal RGB PNG deployment; no training hook, geometry, or sampling."""
from __future__ import annotations
import numpy as np
from common import require

def paste_png(observation, rgb, xy=(26, 137)):
    result = dict(observation)
    wrist = np.asarray(observation["wrist_image"]).copy()
    require(wrist.shape == (224, 224, 3) and wrist.dtype == np.uint8, "Deployment RGB shape changed")
    require(rgb.shape == (50, 50, 3) and rgb.dtype == np.uint8, "Deployment patch must be uint8 RGB50")
    x, y = xy
    require(0 <= x <= 174 and 0 <= y <= 174, "Deployment patch outside image")
    wrist[y:y + 50, x:x + 50] = rgb
    result["wrist_image"] = wrist
    return result
