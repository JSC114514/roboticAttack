#!/usr/bin/env python3
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from common import CONFIG_PATH, ROOT, atomic_json, atomic_torch, load_json, require, sha256_file


def score_map(value: torch.Tensor, size: int) -> torch.Tensor:
    score = value.float().squeeze()
    require(score.ndim == 1, f"Unexpected token-score shape: {tuple(value.shape)}")
    side = int(math.isqrt(score.numel()))
    if side * side != score.numel():
        score = score[1:]
        side = int(math.isqrt(score.numel()))
    require(side * side == score.numel(), f"DINO token count is not square: {score.numel()}")
    return F.interpolate(score.reshape(1, 1, side, side), size=(size, size), mode="bilinear", align_corners=False).squeeze()


def main() -> None:
    cfg = load_json(CONFIG_PATH)
    records, sources = [], []
    for task in range(cfg["task_count"]):
        path = ROOT / "outputs" / "state_bank" / f"task_{task:02d}.pt"
        require(path.is_file(), f"Missing state-bank task {task}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(len(payload["records"]) == 30, f"Incomplete task {task}")
        records.extend(payload["records"])
        sources.append({"task_id": task, "sha256": sha256_file(path)})
    require(len(records) == 300, "Expected 300 training states")
    heat = torch.stack([score_map(r["token_score"], cfg["image_resolution"]) for r in records]).mean(0)
    side = cfg["patch_shape"][1]
    sums = F.conv2d(heat.reshape(1, 1, *heat.shape), torch.ones(1, 1, side, side)).squeeze()
    flat = int(sums.argmax())
    width = sums.shape[1]
    model_y, model_x = divmod(flat, width)
    # DINO tokens describe the post-center-crop 224x224 image. The differentiable
    # and rollout renderers paste on the pre-crop 224x224 image, so map the window
    # origin through the official sqrt(0.9) center crop.
    crop_scale = math.sqrt(0.9)
    offset = cfg["image_resolution"] * (1.0 - crop_scale) / 2.0
    x = round(offset + model_x * crop_scale)
    y = round(offset + model_y * crop_scale)
    x = min(max(0, x), cfg["image_resolution"] - side)
    y = min(max(0, y), cfg["image_resolution"] - side)
    payload = {"experiment_id": cfg["experiment_id"], "records": records, "source_shards": sources,
               "selected_xy": [int(x), int(y)], "selected_model_xy": [int(model_x), int(model_y)],
               "selection": "argmax mean RANGE+LATERAL DINO token response in 50x50 post-crop window, mapped to pre-crop Wrist coordinates",
               "heatmap": heat, "config_sha256": sha256_file(CONFIG_PATH)}
    atomic_torch(payload, ROOT / "artifacts" / "GEOMETRY_TRAIN_BANK.pt")
    atomic_json({"status": "COMPLETE", "training_states": len(records), "selected_xy": [int(x), int(y)],
                 "selected_model_xy": [int(model_x), int(model_y)],
                 "source_shards": sources, "artifact_sha256": sha256_file(ROOT / "artifacts" / "GEOMETRY_TRAIN_BANK.pt")},
                ROOT / "reports" / "GEOMETRY_LOCATION.json")
    print(json.dumps({"status": "COMPLETE", "selected_xy": [int(x), int(y)]}, indent=2))


if __name__ == "__main__":
    main()
