#!/usr/bin/env python3
"""Four-task masked DINO spatial-readout pilot; no patch training or Slurm submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(cfg: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for task_id in cfg["task_ids"]:
        path = Path(cfg["source_root"]) / f"task_{task_id}" / "records.parquet"
        require(path.is_file(), f"Missing masked source shard: {path}")
        frame = pd.read_parquet(path)
        require(set(frame.task_id.astype(int)) == {task_id}, f"Task identity drift: {path}")
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    require(set(records.split) == set(cfg["splits"]), "Source split drift")
    require(not records.duplicated(["task_id", "state_id"]).any(), "Duplicate source state")
    require(records.clean_success.eq(1).all(), "Source contains non-clean-success state")
    require(records.array_path.map(lambda value: Path(value).is_file()).all(), "Missing source array")
    return records.sort_values(["task_id", "split", "state_id"]).reset_index(drop=True)


def transform_mask(mask: np.ndarray, image_size: int = 224, grid: int = 16) -> torch.Tensor:
    """Nearest-label counterpart of resize then sqrt(.9)-side center crop."""
    require(mask.ndim == 2, "Expected 2D oracle mask")
    x = torch.as_tensor(mask.astype(np.float32))[None, None]
    x = F.interpolate(x, size=(image_size, image_size), mode="nearest")
    scale = math.sqrt(0.9)
    axis = torch.linspace(-scale, scale, image_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    x = F.grid_sample(x, torch.stack((xx, yy), dim=-1)[None], mode="nearest",
                      padding_mode="border", align_corners=True)
    return F.avg_pool2d(x, kernel_size=image_size // grid)[0, 0].reshape(-1)


def token_centroid(weights: torch.Tensor, grid: int = 16) -> torch.Tensor:
    require(weights.numel() == grid * grid, "Token grid mismatch")
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing="ij")
    coords = torch.stack((xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5), dim=-1)
    norm = weights.float().sum()
    require(float(norm) > 0, "Empty token distribution")
    return (weights.float().reshape(-1, 1) * coords).sum(dim=0) / norm


def patch_exclusion(cfg: dict[str, Any]) -> torch.Tensor:
    grid, size = int(cfg["token_grid"]), int(cfg["image_resolution"])
    x, y = map(int, cfg["patch_xy"])
    w, h = map(int, cfg["patch_shape"])
    halo = int(cfg["patch_token_halo"])
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    x0, x1 = x * grid / size - halo, (x + w) * grid / size + halo
    y0, y1 = y * grid / size - halo, (y + h) * grid / size + halo
    return ((xx + 0.5 >= x0) & (xx + 0.5 <= x1) &
            (yy + 0.5 >= y0) & (yy + 0.5 <= y1)).reshape(-1)


def readout(features: torch.Tensor, prototypes: dict[str, torch.Tensor],
            temperature: float, exclude: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    features = F.normalize(features.float(), dim=-1)
    output = {}
    for role in ("target_object", "gripper"):
        score = features @ prototypes[role] - features @ prototypes["matched_background"]
        if exclude is not None:
            score = score.masked_fill(exclude.to(score.device), -1e4)
        output[role] = token_centroid(torch.softmax(score / temperature, dim=0))
    return output


def prepare_array(row: Any, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    with np.load(row.array_path, allow_pickle=False) as arrays:
        rgb = np.asarray(arrays["wrist_rgb"]).copy()
        masks = {role: transform_mask(np.asarray(arrays[f"wrist_pixel_mask_{role}"]),
                                      int(cfg["image_resolution"]), int(cfg["token_grid"]))
                 for role in ("target_object", "gripper", "matched_background")}
    require(rgb.shape == (256, 256, 3), f"Wrist source image shape drift: {rgb.shape}")
    require(all(float(mask.sum()) > 0 for mask in masks.values()), f"Empty oracle mask: {row.state_id}")
    return rgb, masks


def gpu_features(wrapper: Any, image: np.ndarray) -> torch.Tensor:
    from VLAAttacker.multiview.openvla_oft_experiments import _processor_normalize_tensor
    raw = torch.as_tensor(image, device=wrapper.device, dtype=torch.float32)
    raw = raw.permute(2, 0, 1)[None].div(255)
    pixels = _processor_normalize_tensor(raw, wrapper.processor.image_processor, center_crop=True)
    with torch.inference_mode():
        result = wrapper.vla.vision_backbone.featurizer(pixels[:, :3].to(dtype=wrapper.dtype))
    require(tuple(result.shape) == (1, 256, 1024), f"Unexpected DINO features: {tuple(result.shape)}")
    return result[0].float().cpu()


def run(cfg: dict[str, Any], records: pd.DataFrame) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one Slurm GPU required")
    model_cfg = json.loads(Path(cfg["model_config"]).read_text())
    from VLAAttacker.UADA_OFT_DINO_wrapper import load_wrapper
    wrapper = load_wrapper(model_cfg)
    require("3090" in torch.cuda.get_device_name(0), "Stage 0a is pinned to RTX3090")

    mgwp_cfg = json.loads(Path(cfg["mgwp_config"]).read_text())
    historical = Path(mgwp_cfg["paths"]["historical_root"])
    for path in (historical / "dgwga_v1", historical):
        sys.path.insert(0, str(path))
    from dgwga.warp import warp_rgb_cuda
    warp_cfg = json.loads(Path(mgwp_cfg["paths"]["warp_config"]).read_text())

    cache: dict[str, tuple[np.ndarray, dict[str, torch.Tensor], torch.Tensor]] = {}
    sums: dict[int, dict[str, torch.Tensor]] = {}
    mass: dict[int, dict[str, float]] = {}
    for row in records.itertuples(index=False):
        rgb, masks = prepare_array(row, cfg)
        features = gpu_features(wrapper, rgb)
        cache[str(row.state_id)] = (rgb, masks, features)
        if row.split != "discovery":
            continue
        task_sums = sums.setdefault(int(row.task_id), {})
        task_mass = mass.setdefault(int(row.task_id), {})
        for role, mask in masks.items():
            task_sums[role] = task_sums.get(role, torch.zeros(features.shape[1])) + (features * mask[:, None]).sum(0)
            task_mass[role] = task_mass.get(role, 0.0) + float(mask.sum())
    prototypes = {task: {role: F.normalize(value / mass[task][role], dim=0)
                         for role, value in sums[task].items()} for task in sums}
    require(set(prototypes) == set(map(int, cfg["task_ids"])), "Discovery prototype coverage drift")
    exclude = patch_exclusion(cfg)

    rows = []
    for row in records.itertuples(index=False):
        rgb, masks, features = cache[str(row.state_id)]
        task_prototypes = prototypes[int(row.task_id)]
        truth = {role: token_centroid(masks[role]) for role in ("target_object", "gripper")}
        for exclusion_name, excluded in (("full", None), ("outside_patch", exclude)):
            predicted = readout(features, task_prototypes, float(cfg["prototype_temperature"]), excluded)
            obj_error = float(torch.linalg.vector_norm(predicted["target_object"] - truth["target_object"]))
            grip_error = float(torch.linalg.vector_norm(predicted["gripper"] - truth["gripper"]))
            item = {"task_id": int(row.task_id), "state_id": str(row.state_id), "split": str(row.split),
                    "exclusion": exclusion_name, "object_error_tokens": obj_error,
                    "gripper_error_tokens": grip_error,
                    "object_oracle_mass": float(masks["target_object"].sum()),
                    "gripper_oracle_mass": float(masks["gripper"].sum())}
            if row.split != "discovery":
                delta = {}
                for name in ("range", "lateral"):
                    altered, _ = warp_rgb_cuda(rgb, cfg=warp_cfg, **cfg[f"{name}_geometry"])
                    altered_features = gpu_features(wrapper, np.asarray(altered, dtype=np.uint8))
                    altered_readout = readout(altered_features, task_prototypes,
                                              float(cfg["prototype_temperature"]), excluded)
                    delta[name] = (altered_readout["target_object"] - altered_readout["gripper"]
                                   - predicted["target_object"] + predicted["gripper"])
                    item[f"{name}_relative_dx"] = float(delta[name][0])
                    item[f"{name}_relative_dy"] = float(delta[name][1])
                    item[f"{name}_relative_norm"] = float(torch.linalg.vector_norm(delta[name]))
                oracle_relative = truth["target_object"] - truth["gripper"]
                item["range_radial_projection"] = float(torch.dot(delta["range"], oracle_relative)
                                                          / torch.linalg.vector_norm(oracle_relative).clamp_min(1e-6))
            rows.append(item)

    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "STATE_READOUTS.csv", index=False)
    heldout = result.loc[result.split.isin(("validation", "test")) & result.exclusion.eq("outside_patch")]
    task_stats = heldout.groupby("task_id").agg(
        object_median_error_tokens=("object_error_tokens", "median"),
        gripper_median_error_tokens=("gripper_error_tokens", "median"),
        range_median_radial_projection=("range_radial_projection", "median"),
        lateral_median_relative_norm=("lateral_relative_norm", "median"),
    )
    task_stats.to_csv(output / "TASK_READOUTS.csv")
    status = {
        "status": "COMPLETE_FEASIBILITY_ONLY", "experiment_id": cfg["experiment_id"],
        "gpu": torch.cuda.get_device_name(0), "source_tasks": list(map(int, cfg["task_ids"])),
        "source_rows": len(records), "heldout_rows": len(heldout),
        "median_object_error_tokens": float(heldout.object_error_tokens.median()),
        "median_gripper_error_tokens": float(heldout.gripper_error_tokens.median()),
        "tasks_positive_range_radial": int((task_stats.range_median_radial_projection > 0).sum()),
        "lateral_relative_median_norm": float(heldout.lateral_relative_norm.median()),
        "pass_four_task_localization": bool(heldout.object_error_tokens.median() <= cfg["max_median_error_tokens"]
                                            and heldout.gripper_error_tokens.median() <= cfg["max_median_error_tokens"]),
        "pass_four_task_range_sign": bool((task_stats.range_median_radial_projection > 0).sum()
                                          >= cfg["minimum_tasks_with_signed_range_response"]),
        "next_gate": "Must repeat on all 10 Object tasks and test DINO-restored action; no patch training authorized by this report",
        "source_hashes": {"model_config": sha256_file(Path(cfg["model_config"])),
                          "mgwp_config": sha256_file(Path(cfg["mgwp_config"]))},
    }
    (output / "STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    records = load_records(cfg)
    if args.check_only:
        counts = records.groupby(["task_id", "split"]).size().to_dict()
        print(json.dumps({"status": "CHECK_ONLY_PASS", "rows": len(records),
                          "counts": {f"{task}:{split}": int(count) for (task, split), count in counts.items()},
                          "gpu_job_submitted": False}, ensure_ascii=False, indent=2))
        return
    run(cfg, records)


if __name__ == "__main__":
    main()
