#!/usr/bin/env python3
"""Collect exact, audited Wrist masks for one task of the frozen Object bank.

Run only through a user-submitted single-3090 Slurm shard. No patch training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from replay_object_bank_mask_pilot import (
    CONFIG_PATH, MGWP_ROOT, PROJECT_ROOT, load_json, load_wrapper,
    make_segmentation_env, replay, require, require_gpu, select_records, sha256_file,
)
from probe_dino_spatial_relation_stage0a import patch_exclusion, transform_mask


OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/spatial_relation_object_bank_masks"
PILOT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/spatial_relation_object_bank_mask_pilot"
FREEZE_SEED = 20260922
MASK_ROLES = ("target_object", "gripper", "matched_background")


def holdout_rank(task_id: int) -> int:
    """Freeze one independent trajectory per task before reading masks.

    Task 0 excludes rank 0 because its pilot masks were already inspected.
    """
    require(0 <= task_id < 10, "Object task outside range")
    candidates = np.arange(1, 5) if task_id == 0 else np.arange(5)
    return int(np.random.default_rng(FREEZE_SEED + task_id).choice(candidates))


def task_records(bank: dict[str, Any], task_id: int) -> list[list[dict[str, Any]]]:
    require(len(bank["records"]) == 300, "Object bank must contain 300 frozen states")
    groups = [select_records(bank, task_id, rank) for rank in range(5)]
    initial_states = [int(group[0]["state_index"]) for group in groups]
    require(len(set(initial_states)) == 5, f"Repeated initial state in task {task_id}")
    require(len({group[0]["task_name"] for group in groups}) == 1, "Task language drift across trajectories")
    return groups


def audit_pilot_reference(task_id: int, rank: int, captured: list[dict[str, Any]], bank_hash: str) -> None:
    if task_id != 0 or rank != 0:
        return
    pilot = PILOT_ROOT / "task_0/trajectory_rank_0/RECEIPT.json"
    require(pilot.is_file(), "Task 0 reference pilot receipt missing")
    receipt = json.loads(pilot.read_text())
    require(receipt["status"] == "EXACT_OBJECT_REPLAY_MASK_PILOT_COMPLETE"
            and receipt["bank_sha256"] == bank_hash, "Task 0 pilot lineage mismatch")
    by_decision = {int(sample["decision_index"]): sample for sample in receipt["samples"]}
    require(set(by_decision) == {int(item["decision_index"]) for item in captured},
            "Task 0 pilot decision coverage drift")
    for item in captured:
        archived = by_decision[int(item["decision_index"])]
        require(sha256_file(archived["array_path"]) == archived["array_sha256"],
                "Task 0 pilot NPZ hash drift")
        with np.load(archived["array_path"], allow_pickle=False) as arrays:
            require(np.array_equal(arrays["wrist_rgb"], item["wrist_rgb"]),
                    "Task 0 pilot Wrist RGB drift")
            for role in MASK_ROLES:
                require(np.array_equal(arrays[f"wrist_pixel_mask_{role}"], item["masks"][role]),
                        f"Task 0 pilot {role} mask drift")


def sample_quality(sample: dict[str, Any], exclusion: torch.Tensor) -> dict[str, Any]:
    masks = sample["masks"]
    require(set(masks) == set(MASK_ROLES), "Unexpected mask roles")
    for role, mask in masks.items():
        require(mask.shape == (256, 256) and mask.dtype == np.uint8,
                f"Invalid {role} mask shape/dtype")
        require(np.isin(mask, [0, 1]).all(), f"Non-binary {role} mask")
    require(sample["wrist_rgb"].shape == (256, 256, 3)
            and sample["wrist_rgb"].dtype == np.uint8, "Invalid Wrist RGB")
    require(not np.any(masks["target_object"] & masks["gripper"]),
            "Target/gripper instance overlap")
    details = {"pixel_area": {role: int(mask.sum()) for role, mask in masks.items()},
               "visible": dict(sample["mask_validity"])}
    for role in ("target_object", "gripper"):
        tokens = transform_mask(masks[role])
        total = float(tokens.sum())
        outside = float(tokens.masked_fill(exclusion, 0).sum())
        details[f"{role}_token_mass"] = total
        details[f"{role}_outside_patch_token_mass"] = outside
        details[f"{role}_patch_excluded_fraction"] = 1.0 - outside / total if total > 0 else None
    details["usable_for_spatial_readout"] = all(
        details[f"{role}_outside_patch_token_mass"] > 0 for role in ("target_object", "gripper")
    ) and details["pixel_area"]["matched_background"] > 0
    return details


def collect_task(cfg: dict[str, Any], bank: dict[str, Any], task_id: int) -> dict[str, Any]:
    require("3090" in torch.cuda.get_device_name(0), "Object mask shards use RTX 3090")
    final = OUTPUT_ROOT / f"task_{task_id:02d}"
    require(not final.exists(), f"Refusing to overwrite existing Object mask shard: {final}")
    groups = task_records(bank, task_id)
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank_hash = sha256_file(bank_path)
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, task_id)
    require(task.language == groups[0][0]["task_name"], "Simulator/Object bank task language mismatch")
    captured_groups = []
    try:
        for rank, group in enumerate(groups):
            captured, metadata = replay(cfg, group, task_id,
                                        resources=(wrapper, env, task, initial_states))
            audit_pilot_reference(task_id, rank, captured, bank_hash)
            captured_groups.append((rank, captured, metadata))
            print(f"EXACT_REPLAY_PASS task={task_id} trajectory_rank={rank} "
                  f"state_index={metadata['initial_state_index']} captured={len(captured)}", flush=True)
    finally:
        env.close()
    require(len(captured_groups) == 5 and sum(len(items) for _, items, _ in captured_groups) == 30,
            "Incomplete exact Object replay")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exclusion = patch_exclusion({"token_grid": 16, "image_resolution": 224,
                                 "patch_xy": [26, 137], "patch_shape": [50, 50],
                                 "patch_token_halo": 1})
    holdout = holdout_rank(task_id)
    receipt = {"status": "PENDING", "suite": cfg["suite"], "checkpoint": cfg["paths"]["checkpoint"],
               "task_id": task_id, "task_language": task.language,
               "bank_sha256": bank_hash, "source_config_sha256": sha256_file(CONFIG_PATH),
               "holdout_freeze_seed": FREEZE_SEED, "heldout_trajectory_rank": holdout,
               "independent_trajectories": 5, "expected_states": 30,
               "gpu": torch.cuda.get_device_name(0), "trajectories": [], "samples": []}
    with tempfile.TemporaryDirectory(prefix=f".task_{task_id:02d}_", dir=OUTPUT_ROOT) as temporary:
        staging = Path(temporary)
        for rank, captured, metadata in captured_groups:
            split = "test" if rank == holdout else "discovery"
            receipt["trajectories"].append({"trajectory_rank": rank, "split": split,
                                            **metadata, "sample_count": len(captured)})
            for sample in captured:
                quality = sample_quality(sample, exclusion)
                name = f"trajectory_{rank}_decision_{sample['decision_index']:03d}.npz"
                path = staging / name
                np.savez_compressed(path, wrist_rgb=sample["wrist_rgb"],
                                    **{f"wrist_pixel_mask_{role}": mask
                                       for role, mask in sample["masks"].items()})
                receipt["samples"].append({"trajectory_rank": rank, "split": split,
                                           "initial_state_index": metadata["initial_state_index"],
                                           "decision_index": sample["decision_index"],
                                           "phase_proxy": sample["phase_proxy"],
                                           "replay_metrics": sample["metrics"], "quality": quality,
                                           "array_path": str(final / name),
                                           "array_sha256": sha256_file(path)})
        require(len(receipt["samples"]) == 30, "Task shard sample count drift")
        usable = sum(int(sample["quality"]["usable_for_spatial_readout"]) for sample in receipt["samples"])
        receipt["usable_spatial_states"] = usable
        receipt["heldout_usable_spatial_states"] = sum(
            int(sample["quality"]["usable_for_spatial_readout"])
            for sample in receipt["samples"] if sample["split"] == "test"
        )
        receipt["status"] = "EXACT_OBJECT_MASK_SHARD_COMPLETE" if usable == 30 else "EXACT_OBJECT_MASK_SHARD_PARTIAL_VISIBILITY"
        (staging / "RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        require(not final.exists(), f"Object mask shard appeared concurrently: {final}")
        os.replace(staging, final)
    print(json.dumps({key: receipt[key] for key in
                      ("status", "task_id", "independent_trajectories", "expected_states",
                       "usable_spatial_states", "heldout_trajectory_rank",
                       "heldout_usable_spatial_states", "bank_sha256")}, indent=2), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and cfg["unnorm_key"] == "libero_object_no_noops",
            "Mask collection requires LIBERO-Object source and model")
    bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt", map_location="cpu", weights_only=False)
    groups = task_records(bank, args.task_id)
    report = {"status": "CHECK_ONLY_PASS", "task_id": args.task_id,
              "task_language": groups[0][0]["task_name"],
              "initial_state_indices": [int(group[0]["state_index"]) for group in groups],
              "heldout_trajectory_rank": holdout_rank(args.task_id),
              "source_suite": cfg["suite"], "source_checkpoint": cfg["paths"]["checkpoint"],
              "source_bank_rows": len(bank["records"]), "sample_count": sum(map(len, groups)),
              "gpu_job_submitted_by_script": False}
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(json.dumps(require_gpu()), flush=True)
    collect_task(cfg, bank, args.task_id)


if __name__ == "__main__":
    main()
