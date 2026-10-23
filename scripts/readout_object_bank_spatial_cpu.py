#!/usr/bin/env python3
"""Frozen Object-bank DINO spatial readout; CPU-only, no patch or Slurm job.

Fit task-specific role prototypes on four discovery trajectories per task and
evaluate on the pre-frozen fifth trajectory. The six frames in a trajectory
are repeated measurements, not six independent replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collect_object_bank_spatial_masks import (
    CONFIG_PATH, MASK_ROLES, MGWP_ROOT, OUTPUT_ROOT, holdout_rank,
    sample_quality, task_records,
)
from probe_dino_spatial_relation_stage0a import (
    patch_exclusion, readout, token_centroid, transform_mask,
)
from replay_object_bank_mask_pilot import require, sha256_file


RESULT_ROOT = OUTPUT_ROOT.parent / "spatial_relation_object_bank_cpu_readout"
TEMPERATURE = 0.05  # Frozen before examining this Object heldout readout.
EXCLUDE = patch_exclusion({"token_grid": 16, "image_resolution": 224,
                           "patch_xy": [26, 137], "patch_shape": [50, 50],
                           "patch_token_halo": 1})


def load_manifest(bank: dict, *, verify_arrays: bool = True) -> list[dict]:
    require(bank.get("config_sha256") == sha256_file(CONFIG_PATH), "Bank/config lineage mismatch")
    bank_hash = sha256_file(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt")
    manifest = []
    for task_id in range(10):
        groups = task_records(bank, task_id)
        path = OUTPUT_ROOT / f"task_{task_id:02d}" / "RECEIPT.json"
        require(path.is_file(), f"Missing Object mask receipt: {path}")
        receipt = json.loads(path.read_text())
        require(receipt["status"] == "EXACT_OBJECT_MASK_SHARD_COMPLETE"
                and receipt["bank_sha256"] == bank_hash
                and receipt["source_config_sha256"] == bank["config_sha256"],
                f"Object mask lineage/status mismatch task {task_id}")
        require(receipt["heldout_trajectory_rank"] == holdout_rank(task_id),
                f"Holdout split drift task {task_id}")
        by_key = {(int(s["trajectory_rank"]), int(s["decision_index"])): s
                  for s in receipt["samples"]}
        require(len(by_key) == 30 and len(receipt["samples"]) == 30,
                f"Sample identity drift task {task_id}")
        for rank, group in enumerate(groups):
            split = "test" if rank == holdout_rank(task_id) else "discovery"
            for row in group:
                key = rank, int(row["decision_index"])
                sample = by_key[key]
                require(sample["split"] == split
                        and int(sample["initial_state_index"]) == int(row["state_index"]),
                        f"Mask/bank trajectory mismatch {task_id}:{key}")
                require(sample["quality"]["usable_for_spatial_readout"],
                        f"Unusable spatial mask {task_id}:{key}")
                array_path = Path(sample["array_path"])
                require(array_path.is_file(), f"Missing mask array {array_path}")
                if verify_arrays:
                    require(sha256_file(array_path) == sample["array_sha256"],
                            f"Mask array hash drift {array_path}")
                manifest.append({"task_id": task_id, "rank": rank, "split": split,
                                 "row": row, "mask_path": array_path,
                                 "mask_sha256": sample["array_sha256"]})
    require(len(manifest) == 300
            and sum(item["split"] == "discovery" for item in manifest) == 240
            and sum(item["split"] == "test" for item in manifest) == 60,
            "Object discovery/test coverage drift")
    return manifest


def state_arrays(item: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = item["row"]["clean_dino"]
    require(isinstance(features, torch.Tensor) and tuple(features.shape) == (1, 256, 1024),
            "Unexpected frozen DINO feature shape")
    with np.load(item["mask_path"], allow_pickle=False) as arrays:
        require(set(arrays.files) == {"wrist_rgb"} | {f"wrist_pixel_mask_{r}" for r in MASK_ROLES},
                "Mask array schema drift")
        masks = {role: transform_mask(np.asarray(arrays[f"wrist_pixel_mask_{role}"]))
                 .masked_fill(EXCLUDE, 0) for role in MASK_ROLES}
    require(all(float(mask.sum()) > 0 for mask in masks.values()), "Empty outside-patch role mask")
    return features[0].float(), masks


def fit_prototypes(manifest: list[dict]) -> dict[int, dict[str, torch.Tensor]]:
    sums = defaultdict(lambda: {role: torch.zeros(1024) for role in MASK_ROLES})
    masses = defaultdict(lambda: {role: 0.0 for role in MASK_ROLES})
    for item in manifest:
        if item["split"] != "discovery":
            continue
        features, masks = state_arrays(item)
        task_id = item["task_id"]
        for role in MASK_ROLES:
            sums[task_id][role] += (features * masks[role][:, None]).sum(0)
            masses[task_id][role] += float(masks[role].sum())
    require(set(sums) == set(range(10)), "Missing task prototype")
    return {task: {role: F.normalize(sums[task][role] / masses[task][role], dim=0)
                   for role in MASK_ROLES} for task in range(10)}


def evaluate(manifest: list[dict], prototypes: dict[int, dict[str, torch.Tensor]]) -> list[dict]:
    results = []
    for item in manifest:
        if item["split"] != "test":
            continue
        features, masks = state_arrays(item)
        truth = {role: token_centroid(masks[role]) for role in ("target_object", "gripper")}
        predicted = readout(features, prototypes[item["task_id"]], TEMPERATURE, EXCLUDE)
        relative_truth = truth["target_object"] - truth["gripper"]
        relative_pred = predicted["target_object"] - predicted["gripper"]
        results.append({"task_id": item["task_id"], "trajectory_rank": item["rank"],
                        "decision_index": int(item["row"]["decision_index"]),
                        "phase_proxy": item["row"]["phase_proxy"],
                        "object_error_tokens": float(torch.linalg.vector_norm(
                            predicted["target_object"] - truth["target_object"])),
                        "gripper_error_tokens": float(torch.linalg.vector_norm(
                            predicted["gripper"] - truth["gripper"])),
                        "relative_error_tokens": float(torch.linalg.vector_norm(
                            relative_pred - relative_truth)),
                        "true_relative_xy": relative_truth.tolist(),
                        "predicted_relative_xy": relative_pred.tolist()})
    require(len(results) == 60, "Incomplete heldout readout")
    return results


def summarize(results: list[dict]) -> dict:
    per_task = []
    for task_id in range(10):
        rows = [row for row in results if row["task_id"] == task_id]
        require(len(rows) == 6 and len({r["trajectory_rank"] for r in rows}) == 1,
                "Heldout trajectory structure drift")
        per_task.append({"task_id": task_id, "trajectory_rank": rows[0]["trajectory_rank"],
                         **{f"median_{name}": float(np.median([r[name] for r in rows]))
                            for name in ("object_error_tokens", "gripper_error_tokens",
                                         "relative_error_tokens")}})
    return {"status": "CPU_LOCALIZATION_GATE_COMPLETE", "heldout_trajectories": 10,
            "heldout_states": 60, "temperature": TEMPERATURE,
            "outside_patch_token_exclusion": True,
            "median_object_error_tokens": float(np.median([r["object_error_tokens"] for r in results])),
            "median_gripper_error_tokens": float(np.median([r["gripper_error_tokens"] for r in results])),
            "median_relative_error_tokens": float(np.median([r["relative_error_tokens"] for r in results])),
            "pass_predeclared_localization_gate": bool(
                np.median([r["object_error_tokens"] for r in results]) <= 2.0
                and np.median([r["gripper_error_tokens"] for r in results]) <= 2.0),
            "per_task": per_task,
            "interpretation_limit": "Localization alone does not establish geometry response, action mediation or attack success."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(bank)
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "bank_sha256": sha256_file(bank_path),
                          "mask_shards": 10, "states": len(manifest),
                          "discovery": sum(x["split"] == "discovery" for x in manifest),
                          "test": sum(x["split"] == "test" for x in manifest),
                          "gpu_job_submitted": False}, indent=2))
        return
    require(not RESULT_ROOT.exists(), f"Refusing to overwrite readout: {RESULT_ROOT}")
    prototypes = fit_prototypes(manifest)
    results = evaluate(manifest, prototypes)
    summary = summarize(results)
    summary["bank_sha256"] = sha256_file(bank_path)
    summary["mask_receipt_sha256"] = {
        str(task): sha256_file(OUTPUT_ROOT / f"task_{task:02d}" / "RECEIPT.json")
        for task in range(10)}
    RESULT_ROOT.mkdir(parents=True)
    (RESULT_ROOT / "HELDOUT_STATES.json").write_text(json.dumps(results, indent=2) + "\n")
    (RESULT_ROOT / "STATUS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
