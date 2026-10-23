#!/usr/bin/env python3
"""Fixed-position and spatial-token-shuffle controls for Object DINO readout."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from readout_object_bank_spatial_cpu import (
    EXCLUDE, MGWP_ROOT, OUTPUT_ROOT, RESULT_ROOT, TEMPERATURE,
    fit_prototypes, load_manifest, readout, require, state_arrays,
    token_centroid,
)
from replay_object_bank_mask_pilot import sha256_file


CONTROL_ROOT = RESULT_ROOT.parent / "spatial_relation_object_bank_cpu_controls"


def evaluate_controls(manifest: list[dict]) -> dict:
    sums = defaultdict(lambda: {role: torch.zeros(2) for role in ("target_object", "gripper")})
    counts = defaultdict(int)
    for item in manifest:
        if item["split"] != "discovery":
            continue
        _, masks = state_arrays(item)
        for role in ("target_object", "gripper"):
            sums[item["task_id"]][role] += token_centroid(masks[role])
        counts[item["task_id"]] += 1
    require(all(counts[task] == 24 for task in range(10)), "Discovery prior coverage drift")
    prior = {task: {role: sums[task][role] / counts[task]
                    for role in ("target_object", "gripper")}
             for task in range(10)}
    prototypes = fit_prototypes(manifest)
    rows = []
    for item in manifest:
        if item["split"] != "test":
            continue
        features, masks = state_arrays(item)
        truth = {role: token_centroid(masks[role]) for role in ("target_object", "gripper")}
        predicted = readout(features, prototypes[item["task_id"]], TEMPERATURE, EXCLUDE)
        seed = 20260922 + 1000 * item["task_id"] + 100 * item["rank"] + int(item["row"]["decision_index"])
        permutation = torch.randperm(256, generator=torch.Generator().manual_seed(seed))
        shuffled = readout(features[permutation], prototypes[item["task_id"]], TEMPERATURE, EXCLUDE)
        for role in ("target_object", "gripper"):
            rows.append({"task_id": item["task_id"], "trajectory_rank": item["rank"],
                         "decision_index": int(item["row"]["decision_index"]), "role": role,
                         "dino_error": float(torch.linalg.vector_norm(predicted[role] - truth[role])),
                         "position_prior_error": float(torch.linalg.vector_norm(prior[item["task_id"]][role] - truth[role])),
                         "token_shuffle_error": float(torch.linalg.vector_norm(shuffled[role] - truth[role]))})
    require(len(rows) == 120, "Control heldout coverage drift")
    per_task = []
    for task in range(10):
        item = {"task_id": task}
        for role in ("target_object", "gripper"):
            subset = [r for r in rows if r["task_id"] == task and r["role"] == role]
            for metric in ("dino_error", "position_prior_error", "token_shuffle_error"):
                item[f"{role}_{metric}_median"] = float(np.median([r[metric] for r in subset]))
        per_task.append(item)
    summary = {"status": "CPU_SPATIAL_CONTROLS_COMPLETE", "heldout_states": 60,
               "heldout_trajectories": 10,
               "controls": "discovery-only per-task coordinate prior and seeded within-image token shuffle",
               "temperature": TEMPERATURE,
               "per_task": per_task,
               "source_readout_sha256": sha256_file(RESULT_ROOT / "STATUS.json")}
    for role in ("target_object", "gripper"):
        for metric in ("dino_error", "position_prior_error", "token_shuffle_error"):
            summary[f"{role}_{metric}_median"] = float(np.median(
                [r[metric] for r in rows if r["role"] == role]))
        summary[f"tasks_dino_better_than_prior_{role}"] = sum(
            p[f"{role}_dino_error_median"] < p[f"{role}_position_prior_error_median"]
            for p in per_task)
        summary[f"tasks_dino_better_than_shuffle_{role}"] = sum(
            p[f"{role}_dino_error_median"] < p[f"{role}_token_shuffle_error_median"]
            for p in per_task)
    return {"summary": summary, "heldout_rows": rows}


def main() -> None:
    require((RESULT_ROOT / "STATUS.json").is_file(), "Primary readout missing")
    require(not CONTROL_ROOT.exists(), "Refusing to overwrite control result")
    bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt", map_location="cpu", weights_only=False)
    result = evaluate_controls(load_manifest(bank))
    CONTROL_ROOT.mkdir(parents=True)
    (CONTROL_ROOT / "HELDOUT_CONTROLS.json").write_text(json.dumps(result["heldout_rows"], indent=2) + "\n")
    (CONTROL_ROOT / "STATUS.json").write_text(json.dumps(result["summary"], indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
