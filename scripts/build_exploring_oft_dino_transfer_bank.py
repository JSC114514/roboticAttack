#!/usr/bin/env python3
"""Merge immutable 3090 shards into one clip bank with matched score controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dino_transfer_common import CONFIG, clip_indices, collection_config, score_and_weights, source_records
from white_patch.UADA_OFT_DINO import atomic_json, atomic_torch, require, sha256_file


def build(config_path: Path = CONFIG, *, preflight: bool = False) -> dict:
    cfg = collection_config(config_path, require_prefix=not preflight)
    if preflight and not Path(cfg["baseline_patch"]).is_file():
        return {"status": "WAITING_FOR_SCRATCH_PREFIX", "gpu_jobs_submitted": False}
    clean = source_records(cfg)
    output = Path(cfg["replay_bank"])
    status_path = output.with_name("STATUS.json")
    if output.exists() or status_path.exists():
        require(output.exists() and status_path.exists(), "Incomplete window bank/status pair")
        status = json.loads(status_path.read_text())
        require(status["status"] == "COMPLETE" and status["bank_sha256"] == sha256_file(output)
                and status["config_sha256"] == sha256_file(config_path), "Existing window bank drift")
        return status
    rows, receipts, episodes = [], {}, []
    for task_id in range(10):
        shard = output.parent / f"task_{task_id:02d}.pt"
        receipt_path = shard.with_suffix(".json")
        if not shard.is_file() or not receipt_path.is_file():
            if preflight:
                return {"status": "WAITING_FOR_WINDOW_SHARDS", "first_missing_task": task_id,
                        "gpu_jobs_submitted": False}
            raise RuntimeError(f"Missing window shard: {task_id}")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "COMPLETE" and receipt["task_id"] == task_id
                and receipt["shard_sha256"] == sha256_file(shard)
                and receipt["config_sha256"] == sha256_file(config_path),
                "Window shard receipt drift")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        require(payload["task_id"] == task_id and payload["experiment_id"] == cfg["experiment_id"]
                and payload["config_sha256"] == sha256_file(config_path)
                and payload["baseline_patch_sha256"] == cfg["baseline_patch_sha256"],
                "Window shard identity drift")
        task_episodes = payload["episodes"]
        expected = {int(r["state_index"]) for r in clean if int(r["task_id"]) == task_id}
        require(len(task_episodes) == 5 and {int(e["state_index"]) for e in task_episodes} == expected,
                "Window source initial state grid drift")
        ep_map = {int(e["state_index"]): e for e in task_episodes}
        survivors = {int(e["state_index"]) for e in task_episodes if e["success"]}
        task_rows = payload["records"]
        require(1 <= len(task_rows) <= 30 and
                len({(int(r["state_index"]), int(r["decision_index"])) for r in task_rows}) == len(task_rows),
                "Duplicate or missing window center")
        for row in task_rows:
            state = int(row["state_index"])
            center = int(row["decision_index"])
            require(int(row["task_id"]) == task_id and state in expected
                    and (not survivors or state in survivors)
                    and bool(row["attacked_episode_success"]) == bool(ep_map[state]["success"])
                    and row["phase_proxy"] in ("early", "middle", "late")
                    and tuple(row["window_indices"]) ==
                    clip_indices(center, int(ep_map[state]["decision_blocks"])),
                    "Window trajectory/phase/decision drift")
            require(len(row["observations"]) == 3, "Window must contain three decisions")
            for obs in row["observations"]:
                require({key: tuple(np.asarray(obs[key]).shape) for key in obs}
                        == {"full_image": (224, 224, 3), "wrist_image": (224, 224, 3),
                            "state": (8,)}, "Window observation shape drift")
                require(np.asarray(obs["full_image"]).dtype == np.uint8
                        and np.asarray(obs["wrist_image"]).dtype == np.uint8
                        and np.isfinite(obs["state"]).all(), "Invalid raw window observation")
            require(all(np.isfinite(float(row[key])) for key in
                        ("dino_score", "dino_delta_rms", "patched_target_mse", "restored_target_mse")),
                    "Invalid DINO restoration score")
        rows.extend(task_rows)
        episodes.extend(task_episodes)
        receipts[str(receipt_path)] = sha256_file(receipt_path)
    selected = score_and_weights(rows, int(cfg["data_seed"]))
    report = {"status": "COMPLETE", "experiment_id": cfg["experiment_id"],
              "config_sha256": sha256_file(config_path),
              "source_bank_sha256": cfg["training_bank_sha256"],
              "baseline_patch_sha256": cfg["baseline_patch_sha256"],
              "shard_receipt_sha256": receipts, "episodes": len(episodes),
              "records": len(selected),
              "survivor_episodes": sum(int(e["success"]) for e in episodes),
              "positive_scores": sum(r["dino_score"] > 0 for r in selected),
              "double_weight_records": sum(r["dino_weight"] == 2 for r in selected),
              "shuffled_mismatch_records": sum(r["dino_weight"] != r["shuffled_weight"] for r in selected)}
    if report["positive_scores"] == 0 or report["shuffled_mismatch_records"] == 0:
        atomic_json({**report, "status": "NO_USABLE_DINO_RANKING"},
                    output.with_name("DIAGNOSTIC_STOP.json"))
        raise RuntimeError("No usable DINO ranking in new clips")
    if preflight:
        return {**report, "status": "READY_TO_BUILD", "gpu_jobs_submitted": False}
    atomic_torch({"records": selected, "episodes": episodes, "report": report}, output)
    report["bank_sha256"] = sha256_file(output)
    atomic_json(report, status_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.config, preflight=args.preflight), indent=2))
