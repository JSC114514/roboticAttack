#!/usr/bin/env python3
"""CPU-only immutable merge of ten source-task replay shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dino_replay_common import CONFIG, config, score_and_weights, source_records
from white_patch.UADA_OFT_DINO import atomic_json, atomic_torch, require, sha256_file


def build(config_path: Path, *, preflight: bool = False) -> dict:
    cfg = config(config_path)
    clean = source_records(cfg)
    output = Path(cfg["replay_bank"])
    status_path = output.with_name("STATUS.json")
    if output.exists() or status_path.exists():
        require(output.exists() and status_path.exists(), "Incomplete replay bank/status pair")
        status = json.loads(status_path.read_text())
        require(status["status"] == "COMPLETE" and status["bank_sha256"] == sha256_file(output)
                and status["config_sha256"] == sha256_file(config_path), "Existing replay bank drift")
        return status
    rows, receipts, episodes = [], {}, []
    for task_id in range(10):
        shard = output.parent / f"task_{task_id:02d}.pt"
        receipt_path = shard.with_suffix(".json")
        if not shard.exists() or not receipt_path.exists():
            if preflight:
                return {"status": "WAITING_FOR_REPLAY_SHARDS", "first_missing_task": task_id,
                        "config_sha256": sha256_file(config_path), "gpu_jobs_submitted": False}
            raise RuntimeError(f"Missing replay task shard: {task_id}")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "COMPLETE" and receipt["task_id"] == task_id
                and receipt["shard_sha256"] == sha256_file(shard)
                and receipt["config_sha256"] == sha256_file(config_path), "Replay shard receipt drift")
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        require(payload["task_id"] == task_id and payload["experiment_id"] == cfg["experiment_id"]
                and payload["config_sha256"] == sha256_file(config_path)
                and payload["baseline_patch_sha256"] == cfg["baseline_patch_sha256"],
                "Replay shard identity drift")
        task_episodes = payload["episodes"]
        expected = {int(r["state_index"]) for r in clean if int(r["task_id"]) == task_id}
        require(len(task_episodes) == 5 and {int(e["state_index"]) for e in task_episodes} == expected,
                "Replay episode source-state grid drift")
        ep_map = {int(e["state_index"]): e for e in task_episodes}
        task_rows = payload["records"]
        require(1 <= len(task_rows) <= 30 and len({(int(r["state_index"]), int(r["decision_index"]))
                                                for r in task_rows}) == len(task_rows),
                "Duplicate or missing replay decisions")
        for row in task_rows:
            state = int(row["state_index"])
            obs = row["observation"]
            require(int(row["task_id"]) == task_id and state in expected
                    and 0 <= int(row["decision_index"]) < int(ep_map[state]["decision_blocks"])
                    and bool(row["attacked_episode_success"]) == bool(ep_map[state]["success"])
                    and row["phase_proxy"] in ("early", "middle", "late"),
                    "Replay frame identity/phase drift")
            require({key: tuple(np.asarray(obs[key]).shape) for key in obs}
                    == {"full_image": (224, 224, 3), "wrist_image": (224, 224, 3), "state": (8,)},
                    "Replay observation shape drift")
            require(np.asarray(obs["full_image"]).dtype == np.uint8
                    and np.asarray(obs["wrist_image"]).dtype == np.uint8
                    and np.isfinite(obs["state"]).all()
                    and all(np.isfinite(float(row[key])) for key in
                            ("dino_score", "dino_delta_rms", "patched_target_mse", "restored_target_mse")),
                    "Invalid raw replay observation or score")
        require(all(bool(r["attacked_episode_success"]) for r in task_rows)
                if any(e["success"] for e in task_episodes) else True,
                "Survivor-priority selection drift")
        rows.extend(task_rows)
        episodes.extend(task_episodes)
        receipts[str(receipt_path)] = sha256_file(receipt_path)
    selected = score_and_weights(rows, int(cfg["data_seed"]))
    require(len(selected) <= 300 and {int(r["task_id"]) for r in selected} == set(range(10)),
            "Selected replay coverage drift")
    report = {"status": "COMPLETE", "experiment_id": cfg["experiment_id"],
              "config_sha256": sha256_file(config_path), "source_bank_sha256": cfg["training_bank_sha256"],
              "baseline_patch_sha256": cfg["baseline_patch_sha256"],
              "shard_receipt_sha256": receipts, "episodes": len(episodes), "records": len(selected),
              "survivor_episodes": sum(int(e["success"]) for e in episodes),
              "positive_scores": sum(r["dino_score"] > 0 for r in selected),
              "double_weight_records": sum(r["dino_weight"] == 2 for r in selected),
              "shuffled_mismatch_records": sum(r["dino_weight"] != r["shuffled_weight"] for r in selected)}
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
