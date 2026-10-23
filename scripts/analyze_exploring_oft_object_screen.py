#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAMP_SCRIPTS = ROOT.parent / "oft_ramp_recovery_aware" / "scripts"
sys.path.insert(0, str(RAMP_SCRIPTS))

from common import atomic_json, atomic_parquet, load_json, require, sha256_file  # noqa: E402
from statistics import paired_contrast_bootstrap, task_episode_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = Path(args.config).resolve()
    cfg = load_json(config)
    output = Path(cfg["output_root"])
    shards, trace_shards = [], []
    for task_id in range(int(cfg["task_count"])):
        rpath = output / f"task_{task_id:02d}_rollouts.parquet"
        tpath = output / f"task_{task_id:02d}_traces.parquet"
        require(rpath.is_file() and tpath.is_file(), f"Missing Object screen shard {task_id}")
        shards.append(pd.read_parquet(rpath))
        trace_shards.append(pd.read_parquet(tpath))
    rollouts = pd.concat(shards, ignore_index=True)
    traces = pd.concat(trace_shards, ignore_index=True)
    conditions = list(cfg["conditions"])
    expected = int(cfg["task_count"]) * len(cfg["state_indices"]) * len(conditions)
    require(len(rollouts) == expected and set(rollouts.condition) == set(conditions),
            "Combined Object screen grid drift")
    require(not rollouts.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate combined Object screen rollout")
    require(rollouts.frozen_clean_success.isin([0, 1]).all(), "Invalid clean eligibility")
    eligible = rollouts.loc[rollouts.frozen_clean_success == 1].copy()
    eligible["casr"] = eligible.failure.astype(float)

    atomic_parquet(rollouts, output / "OBJECT_SCREEN_ROLLOUTS.parquet")
    atomic_parquet(traces, output / "OBJECT_SCREEN_TRACES.parquet")
    atomic_parquet(eligible, output / "OBJECT_SCREEN_ELIGIBLE.parquet")
    summary = task_episode_bootstrap(
        eligible, "casr", "condition", 10_000, int(cfg["bootstrap_seed"]))
    summary_path = output / "OBJECT_SCREEN_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)
    contrast = paired_contrast_bootstrap(
        eligible, "casr", "condition", conditions[1], conditions[0], 10_000,
        int(cfg["bootstrap_seed"]) + 1)
    contrast["contrast_role"] = "DINO_CF_MINUS_UADA"
    contrast_path = output / "OBJECT_SCREEN_CONTRAST.csv"
    pd.DataFrame([contrast]).to_csv(contrast_path, index=False)
    (eligible.groupby(["task_id", "condition"], as_index=False)
     .agg(clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
          attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean"))
     .to_csv(output / "OBJECT_SCREEN_PER_TASK.csv", index=False))
    status = {
        "status": "COMPLETE", "experiment_id": cfg["experiment_id"],
        "new_episodes": len(rollouts), "eligible_rows": len(eligible),
        "conditions": conditions, "manual_visual_review": cfg["manual_visual_review"],
        "summary_sha256": sha256_file(summary_path),
        "contrast_sha256": sha256_file(contrast_path),
        "claim_boundary": "Object outer-200 screen only; no cross-suite or full-reproduction claim",
    }
    atomic_json(status, output / "STATUS.json")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

