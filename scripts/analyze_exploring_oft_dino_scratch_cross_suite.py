#!/usr/bin/env python3
"""Same-budget CASR contrasts and checkpoint curves; one training seed."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_exploring_oft_dino_scratch_cross_suite import (
    CONFIG, METHODS, CHECKPOINTS, CONDITIONS, DENOMINATORS, EXPERIMENT_ID,
    load_config, preflight, frozen_clean, validate_result, require, atomic_json, sha256_file,
)
from analyze_exploring_oft_dino_window_cross_suite import stats, STATS_PATH
from white_patch.UADA_OFT_DINO import atomic_parquet

CONTRASTS = tuple((f"{left}_minus_{right}_{step}", f"{left}_{step}", f"{right}_{step}")
                 for step in CHECKPOINTS for left, right in
                 (("dino", "uada"), ("dino", "uniform"), ("uniform", "uada"))) + tuple(
    (f"{method}_1000_minus_500", f"{method}_1000", f"{method}_500") for method in METHODS)


def analyze():
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training/visual review incomplete")
    output = Path(cfg["output_root"])
    parts, trace_parts = [], []
    for suite in cfg["suites"]:
        for task in range(10):
            folder = output / suite
            rows = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
            traces = pd.read_parquet(folder / f"task_{task:02d}_traces.parquet")
            validate_result(rows, traces, suite, task, cfg)
            parts.append(rows)
            trace_parts.append(traces)
    frame = pd.concat(parts, ignore_index=True)
    require(len(frame) == 3600 and not frame.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Incomplete/duplicate grid")
    summaries, contrasts, per_tasks, eligible_parts = [], [], [], []
    for i, suite in enumerate(cfg["suites"]):
        eligible = frame.loc[(frame.suite == suite) & (frame.frozen_clean_success == 1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        require(len(eligible) == DENOMINATORS[suite] * len(CONDITIONS), "Eligible count drift")
        eligible_parts.append(eligible)
        summary = stats.task_episode_bootstrap(eligible, "casr", "condition", 10000,
                                               cfg["bootstrap_seed"] + i)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        task = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean"))
        task.insert(0, "suite", suite)
        per_tasks.append(task)
        for j, (role, left, right) in enumerate(CONTRASTS):
            value = stats.paired_contrast_bootstrap(eligible, "casr", "condition", left, right,
                                                    10000, cfg["bootstrap_seed"] + 100 + i * 100 + j)
            require(value["paired_episodes"] == DENOMINATORS[suite], "Pair count changed")
            contrasts.append({"suite": suite, "contrast_role": role, **value})
    prefix = "DINO_SCRATCH_XSUITE"
    eligible = pd.concat(eligible_parts, ignore_index=True)
    require(len(eligible) == 3438, "Eligible total changed")
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    summary_path, contrast_path = (output / f"{prefix}_{name}.csv" for name in ("PER_SUITE", "CONTRASTS"))
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(per_tasks, ignore_index=True).to_csv(output / f"{prefix}_PER_TASK.csv", index=False)
    train = json.loads(CONFIG.read_text())
    root = Path(train["output_root"])
    collection_s = sum(json.loads(p.read_text())["elapsed_s"] for p in (root / "data").glob("task_*.json"))
    costs = []
    for method in METHODS:
        status = json.loads((root / "training" / method / "status.json").read_text())
        rows = pd.read_parquet(root / "training" / method / "training.parquet")
        for step in CHECKPOINTS:
            elapsed = float(rows.loc[rows.outer_iteration == step, "elapsed_s"].iloc[0])
            costs.append({"condition": f"{method}_{step}", "total_outer": step,
                          "optimizer_updates": step * 50, "training_including_prefix_s": elapsed,
                          "shared_prefix_s": status["prefix_elapsed_s"],
                          "shared_collection_and_scoring_gpu_s": collection_s if method != "uada" else 0.})
    pd.DataFrame(costs).to_csv(output / f"{prefix}_COSTS.csv", index=False)
    report = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "new_attack_episodes": len(frame), "eligible_attack_rows": len(eligible),
              "conditions": CONDITIONS, "training_seed_count": 1, "checkpoints_are_repeated_measures": True,
              "common_counted_prefix_outer": 250, "historical_trained_patch_used": False,
              "primary_suites": cfg["suites"][:3], "long_secondary": True,
              "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(summary_path), "contrasts_sha256": sha256_file(contrast_path)}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
