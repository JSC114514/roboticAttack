#!/usr/bin/env python3
"""Task-equal paired CASR at matched 1000/2000 training checkpoints."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from run_exploring_oft_dino_transfer_cross_suite import (
    ARMS, CONDITIONS, DENOMINATORS, EXPERIMENT_ID, ORIGINAL_CONFIG, ROOT,
    config, frozen_clean, load_config, preflight, require, sha256_file, validate,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet

STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_transfer2000_stats", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)


def analyze(draws=10000):
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training incomplete")
    output = Path(cfg["output_root"])
    parts, trace_parts = [], []
    for suite in cfg["suites"]:
        for task in range(10):
            folder = output / suite
            rows = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
            traces = pd.read_parquet(folder / f"task_{task:02d}_traces.parquet")
            validate(rows, traces, suite, task, cfg)
            require(rows.experiment_id.eq(EXPERIMENT_ID).all(), "Experiment identity drift")
            parts.append(rows)
            trace_parts.append(traces)
    frame = pd.concat(parts, ignore_index=True)
    require(len(frame) == 4000 and not frame.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Incomplete 4000-episode grid")
    eligible_parts, summaries, contrasts, tasks = [], [], [], []
    for suite_index, suite in enumerate(cfg["suites"]):
        eligible = frame.loc[(frame.suite == suite) & (frame.frozen_clean_success == 1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        require(len(eligible) == DENOMINATORS[suite] * len(CONDITIONS), "Clean pair drift")
        eligible_parts.append(eligible)
        suite_summary = stats.task_episode_bootstrap(
            eligible, "casr", "condition", draws, cfg["bootstrap_seed"] + suite_index)
        suite_summary.insert(0, "suite", suite)
        summaries.append(suite_summary)
        per_task = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"))
        per_task.insert(0, "suite", suite)
        tasks.append(per_task)
        comparisons = [(f"{arm}_minus_original_{step}", f"{arm}_{step}", f"original_uada_{step}")
                       for step in (1000, 2000) for arm in ARMS[1:]]
        comparisons += [(f"dynamic_minus_static_{step}", f"dynamic_{step}", f"static_{step}")
                        for step in (1000, 2000)]
        comparisons += [(f"diverse_minus_dynamic_{step}", f"diverse_{step}", f"dynamic_{step}")
                        for step in (1000, 2000)]
        comparisons += [(f"{arm}_2000_minus_1000", f"{arm}_2000", f"{arm}_1000")
                        for arm in ARMS]
        for compare_index, (name, left, right) in enumerate(comparisons):
            result = stats.paired_contrast_bootstrap(
                eligible, "casr", "condition", left, right, draws,
                cfg["bootstrap_seed"] + 100 + suite_index * 100 + compare_index)
            require(result["paired_episodes"] == DENOMINATORS[suite], "Matched episode count drift")
            contrasts.append({"suite": suite, "contrast": name, **result})
    eligible = pd.concat(eligible_parts, ignore_index=True)
    require(len(eligible) == 3820, "Wrong eligible total")
    prefix = "DINO_TRANSFER2000_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    summary_path = output / f"{prefix}_PER_SUITE.csv"
    contrast_path = output / f"{prefix}_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(tasks, ignore_index=True).to_csv(output / f"{prefix}_PER_TASK.csv", index=False)
    root = Path(config()["output_root"])
    original = json.loads(ORIGINAL_CONFIG.read_text())
    costs = []
    for arm in ARMS:
        folder = (Path(original["output_root"]) / "exploring_oft_uada" if arm == "original_uada"
                  else root / "training" / arm)
        rows = pd.read_parquet(folder / "training.parquet")
        for step in (1000, 2000):
            row = rows.loc[rows.outer_iteration == step].iloc[0]
            costs.append({"condition": f"{arm}_{step}", "optimizer_updates": step * 50,
                          "training_elapsed_s": float(row.elapsed_s),
                          "shared_collection_excluded": arm != "original_uada"})
    pd.DataFrame(costs).to_csv(output / f"{prefix}_COSTS.csv", index=False)
    report = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "attack_episodes": len(frame), "eligible_rows": len(eligible),
              "checkpoints_are_repeated_measures": True, "training_seed_count": 1,
              "source_suite": "libero_object", "transfer_focus": ["libero_spatial", "libero_goal"],
              "long_reported_secondary": True,
              "per_suite_sha256": sha256_file(summary_path),
              "contrasts_sha256": sha256_file(contrast_path)}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
