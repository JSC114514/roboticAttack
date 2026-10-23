#!/usr/bin/env python3
"""Original-UADA checkpoint CASR and optional paired v2 contrasts."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from run_exploring_oft_uada_original1000_cross_suite import (
    CONDITIONS, DENOMINATORS, EXPERIMENT_ID, RUN, STEPS, config, frozen_clean,
    preflight, require, sha256_file, validate,
)
from common import atomic_json, atomic_parquet

STATS_PATH = RUN.parents[1].parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_original_uada_bootstrap_statistics", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)


def analyze(draws=10000):
    cfg = config()
    require(preflight(cfg)["status"] == "PASS", "Baseline training incomplete")
    output = Path(cfg["output_root"])
    parts = []
    for suite in cfg["suites"]:
        for task in range(10):
            folder = output / suite
            rows = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
            traces = pd.read_parquet(folder / f"task_{task:02d}_traces.parquet")
            validate(rows, traces, suite, task, cfg)
            require(rows.experiment_id.eq(EXPERIMENT_ID).all(), "Wrong baseline evaluation")
            parts.append(rows)
    frame = pd.concat(parts, ignore_index=True)
    require(len(frame) == 1200 and not frame.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Baseline grid incomplete")
    v2_root = Path(cfg["output_root"]).parents[2] / "wrist_dino_scratch_v2/evaluation/four_suite"
    v2_status = v2_root / "STATUS.json"
    compare = v2_status.is_file() and json.loads(v2_status.read_text()).get("status") == "COMPLETE"
    if compare:
        v2 = pd.read_parquet(v2_root / "DINO_SCRATCH_XSUITE_ROLLOUTS.parquet")
        require(len(v2) == 3600 and v2.condition.nunique() == 9
                and v2.experiment_id.eq("OFT-DINO-SCRATCH1000-XSUITE-V2").all(), "V2 grid incomplete")
        keys = ["suite", "task_id", "state_index"]
        anchors = frame.loc[frame.condition == CONDITIONS[0], keys + ["rollout_seed", "frozen_clean_success"]]
        matched = v2.merge(anchors, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
        require(len(matched) == len(v2) and matched.rollout_seed.eq(matched.rollout_seed_baseline).all()
                and matched.frozen_clean_success.eq(matched.frozen_clean_success_baseline).all(),
                "V2 is not paired on task/state/seed/clean eligibility")
        frame = pd.concat([frame, v2], ignore_index=True)
    summaries, contrasts, tasks, eligible_parts = [], [], [], []
    for i, suite in enumerate(cfg["suites"]):
        eligible = frame.loc[(frame.suite == suite) & (frame.frozen_clean_success == 1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        expected = DENOMINATORS[suite] * (12 if compare else 3)
        require(len(eligible) == expected, "Clean-success denominator drift")
        eligible_parts.append(eligible)
        summary = stats.task_episode_bootstrap(eligible, "casr", "condition", draws,
                                               cfg["bootstrap_seed"] + i)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        task = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"))
        task.insert(0, "suite", suite)
        tasks.append(task)
        pairs = [(f"original_1000_minus_500", CONDITIONS[2], CONDITIONS[0]),
                 (f"original_750_minus_500", CONDITIONS[1], CONDITIONS[0])]
        if compare:
            for step in STEPS:
                pairs.extend((f"{method}_{step}_minus_original", f"{method}_{step}",
                              f"original_uada_{step}") for method in ("dino", "uniform", "uada"))
        for j, (role, left, right) in enumerate(pairs):
            result = stats.paired_contrast_bootstrap(eligible, "casr", "condition", left, right,
                                                     draws, cfg["bootstrap_seed"] + 100 * i + j)
            require(result["paired_episodes"] == DENOMINATORS[suite], "Unpaired contrast")
            contrasts.append({"suite": suite, "contrast_role": role, **result})
    atomic_parquet(frame, output / "ORIGINAL_UADA1000_COMBINED_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(eligible_parts, ignore_index=True),
                   output / "ORIGINAL_UADA1000_COMBINED_ELIGIBLE.parquet")
    suite_path = output / "ORIGINAL_UADA1000_PER_SUITE.csv"
    contrast_path = output / "ORIGINAL_UADA1000_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(suite_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(tasks, ignore_index=True).to_csv(output / "ORIGINAL_UADA1000_PER_TASK.csv", index=False)
    history = pd.read_parquet(RUN / "training/exploring_oft_uada/training.parquet")
    costs = [{"condition": f"original_uada_{step}", "outer": step, "updates": step * 50,
              "training_elapsed_s": float(history.loc[history.outer_iteration == step, "elapsed_s"].iloc[0])}
             for step in STEPS]
    pd.DataFrame(costs).to_csv(output / "ORIGINAL_UADA1000_TRAINING_COSTS.csv", index=False)
    report = {"status": "COMPLETE" if compare else "BASELINE_COMPLETE_V2_PENDING",
              "baseline_experiment_id": EXPERIMENT_ID, "baseline_attack_episodes": 1200,
              "v2_paired": compare, "clean_success_denominators": DENOMINATORS,
              "training_seed_count": 1, "checkpoints_are_repeated_measures": True,
              "not_identical_to_historical_uada500": "1000-outer cosine schedule; old 500 used 500-outer schedule",
              "not_pure_mechanism_ablation": "V2 differs in bank/placement/TV/shared prefix; use V2 uada for internal ablation",
              "long_secondary": True, "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(suite_path),
              "contrasts_sha256": sha256_file(contrast_path)}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
