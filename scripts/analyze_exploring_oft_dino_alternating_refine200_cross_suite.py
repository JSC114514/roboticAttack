#!/usr/bin/env python3
"""Pair 800 new attacks with 1200 frozen controls, preserving provenance and appearance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from run_exploring_oft_dino_alternating_refine200_cross_suite import (
    ROOT, EXPERIMENT_ID, CONDITIONS, CONTROLS, ALL_CONDITIONS, DENOMINATORS,
    atomic_json, frozen_clean, frozen_controls, evaluation_manifest, load_config,
    load_refinement_patches, require, sha256_file, training_costs, validate_result,
)
from common import atomic_parquet

STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_oft_alternating_bootstrap", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)

PREFIX = "DINO_ALTERNATING200_XSUITE"
CONTRASTS = (
    ("ROLLING_MINUS_FIXED", CONDITIONS[1], CONDITIONS[0]),
    ("FIXED_MINUS_UADA_CONTINUE", CONDITIONS[0], CONTROLS[1]),
    ("ROLLING_MINUS_UADA_CONTINUE", CONDITIONS[1], CONTROLS[1]),
    ("FIXED_MINUS_ILA", CONDITIONS[0], CONTROLS[2]),
    ("ROLLING_MINUS_ILA", CONDITIONS[1], CONTROLS[2]),
    ("FIXED_MINUS_UADA500", CONDITIONS[0], CONTROLS[0]),
    ("ROLLING_MINUS_UADA500", CONDITIONS[1], CONTROLS[0]),
)


def analyze(config: Path, draws=10000):
    cfg = load_config(config)
    output = Path(cfg["output_root"])
    signature = evaluation_manifest(cfg)
    load_refinement_patches(cfg, torch.device("cpu"))
    control_rows, control_traces = frozen_controls(cfg)
    all_rows, all_traces = [], []
    for suite in cfg["suites"]:
        for task in range(10):
            rp = output / suite / f"task_{task:02d}_rollouts.parquet"
            tp = output / suite / f"task_{task:02d}_traces.parquet"
            require(rp.is_file() and tp.is_file(), f"Missing evaluation shard: {suite} {task}")
            r, t = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_result(r, t, suite, task, cfg)
            all_rows.append(r)
            all_traces.append(t)
    new_rows, new_traces = pd.concat(all_rows, ignore_index=True), pd.concat(all_traces, ignore_index=True)
    require(len(new_rows) == 800, "New rollout grid drift")
    for frame in (new_rows, new_traces):
        frame["evaluation_origin"] = "NEW_ALTERNATING_EVALUATION"
        frame["appearance_assessment"] = frame.condition.map(cfg["appearance_assessment"])
    for frame in (control_rows, control_traces):
        frame["evaluation_origin"] = "REUSED_ILA_V2_EVALUATION"
        frame["appearance_assessment"] = "PASS_USER_20260927_ILA_REFINE200_V2"
    rollouts = pd.concat([control_rows, new_rows], ignore_index=True)
    traces = pd.concat([control_traces, new_traces], ignore_index=True)
    keys = ["suite", "task_id", "state_index", "condition"]
    require(len(rollouts) == 2000 and not rollouts.duplicated(keys).any()
            and set(rollouts.condition) == set(ALL_CONDITIONS), "Combined rollout grid drift")
    pairs = rollouts.groupby(keys[:3])
    require(pairs.size().eq(5).all() and pairs.rollout_seed.nunique().eq(1).all()
            and pairs.frozen_clean_success.nunique().eq(1).all(), "Historical/new pairing mismatch")
    summaries, contrasts, per_tasks, eligible_frames, denominators = [], [], [], [], {}
    for suite_index, suite in enumerate(cfg["suites"]):
        denominators[suite] = int(frozen_clean(cfg, suite).success.sum())
        eligible = rollouts.loc[(rollouts.suite == suite) & (rollouts.frozen_clean_success == 1)].copy()
        require(len(eligible) == denominators[suite] * 5, "Eligible row count drift")
        eligible["casr"] = eligible.failure.astype(float)
        eligible_frames.append(eligible)
        summary = stats.task_episode_bootstrap(eligible, "casr", "condition", draws,
                                               cfg["bootstrap_seed"] + suite_index)
        summary.insert(0, "suite", suite)
        summary["appearance_assessment"] = summary.condition.map(
            {**{m: "PASS_USER_20260927_ILA_REFINE200_V2" for m in CONTROLS}, **cfg["appearance_assessment"]})
        summaries.append(summary)
        task_summary = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean"))
        task_summary.insert(0, "suite", suite)
        per_tasks.append(task_summary)
        for index, (role, left, right) in enumerate(CONTRASTS):
            result = stats.paired_contrast_bootstrap(eligible, "casr", "condition", left, right, draws,
                                                    cfg["bootstrap_seed"] + 10 + 10 * suite_index + index)
            require(result["paired_episodes"] == denominators[suite], f"Contrast pairing drift: {suite} {role}")
            contrasts.append({"suite": suite, "contrast_role": role, **result})
    require(denominators == DENOMINATORS, "Frozen denominators changed")
    eligible = pd.concat(eligible_frames, ignore_index=True)
    require(len(eligible) == 1910 and int(new_rows.frozen_clean_success.sum()) == 764,
            "Combined eligibility count drift")
    atomic_parquet(new_rows, output / f"{PREFIX}_NEW_ROLLOUTS.parquet")
    atomic_parquet(rollouts, output / f"{PREFIX}_ROLLOUTS.parquet")
    atomic_parquet(traces, output / f"{PREFIX}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{PREFIX}_ELIGIBLE.parquet")
    summary_path, contrast_path = output / f"{PREFIX}_PER_SUITE.csv", output / f"{PREFIX}_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(per_tasks, ignore_index=True).to_csv(output / f"{PREFIX}_PER_TASK.csv", index=False)
    costs = training_costs(cfg)
    pd.DataFrame(costs).to_csv(output / f"{PREFIX}_TRAINING_COSTS.csv", index=False)
    status = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID, "evaluation_signature": signature,
              "new_attack_episodes": 800, "reused_attack_episodes": 1200, "combined_attack_episodes": 2000,
              "eligible_attack_rows": 1910, "new_eligible_attack_rows": 764,
              "frozen_clean_success_denominators": denominators, "conditions": list(ALL_CONDITIONS),
              "training_costs": costs, "evaluation_authorization": cfg["evaluation_authorization"],
              "appearance_assessment": cfg["appearance_assessment"], "reused_controls": cfg["reused_controls"],
              "bootstrap_draws": draws, "contrast_roles": [r for r, _, _ in CONTRASTS],
              "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(summary_path), "contrasts_sha256": sha256_file(contrast_path),
              "claim_boundary": "Object source; Spatial/Goal zero-shot; Long secondary; reused development states; "
                                "single training seed; historical controls, not concurrent; rolling appearance not accepted",
              "multiplicity_adjustment": "none; all seven contrasts and four suites reported"}
    atomic_json(status, output / "STATUS.json")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
