#!/usr/bin/env python3
"""Five planned paired CASR contrasts, separately for all four suites."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from run_exploring_oft_dino_ila_refine200_cross_suite import (
    ROOT, CONDITIONS, DENOMINATORS, EXPERIMENT_ID, atomic_json, frozen_clean,
    load_config, load_refinement_patches, require, sha256_file, validate_result, training_costs,
)
from common import atomic_parquet

# Explicit path/name: importing stdlib statistics first must not break analysis.
STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_oft_ila_refinement_bootstrap", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)
task_episode_bootstrap = stats.task_episode_bootstrap
paired_contrast_bootstrap = stats.paired_contrast_bootstrap

PREFIX = "DINO_ILA_REFINE200_XSUITE"
CONTRASTS = (
    ("C_MINUS_B", "dino_ila_tv", "uada_continue_tv"),
    ("C_MINUS_D", "dino_ila_tv", "dino_disruption_tv"),
    ("B_MINUS_A", "uada_continue_tv", "exploring_oft_uada"),
    ("C_MINUS_A", "dino_ila_tv", "exploring_oft_uada"),
    ("D_MINUS_A", "dino_disruption_tv", "exploring_oft_uada"),
)


def analyze(config: Path, draws: int = 10000):
    cfg = load_config(config)
    output = Path(cfg["output_root"])
    all_rows, all_traces = [], []
    for suite in cfg["suites"]:
        for task_id in range(10):
            rpath = output / suite / f"task_{task_id:02d}_rollouts.parquet"
            tpath = output / suite / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing evaluation shard: {suite} {task_id}")
            rows, traces = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate_result(rows, traces, suite, task_id, cfg)
            all_rows.append(rows)
            all_traces.append(traces)
    rollouts, traces = pd.concat(all_rows, ignore_index=True), pd.concat(all_traces, ignore_index=True)
    require(len(rollouts) == 1600 and not rollouts.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Combined rollout grid drift")
    eligible_frames, summaries, contrasts, task_summaries = [], [], [], []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        subset = rollouts.loc[rollouts.suite == suite].copy()
        eligible = subset.loc[subset.frozen_clean_success == 1].copy()
        eligible["casr"] = eligible.failure.astype(float)
        denominators[suite] = int(clean.success.sum())
        require(len(eligible) == denominators[suite] * 4, "Eligible row count drift")
        eligible_frames.append(eligible)
        summary = task_episode_bootstrap(eligible, "casr", "condition", draws, cfg["bootstrap_seed"] + suite_index)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        per_task = (eligible.groupby(["task_id", "condition"], as_index=False)
                    .agg(clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
                         attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean")))
        per_task.insert(0, "suite", suite)
        task_summaries.append(per_task)
        for contrast_index, (role, left, right) in enumerate(CONTRASTS):
            result = paired_contrast_bootstrap(eligible, "casr", "condition", left, right, draws,
                                              cfg["bootstrap_seed"] + 10 + 10 * suite_index + contrast_index)
            require(result["paired_episodes"] == denominators[suite], f"Contrast pairing drift: {suite} {role}")
            contrasts.append({"suite": suite, "contrast_role": role, **result})
    require(denominators == DENOMINATORS, "Frozen denominators changed")
    eligible = pd.concat(eligible_frames, ignore_index=True)
    require(len(eligible) == 1528, "Combined eligibility count drift")
    atomic_parquet(rollouts, output / f"{PREFIX}_ROLLOUTS.parquet")
    atomic_parquet(traces, output / f"{PREFIX}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{PREFIX}_ELIGIBLE.parquet")
    summary_path = output / f"{PREFIX}_PER_SUITE.csv"
    contrast_path = output / f"{PREFIX}_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(task_summaries, ignore_index=True).to_csv(output / f"{PREFIX}_PER_TASK.csv", index=False)
    costs = training_costs(cfg)
    pd.DataFrame(costs).to_csv(output / f"{PREFIX}_TRAINING_COSTS.csv", index=False)
    status = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID, "new_attack_episodes": len(rollouts),
              "eligible_attack_rows": len(eligible), "frozen_clean_success_denominators": denominators,
              "conditions": list(CONDITIONS), "training_costs": costs, "manual_visual_review": cfg["manual_visual_review"],
              "bootstrap_draws": draws, "contrast_roles": [role for role, _, _ in CONTRASTS],
              "patch_sha256": {m: cfg["patch_sources"][m]["sha256"] for m in CONDITIONS},
              "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(summary_path), "contrasts_sha256": sha256_file(contrast_path),
              "claim_boundary": "Object source; Spatial/Goal zero-shot; Long secondary; reused development states; single training seed",
              "multiplicity_adjustment": "none; report all planned contrasts and all suites"}
    atomic_json(status, output / "STATUS.json")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import torch
    cfg = load_config(args.config)
    load_refinement_patches(cfg, torch.device("cpu"))
    print(json.dumps(analyze(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
