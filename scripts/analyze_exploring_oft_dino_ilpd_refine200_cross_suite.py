#!/usr/bin/env python3
"""Paired clean-success CASR analysis for all four frozen ILPD-V2 patches."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from run_exploring_oft_dino_ilpd_refine200_cross_suite import (
    CONDITIONS, DENOMINATORS, EXPERIMENT_ID, ROOT, atomic_json, frozen_clean,
    load_config, load_ilpd_patches, require, sha256_file, validate_result,
)
from common import atomic_parquet

STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_oft_ilpd_refinement_bootstrap", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)

PREFIX = "DINO_ILPD_REFINE200_XSUITE"
CONTRASTS = (
    ("DINO_ILPD_MINUS_UADA", "dino_ilpd_bounded", "uada_bounded"),
    ("DINO_ILPD_MINUS_DINO_GRADSCALE", "dino_ilpd_bounded", "dino_gradscale_bounded"),
    ("DINO_ILPD_MINUS_SIGLIP_ILPD", "dino_ilpd_bounded", "siglip_ilpd_bounded"),
    ("DINO_GRADSCALE_MINUS_UADA", "dino_gradscale_bounded", "uada_bounded"),
    ("SIGLIP_ILPD_MINUS_UADA", "siglip_ilpd_bounded", "uada_bounded"),
)


def analyze(config: Path, draws: int = 10000) -> dict:
    cfg = load_config(config)
    output = Path(cfg["output_root"])
    rollout_parts, trace_parts = [], []
    for suite in cfg["suites"]:
        for task_id in range(10):
            rpath = output / suite / f"task_{task_id:02d}_rollouts.parquet"
            tpath = output / suite / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing evaluation shard: {suite} {task_id}")
            rollouts, traces = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate_result(rollouts, traces, suite, task_id, cfg)
            rollout_parts.append(rollouts)
            trace_parts.append(traces)
    rollouts = pd.concat(rollout_parts, ignore_index=True)
    traces = pd.concat(trace_parts, ignore_index=True)
    require(len(rollouts) == 1600 and not rollouts.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Combined rollout grid drift")
    eligible_parts, summaries, contrasts, tasks = [], [], [], []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        eligible = rollouts.loc[(rollouts.suite == suite) & (rollouts.frozen_clean_success == 1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        denominators[suite] = int(clean.success.sum())
        require(len(eligible) == denominators[suite] * len(CONDITIONS), "Eligible row count drift")
        eligible_parts.append(eligible)
        summary = stats.task_episode_bootstrap(
            eligible, "casr", "condition", draws, cfg["bootstrap_seed"] + suite_index)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        per_task = (eligible.groupby(["task_id", "condition"], as_index=False)
                    .agg(clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
                         attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean")))
        per_task.insert(0, "suite", suite)
        tasks.append(per_task)
        for contrast_index, (role, left, right) in enumerate(CONTRASTS):
            result = stats.paired_contrast_bootstrap(
                eligible, "casr", "condition", left, right, draws,
                cfg["bootstrap_seed"] + 10 + 10 * suite_index + contrast_index)
            require(result["paired_episodes"] == denominators[suite], f"Contrast pairing drift: {suite} {role}")
            contrasts.append({"suite": suite, "contrast_role": role, **result})
    require(denominators == DENOMINATORS, "Frozen clean denominators changed")
    eligible = pd.concat(eligible_parts, ignore_index=True)
    require(len(eligible) == 1528, "Combined eligible count drift")
    atomic_parquet(rollouts, output / f"{PREFIX}_ROLLOUTS.parquet")
    atomic_parquet(traces, output / f"{PREFIX}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{PREFIX}_ELIGIBLE.parquet")
    per_suite_path = output / f"{PREFIX}_PER_SUITE.csv"
    contrasts_path = output / f"{PREFIX}_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(per_suite_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrasts_path, index=False)
    pd.concat(tasks, ignore_index=True).to_csv(output / f"{PREFIX}_PER_TASK.csv", index=False)
    training_costs = []
    for method in CONDITIONS:
        status = json.loads((Path(cfg["training_root"]) / method / "status.json").read_text())
        training_costs.append({"condition": method, "base_outer_iterations": 500,
                               "refinement_outer_iterations": 200, "total_outer_iterations": 700,
                               "refinement_optimizer_updates": 10000,
                               "logged_refinement_s": float(status["elapsed_s"]),
                               "known_cost_including_base_s": float(status["cost_including_base_s"]),
                               "cost_boundary": status["cost_boundary"]})
    pd.DataFrame(training_costs).to_csv(output / f"{PREFIX}_TRAINING_COSTS.csv", index=False)
    status = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "new_attack_episodes": len(rollouts), "eligible_attack_rows": len(eligible),
              "frozen_clean_success_denominators": denominators, "conditions": list(CONDITIONS),
              "manual_test_request": cfg["manual_test_request"], "bootstrap_draws": draws,
              "contrast_roles": [role for role, _, _ in CONTRASTS], "patch_sha256": cfg["patch_sha256"],
              "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(per_suite_path), "contrasts_sha256": sha256_file(contrasts_path),
              "claim_boundary": "Object source; Spatial/Goal zero-shot; Long secondary; development states; one training seed",
              "multiplicity_adjustment": "none; report all planned contrasts and suites"}
    atomic_json(status, output / "STATUS.json")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    import torch
    cfg = load_config(args.config)
    load_ilpd_patches(cfg, torch.device("cpu"))
    print(json.dumps(analyze(args.config.resolve()), indent=2))


if __name__ == "__main__":
    main()
