#!/usr/bin/env python3
"""Task-equal CASR and paired planned contrasts for replay-v1 evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from run_exploring_oft_dino_replay_cross_suite import (
    CONDITIONS, DENOMINATORS, EXPERIMENT_ID, ROOT, atomic_json,
    frozen_clean, load_config, preflight, require, sha256_file, validate_result,
)
sys.path.append(str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import atomic_parquet

STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_replay_bootstrap_statistics", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)

CONTRASTS = (("REPLAY_MINUS_CLEAN_CONTINUE", "replay_uniform", "clean_continue"),
             ("DINO_MINUS_REPLAY_UNIFORM", "replay_dino", "replay_uniform"),
             ("DINO_MINUS_SHUFFLED", "replay_dino", "replay_shuffled"),
             ("CLEAN_CONTINUE_MINUS_FROZEN", "clean_continue", "uada_2000_frozen"),
             ("DINO_MINUS_FROZEN", "replay_dino", "uada_2000_frozen"))


def analyze(config_path, draws=10000):
    cfg = load_config(config_path)
    require(preflight(cfg)["status"] == "PASS", "Evaluation visual gate not complete")
    output = Path(cfg["output_root"])
    parts, traces = [], []
    for suite in cfg["suites"]:
        for task_id in range(10):
            folder = output / suite
            rpath = folder / f"task_{task_id:02d}_rollouts.parquet"
            tpath = folder / f"task_{task_id:02d}_traces.parquet"
            require(rpath.is_file() and tpath.is_file(), f"Missing evaluation shard: {suite} {task_id}")
            rollouts, trace = pd.read_parquet(rpath), pd.read_parquet(tpath)
            validate_result(rollouts, trace, suite, task_id, cfg)
            parts.append(rollouts)
            traces.append(trace)
    frame = pd.concat(parts, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    require(len(frame) == 2000 and not frame.duplicated(
        ["suite", "task_id", "state_index", "condition"]).any(), "Evaluation episode grid drift")
    summaries, contrasts, task_rows, eligible_parts = [], [], [], []
    denominators = {}
    for suite_index, suite in enumerate(cfg["suites"]):
        clean = frozen_clean(cfg, suite)
        eligible = frame.loc[(frame.suite == suite) & (frame.frozen_clean_success == 1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        denominators[suite] = int(clean.success.sum())
        require(len(eligible) == denominators[suite] * len(CONDITIONS), "Eligible count drift")
        eligible_parts.append(eligible)
        summary = stats.task_episode_bootstrap(eligible, "casr", "condition", draws,
                                               cfg["bootstrap_seed"] + suite_index)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        per_task = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean"))
        per_task.insert(0, "suite", suite)
        task_rows.append(per_task)
        for contrast_index, (name, left, right) in enumerate(CONTRASTS):
            result = stats.paired_contrast_bootstrap(eligible, "casr", "condition", left, right,
                draws, cfg["bootstrap_seed"] + 10 + 10 * suite_index + contrast_index)
            require(result["paired_episodes"] == denominators[suite], "Paired episodes drift")
            contrasts.append({"suite": suite, "contrast_role": name, **result})
    require(denominators == DENOMINATORS, "Clean denominator drift")
    eligible = pd.concat(eligible_parts, ignore_index=True)
    require(len(eligible) == 1910, "Four-suite eligible episode count drift")
    prefix = "DINO_REPLAY_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(trace, output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    suite_path = output / f"{prefix}_PER_SUITE.csv"
    contrast_path = output / f"{prefix}_CONTRASTS.csv"
    pd.concat(summaries, ignore_index=True).to_csv(suite_path, index=False)
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    pd.concat(task_rows, ignore_index=True).to_csv(output / f"{prefix}_PER_TASK.csv", index=False)
    train = json.loads(Path(cfg["train_config"]).read_text())
    cost_rows = []
    for method in CONDITIONS[1:]:
        status = json.loads((Path(train["output_root"]) / "training" / method / "status.json").read_text())
        cost_rows.append({"condition": method, "base_outer": 2000, "refinement_outer": 200,
                          "new_optimizer_updates": 10000,
                          "baseline_logged_s": train["baseline_elapsed_s"],
                          "refinement_elapsed_s": status["elapsed_s"],
                          "known_total_s": train["baseline_elapsed_s"] + status["elapsed_s"]})
    pd.DataFrame(cost_rows).to_csv(output / f"{prefix}_TRAINING_COSTS.csv", index=False)
    report = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "new_attack_episodes": len(frame), "eligible_attack_rows": len(eligible),
              "clean_success_denominators": denominators, "conditions": list(CONDITIONS),
              "contrast_roles": [x[0] for x in CONTRASTS],
              "object_spatial_goal_primary": True, "long_secondary": True,
              "equal_budget_contrasts": list(x[0] for x in CONTRASTS[:3]),
              "frozen_baseline_budget_difference": "2200 versus 2000 outer; not equal budget",
              "statistics_source_sha256": sha256_file(STATS_PATH),
              "per_suite_sha256": sha256_file(suite_path),
              "contrasts_sha256": sha256_file(contrast_path)}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/exploring_oft_dino/object_dino_replay_refine200_cross_suite_v1.json")
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.draws), indent=2))
