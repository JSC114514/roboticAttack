#!/usr/bin/env python3
"""Task-equal paired statistics, preserving frozen-control provenance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_exploring_oft_dino_repair_cross_suite import (
    CONDITIONS, DENOMINATORS, EXPERIMENT_ID, METHODS, ROOT, config, load_config,
    patches, preflight, reference_frames, require, sha256_file, validate_new_shard)
from dino_repair_common import frozen_digest
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet

STATS_PATH = ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"
spec = importlib.util.spec_from_file_location("_repair_stats", STATS_PATH)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)
PAIRS = [
    ("B_minus_A", "clear", "diverse_reference_1000"),
    ("B_minus_C", "clear", "uniform_clear"),
    ("D_minus_B", "dino_causal_clear", "clear"),
    ("E_minus_D", "dino_repair_clear", "dino_causal_clear"),
    ("E_minus_B", "dino_repair_clear", "clear"),
]
PAIRS += [(f"{method}_minus_A", method, "diverse_reference_1000") for method in METHODS[1:]]
PAIRS += [(f"{method}_minus_original", method, "original_uada_1000") for method in METHODS]


def hierarchical_samples(eligible, draws, rng):
    """Use common task and episode resampling for all conditions in each draw."""
    wide = eligible.pivot(index=["task_id", "state_index"], columns="condition", values="casr")
    wide = wide.reindex(columns=CONDITIONS)
    require(not wide.isna().any().any(), "Unpaired focus episode")
    arrays = [g.to_numpy(float) for _, g in wide.groupby(level="task_id")]
    require(len(arrays) == 10, "Missing focus task")
    selected = rng.integers(0, 10, size=(draws, 10))
    samples = np.zeros((draws, len(CONDITIONS)))
    for position in range(10):
        for task, values in enumerate(arrays):
            draw_ids = np.flatnonzero(selected[:, position] == task)
            episodes = rng.integers(0, len(values), size=(len(draw_ids), len(values)))
            samples[draw_ids] += values[episodes].mean(axis=1) / 10
    points = np.stack([v.mean(axis=0) for v in arrays]).mean(axis=0)
    return points, samples


def focus_statistics(eligible, draws, seed):
    rng = np.random.default_rng(seed)
    points, samples = zip(*(hierarchical_samples(
        eligible.loc[eligible.suite.eq(suite)], draws, rng)
        for suite in ("libero_spatial", "libero_goal")))
    points, samples = np.stack(points), np.stack(samples, axis=1)
    macro, worst = points.mean(axis=0), points.min(axis=0)
    macro_samples, worst_samples = samples.mean(axis=1), samples.min(axis=1)
    rows = []
    for j, condition in enumerate(CONDITIONS):
        for metric, value, distribution in (
            ("spatial_goal_macro", macro[j], macro_samples[:, j]),
            ("spatial_goal_worst_suite", worst[j], worst_samples[:, j]),
        ):
            low, high = np.percentile(distribution, [2.5, 97.5])
            rows.append({"condition": condition, "metric": metric,
                         "task_equal_mean": float(value), "ci95_low": float(low), "ci95_high": float(high)})
    contrasts = []
    for name, left, right in PAIRS:
        li, ri = CONDITIONS.index(left), CONDITIONS.index(right)
        for metric, value, distribution in (
            ("spatial_goal_macro", macro[li] - macro[ri], macro_samples[:, li] - macro_samples[:, ri]),
            ("spatial_goal_worst_suite", worst[li] - worst[ri], worst_samples[:, li] - worst_samples[:, ri]),
        ):
            low, high = np.percentile(distribution, [2.5, 97.5])
            contrasts.append({"contrast": name, "left": left, "right": right, "metric": metric,
                              "difference_left_minus_right": float(value),
                              "ci95_low": float(low), "ci95_high": float(high)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def analyze(draws=10000):
    require(draws >= 100, "Bootstrap budget too small")
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training not complete")
    output = Path(cfg["output_root"])
    _, hashes, _ = patches(cfg, "cpu")
    references, reference_traces = reference_frames(cfg)
    parts, traces = [references], [reference_traces]
    for suite in cfg["suites"]:
        for task in range(10):
            folder = output / suite
            rp, tp = folder / f"task_{task:02d}_rollouts.parquet", folder / f"task_{task:02d}_traces.parquet"
            rows, trace = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rows, trace, suite, task, cfg, hashes)
            for frame, path in ((rows, rp), (trace, tp)):
                frame["source_experiment_id"] = frame.experiment_id
                frame["source_condition"] = frame.condition
                frame["source_path"] = str(path)
                frame["source_sha256"] = sha256_file(path)
                frame["reused"] = False
            parts.append(rows)
            traces.append(trace)
    frame = pd.concat(parts, ignore_index=True)
    require(len(frame) == 2400 and int(frame.reused.sum()) == 800
            and set(frame.condition) == set(CONDITIONS) and not frame.duplicated(
                ["suite", "task_id", "state_index", "condition"]).any(), "Incomplete 2400 episode grid")
    require(frame.groupby(["suite", "condition"]).size().eq(100).all(), "Suite/condition grid incomplete")
    eligible_parts, summaries, contrasts, task_rows = [], [], [], []
    for si, suite in enumerate(cfg["suites"]):
        eligible = frame.loc[frame.suite.eq(suite) & frame.frozen_clean_success.eq(1)].copy()
        eligible["casr"] = eligible.failure.astype(float)
        require(len(eligible) == DENOMINATORS[suite] * len(CONDITIONS), "Clean eligibility changed")
        eligible_parts.append(eligible)
        summary = stats.task_episode_bootstrap(eligible, "casr", "condition", draws, cfg["bootstrap_seed"] + si)
        summary.insert(0, "suite", suite)
        summaries.append(summary)
        per_task = eligible.groupby(["task_id", "condition"], as_index=False).agg(
            clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
            attacked_success=("success", "mean"))
        per_task.insert(0, "suite", suite)
        task_rows.append(per_task)
        for pi, (name, left, right) in enumerate(PAIRS):
            result = stats.paired_contrast_bootstrap(eligible, "casr", "condition", left, right,
                draws, cfg["bootstrap_seed"] + 100 + si * 100 + pi)
            require(result["paired_episodes"] == DENOMINATORS[suite], "Paired denominator changed")
            contrasts.append({"suite": suite, "contrast": name, **result})
    eligible = pd.concat(eligible_parts, ignore_index=True)
    require(len(eligible) == 2292, "Eligible row count changed")
    prefix = "DINO_REPAIR1000_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(traces, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    for suffix, values in (("PER_SUITE", pd.concat(summaries, ignore_index=True)),
                           ("CONTRASTS", pd.DataFrame(contrasts)),
                           ("PER_TASK", pd.concat(task_rows, ignore_index=True))):
        values.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
    focus, focus_contrasts = focus_statistics(eligible, draws, cfg["bootstrap_seed"] + 5000)
    focus.to_csv(output / f"{prefix}_SPATIAL_GOAL.csv", index=False)
    focus_contrasts.to_csv(output / f"{prefix}_SPATIAL_GOAL_CONTRASTS.csv", index=False)
    train = config()
    costs = []
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        costs.append({"condition": method, "optimizer_updates": status["optimizer_updates"],
                      "elapsed_s": status["elapsed_s"], "prefix_elapsed_s": status.get("prefix_elapsed_s"),
                      "costs": status.get("costs", {}), "reused": False})
    calibration = json.loads((Path(train["output_root"]) /
        "reports/APPEARANCE_CALIBRATION.json").read_text())
    costs.append({"calibration": calibration.get("costs", {}),
                  "calibration_elapsed_s": calibration.get("calibration_elapsed_s")})
    for method in METHODS:
        attack = frame.loc[frame.condition.eq(method)]
        costs.append({"condition": method, "new_evaluation_elapsed_s": float(attack.elapsed_s.sum()),
                      "evaluation_episodes": len(attack),
                      "evaluation_model_load_environment_setup_s": "not recorded by legacy normal runner"})
    old = json.loads(Path(train["prefix_config"]).read_text())
    original = json.loads(Path(train["original_config"]).read_text())
    historical_folders = {
        "diverse_reference_1000": Path(old["output_root"]) / "training/diverse",
        "original_uada_1000": Path(original["output_root"]) / "exploring_oft_uada",
    }
    for condition in ("diverse_reference_1000", "original_uada_1000"):
        attack = frame.loc[frame.condition.eq(condition)]
        history_path = historical_folders[condition] / "training.parquet"
        frozen_digest(train, history_path)
        history = pd.read_parquet(history_path)
        row = history.loc[history.outer_iteration.eq(1000)]
        require(len(row) == 1, "Missing historical matched-checkpoint cost")
        costs.append({"condition": condition, "reused": True, "new_training_elapsed_s": 0.,
                      "new_evaluation_elapsed_s": 0., "recorded_evaluation_elapsed_s": float(attack.elapsed_s.sum()),
                      "recorded_training_to_1000_elapsed_s": float(row.iloc[0].elapsed_s),
                      "optimizer_updates_at_1000": 50000,
                      "historical_training_parquet_sha256": sha256_file(history_path),
                      "note": "Frozen 1000 checkpoint; matched-checkpoint training cost is historical."})
    atomic_json(costs, output / f"{prefix}_COSTS.json")
    report = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
              "new_attack_episodes": 1600, "reused_attack_episodes": 800,
              "attack_episodes": 2400, "eligible_rows": 2292, "training_seed_count": 1,
              "source_suite": "libero_object", "long_reported_secondary": True,
              "shared_prefix_counted": 250, "bootstrap_draws": draws,
              "provenance_columns_preserved": True,
              "historical_source_hash_missing": train["unrecorded_legacy_sources"],
              "engineering_completion_is_not_mechanism_or_visual_success": True}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    print(json.dumps(analyze(parser.parse_args().draws), indent=2))
