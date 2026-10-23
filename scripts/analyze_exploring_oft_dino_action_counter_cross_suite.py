#!/usr/bin/env python3
"""Frozen task->state paired bootstrap for the six fresh arms and two 500 controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dino_action_counter_common import CONFIG, config, digest_json, external_baselines, require_preparation
from run_exploring_oft_dino_action_counter_cross_suite import (
    CONDITIONS, DENOMINATORS, EXPERIMENT_ID, METHODS, KEY, load_config, patches,
    validate_new_shard, validate_runtime, require, sha256_file,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet

PAIRS = [
    ("D0_minus_R0", "dino_original", "random_original"),
    ("D1_minus_R1", "dino_clear", "random_clear"),
    ("R0_minus_N0", "random_original", "none_original"),
    ("R1_minus_N1", "random_clear", "none_clear"),
    ("N1_minus_N0", "none_clear", "none_original"),
    ("R1_minus_R0", "random_clear", "random_original"),
    ("D1_minus_D0", "dino_clear", "dino_original"),
    ("D0_minus_N0", "dino_original", "none_original"),
    ("D1_minus_N1", "dino_clear", "none_clear"),
]
PAIRS += [(f"{m}_minus_original_uada500", m, "original_uada_500") for m in METHODS]
PAIRS += [(f"{m}_minus_legacy_A500", m, "diverse_reference_500") for m in METHODS]
PAIRS += [("legacy_A500_minus_original_uada500", "diverse_reference_500", "original_uada_500")]


def contrast_vectors():
    vectors = {}
    for name, left, right in PAIRS:
        value = np.zeros(len(CONDITIONS))
        value[CONDITIONS.index(left)] = 1
        value[CONDITIONS.index(right)] = -1
        vectors[name] = (value, left, right)
    value = np.zeros(len(CONDITIONS))
    for method, weight in (("dino_clear", 1), ("random_clear", -1),
                           ("dino_original", -1), ("random_original", 1)):
        value[CONDITIONS.index(method)] = weight
    vectors["appearance_x_direction_interaction"] = (value, "D1-R1", "D0-R0")
    return vectors


def hierarchical_samples(eligible, draws, rng):
    """Share every resampled task/state index across all eight conditions."""
    require(draws >= 1 and not eligible.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate paired bootstrap key")
    wide = eligible.pivot(index=["task_id", "state_index"], columns="condition", values="casr")
    wide = wide.reindex(columns=CONDITIONS)
    require(not wide.isna().any().any() and np.isfinite(wide.to_numpy(float)).all()
            and wide.isin([0, 1]).all().all(), "Unpaired/nonfinite/binary CASR data")
    tasks = sorted(wide.index.get_level_values("task_id").unique())
    require(tasks == list(range(10)), "All ten tasks require clean-success states")
    arrays = [wide.xs(task, level="task_id").to_numpy(float) for task in tasks]
    selected = rng.integers(0, 10, size=(draws, 10))
    samples = np.zeros((draws, len(CONDITIONS)))
    for position in range(10):
        for task, values in enumerate(arrays):
            ids = np.flatnonzero(selected[:, position] == task)
            states = rng.integers(0, len(values), size=(len(ids), len(values)))
            samples[ids] += values[states].mean(axis=1) / 10
    per_task = np.stack([values.mean(axis=0) for values in arrays])
    return per_task.mean(axis=0), samples, per_task


def paired_statistics(eligible, draws=10000, seed=20260907):
    rng = np.random.default_rng(seed)
    points, samples, tasks = {}, {}, {}
    for suite in DENOMINATORS:
        point, sample, per_task = hierarchical_samples(eligible.loc[eligible.suite.eq(suite)], draws, rng)
        points[suite], samples[suite], tasks[suite] = point, sample, per_task
    groups = {suite: [suite] for suite in DENOMINATORS}
    groups["spatial_goal_macro"] = ["libero_spatial", "libero_goal"]
    groups["four_suite_macro"] = list(DENOMINATORS)
    summaries, comparisons = [], []
    for group, suites in groups.items():
        point = np.mean([points[s] for s in suites], axis=0)
        sample = np.mean([samples[s] for s in suites], axis=0)
        task_values = np.concatenate([tasks[s] for s in suites], axis=0)
        for j, method in enumerate(CONDITIONS):
            low, high = np.percentile(sample[:, j], [2.5, 97.5])
            summaries.append({"suite_or_endpoint": group, "condition": method,
                "task_equal_casr": float(point[j]), "ci95_low": float(low), "ci95_high": float(high),
                "clean_success_states": sum(DENOMINATORS[s] for s in suites),
                "tasks": 10 * len(suites), "suite_equal": len(suites) > 1})
        for name, (vector, left, right) in contrast_vectors().items():
            delta = task_values @ vector
            distribution = sample @ vector
            low, high = np.percentile(distribution, [2.5, 97.5])
            comparisons.append({"suite_or_endpoint": group, "contrast": name, "left": left, "right": right,
                "difference_left_minus_right": float(point @ vector),
                "ci95_low": float(low), "ci95_high": float(high),
                "positive_tasks": int((delta > 0).sum()), "negative_tasks": int((delta < 0).sum()),
                "tied_tasks": int((delta == 0).sum()), "tasks": len(delta),
                "paired_clean_success_states": sum(DENOMINATORS[s] for s in suites)})
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def validate_raw(frame, cfg):
    require(len(frame) == 3200 and set(frame.condition) == set(CONDITIONS)
            and set(frame.suite) == set(DENOMINATORS) and not frame.duplicated(KEY).any(),
            "Incomplete/duplicate 3200-episode raw grid")
    require(frame.groupby(["suite", "condition"]).size().eq(100).all()
            and frame.groupby(["suite", "task_id", "condition"]).size().eq(10).all()
            and frame.success.isin([0, 1]).all() and frame.failure.eq(1-frame.success).all()
            and frame.frozen_clean_success.isin([0, 1]).all()
            and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Raw coverage/success/finite data failed")
    require(frame.groupby(["suite", "task_id", "state_index"]).frozen_clean_success.nunique().eq(1).all()
            and frame.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(),
            "Unpaired clean qualification or seeds")
    for suite, expected in DENOMINATORS.items():
        eligible = frame.loc[frame.suite.eq(suite) & frame.frozen_clean_success.eq(1)]
        require(eligible.groupby("condition").size().eq(expected).all(), "Frozen eligible count changed")
    return frame.loc[frame.frozen_clean_success.eq(1)].assign(casr=lambda f: f.failure.astype(float)).copy()


def analyze(draws=10000):
    train, cfg = config(), load_config()
    require(draws == train["bootstrap_draws"] == 10000, "Formal bootstrap is frozen at 10000 draws")
    preparation = require_preparation(train)
    _, hashes, _ = patches(cfg, "cpu")
    output = Path(cfg["output_root"])
    require(json.loads((output / "EVAL_CONFIG.json").read_text()) == cfg, "Final evaluation config changed")
    parts, trace_parts, artifact_hashes = [], [], {}
    for task in range(10):
        receipt = json.loads((output / f"task_{task:02d}_STATUS.json").read_text())
        require(receipt["status"] == "COMPLETE" and receipt["task_id"] == task
                and receipt["config_signature"] == digest_json(cfg) and receipt["patches"] == hashes,
                "Task receipt incomplete or identity changed")
        for suite in cfg["suites"]:
            rp, tp = [output / suite / f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
            require(sha256_file(rp) == receipt["suites"][suite]["rollouts_sha256"]
                    and sha256_file(tp) == receipt["suites"][suite]["traces_sha256"],
                    "Committed evaluation shard bytes changed")
            rows, traces = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rows, traces, suite, task, cfg, hashes)
            validate_runtime(rows, traces, cfg, suite, task)
            for frame, path in ((rows, rp), (traces, tp)):
                if "source_path" not in frame:
                    frame["source_path"] = ""
                if "source_sha256" not in frame:
                    frame["source_sha256"] = ""
                frame.loc[~frame.reused.astype(bool), "source_path"] = str(path)
                frame.loc[~frame.reused.astype(bool), "source_sha256"] = sha256_file(path)
                artifact_hashes[str(path)] = sha256_file(path)
            parts.append(rows)
            trace_parts.append(traces)
    frame = pd.concat(parts, ignore_index=True)
    eligible = validate_raw(frame, cfg)
    require(len(eligible) == 3056, "Expected 8 x 382 eligible repeated-measure rows")
    summary, contrasts = paired_statistics(eligible, draws, train["bootstrap_seed"])
    per_task = eligible.groupby(["suite", "task_id", "condition"], as_index=False).agg(
        clean_success_states=("state_index", "nunique"), casr=("casr", "mean"), attacked_success=("success", "mean"))
    prefix = "DINO_ACTION_COUNTER500_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    for suffix, data in (("SUMMARY", summary), ("CONTRASTS", contrasts), ("PER_TASK", per_task)):
        data.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
    costs, training_metrics = [], []
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        history = pd.read_parquet(folder / "training.parquet")
        require(len(history) == 500 and np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Training metrics incomplete/nonfinite")
        attack = frame.loc[frame.condition.eq(method)]
        costs.append({"condition": method, "new_training_elapsed_s": status["elapsed_s"],
            "optimizer_updates": 25000, "fresh_start_outer": 0, "costs": status["costs"],
            "new_evaluation_elapsed_s": float(attack.loc[~attack.reused.astype(bool)].elapsed_s.sum()),
            "evaluation_episodes": len(attack), "training_seed_count": 1})
        metrics = {"condition": method, "training_seed_count": 1,
            "tv_threshold": train["tv_thresholds"][method.rsplit("_", 1)[1]],
            "zero_update_outer_count": int(history.zero_update_outer.sum()),
            "final_tv": float(history.iloc[-1].tv), "final_hf_energy": float(history.iloc[-1].hf_energy)}
        for name in ("tv_violated", "counter_loss_delta", "counter_loss_before", "counter_loss_after",
                     "counter_effective", "random_match_linf_error", "random_match_rms_error",
                     "random_match_mean_abs_error", "random_zero_eta",
                     "augmentation_loss_delta", "clean_action_distance", "clean_action_distance_before_eta",
                     "counter_selected_clean_action_distance", "eta_linf", "eta_rms", "eta_mean_abs",
                     "rgb_outer_mean_abs", "rgb_outer_max_abs", "png_changed_fraction"):
            if name in history:
                metrics[f"mean_{name}"] = float(history[name].mean())
        training_metrics.append(metrics)
    for method, item in external_baselines(train).items():
        attack = frame.loc[frame.condition.eq(method)]
        costs.append({"condition": method, "new_training_elapsed_s": 0., "optimizer_updates_at_500": 25000,
            "recorded_training_to_500_elapsed_s": item["recorded_training_elapsed_s"],
            "new_evaluation_elapsed_s": float(attack.loc[~attack.reused.astype(bool)].elapsed_s.sum()),
            "recorded_reused_evaluation_elapsed_s": float(attack.loc[attack.reused.astype(bool)].elapsed_s.sum()),
            "evaluation_episodes": len(attack), "pt_sha256": item["pt"],
            "note": "Existing matched-500 checkpoint, no retraining."})
    atomic_json(costs, output / f"{prefix}_COSTS.json")
    pd.DataFrame(training_metrics).to_csv(output / f"{prefix}_TRAINING_METRICS.csv", index=False)
    reused = int(frame.reused.astype(bool).sum())
    report = {"status": "COMPLETE", "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG), "preparation_signature": preparation["signature"],
        "evaluation_config_signature": digest_json(cfg), "checkpoints": hashes,
        "attack_episodes": 3200, "new_attack_episodes": 3200-reused, "reused_attack_episodes": reused,
        "eligible_rows": len(eligible), "frozen_clean_episodes": 400, "denominators": DENOMINATORS,
        "bootstrap_draws": draws, "bootstrap_seed": 20260907, "bootstrap_unit": "task then clean-success state",
        "paired_resampling_all_conditions": True, "task_equal_then_suite_equal": True,
        "primary_endpoint": "spatial_goal_macro", "training_seed_count": 1, "long_secondary": True,
        "historical_source_hash_missing": train["historical_source_hash_missing"] if reused else [],
        "raw_shard_hashes": artifact_hashes, "stop_after_outer": 500,
        "no_next_stage_submission": True,
        "interpretation": "Single-seed screening conditional on the frozen historical source schedule; "
        "episode replication is not independent training replication. Engineering completion does not establish mechanism, "
        "visual quality, equivalence, or a blind confirmation result."}
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    print(json.dumps(analyze(parser.parse_args().draws), indent=2))
