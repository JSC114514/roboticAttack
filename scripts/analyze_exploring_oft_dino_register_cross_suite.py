#!/usr/bin/env python3
"""Task/state paired bootstrap for OFT-DINO-REGISTER500-V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dino_register_common import (
    CHECKPOINT, CONDITIONS, CONFIG, DENOMINATORS, METHODS, REFERENCES, config,
    digest_json, reference_patches, require_preparation,
)
from run_exploring_oft_dino_register_cross_suite import (
    EXPERIMENT_ID, KEY, load_config, patches, validate_new_shard, validate_runtime,
)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_parquet, require, sha256_file

PAIRS = [
    ("register_g10_minus_uniform_axis0", "register_g10", "uniform_axis0_500"),
    ("register_g30_minus_uniform_axis0", "register_g30", "uniform_axis0_500"),
    ("spatial_g30_minus_uniform_axis0", "spatial_g30", "uniform_axis0_500"),
    ("register_g10_minus_original_uada", "register_g10", "original_uada_500"),
    ("register_g30_minus_original_uada", "register_g30", "original_uada_500"),
    ("spatial_g30_minus_original_uada", "spatial_g30", "original_uada_500"),
    ("register_g30_minus_spatial_g30", "register_g30", "spatial_g30"),
    ("register_g10_minus_spatial_g30", "register_g10", "spatial_g30"),
    ("uniform_axis0_minus_original_uada", "uniform_axis0_500", "original_uada_500"),
]


def contrast_vectors():
    vectors = {}
    for name, left, right in PAIRS:
        value = np.zeros(len(CONDITIONS))
        value[CONDITIONS.index(left)] = 1
        value[CONDITIONS.index(right)] = -1
        vectors[name] = (value, left, right)
    return vectors


def hierarchical_samples(eligible, draws, rng):
    require(draws >= 1 and not eligible.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate paired bootstrap key")
    wide = eligible.pivot(index=["task_id", "state_index"], columns="condition", values="casr")
    wide = wide.reindex(columns=list(CONDITIONS))
    require(not wide.isna().any().any()
            and np.isfinite(wide.to_numpy(float)).all()
            and wide.isin([0, 1]).all().all(),
            "Unpaired/nonfinite/binary CASR data")
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
        for index, condition in enumerate(CONDITIONS):
            low, high = np.percentile(sample[:, index], [2.5, 97.5])
            summaries.append({
                "checkpoint_outer": CHECKPOINT,
                "suite_or_endpoint": group,
                "condition": condition,
                "task_equal_casr": float(point[index]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "clean_success_states": sum(DENOMINATORS[s] for s in suites),
                "tasks": 10 * len(suites),
                "suite_equal": len(suites) > 1,
            })
        for name, (vector, left, right) in contrast_vectors().items():
            delta, distribution = task_values @ vector, sample @ vector
            low, high = np.percentile(distribution, [2.5, 97.5])
            comparisons.append({
                "checkpoint_outer": CHECKPOINT,
                "suite_or_endpoint": group,
                "contrast": name,
                "left": left,
                "right": right,
                "difference_left_minus_right": float(point @ vector),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_tasks": int((delta > 0).sum()),
                "negative_tasks": int((delta < 0).sum()),
                "tied_tasks": int((delta == 0).sum()),
                "tasks": len(delta),
                "paired_clean_success_states": sum(DENOMINATORS[s] for s in suites),
            })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def validate_raw(frame, cfg):
    expected_rows = len(CONDITIONS) * 400
    require(len(frame) == expected_rows
            and set(frame.condition) == set(CONDITIONS)
            and set(frame.suite) == set(DENOMINATORS)
            and frame.checkpoint_outer.eq(CHECKPOINT).all()
            and not frame.duplicated(KEY).any(),
            "Incomplete/duplicate raw register grid")
    require(frame.groupby(["suite", "condition"]).size().eq(100).all()
            and frame.groupby(["suite", "task_id", "condition"]).size().eq(10).all()
            and frame.success.isin([0, 1]).all()
            and frame.failure.eq(1 - frame.success).all()
            and frame.frozen_clean_success.isin([0, 1]).all()
            and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Raw coverage/success/finite data failed")
    require(frame.groupby(["suite", "task_id", "state_index"]).frozen_clean_success.nunique().eq(1).all()
            and frame.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(),
            "Unpaired clean qualification or seeds")
    for suite, expected in DENOMINATORS.items():
        eligible = frame.loc[frame.suite.eq(suite) & frame.frozen_clean_success.eq(1)]
        require(eligible.groupby("condition").size().eq(expected).all(),
                f"Frozen eligible count changed: {suite}")
    return frame.loc[frame.frozen_clean_success.eq(1)].assign(casr=lambda f: f.failure.astype(float)).copy()


def analyze(draws=10000):
    train, cfg = config(), load_config()
    require(draws == train["bootstrap_draws"] == 10000, "Formal bootstrap is frozen at 10000 draws")
    preparation = require_preparation(train)
    _, hashes = patches(cfg, "cpu")
    output = Path(cfg["output_root"])
    require(json.loads((output / "EVAL_CONFIG.json").read_text()) == cfg,
            "Final evaluation config changed")
    parts, trace_parts, artifact_hashes = [], [], {}
    for task in range(10):
        receipt = json.loads((output / f"task_{task:02d}_STATUS.json").read_text())
        require(receipt["status"] == "COMPLETE"
                and receipt["task_id"] == task
                and receipt["config_signature"] == digest_json(cfg)
                and receipt["patches"] == hashes,
                "Task receipt incomplete or patch identity changed")
        for suite in cfg["suites"]:
            rp, tp = [output / suite / f"task_{task:02d}_{kind}.parquet" for kind in ("rollouts", "traces")]
            require(sha256_file(rp) == receipt["suites"][suite]["rollouts_sha256"]
                    and sha256_file(tp) == receipt["suites"][suite]["traces_sha256"],
                    "Committed evaluation shard bytes changed")
            rows, traces = pd.read_parquet(rp), pd.read_parquet(tp)
            validate_new_shard(rows, traces, suite, task, cfg, hashes)
            validate_runtime(rows, traces, cfg, suite, task)
            for frame, path in ((rows, rp), (traces, tp)):
                frame["source_path"], frame["source_sha256"] = str(path), sha256_file(path)
                artifact_hashes[str(path)] = sha256_file(path)
            parts.append(rows)
            trace_parts.append(traces)
    frame = pd.concat(parts, ignore_index=True)
    frame["eligibility_reason"] = np.where(frame.frozen_clean_success.eq(1),
                                           "eligible_clean_success", "frozen_clean_failure")
    eligible = validate_raw(frame, cfg)
    require(len(eligible) == len(CONDITIONS) * sum(DENOMINATORS.values()),
            "Expected all eligible repeated-measure rows")
    summary, contrasts = paired_statistics(eligible, draws, train["bootstrap_seed"])
    suite_summary = summary.loc[summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    endpoints = summary.loc[~summary.suite_or_endpoint.isin(DENOMINATORS)].copy()
    per_task = eligible.groupby(["suite", "task_id", "condition"], as_index=False).agg(
        clean_success_states=("state_index", "nunique"),
        casr=("casr", "mean"),
        attacked_success=("success", "mean"),
    )
    prefix = "DINO_REGISTER500_XSUITE"
    atomic_parquet(frame, output / f"{prefix}_ROLLOUTS.parquet")
    atomic_parquet(pd.concat(trace_parts, ignore_index=True), output / f"{prefix}_TRACES.parquet")
    atomic_parquet(eligible, output / f"{prefix}_ELIGIBLE.parquet")
    for suffix, data in (("SUMMARY", suite_summary), ("ENDPOINTS", endpoints),
                         ("CONTRASTS", contrasts), ("PER_TASK", per_task)):
        data.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
    costs = []
    training = pd.read_csv(Path(train["output_root"]) / "REGISTER_TRAINING_METRICS.csv")
    training.to_csv(output / f"{prefix}_TRAINING_METRICS.csv", index=False)
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        history = pd.read_parquet(folder / "training.parquet")
        attack = frame.loc[frame.condition.eq(method)]
        row = history.loc[history.outer_iteration.eq(CHECKPOINT)].iloc[0]
        costs.append({
            "checkpoint_outer": CHECKPOINT,
            "condition": method,
            "fresh_start_outer": 0,
            "optimizer_updates": CHECKPOINT * train["inner_loop"],
            "cumulative_training_elapsed_s": float(row.elapsed_s),
            "whole_training_elapsed_s": float(status["elapsed_s"]),
            "whole_training_costs": status["costs"],
            "evaluation_elapsed_s": float(attack.elapsed_s.sum()),
            "evaluation_episodes": len(attack),
            "training_seed_count": 1,
        })
    for condition, item in reference_patches(train).items():
        attack = frame.loc[frame.condition.eq(condition)]
        costs.append({
            "checkpoint_outer": CHECKPOINT,
            "condition": condition,
            "new_training_elapsed_s": 0.,
            "optimizer_updates": CHECKPOINT * train["inner_loop"],
            "recorded_training_elapsed_s": item["recorded_training_elapsed_s"],
            "evaluation_elapsed_s": float(attack.elapsed_s.sum()),
            "evaluation_episodes": len(attack),
            "pt_sha256": item["pt"],
            "note": "Existing matched 500 checkpoint; reevaluated without retraining.",
        })
    atomic_json(costs, output / f"{prefix}_COSTS.json")
    report = {
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG),
        "preparation_signature": preparation["signature"],
        "evaluation_config_signature": digest_json(cfg),
        "checkpoints": hashes,
        "evaluation_checkpoint_outer_iterations": [CHECKPOINT],
        "attack_episodes": len(CONDITIONS) * 400,
        "new_patch_attack_episodes": len(METHODS) * 400,
        "reference_attack_episodes": len(REFERENCES) * 400,
        "eligible_rows": len(eligible),
        "artifact_hashes": artifact_hashes,
        "prefix": prefix,
    }
    atomic_json(report, output / "STATUS.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.draws), indent=2))
