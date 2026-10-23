#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, atomic_json, atomic_parquet, load_json, require, sha256_file
from statistics import paired_contrast_bootstrap, task_episode_bootstrap


CONFIG = ROOT / "configs" / "CID_CF_STAGE_D.json"
DRAWS = 10_000
KEYS = ["suite", "task_id", "state_index"]


def eligible_attacks(rollouts: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    clean = (rollouts.loc[rollouts.condition == "clean", KEYS + ["success"]]
             .rename(columns={"success": "clean_success"}))
    require(not clean.duplicated(KEYS).any(), "Duplicate clean eligibility row")
    attacked = rollouts.loc[rollouts.condition.isin(conditions)].merge(
        clean, on=KEYS, validate="many_to_one")
    eligible = attacked.loc[attacked.clean_success == 1].copy()
    eligible["casr"] = eligible.failure.astype(float)
    return eligible


def _nested(frame: pd.DataFrame, value: str):
    return {(str(suite), int(task)): rows[value].to_numpy(float)
            for (suite, task), rows in frame.groupby(["suite", "task_id"])}


def _fixed_suite_draws(frame: pd.DataFrame, value: str, suites: list[str], draws: int,
                       seed: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    nested = _nested(frame, value)
    rng = np.random.default_rng(seed)
    suite_points, sampled = [], np.empty((draws, len(suites)), dtype=np.float64)
    for suite_index, suite in enumerate(suites):
        tasks = sorted(task for name, task in nested if name == suite)
        require(len(tasks) > 0, f"No eligible rows for {suite}")
        suite_points.append(float(np.mean([nested[(suite, task)].mean() for task in tasks])))
        for draw in range(draws):
            chosen = rng.choice(tasks, size=len(tasks), replace=True)
            sampled[draw, suite_index] = np.mean([
                rng.choice(nested[(suite, int(task))],
                           size=len(nested[(suite, int(task))]), replace=True).mean()
                for task in chosen
            ])
    points = np.asarray(suite_points)
    return float(points.mean()), float(points.min()), sampled.mean(axis=1), sampled.min(axis=1)


def development_summary(frame: pd.DataFrame, conditions: list[str], suites: list[str],
                        seed: int, draws: int = DRAWS) -> pd.DataFrame:
    rows = []
    for index, condition in enumerate(conditions):
        subset = frame.loc[frame.condition == condition]
        macro, worst, macro_draws, worst_draws = _fixed_suite_draws(
            subset, "casr", suites, draws, seed + index)
        rows.extend([
            {"condition": condition, "metric": "macro_casr", "estimate": macro,
             "ci95_low": float(np.quantile(macro_draws, 0.025)),
             "ci95_high": float(np.quantile(macro_draws, 0.975)),
             "suites": len(suites), "eligible_episodes": len(subset)},
            {"condition": condition, "metric": "worst_suite_casr", "estimate": worst,
             "ci95_low": float(np.quantile(worst_draws, 0.025)),
             "ci95_high": float(np.quantile(worst_draws, 0.975)),
             "suites": len(suites), "eligible_episodes": len(subset)},
        ])
    return pd.DataFrame(rows)


def development_contrast(frame: pd.DataFrame, left: str, right: str, suites: list[str],
                         seed: int, draws: int = DRAWS) -> list[dict]:
    wide = frame.pivot_table(index=KEYS, columns="condition", values="casr", aggfunc="mean")
    paired = wide[[left, right]].dropna().reset_index()
    nested = {(str(suite), int(task)): rows[[left, right]].to_numpy(float)
              for (suite, task), rows in paired.groupby(["suite", "task_id"])}
    point = np.empty((len(suites), 2), dtype=np.float64)
    sampled = np.empty((draws, len(suites), 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for suite_index, suite in enumerate(suites):
        tasks = sorted(task for name, task in nested if name == suite)
        require(len(tasks) > 0, f"No paired development rows for {suite}")
        point[suite_index] = np.mean([nested[(suite, task)].mean(axis=0) for task in tasks], axis=0)
        for draw in range(draws):
            chosen = rng.choice(tasks, size=len(tasks), replace=True)
            task_means = []
            for task in chosen:
                values = nested[(suite, int(task))]
                indices = rng.choice(len(values), size=len(values), replace=True)
                task_means.append(values[indices].mean(axis=0))
            sampled[draw, suite_index] = np.mean(task_means, axis=0)
    macro = float(point[:, 0].mean() - point[:, 1].mean())
    worst = float(point[:, 0].min() - point[:, 1].min())
    macro_draws = sampled[:, :, 0].mean(axis=1) - sampled[:, :, 1].mean(axis=1)
    worst_draws = sampled[:, :, 0].min(axis=1) - sampled[:, :, 1].min(axis=1)
    return [
        {"scope": "Spatial+Goal", "metric": "macro_casr_difference",
         "left": left, "right": right, "difference_left_minus_right": macro,
         "ci95_low": float(np.quantile(macro_draws, 0.025)),
         "ci95_high": float(np.quantile(macro_draws, 0.975)),
         "paired_episodes": len(paired)},
        {"scope": "Spatial+Goal", "metric": "worst_suite_difference",
         "left": left, "right": right, "difference_left_minus_right": worst,
         "ci95_low": float(np.quantile(worst_draws, 0.025)),
         "ci95_high": float(np.quantile(worst_draws, 0.975)),
         "paired_episodes": len(paired)},
    ]


def main() -> None:
    cfg = load_json(CONFIG)
    require(cfg["long_required"] is False and "libero_10" not in cfg["evaluation_suites"],
            "Stage-D analysis Long boundary drift")
    new_frames, trace_frames = [], []
    for task_id in range(int(cfg["task_count"])):
        base = ROOT / "outputs" / "cid_cf" / "stage_d"
        rpath = base / f"task_{task_id:02d}_rollouts.parquet"
        tpath = base / f"task_{task_id:02d}_traces.parquet"
        require(rpath.is_file() and tpath.is_file(), f"Missing Stage-D shard: task {task_id}")
        new_frames.append(pd.read_parquet(rpath))
        trace_frames.append(pd.read_parquet(tpath))
    new = pd.concat(new_frames, ignore_index=True)
    traces = pd.concat(trace_frames, ignore_index=True)
    expected_new = (len(cfg["evaluation_suites"]) * int(cfg["task_count"])
                    * len(cfg["state_indices"]) * len(cfg["new_methods"]))
    require(len(new) == expected_new, f"Expected {expected_new} new rollouts, got {len(new)}")
    require(not new.duplicated(KEYS + ["condition"]).any(), "Duplicate Stage-D rollout")

    old_all = pd.read_parquet(cfg["paths"]["existing_rollouts"])
    old = old_all.loc[old_all.suite.isin(cfg["evaluation_suites"])].copy()
    require(len(old) == 2400, "Frozen three-suite baseline grid drift")
    require(not old.duplicated(KEYS + ["condition"]).any(), "Duplicate frozen baseline rollout")
    require(set(new.condition) == set(cfg["new_methods"]), "New method coverage drift")
    rollouts = pd.concat([old, new], ignore_index=True, sort=False)
    all_conditions = old.condition.drop_duplicates().tolist() + list(cfg["new_methods"])
    require(not rollouts.duplicated(KEYS + ["condition"]).any(), "Combined Stage-D grid duplicate")
    eligible = eligible_attacks(rollouts, [c for c in all_conditions if c != "clean"])

    results = ROOT / "results"
    raw_path = results / "CID_CF_STAGE_D_NEW_ROLLOUTS.parquet"
    trace_path = results / "CID_CF_STAGE_D_NEW_TRACES.parquet"
    eligible_path = results / "CID_CF_STAGE_D_ELIGIBLE.parquet"
    atomic_parquet(new, raw_path)
    atomic_parquet(traces, trace_path)
    atomic_parquet(eligible, eligible_path)

    per_suite_rows = []
    for suite_index, suite in enumerate(cfg["evaluation_suites"]):
        subset = eligible.loc[eligible.suite == suite]
        summary = task_episode_bootstrap(
            subset, "casr", "condition", DRAWS, int(cfg["bootstrap_seed"]) + suite_index)
        summary.insert(0, "scope", suite)
        per_suite_rows.append(summary)
    per_suite_path = results / "CID_CF_STAGE_D_PER_SUITE.csv"
    pd.concat(per_suite_rows, ignore_index=True).to_csv(per_suite_path, index=False)

    development = eligible.loc[eligible.suite.isin(cfg["development_targets"])].copy()
    development_conditions = [cfg["frozen_reference"], *cfg["new_methods"]]
    aggregate = development_summary(
        development, development_conditions, cfg["development_targets"],
        int(cfg["bootstrap_seed"]))
    aggregate_path = results / "CID_CF_STAGE_D_DEVELOPMENT_AGGREGATE.csv"
    aggregate.to_csv(aggregate_path, index=False)

    contrast_rows = []
    seed = int(cfg["bootstrap_seed"])
    for suite_index, suite in enumerate(cfg["evaluation_suites"]):
        subset = eligible.loc[eligible.suite == suite]
        for ref_index, reference in enumerate(cfg["primary_contrasts"]):
            row = paired_contrast_bootstrap(
                subset, "casr", "condition", cfg["primary_method"], reference,
                DRAWS, seed + suite_index * 100 + ref_index)
            row["scope"] = suite
            row["metric"] = "casr_difference"
            contrast_rows.append(row)
    for ref_index, reference in enumerate(cfg["primary_contrasts"]):
        contrast_rows.extend(development_contrast(
            development, cfg["primary_method"], reference,
            cfg["development_targets"], seed + 1000 + ref_index))
    contrast_rows.extend(development_contrast(
        development, cfg["component_contrast"][0], cfg["component_contrast"][1],
        cfg["development_targets"], seed + 2000))
    contrast_path = results / "CID_CF_STAGE_D_CONTRASTS.csv"
    pd.DataFrame(contrast_rows).to_csv(contrast_path, index=False)

    clean_counts = (old.loc[old.condition == "clean"].groupby("suite").success
                    .agg(["sum", "count"]).reset_index().to_dict("records"))
    status = {
        "status": "COMPLETE", "stage": "CID_CF_STAGE_D_CLOSED_LOOP",
        "claim_scope": cfg["claim_scope"], "new_rollouts": len(new),
        "new_trace_rows": len(traces), "reused_old_rollouts": len(old),
        "eligible_attack_rows": len(eligible), "evaluation_suites": cfg["evaluation_suites"],
        "development_targets": cfg["development_targets"], "clean_success": clean_counts,
        "long_required": False, "long_accessed": False,
        "per_suite_sha256": sha256_file(per_suite_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "contrasts_sha256": sha256_file(contrast_path),
        "stop": "STAGE_D_COMPLETE_OPTIONAL_LONG_NOT_REQUIRED",
    }
    atomic_json(status, ROOT / "reports" / "CID_CF_STAGE_D_STATUS.json")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
