#!/usr/bin/env python3
from __future__ import annotations

import json

import pandas as pd

from common import (CONFIG_PATH, ROOT, atomic_json, atomic_parquet, load_json,
                    require, sha256_file)
from statistics import paired_contrast_bootstrap, task_episode_bootstrap

ATTACK_CONDITIONS = ("dino_norm_1500", "dino_continue_u0100",
                     "ramp_t_u0100", "ramp_tg_u0100")
CONTRASTS = (("ramp_t_u0100", "dino_continue_u0100"),
             ("ramp_tg_u0100", "dino_continue_u0100"),
             ("ramp_tg_u0100", "ramp_t_u0100"),
             ("dino_continue_u0100", "dino_norm_1500"))
MECHANISM_METRICS = ("d_action", "d_persistent", "dino_displacement",
                     "geometry_energy", "dino_restore_reduction_fraction",
                     "trajectory_cumulative_drift")


def eligible_attacks(rollouts: pd.DataFrame) -> pd.DataFrame:
    """Restrict CASR to paired states whose current clean rerun succeeds."""
    clean = (rollouts.loc[rollouts.condition == "clean",
                          ["task_id", "state_index", "success"]]
             .rename(columns={"success": "clean_success"}))
    attacks = rollouts.loc[rollouts.condition.isin(ATTACK_CONDITIONS)].copy()
    attacks = attacks.merge(clean, on=["task_id", "state_index"], validate="many_to_one")
    return attacks.loc[attacks.clean_success == 1].copy()


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rollout_paths = [ROOT / "outputs" / "continuous_cldev" /
                     f"task_{task:02d}_rollouts.parquet" for task in range(10)]
    trace_paths = [ROOT / "outputs" / "continuous_cldev" /
                   f"task_{task:02d}_traces.parquet" for task in range(10)]
    require(all(path.is_file() for path in rollout_paths), "CL-Dev rollout shards are incomplete")
    require(all(path.is_file() for path in trace_paths), "CL-Dev trace shards are incomplete")
    rollouts = pd.concat([pd.read_parquet(path) for path in rollout_paths], ignore_index=True)
    traces = pd.concat([pd.read_parquet(path) for path in trace_paths], ignore_index=True)
    require(len(rollouts) == 250, f"Expected 250 CL-Dev rollouts, got {len(rollouts)}")
    require(not rollouts.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate CL-Dev rollout")
    require(set(rollouts.task_id.astype(int)) == set(range(10)), "Task coverage drift")
    require(set(rollouts.condition) == {"clean", *ATTACK_CONDITIONS}, "Condition coverage drift")
    require(not traces.duplicated(["task_id", "state_index", "condition",
                                   "decision_block"]).any(), "Duplicate CL-Dev trace")
    return rollouts, traces


def main() -> None:
    cfg = load_json(CONFIG_PATH)
    seed = int(cfg["bootstrap_seed"])
    rollouts, traces = load_and_validate()
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    raw_path = results / "CLDEV_CONTINUOUS_ROLLOUTS.parquet"
    trace_path = results / "CLDEV_CONTINUOUS_TRACES.parquet"
    atomic_parquet(rollouts, raw_path)
    atomic_parquet(traces, trace_path)

    attacks_all = rollouts.loc[rollouts.condition.isin(ATTACK_CONDITIONS)].copy()
    eligible = eligible_attacks(rollouts)
    require(set(eligible.task_id.astype(int)) == set(range(10)),
            "At least one current clean-success state per task is required")
    eligible["casr"] = eligible["failure"].astype(float)
    eligible["attacked_success"] = eligible["success"].astype(float)
    attacks_all["raw_failure"] = attacks_all["failure"].astype(float)

    summary_parts = []
    for frame, metric, population in ((eligible, "casr", "current_clean_success"),
                                      (eligible, "attacked_success", "current_clean_success"),
                                      (attacks_all, "raw_failure", "all_selected_states")):
        part = task_episode_bootstrap(frame, metric, "condition", 10000, seed)
        part["analysis_population"] = population
        summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary_path = results / "CLDEV_CONTINUOUS_SUMMARY.csv"
    summary.to_csv(summary_path, index=False)

    contrasts = []
    for index, (left, right) in enumerate(CONTRASTS):
        row = paired_contrast_bootstrap(eligible, "casr", "condition", left, right,
                                        10000, seed + index)
        row["analysis_population"] = "current_clean_success"
        contrasts.append(row)
    contrast_path = results / "CLDEV_CONTINUOUS_CONTRASTS.csv"
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)

    per_task = (eligible.groupby(["task_id", "condition"], as_index=False)
                .agg(clean_success_denominator=("state_index", "nunique"),
                     casr=("casr", "mean"), attacked_success=("success", "mean"),
                     mean_episode_steps=("episode_steps", "mean")))
    per_task_path = results / "CLDEV_CONTINUOUS_PER_TASK.csv"
    per_task.to_csv(per_task_path, index=False)

    open_paths = [ROOT / "outputs" / "open_loop" / f"task_{task:02d}.parquet"
                  for task in range(10)]
    require(all(path.is_file() for path in open_paths), "R2 mechanism shards are incomplete")
    open_loop = pd.concat([pd.read_parquet(path) for path in open_paths], ignore_index=True)
    mechanism = (open_loop.loc[open_loop.condition.isin(ATTACK_CONDITIONS)]
                 .groupby(["task_id", "state_index", "condition"], as_index=False)
                 [list(MECHANISM_METRICS)].mean())
    joined = eligible.merge(mechanism, on=["task_id", "state_index", "condition"],
                            validate="one_to_one")
    require(len(joined) == len(eligible), "Mechanism/closed-loop join lost eligible episodes")
    joined_path = results / "CLDEV_MECHANISM_JOINED.parquet"
    atomic_parquet(joined, joined_path)

    joined["outcome"] = joined.failure.map({0: "attack_success_no", 1: "attack_success_yes"})
    joined["condition_outcome"] = joined.condition.astype(str) + "|" + joined.outcome
    mechanism_summary = []
    for metric in MECHANISM_METRICS:
        part = task_episode_bootstrap(joined, metric, "condition_outcome", 10000, seed)
        mechanism_summary.append(part)
    mechanism_path = results / "CLDEV_MECHANISM_BY_OUTCOME.csv"
    pd.concat(mechanism_summary, ignore_index=True).to_csv(mechanism_path, index=False)

    status_path = ROOT / "reports" / "CLDEV_STATUS.json"
    atomic_json({"status": "COMPLETE", "stage": "CL-Dev", "rows": len(rollouts),
                 "trace_rows": len(traces), "eligible_attack_rows": len(eligible),
                 "clean_success_states": int(eligible[["task_id", "state_index"]]
                                             .drop_duplicates().shape[0]),
                 "summary_sha256": sha256_file(summary_path),
                 "contrasts_sha256": sha256_file(contrast_path),
                 "stop": "AWAIT_USER_BRANCH_A_OR_B"}, status_path)
    print(json.dumps({"status": "COMPLETE", "stage": "CL-Dev", "rows": len(rollouts),
                      "eligible_attack_rows": len(eligible),
                      "stop": "AWAIT_USER_BRANCH_A_OR_B"}, indent=2))


if __name__ == "__main__":
    main()
