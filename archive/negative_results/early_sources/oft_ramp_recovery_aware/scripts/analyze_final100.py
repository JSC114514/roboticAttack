#!/usr/bin/env python3
from __future__ import annotations

import json

import pandas as pd

from common import ROOT, atomic_json, atomic_parquet, load_json, require, sha256_file
from run_final100 import FINAL_CONFIG_PATH
from statistics import paired_contrast_bootstrap, task_episode_bootstrap


def eligible_attacks(rollouts: pd.DataFrame, attacks: tuple[str, ...]) -> pd.DataFrame:
    clean = (rollouts.loc[rollouts.condition == "clean", ["task_id", "state_index", "success"]]
             .rename(columns={"success": "clean_success"}))
    attacked = rollouts.loc[rollouts.condition.isin(attacks)].copy()
    attacked = attacked.merge(clean, on=["task_id", "state_index"], validate="many_to_one")
    return attacked.loc[attacked.clean_success == 1].copy()


def main() -> None:
    cfg = load_json(FINAL_CONFIG_PATH)
    conditions = tuple(cfg["conditions"])
    attacks = tuple(condition for condition in conditions if condition != "clean")
    rollout_paths = [ROOT / "outputs" / "final100" / f"task_{task:02d}_rollouts.parquet"
                     for task in range(int(cfg["task_count"]))]
    trace_paths = [ROOT / "outputs" / "final100" / f"task_{task:02d}_traces.parquet"
                   for task in range(int(cfg["task_count"]))]
    require(all(path.is_file() for path in rollout_paths), "Final-100 rollout shards incomplete")
    require(all(path.is_file() for path in trace_paths), "Final-100 trace shards incomplete")
    rollouts = pd.concat([pd.read_parquet(path) for path in rollout_paths], ignore_index=True)
    traces = pd.concat([pd.read_parquet(path) for path in trace_paths], ignore_index=True)
    expected = int(cfg["task_count"]) * len(cfg["final_test_state_indices"]) * len(conditions)
    require(len(rollouts) == expected, f"Expected {expected} Final-100 rollouts, got {len(rollouts)}")
    require(not rollouts.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate Final-100 rollout")
    require(set(rollouts.condition) == set(conditions), "Final-100 condition drift")
    require(set(rollouts.state_index.astype(int)) == set(cfg["final_test_state_indices"]),
            "Final-100 split drift")

    eligible = eligible_attacks(rollouts, attacks)
    require(set(eligible.task_id.astype(int)) == set(range(int(cfg["task_count"]))),
            "Every task requires at least one clean-success Final-100 state")
    eligible["casr"] = eligible.failure.astype(float)
    eligible["attacked_success"] = eligible.success.astype(float)
    raw_attacks = rollouts.loc[rollouts.condition.isin(attacks)].copy()
    raw_attacks["raw_failure"] = raw_attacks.failure.astype(float)

    results = ROOT / "results"
    raw_path = results / "FINAL_100_ROLLOUTS.parquet"
    trace_path = results / "FINAL_100_TRACES.parquet"
    atomic_parquet(rollouts, raw_path)
    atomic_parquet(traces, trace_path)
    summaries = [task_episode_bootstrap(
        eligible, metric, "condition", 10000, int(cfg["bootstrap_seed"]))
        for metric in ("casr", "attacked_success")]
    summaries.append(task_episode_bootstrap(
        raw_attacks, "raw_failure", "condition", 10000, int(cfg["bootstrap_seed"])))
    summary_path = results / "FINAL_100_SUMMARY.csv"
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)

    references = tuple(cfg["primary_references"]) + tuple(cfg["secondary_references"])
    primary = str(cfg["primary_method"])
    contrasts = [paired_contrast_bootstrap(
        eligible, "casr", "condition", primary, reference, 10000,
        int(cfg["bootstrap_seed"]) + index)
        for index, reference in enumerate(references)]
    for row, reference in zip(contrasts, references, strict=True):
        row["contrast_role"] = "primary" if reference in cfg["primary_references"] else "secondary"
    contrast_path = results / "FINAL_100_CONTRASTS.csv"
    pd.DataFrame(contrasts).to_csv(contrast_path, index=False)
    per_task_path = results / "FINAL_100_PER_TASK.csv"
    (eligible.groupby(["task_id", "condition"], as_index=False)
     .agg(clean_success_denominator=("state_index", "nunique"), casr=("casr", "mean"),
          attacked_success=("success", "mean"), mean_episode_steps=("episode_steps", "mean"))
     .to_csv(per_task_path, index=False))

    clean_states = int(eligible[["task_id", "state_index"]].drop_duplicates().shape[0])
    status_path = ROOT / "reports" / "FINAL_100_STATUS.json"
    atomic_json({"status": "COMPLETE", "stage": "FINAL_100", "rollouts": len(rollouts),
                 "trace_rows": len(traces), "clean_success_states": clean_states,
                 "eligible_attack_rows": len(eligible), "primary_method": primary,
                 "final_test_state_indices": cfg["final_test_state_indices"],
                 "summary_sha256": sha256_file(summary_path),
                 "contrasts_sha256": sha256_file(contrast_path),
                 "stop": "FINAL_100_COMPLETE_NO_AUTOMATIC_EXTENSION"}, status_path)
    print(json.dumps(load_json(status_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
