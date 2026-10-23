#!/usr/bin/env python3
"""Aggregate the ten pre-registered discovery Jacobian shards."""
from __future__ import annotations

import argparse
import json

from audit_object_pose_jacobian_discovery import OUTPUT_ROOT
from collect_object_bank_spatial_masks import holdout_rank
from replay_object_bank_mask_pilot import require


MIN_OVERALL_FRACTION = 0.50
MIN_TASKS_WITH_ANY = 8


def decision_gate(states: list[dict]) -> dict:
    require(len(states) == 40, "Expected 40 independent discovery trajectories")
    passed = sum(bool(row["all_four_accepted"]) for row in states)
    tasks_any = sum(any(row["task_id"] == task and row["all_four_accepted"] for row in states)
                    for task in range(10))
    fraction = passed / len(states)
    return {"passed_trajectories": passed, "total_trajectories": len(states),
            "overall_fraction": fraction, "tasks_with_any_pass": tasks_any,
            "minimum_overall_fraction": MIN_OVERALL_FRACTION,
            "minimum_tasks_with_any_pass": MIN_TASKS_WITH_ANY,
            "advance_to_independent_holdout": bool(fraction >= MIN_OVERALL_FRACTION
                                                    and tasks_any >= MIN_TASKS_WITH_ANY)}


def summarize() -> dict:
    summary_path = OUTPUT_ROOT / "SUMMARY.json"
    require(not summary_path.exists(), f"Refusing to overwrite {summary_path}")
    states, per_task = [], []
    for task_id in range(10):
        path = OUTPUT_ROOT / f"task_{task_id:02d}.json"
        require(path.is_file(), f"Missing discovery shard {path}")
        receipt = json.loads(path.read_text())
        require(receipt["status"] == "OBJECT_POSE_JACOBIAN_DISCOVERY_TASK_COMPLETE"
                and receipt["task_id"] == task_id
                and receipt["heldout_trajectory_rank_untouched"] == holdout_rank(task_id)
                and receipt["no_counterfactual_model_inference"], "Discovery receipt drift")
        require(len(receipt["states"]) == 4
                and holdout_rank(task_id) not in {row["trajectory_rank"]
                                                  for row in receipt["states"]},
                "Heldout leakage or trajectory count drift")
        task_rows = []
        for row in receipt["states"]:
            item = {"task_id": task_id, "trajectory_rank": row["trajectory_rank"],
                    "decision_index": row["decision_index"],
                    "all_four_accepted": row["symmetric_pair_gate"]["all_four_accepted"],
                    "jacobian_condition_number": row["jacobian"]["condition_number"],
                    "accepted_directions": sum(candidate["accepted"]
                                               for candidate in row["candidates"])}
            states.append(item)
            task_rows.append(item)
        per_task.append({"task_id": task_id,
                         "passed_trajectories": sum(row["all_four_accepted"] for row in task_rows),
                         "total_trajectories": 4})
    gate = decision_gate(states)
    report = {"status": "OBJECT_POSE_JACOBIAN_DISCOVERY_SUMMARY_COMPLETE",
              "unit_of_replication": "independent initial-state trajectory",
              "heldout_trajectories_untouched": 10,
              "no_dino_or_action_based_selection": True,
              "gate": gate, "per_task": per_task, "states": states,
              "decision": ("Proceed once to all ten frozen heldout trajectories."
                           if gate["advance_to_independent_holdout"] else
                           "Close the world-XY-only counterfactual construction route; do not tune thresholds on heldout data.")}
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS",
                          "expected_task_shards": 10,
                          "expected_discovery_trajectories": 40,
                          "minimum_overall_fraction": MIN_OVERALL_FRACTION,
                          "minimum_tasks_with_any_pass": MIN_TASKS_WITH_ANY}, indent=2))
        return
    summarize()


if __name__ == "__main__":
    main()
