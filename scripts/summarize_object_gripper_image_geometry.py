#!/usr/bin/env python3
"""Summarize the ten discovery-only image-shift geometry shards."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from audit_object_gripper_image_geometry import OUTPUT_ROOT
from collect_object_bank_spatial_masks import holdout_rank
from replay_object_bank_mask_pilot import require


MIN_VALID_TRAJECTORIES = 20
MIN_TASKS_WITH_VALID = 8


def evaluate_gate(rows: list[dict]) -> dict:
    require(len(rows) == 40, "Expected exactly 40 discovery trajectories")
    valid = [row for row in rows if row["valid"]]
    tasks = {row["task_id"] for row in valid}
    return {"valid_trajectories": len(valid), "total_trajectories": 40,
            "valid_fraction": len(valid) / 40,
            "tasks_with_valid_trajectory": len(tasks),
            "minimum_valid_trajectories": MIN_VALID_TRAJECTORIES,
            "minimum_tasks_with_valid": MIN_TASKS_WITH_VALID,
            "broad_diagnostic_feasible": bool(len(valid) >= MIN_VALID_TRAJECTORIES
                                              and len(tasks) >= MIN_TASKS_WITH_VALID)}


def summarize() -> dict:
    final = OUTPUT_ROOT / "SUMMARY.json"
    require(not final.exists(), f"Refusing to overwrite {final}")
    rows, hashes = [], set()
    for task_id in range(10):
        path = OUTPUT_ROOT / f"task_{task_id:02d}.json"
        require(path.is_file(), f"Missing task receipt: {path}")
        receipt = json.loads(path.read_text())
        require(receipt["status"] == "OBJECT_GRIPPER_IMAGE_GEOMETRY_TASK_COMPLETE"
                and receipt["task_id"] == task_id
                and receipt["heldout_trajectory_rank_untouched"] == holdout_rank(task_id)
                and receipt["no_edited_image_model_inference"],
                f"Task receipt lineage/split drift: {task_id}")
        hashes.add(receipt["bank_sha256"])
        require(len(receipt["states"]) == 4
                and len({row["trajectory_rank"] for row in receipt["states"]}) == 4,
                f"Task receipt trajectory count drift: {task_id}")
        rows.extend({"task_id": task_id, "trajectory_rank": state["trajectory_rank"],
                     "decision_index": state["decision_index"],
                     "valid": state["valid"], "reason": state["reason"]}
                    for state in receipt["states"])
    require(len(hashes) == 1, "Source bank differs across geometry shards")
    gate = evaluate_gate(rows)
    reasons = Counter(row["reason"] for row in rows if not row["valid"])
    valid_keys = [[row["task_id"], row["trajectory_rank"], row["decision_index"]]
                  for row in rows if row["valid"]]
    report = {"status": "OBJECT_GRIPPER_IMAGE_GEOMETRY_DISCOVERY_COMPLETE",
              "unit_of_replication": "independent initial-state trajectory",
              "heldout_trajectories_untouched": 10,
              "no_edited_image_model_inference": True,
              "bank_sha256": next(iter(hashes)), "gate": gate,
              "valid_state_keys": valid_keys,
              "invalid_reasons": dict(sorted(reasons.items())), "states": rows,
              "decision": ("Freeze valid discovery states before multi-trajectory DINO/action diagnosis."
                           if gate["broad_diagnostic_feasible"] else
                           "Do not generalize from handpicked states; deprioritize this four-direction image-shift diagnostic and consider a different DINO hypothesis.")}
    final.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "expected_task_shards": 10,
                          "expected_trajectory_units": 40,
                          "minimum_valid_trajectories": MIN_VALID_TRAJECTORIES,
                          "minimum_tasks_with_valid": MIN_TASKS_WITH_VALID}, indent=2))
        return
    summarize()


if __name__ == "__main__":
    main()
