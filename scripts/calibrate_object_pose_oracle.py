#!/usr/bin/env python3
"""Oracle-only magnitude calibration for one Object pose counterfactual.

Replay uses OFT to reach the bank state; no model evaluation occurs on altered
images. Every predeclared direction/magnitude is reported, including exclusions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot_object_pose_counterfactual import (
    CONFIG_PATH, MGWP_ROOT, PROJECT_ROOT, DECISION, RANK, TASK_ID,
    load_json, load_wrapper, make_segmentation_env, mask_relative, render_integrity,
    replay_to_decision, require, require_gpu, rerender, segmentation_masks,
    target_goal_subject, translated_render,
)
from replay_object_bank_mask_pilot import sha256_file
from collect_object_bank_spatial_masks import holdout_rank, task_records


MAGNITUDES_M = (0.04, 0.05, 0.06)
DIRECTIONS = (("x_plus", 1.0, 0.0), ("x_minus", -1.0, 0.0),
              ("y_plus", 0.0, 1.0), ("y_minus", 0.0, -1.0))
OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_pose_oracle_calibration_v1"


def select_symmetric_pairs(rows: list[dict]) -> dict[str, float | None]:
    """Predeclared choice: smallest magnitude with both signs valid per axis."""
    by_key = {(row["magnitude_m"], row["direction"]): row for row in rows}
    require(len(by_key) == len(rows), "Duplicate oracle candidate")
    require(set(by_key) == {(magnitude, direction) for magnitude in MAGNITUDES_M
                            for direction, _, _ in DIRECTIONS}, "Incomplete oracle candidate grid")
    selected: dict[str, float | None] = {}
    for axis in ("x", "y"):
        selected[axis] = next((magnitude for magnitude in MAGNITUDES_M
                               if by_key[(magnitude, f"{axis}_plus")]["valid"]
                               and by_key[(magnitude, f"{axis}_minus")]["valid"]), None)
    return selected


def run() -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID),
            "Calibration must use Object discovery trajectory")
    require(not OUTPUT_ROOT.exists(), f"Refusing to overwrite {OUTPUT_ROOT}")
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    require(any(int(row["decision_index"]) == DECISION
                and row["phase_proxy"] == "pre_contact_proxy" for row in records),
            "Calibration decision absent from bank")
    require_gpu()
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, TASK_ID)
    try:
        anchor = replay_to_decision(cfg, wrapper, env, task, initial_states, records)
        clean_raw = rerender(env, anchor["sim_state"])
        repeated_raw = rerender(env, anchor["sim_state"])
        require(all(np.array_equal(clean_raw[key], repeated_raw[key])
                    for key in ("agentview_image", "robot0_eye_in_hand_image")),
                "Same-state renderer is not deterministic")
        subject = target_goal_subject(env)
        target = env.env.objects_dict[subject]
        require(len(target.joints) == 1, "Target requires one free joint")
        joint = target.joints[0]
        joint_addr = env.sim.model.get_joint_qpos_addr(joint)
        require(isinstance(joint_addr, tuple) and joint_addr[1] - joint_addr[0] == 7,
                "Target requires a 7D free joint")
        clean_qpos = env.sim.data.qpos.copy()
        clean_masks = segmentation_masks(env, clean_raw, subject)
        clean_relative, _ = mask_relative(clean_masks)
        rows = []
        for magnitude in MAGNITUDES_M:
            for direction, vx, vy in DIRECTIONS:
                row = {"magnitude_m": magnitude, "direction": direction,
                       "world_delta_xy_m": [magnitude * vx, magnitude * vy],
                       "valid": False, "rejection_reason": None}
                try:
                    raw, _ = translated_render(env, anchor["sim_state"], joint,
                                               magnitude * vx, magnitude * vy)
                    masks = segmentation_masks(env, raw, subject)
                    row["integrity"] = render_integrity(
                        clean_raw, raw, clean_masks, masks, clean_qpos,
                        env.sim.data.qpos.copy(), joint_addr)
                    relative, token_mass = mask_relative(masks)
                    delta = relative - clean_relative
                    row["relative_delta_xy_tokens"] = delta.tolist()
                    row["oracle_shift_tokens"] = float(np.linalg.norm(delta))
                    row["visible_token_mass"] = token_mass
                    row["valid"] = row["oracle_shift_tokens"] >= 2.0
                    if not row["valid"]:
                        row["rejection_reason"] = "oracle_relative_shift_below_2_tokens"
                except RuntimeError as exc:
                    row["rejection_reason"] = str(exc)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        env.close()
    selected = select_symmetric_pairs(rows)
    report = {"status": "ORACLE_CALIBRATION_COMPLETE", "diagnostic_not_attack": True,
              "no_counterfactual_model_inference": True,
              "task_id": TASK_ID, "trajectory_rank": RANK, "decision_index": DECISION,
              "state_index": anchor["state_index"], "bank_sha256": sha256_file(bank_path),
              "source_config_sha256": sha256_file(CONFIG_PATH),
              "magnitude_grid_m": MAGNITUDES_M, "oracle_shift_min_tokens": 2.0,
              "selection_rule": "smallest magnitude where plus and minus are both valid per axis",
              "selected_symmetric_pair_m": selected, "candidates": rows,
              "limitation": "One discovery state only; stimulus calibration, not model mechanism evidence."}
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "STATUS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "selected_symmetric_pair_m")},
                     indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        cfg = load_json(CONFIG_PATH)
        bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt",
                          map_location="cpu", weights_only=False)
        records = task_records(bank, TASK_ID)[RANK]
        require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID)
                and any(int(row["decision_index"]) == DECISION for row in records),
                "Oracle calibration source/split drift")
        print(json.dumps({"status": "CHECK_ONLY_PASS", "task_id": TASK_ID,
                          "trajectory_rank": RANK, "decision_index": DECISION,
                          "candidate_count": len(MAGNITUDES_M) * len(DIRECTIONS),
                          "magnitude_grid_m": MAGNITUDES_M,
                          "no_counterfactual_model_inference": True,
                          "gpu_job_submitted": False}, indent=2))
        return
    run()


if __name__ == "__main__":
    main()
