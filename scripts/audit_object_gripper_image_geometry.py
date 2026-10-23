#!/usr/bin/env python3
"""Discovery-only geometry audit for four signed Wrist object image shifts.

The policy is used to replay frozen clean trajectories. No edited image is
passed through DINO or the action model. Each trajectory is one audit unit.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from audit_object_pose_jacobian_discovery import discovery_selections, replay_to_selected
from pilot_object_gripper_image_cf import (
    RAW_SHIFT_PIXELS, SHIFTS, OUTPUT_ROOT as PILOT_ROOT, prepare_conditions,
    robot_instance_mask,
)
from pilot_object_pose_counterfactual import (
    CONFIG_PATH, MGWP_ROOT, PROJECT_ROOT, load_json, load_wrapper,
    make_segmentation_env, require, require_gpu, rerender,
    segmentation_masks, target_goal_subject,
)
from replay_object_bank_mask_pilot import sha256_file


OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_gripper_image_geometry_discovery_v1"


def audit_state(env, anchor: dict) -> dict:
    clean_raw = rerender(env, anchor["sim_state"])
    repeated = rerender(env, anchor["sim_state"])
    require(all(np.array_equal(clean_raw[key], repeated[key])
                for key in ("agentview_image", "robot0_eye_in_hand_image")),
            "Same-state renderer is not deterministic")
    subject = target_goal_subject(env)
    target = env.env.objects_dict[subject]
    require(len(target.joints) == 1, "Target requires one free joint")
    joint = target.joints[0]
    clean_qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
    require(clean_qpos.shape == (7,), "Target is not a free joint")
    clean_masks = segmentation_masks(env, clean_raw, subject)
    robot_mask = robot_instance_mask(env, clean_raw)
    clean_rgb = np.asarray(clean_raw["robot0_eye_in_hand_image"])[::-1, ::-1].copy()

    rerender(env, anchor["sim_state"])
    parked = clean_qpos.copy()
    parked[2] = -1.0
    env.sim.data.set_joint_qpos(joint, parked)
    env.sim.forward()
    env._update_observables(force=True)
    background_raw = env.env._get_observations()
    background_masks = segmentation_masks(env, background_raw, subject)
    require(not background_masks["target_object"].any(), "Parked target remains visible")
    background_rgb = np.asarray(background_raw["robot0_eye_in_hand_image"])[::-1, ::-1].copy()
    gripper_pixels = clean_masks["gripper"].astype(bool)
    require(gripper_pixels.any(), "Gripper mask empty")
    gripper_max_abs = int(np.abs(background_rgb.astype(np.int16)
                                 - clean_rgb.astype(np.int16))[gripper_pixels].max())
    require(gripper_max_abs <= 2, f"Object removal changed gripper pixels: {gripper_max_abs}")

    prepared = prepare_conditions(clean_rgb, background_rgb, clean_masks, robot_mask)
    return {"valid": True, "reason": None,
            "background_gripper_max_abs": gripper_max_abs,
            "oracle_conditions": [
                {"name": item["name"], "raw_shift_xy_pixels": [item["dx"], item["dy"]],
                 "oracle_delta_xy_tokens": item["oracle_delta"].tolist(),
                 "oracle_metrics": item["oracle_metrics"],
                 "integrity": item["integrity"],
                 "sham_integrity": item["sham_integrity"]}
                for item in prepared]}


def run_task(task_id: int) -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object", "Geometry audit requires Object suite")
    require(not PILOT_ROOT.joinpath("STATUS.json").exists() or
            json.loads(PILOT_ROOT.joinpath("STATUS.json").read_text()).get("status")
            == "IMAGE_CF_TECHNICAL_STIMULUS_INVALID",
            "Unexpected pilot status; review before geometry audit")
    final = OUTPUT_ROOT / f"task_{task_id:02d}.json"
    require(not final.exists(), f"Refusing to overwrite {final}")
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    selections = discovery_selections(bank, task_id)
    require_gpu()
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, task_id)
    rows = []
    try:
        for rank, decision, records in selections:
            anchor = replay_to_selected(cfg, wrapper, env, task, initial_states,
                                        task_id, decision, records)
            row = {"trajectory_rank": rank, "decision_index": decision,
                   "state_index": anchor["state_index"], "seed": anchor["seed"],
                   "clean_replay": {key: anchor[key] for key in
                                    ("action_replay_max_abs", "dino_replay_max_abs")}}
            try:
                row.update(audit_state(env, anchor))
            except RuntimeError as exc:
                row.update({"valid": False, "reason": str(exc)})
            rows.append(row)
            print(json.dumps({"task_id": task_id, "trajectory_rank": rank,
                              "valid": row["valid"], "reason": row["reason"]}), flush=True)
    finally:
        env.close()
    require(len(rows) == 4, "Discovery audit trajectory count drift")
    receipt = {"status": "OBJECT_GRIPPER_IMAGE_GEOMETRY_TASK_COMPLETE",
               "task_id": task_id,
               "heldout_trajectory_rank_untouched":  next(
                   rank for rank in range(5) if rank not in {row["trajectory_rank"] for row in rows}),
               "discovery_trajectory_units": 4,
               "no_edited_image_model_inference": True,
               "raw_shift_pixels": RAW_SHIFT_PIXELS,
               "signed_directions": [name for name, _, _ in SHIFTS],
               "bank_sha256": sha256_file(bank_path),
               "source_config_sha256": sha256_file(CONFIG_PATH),
               "states": rows}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt",
                      map_location="cpu", weights_only=False)
    selections = discovery_selections(bank, args.task_id)
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "task_id": args.task_id,
                          "discovery_rank_decisions": [[rank, decision]
                                                       for rank, decision, _ in selections],
                          "expected_trajectory_units": 4,
                          "no_edited_image_model_inference": True,
                          "gpu_job_submitted": False}, indent=2))
        return
    print(json.dumps(run_task(args.task_id), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
