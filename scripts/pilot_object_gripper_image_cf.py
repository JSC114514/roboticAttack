#!/usr/bin/env python3
"""One-state image-space object--gripper counterfactual mechanism pilot.

This is a diagnostic intervention, not the attack threat model. Simulator masks
and an object-removed background construct symmetric Wrist-only target shifts.
Front, proprioception, language, and gripper pixels remain clean. The resulting
images are then used to test signed DINO readout, action response, and clean-DINO
restoration on one oracle-selected discovery state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_object_pose_jacobian_discovery import replay_to_selected
from collect_object_bank_spatial_masks import holdout_rank, task_records
from pilot_object_pose_counterfactual import (
    CONFIG_PATH, MGWP_ROOT, PROJECT_ROOT, load_json, load_wrapper,
    make_segmentation_env, require, require_gpu, rerender, segmentation_masks,
    target_goal_subject,
)
from probe_dino_spatial_relation_stage0a import readout, token_centroid, transform_mask
from readout_object_bank_spatial_cpu import fit_prototypes, load_manifest, TEMPERATURE
from replay_object_bank_mask_pilot import observation, sha256_file
from runtime import action_with_dino_value, differentiable


TASK_ID = 1
RANK = 3
DECISION = 3
RAW_SHIFT_PIXELS = 38
SUPPORT_DILATION_PIXELS = 2
SHIFTS = (("image_x_plus", RAW_SHIFT_PIXELS, 0),
          ("image_x_minus", -RAW_SHIFT_PIXELS, 0),
          ("image_y_plus", 0, RAW_SHIFT_PIXELS),
          ("image_y_minus", 0, -RAW_SHIFT_PIXELS))
ORACLE_TARGET_TOKENS = 2.5
ORACLE_TOLERANCE_TOKENS = 0.30
MIN_ORACLE_DIRECTION_COSINE = 0.995
MIN_DINO_SIGNED_PROJECTION_TOKENS = 0.50
MIN_DINO_DIRECTION_COSINE = 0.50
MIN_ACTION_PAIR_L2 = 0.02
MIN_ACTION_CHANGE_FOR_MEDIATION = 0.01
MIN_MEDIAN_RESTORED_FRACTION = 0.50
OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_gripper_image_cf_pilot_v1"


def translate(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Integer image translation with zero fill; dx right, dy down."""
    require(array.ndim in (2, 3), "Expected image or mask")
    height, width = array.shape[:2]
    require(abs(dx) < width and abs(dy) < height, "Translation outside canvas")
    output = np.zeros_like(array)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    dst_x0, dst_x1 = src_x0 + dx, src_x1 + dx
    dst_y0, dst_y1 = src_y0 + dy, src_y1 + dy
    output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return output


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    require(mask.ndim == 2 and radius >= 0, "Invalid support mask")
    tensor = torch.as_tensor(mask.astype(np.float32))[None, None]
    expanded = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1,
                            padding=radius)[0, 0]
    return expanded.numpy() > 0


def composite_shift(clean_rgb: np.ndarray, background_rgb: np.ndarray,
                    target_mask: np.ndarray, gripper_mask: np.ndarray,
                    dx: int, dy: int) -> tuple[np.ndarray, np.ndarray, dict]:
    require(clean_rgb.shape == background_rgb.shape == (256, 256, 3), "RGB shape drift")
    require(target_mask.shape == gripper_mask.shape == (256, 256), "Mask shape drift")
    support = dilate(target_mask, SUPPORT_DILATION_PIXELS)
    shifted_support = translate(support.astype(np.uint8), dx, dy).astype(bool)
    shifted_target = translate(target_mask.astype(np.uint8), dx, dy).astype(bool)
    require(int(shifted_support.sum()) == int(support.sum())
            and int(shifted_target.sum()) == int(target_mask.sum()), "Shift clipped by image boundary")
    require(not np.any(shifted_target & gripper_mask.astype(bool)),
            "Shifted target overlaps the gripper")
    shifted_clean = translate(clean_rgb, dx, dy)
    output = clean_rgb.copy()
    output[support] = background_rgb[support]
    output[shifted_support] = shifted_clean[shifted_support]
    union = support | shifted_support
    require(np.array_equal(output[~union], clean_rgb[~union]),
            "Pixels outside original/shifted object support changed")
    return output, shifted_target.astype(np.uint8), {
        "source_support_pixels": int(support.sum()),
        "shifted_support_pixels": int(shifted_support.sum()),
        "target_pixels": int(shifted_target.sum()),
        "target_gripper_overlap_pixels": int(np.count_nonzero(shifted_target & gripper_mask)),
        "outside_union_exact": True,
    }


def relative(mask_target: np.ndarray, mask_gripper: np.ndarray) -> np.ndarray:
    target = transform_mask(mask_target)
    gripper = transform_mask(mask_gripper)
    require(float(target.sum()) > 0 and float(gripper.sum()) > 0, "Empty transformed mask")
    return (token_centroid(target) - token_centroid(gripper)).numpy()


def direction_metrics(delta: np.ndarray, desired: np.ndarray) -> dict:
    magnitude = float(np.linalg.norm(delta))
    desired_magnitude = float(np.linalg.norm(desired))
    cosine = float(np.dot(delta, desired) / max(magnitude * desired_magnitude, 1e-12))
    projection = float(np.dot(delta, desired) / desired_magnitude)
    return {"magnitude_tokens": magnitude, "direction_cosine": cosine,
            "signed_projection_tokens": projection}


def mechanism_gate(rows: list[dict], zero: dict) -> dict:
    require({row["name"] for row in rows} == {name for name, _, _ in SHIFTS},
            "Incomplete signed image counterfactual set")
    by_name = {row["name"]: row for row in rows}
    stimulus = all(row["oracle_valid"] for row in rows)
    dino_signed = all(row["dino_relation_metrics"]["direction_cosine"] >= MIN_DINO_DIRECTION_COSINE
                      and row["dino_relation_metrics"]["signed_projection_tokens"]
                      >= MIN_DINO_SIGNED_PROJECTION_TOKENS for row in rows)
    action_pairs = {
        "image_x": float(torch.linalg.vector_norm(
            by_name["image_x_plus"]["action"] - by_name["image_x_minus"]["action"])),
        "image_y": float(torch.linalg.vector_norm(
            by_name["image_y_plus"]["action"] - by_name["image_y_minus"]["action"])),
    }
    action_signed = all(value >= MIN_ACTION_PAIR_L2 for value in action_pairs.values())
    eligible = [row["restored_fraction"] for row in rows
                if row["action_l2_change"] >= MIN_ACTION_CHANGE_FOR_MEDIATION
                and row["restored_fraction"] is not None]
    median_restored = float(np.median(eligible)) if eligible else None
    mediation = bool(len(eligible) == 4 and median_restored >= MIN_MEDIAN_RESTORED_FRACTION)
    zero_control = bool(zero["wrist_exact"] and zero["action_max_abs"] <= 1e-5
                        and zero["dino_max_abs"] <= 1e-3)
    return {"zero_reconstruction_control": zero_control,
            "all_four_stimuli_valid": stimulus,
            "all_four_signed_dino_relation": dino_signed,
            "action_plus_minus_pair_l2": action_pairs,
            "both_action_pairs_nontrivial": action_signed,
            "mediation_eligible_conditions": len(eligible),
            "median_clean_dino_restored_fraction": median_restored,
            "clean_dino_mediation": mediation,
            "pilot_pass": bool(zero_control and stimulus and dino_signed
                               and action_signed and mediation)}


def run() -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID),
            "Image counterfactual pilot must use discovery data")
    require(not OUTPUT_ROOT.exists(), f"Refusing to overwrite {OUTPUT_ROOT}")
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    require(any(int(row["decision_index"]) == DECISION
                and row["phase_proxy"] == "pre_contact_proxy" for row in records),
            "Frozen pilot state drift")
    prototypes = fit_prototypes(load_manifest(bank))[TASK_ID]
    require_gpu()
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, TASK_ID)
    try:
        anchor = replay_to_selected(cfg, wrapper, env, task, initial_states,
                                    TASK_ID, DECISION, records)
        clean_raw = rerender(env, anchor["sim_state"])
        clean_obs = observation(clean_raw, cfg)
        subject = target_goal_subject(env)
        target = env.env.objects_dict[subject]
        require(len(target.joints) == 1, "Target requires one free joint")
        joint = target.joints[0]
        clean_qpos = np.asarray(env.sim.data.get_joint_qpos(joint), dtype=np.float64).copy()
        require(clean_qpos.shape == (7,), "Target is not a free joint")
        clean_masks = segmentation_masks(env, clean_raw, subject)
        clean_rgb = np.asarray(clean_raw["robot0_eye_in_hand_image"])[::-1, ::-1].copy()
        clean_relative = relative(clean_masks["target_object"], clean_masks["gripper"])

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

        with torch.no_grad():
            clean_action, clean_dino = differentiable(wrapper, clean_obs, task.language, None, (0, 0))
        predicted_clean = readout(clean_dino[0].float().cpu(), prototypes, TEMPERATURE)
        predicted_clean_relative = (predicted_clean["target_object"]
                                    - predicted_clean["gripper"]).numpy()

        zero_rgb, zero_mask, zero_integrity = composite_shift(
            clean_rgb, background_rgb, clean_masks["target_object"],
            clean_masks["gripper"], 0, 0)
        require(np.array_equal(zero_rgb, clean_rgb)
                and np.array_equal(zero_mask, clean_masks["target_object"]),
                "Zero-shift reconstruction is not exact")
        zero_raw = dict(clean_raw, robot0_eye_in_hand_image=zero_rgb[::-1, ::-1])
        zero_obs = observation(zero_raw, cfg)
        with torch.no_grad():
            zero_action, zero_dino = differentiable(wrapper, zero_obs, task.language, None, (0, 0))
        zero = {"wrist_exact": bool(np.array_equal(zero_obs["wrist_image"], clean_obs["wrist_image"])),
                "action_max_abs": float((zero_action - clean_action).abs().max()),
                "dino_max_abs": float((zero_dino - clean_dino).abs().max()),
                "integrity": zero_integrity}

        rows = []
        for name, dx, dy in SHIFTS:
            rgb, shifted_target, integrity = composite_shift(
                clean_rgb, background_rgb, clean_masks["target_object"],
                clean_masks["gripper"], dx, dy)
            oracle_delta = relative(shifted_target, clean_masks["gripper"]) - clean_relative
            desired = np.asarray([np.sign(dx) if dx else 0.0,
                                  np.sign(dy) if dy else 0.0], dtype=np.float64)
            oracle_metrics = direction_metrics(oracle_delta, desired)
            oracle_valid = bool(abs(oracle_metrics["magnitude_tokens"] - ORACLE_TARGET_TOKENS)
                                <= ORACLE_TOLERANCE_TOKENS
                                and oracle_metrics["direction_cosine"] >= MIN_ORACLE_DIRECTION_COSINE)
            altered_raw = dict(clean_raw, robot0_eye_in_hand_image=rgb[::-1, ::-1])
            altered_obs = observation(altered_raw, cfg)
            hybrid = dict(clean_obs, wrist_image=altered_obs["wrist_image"])
            require(np.array_equal(hybrid["full_image"], clean_obs["full_image"])
                    and np.array_equal(hybrid["state"], clean_obs["state"]),
                    "Non-Wrist model input changed")
            with torch.no_grad():
                action, dino = differentiable(wrapper, hybrid, task.language, None, (0, 0))
                restored = action_with_dino_value(wrapper, hybrid, task.language, clean_dino)
            predicted = readout(dino[0].float().cpu(), prototypes, TEMPERATURE)
            predicted_relative = (predicted["target_object"] - predicted["gripper"]).numpy()
            predicted_delta = predicted_relative - predicted_clean_relative
            dino_metrics = direction_metrics(predicted_delta, desired)
            action_change = float(torch.linalg.vector_norm(action - clean_action))
            residual = float(torch.linalg.vector_norm(restored - clean_action))
            restored_fraction = (float(1.0 - residual / action_change)
                                 if action_change > 1e-8 else None)
            rows.append({"name": name, "raw_shift_xy_pixels": [dx, dy],
                         "oracle_delta_xy_tokens": oracle_delta.tolist(),
                         "oracle_metrics": oracle_metrics, "oracle_valid": oracle_valid,
                         "predicted_dino_relation_delta_xy_tokens": predicted_delta.tolist(),
                         "dino_relation_metrics": dino_metrics,
                         "action_l2_change": action_change,
                         "clean_dino_restored_action_l2_residual": residual,
                         "restored_fraction": restored_fraction,
                         "integrity": integrity, "action": action.detach().cpu()})
    finally:
        env.close()

    gate = mechanism_gate(rows, zero)
    serial_rows = []
    for row in rows:
        serial = dict(row)
        serial["action"] = serial["action"].tolist()
        serial_rows.append(serial)
    report = {"status": "OBJECT_GRIPPER_IMAGE_CF_PILOT_COMPLETE",
              "diagnostic_not_attack": True, "discovery_state_only": True,
              "task_id": TASK_ID, "trajectory_rank": RANK, "decision_index": DECISION,
              "state_index": anchor["state_index"], "seed": anchor["seed"],
              "selection_reason": "Oracle-only prior discovery state with four feasible directions; no prior altered-image DINO/action inference.",
              "bank_sha256": sha256_file(bank_path),
              "source_config_sha256": sha256_file(CONFIG_PATH),
              "frozen_thresholds": {"raw_shift_pixels": RAW_SHIFT_PIXELS,
                                    "oracle_target_tokens": ORACLE_TARGET_TOKENS,
                                    "oracle_tolerance_tokens": ORACLE_TOLERANCE_TOKENS,
                                    "minimum_oracle_direction_cosine": MIN_ORACLE_DIRECTION_COSINE,
                                    "minimum_dino_signed_projection_tokens": MIN_DINO_SIGNED_PROJECTION_TOKENS,
                                    "minimum_dino_direction_cosine": MIN_DINO_DIRECTION_COSINE,
                                    "minimum_action_pair_l2": MIN_ACTION_PAIR_L2,
                                    "minimum_action_change_for_mediation": MIN_ACTION_CHANGE_FOR_MEDIATION,
                                    "minimum_median_restored_fraction": MIN_MEDIAN_RESTORED_FRACTION},
              "zero_shift_control": zero, "conditions": serial_rows,
              "gate": gate,
              "next_step": ("Implement a frozen multi-state discovery/heldout mechanism test; still do not train a patch."
                            if gate["pilot_pass"] else
                            "Stop the object--gripper relation mechanism branch; do not train a relation patch."),
              "limitation": "Synthetic image-space diagnostic with simulator oracle masks; not a realistic attacker capability or attack result."}
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "STATUS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CONFIG_PATH)
    bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt",
                      map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID)
            and any(int(row["decision_index"]) == DECISION for row in records),
            "Image counterfactual pilot source/split drift")
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "task_id": TASK_ID,
                          "trajectory_rank": RANK, "decision_index": DECISION,
                          "state_index": int(records[0]["state_index"]),
                          "raw_shift_pixels": RAW_SHIFT_PIXELS,
                          "signed_conditions": len(SHIFTS),
                          "diagnostic_not_attack": True,
                          "gpu_job_submitted": False}, indent=2))
        return
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
