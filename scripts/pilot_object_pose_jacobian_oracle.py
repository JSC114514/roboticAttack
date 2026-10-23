#!/usr/bin/env python3
"""Oracle-only local-Jacobian pilot for object--gripper image relations.

The OFT policy is used only to replay one discovery trajectory to a frozen
state. Altered renders are never passed through DINO or the action model. A
central-difference Jacobian maps object world-XY displacement to the target
minus gripper centroid in Wrist token coordinates. Four signed image-space
targets are attempted under one symmetric, predeclared rule.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from collect_object_bank_spatial_masks import holdout_rank, task_records
from pilot_object_pose_counterfactual import (
    CONFIG_PATH, DECISION, MGWP_ROOT, PROJECT_ROOT, RANK, TASK_ID,
    load_json, load_wrapper, make_segmentation_env, mask_relative,
    render_integrity, replay_to_decision, require, require_gpu, rerender,
    segmentation_masks, target_goal_subject, translated_render,
)
from replay_object_bank_mask_pilot import sha256_file


PROBE_METERS = 0.01
TARGET_SHIFT_TOKENS = 2.5
TARGET_TOLERANCE_TOKENS = 0.5
MIN_DIRECTION_COSINE = 0.95
MAX_WORLD_NORM_METERS = 0.08
MAX_CONDITION_NUMBER = 30.0
MAX_CORRECTIONS = 2
TARGETS = (
    ("image_x_plus", (TARGET_SHIFT_TOKENS, 0.0)),
    ("image_x_minus", (-TARGET_SHIFT_TOKENS, 0.0)),
    ("image_y_plus", (0.0, TARGET_SHIFT_TOKENS)),
    ("image_y_minus", (0.0, -TARGET_SHIFT_TOKENS)),
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_pose_jacobian_oracle_pilot_v1"


def central_jacobian(probes: dict[str, np.ndarray], epsilon: float) -> np.ndarray:
    require(epsilon > 0, "Jacobian epsilon must be positive")
    require(set(probes) == {"x_plus", "x_minus", "y_plus", "y_minus"},
            "Incomplete central-difference probes")
    jacobian = np.column_stack(((probes["x_plus"] - probes["x_minus"]) / (2 * epsilon),
                                (probes["y_plus"] - probes["y_minus"]) / (2 * epsilon)))
    require(jacobian.shape == (2, 2) and np.isfinite(jacobian).all(),
            "Invalid local Jacobian")
    return jacobian


def jacobian_diagnostics(jacobian: np.ndarray) -> dict:
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    return {"matrix_token_per_meter": jacobian.tolist(),
            "singular_values_token_per_meter": singular.tolist(),
            "condition_number": condition,
            "valid": bool(np.isfinite(condition) and condition <= MAX_CONDITION_NUMBER)}


def solve_world_delta(jacobian: np.ndarray, desired_token_delta: np.ndarray) -> np.ndarray:
    require(jacobian.shape == (2, 2) and desired_token_delta.shape == (2,),
            "Jacobian solve shape mismatch")
    return np.linalg.pinv(jacobian) @ desired_token_delta


def candidate_metrics(actual: np.ndarray, desired: np.ndarray) -> dict:
    magnitude = float(np.linalg.norm(actual))
    target_magnitude = float(np.linalg.norm(desired))
    cosine = float(np.dot(actual, desired) / max(magnitude * target_magnitude, 1e-12))
    error = float(np.linalg.norm(actual - desired))
    return {"actual_shift_tokens": magnitude, "direction_cosine": cosine,
            "target_error_tokens": error,
            "accepted": bool(magnitude >= TARGET_SHIFT_TOKENS - TARGET_TOLERANCE_TOKENS
                             and error <= TARGET_TOLERANCE_TOKENS
                             and cosine >= MIN_DIRECTION_COSINE)}


def symmetric_pair_gate(rows: list[dict]) -> dict:
    by_name = {row["name"]: row for row in rows}
    require(set(by_name) == {name for name, _ in TARGETS}, "Incomplete signed target set")
    result = {}
    for axis in ("image_x", "image_y"):
        plus, minus = by_name[f"{axis}_plus"], by_name[f"{axis}_minus"]
        both = bool(plus["accepted"] and minus["accepted"])
        ratio = None
        if both:
            values = [plus["metrics"]["actual_shift_tokens"],
                      minus["metrics"]["actual_shift_tokens"]]
            ratio = float(max(values) / min(values))
        result[axis] = {"both_directions_accepted": both,
                        "achieved_magnitude_ratio": ratio}
    result["all_four_accepted"] = all(value["both_directions_accepted"]
                                       for key, value in result.items() if key != "all_four_accepted")
    return result


def run() -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID),
            "Jacobian pilot must use an Object discovery trajectory")
    require(not OUTPUT_ROOT.exists(), f"Refusing to overwrite {OUTPUT_ROOT}")
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    require(any(int(row["decision_index"]) == DECISION
                and row["phase_proxy"] == "pre_contact_proxy" for row in records),
            "Pilot decision absent from frozen discovery bank")
    require_gpu()
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, TASK_ID)
    try:
        anchor = replay_to_decision(cfg, wrapper, env, task, initial_states, records)
        clean_raw = rerender(env, anchor["sim_state"])
        repeat_raw = rerender(env, anchor["sim_state"])
        require(all(np.array_equal(clean_raw[key], repeat_raw[key])
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
        clean_relative, clean_mass = mask_relative(clean_masks)

        probe_rows, probe_vectors = [], {}
        for name, (dx, dy) in (("x_plus", (PROBE_METERS, 0.0)),
                               ("x_minus", (-PROBE_METERS, 0.0)),
                               ("y_plus", (0.0, PROBE_METERS)),
                               ("y_minus", (0.0, -PROBE_METERS))):
            raw, _ = translated_render(env, anchor["sim_state"], joint, dx, dy)
            masks = segmentation_masks(env, raw, subject)
            integrity = render_integrity(clean_raw, raw, clean_masks, masks, clean_qpos,
                                         env.sim.data.qpos.copy(), joint_addr)
            relative, mass = mask_relative(masks)
            delta = relative - clean_relative
            probe_vectors[name] = delta
            probe_rows.append({"name": name, "world_delta_xy_m": [dx, dy],
                               "token_delta_xy": delta.tolist(),
                               "visible_token_mass": mass, "integrity": integrity})

        jacobian = central_jacobian(probe_vectors, PROBE_METERS)
        diagnostics = jacobian_diagnostics(jacobian)
        candidates = []
        if diagnostics["valid"]:
            for name, desired_pair in TARGETS:
                desired = np.asarray(desired_pair, dtype=np.float64)
                world_delta = solve_world_delta(jacobian, desired)
                row = {"name": name, "desired_token_delta_xy": desired.tolist(),
                       "accepted": False, "rejection_reason": None, "iterations": []}
                for iteration in range(MAX_CORRECTIONS + 1):
                    norm = float(np.linalg.norm(world_delta))
                    if norm > MAX_WORLD_NORM_METERS:
                        row["rejection_reason"] = "world_displacement_cap_exceeded"
                        row["proposed_world_delta_xy_m"] = world_delta.tolist()
                        row["proposed_world_norm_m"] = norm
                        break
                    try:
                        raw, _ = translated_render(env, anchor["sim_state"], joint,
                                                   float(world_delta[0]), float(world_delta[1]))
                        masks = segmentation_masks(env, raw, subject)
                        integrity = render_integrity(
                            clean_raw, raw, clean_masks, masks, clean_qpos,
                            env.sim.data.qpos.copy(), joint_addr)
                        relative, mass = mask_relative(masks)
                        actual = relative - clean_relative
                        metrics = candidate_metrics(actual, desired)
                        row["iterations"].append({"iteration": iteration,
                                                  "world_delta_xy_m": world_delta.tolist(),
                                                  "world_norm_m": norm,
                                                  "actual_token_delta_xy": actual.tolist(),
                                                  "metrics": metrics,
                                                  "visible_token_mass": mass,
                                                  "integrity": integrity})
                        row["metrics"] = metrics
                        if metrics["accepted"]:
                            row["accepted"] = True
                            break
                        world_delta = world_delta + solve_world_delta(jacobian, desired - actual)
                    except RuntimeError as exc:
                        row["rejection_reason"] = str(exc)
                        break
                if not row["accepted"] and row["rejection_reason"] is None:
                    row["rejection_reason"] = "target_not_reached_after_fixed_corrections"
                candidates.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        pair_gate = symmetric_pair_gate(candidates) if candidates else {
            "all_four_accepted": False, "reason": "invalid_local_jacobian"}
    finally:
        env.close()

    report = {
        "status": "OBJECT_POSE_JACOBIAN_ORACLE_PILOT_COMPLETE",
        "diagnostic_not_attack": True,
        "no_counterfactual_model_inference": True,
        "task_id": TASK_ID, "trajectory_rank": RANK, "decision_index": DECISION,
        "state_index": anchor["state_index"], "seed": anchor["seed"],
        "bank_sha256": sha256_file(bank_path),
        "source_config_sha256": sha256_file(CONFIG_PATH),
        "frozen_rules": {"probe_m": PROBE_METERS,
                         "target_shift_tokens": TARGET_SHIFT_TOKENS,
                         "target_tolerance_tokens": TARGET_TOLERANCE_TOKENS,
                         "minimum_direction_cosine": MIN_DIRECTION_COSINE,
                         "maximum_world_norm_m": MAX_WORLD_NORM_METERS,
                         "maximum_condition_number": MAX_CONDITION_NUMBER,
                         "maximum_corrections": MAX_CORRECTIONS},
        "clean_relative_xy_tokens": clean_relative.tolist(),
        "clean_visible_token_mass": clean_mass,
        "probes": probe_rows, "jacobian": diagnostics,
        "candidates": candidates, "symmetric_pair_gate": pair_gate,
        "next_gate": "Only if all four oracle targets pass, freeze this solver and evaluate DINO/action mediation on independent whole-trajectory holdouts.",
        "limitation": "One discovery state; tests stimulus construction only and supplies no DINO, action, or attack evidence."
    }
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "STATUS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
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
            "Jacobian pilot source/split drift")
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "task_id": TASK_ID,
                          "trajectory_rank": RANK, "decision_index": DECISION,
                          "probe_m": PROBE_METERS, "target_shift_tokens": TARGET_SHIFT_TOKENS,
                          "world_norm_cap_m": MAX_WORLD_NORM_METERS,
                          "signed_targets": len(TARGETS),
                          "no_counterfactual_model_inference": True,
                          "gpu_job_submitted": False}, indent=2))
        return
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
