#!/usr/bin/env python3
"""One discovery-state object-pose counterfactual; diagnostic, not an attack.

No optimization or Slurm submission. The model sees clean Front/proprio/language
and only the counterfactual Wrist image. Object translation is an oracle probe,
not attacker control. Abort when object-only integrity checks fail.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from collect_object_bank_spatial_masks import CONFIG_PATH, MGWP_ROOT, holdout_rank, task_records
from probe_dino_spatial_relation_stage0a import patch_exclusion, readout, token_centroid, transform_mask
from readout_object_bank_spatial_cpu import fit_prototypes, load_manifest, TEMPERATURE
from replay_object_bank_mask_pilot import (
    PROJECT_ROOT, check_observation, load_json, load_wrapper, make_segmentation_env,
    observation, require, require_gpu, reset_wait, segmentation_masks, sha256_file,
    target_goal_subject,
)
from runtime import action_with_dino_value, differentiable


TASK_ID = 0
RANK = 1  # Discovery; rank 3 is the frozen task-0 heldout trajectory.
DECISION = 4  # Pre-contact; object not yet grasped.
DELTA_METERS = 0.03
OFFSETS = (("x_plus", (DELTA_METERS, 0.0)),
           ("x_minus", (-DELTA_METERS, 0.0)),
           ("y_plus", (0.0, DELTA_METERS)),
           ("y_minus", (0.0, -DELTA_METERS)))
OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_pose_cf_pilot_v1"
EXCLUDE = patch_exclusion({"token_grid": 16, "image_resolution": 224,
                           "patch_xy": [26, 137], "patch_shape": [50, 50],
                           "patch_token_halo": 1})


def rerender(env, state: np.ndarray) -> dict:
    env.set_state(state)
    env.sim.forward()
    env._update_observables(force=True)
    return env.env._get_observations()


def translated_render(env, state: np.ndarray, object_joint: str, dx: float, dy: float) -> tuple[dict, np.ndarray]:
    rerender(env, state)
    original = np.asarray(env.sim.data.get_joint_qpos(object_joint), dtype=np.float64).copy()
    require(original.shape == (7,), f"Expected movable free joint: {object_joint} {original.shape}")
    translated = original.copy()
    translated[:2] += (dx, dy)
    env.sim.data.set_joint_qpos(object_joint, translated)
    env.sim.forward()
    env._update_observables(force=True)
    current = np.asarray(env.sim.data.get_joint_qpos(object_joint), dtype=np.float64).copy()
    require(np.allclose(current, translated, atol=1e-8), "Object translation did not persist")
    raw = env.env._get_observations()
    return raw, current


def mask_relative(masks: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    token = {role: transform_mask(mask).masked_fill(EXCLUDE, 0)
             for role, mask in masks.items()}
    require(all(float(token[role].sum()) > 0 for role in ("target_object", "gripper")),
            "Target or gripper not visible outside fixed patch exclusion")
    centers = {role: token_centroid(token[role]).numpy()
               for role in ("target_object", "gripper")}
    return centers["target_object"] - centers["gripper"], {
        role: float(token[role].sum()) for role in ("target_object", "gripper")}


def render_integrity(clean_raw: dict, altered_raw: dict, clean_masks: dict, altered_masks: dict,
                     clean_state: np.ndarray, changed_state: np.ndarray,
                     joint_addr: tuple[int, int]) -> dict:
    start, end = joint_addr
    unchanged = np.ones(clean_state.size, dtype=bool)
    unchanged[start:end] = False
    require(np.array_equal(clean_state[unchanged], changed_state[unchanged]),
            "Non-target MuJoCo state changed")
    clean_grip = transform_mask(clean_masks["gripper"]).masked_fill(EXCLUDE, 0)
    changed_grip = transform_mask(altered_masks["gripper"]).masked_fill(EXCLUDE, 0)
    grip_shift = float(torch.linalg.vector_norm(token_centroid(clean_grip) - token_centroid(changed_grip)))
    require(grip_shift <= 0.5, f"Gripper moved in Wrist image: {grip_shift:.3f} tokens")
    old = np.asarray(clean_raw["robot0_eye_in_hand_image"])[::-1, ::-1]
    new = np.asarray(altered_raw["robot0_eye_in_hand_image"])[::-1, ::-1]
    object_union = clean_masks["target_object"].astype(bool) | altered_masks["target_object"].astype(bool)
    outside_changed = np.max(np.abs(old.astype(np.int16) - new.astype(np.int16)), axis=2) > 2
    outside_fraction = float(np.mean(outside_changed & ~object_union))
    # Shadows/revealed background can change; >2% is too broad for an object-only probe.
    require(outside_fraction <= 0.02, f"Non-target Wrist pixels changed: {outside_fraction:.3%}")
    return {"gripper_shift_tokens": grip_shift,
            "outside_object_union_changed_fraction": outside_fraction,
            "non_target_sim_state_exact": True}


def replay_to_decision(cfg: dict, wrapper, env, task, initial_states, records: list[dict]) -> dict:
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action

    state_index = int(records[0]["state_index"])
    seed = int(cfg["seed"]) + TASK_ID * 100 + state_index
    raw = reset_wait(env, initial_states[state_index], seed, int(cfg["num_steps_wait"]))
    queue = deque()
    decision = 0
    frozen = next(row for row in records if int(row["decision_index"]) == DECISION)
    for _ in range(int(cfg["max_steps"])):
        if not queue:
            obs = observation(raw, cfg)
            if decision == DECISION:
                check_observation(obs, frozen["observation"])
                with torch.no_grad():
                    action, dino = differentiable(wrapper, obs, task.language, None, (0, 0))
                diff = float((action.cpu() - frozen["clean_action"].float().cpu()).abs().max())
                require(diff <= 0.02, f"Pilot clean action mismatch: {diff}")
                return {"raw": raw, "obs": obs, "sim_state": env.get_sim_state().copy(),
                        "clean_action": action.detach(), "clean_dino": dino.detach(),
                        "action_replay_max_abs": diff, "state_index": state_index,
                        "seed": seed}
            chunk = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(-1, 7)
            require(chunk.shape == (cfg["num_open_loop_steps"], 7), "OFT chunk shape drift")
            queue.extend(chunk)
            decision += 1
        raw, _, done, _ = env.step(process_libero_action(np.asarray(queue.popleft())).tolist())
        require(not done, "Episode ended before counterfactual decision")
    raise RuntimeError("Counterfactual decision not reached")


def run() -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID),
            "Pilot must use Object discovery trajectory")
    bank_path = MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    require(not OUTPUT_ROOT.exists(), f"Refusing to overwrite pilot {OUTPUT_ROOT}")
    prototypes = fit_prototypes(load_manifest(bank))[TASK_ID]
    require_gpu()
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, TASK_ID)
    try:
        anchor = replay_to_decision(cfg, wrapper, env, task, initial_states, records)
        clean_raw = rerender(env, anchor["sim_state"])
        clean_obs = observation(clean_raw, cfg)
        check_observation(clean_obs, anchor["obs"])
        subject = target_goal_subject(env)
        target = env.env.objects_dict[subject]
        require(len(target.joints) == 1, "Target requires a single free joint")
        joint = target.joints[0]
        addr = env.sim.model.get_joint_qpos_addr(joint)
        require(isinstance(addr, tuple) and addr[1] - addr[0] == 7,
                "Target is not a 7D free joint")
        clean_masks = segmentation_masks(env, clean_raw, subject)
        truth_clean, _ = mask_relative(clean_masks)
        clean_pred = readout(anchor["clean_dino"][0].float().cpu(), prototypes,
                             TEMPERATURE, EXCLUDE)
        pred_clean = (clean_pred["target_object"] - clean_pred["gripper"]).numpy()
        results = []
        for name, (dx, dy) in OFFSETS:
            raw, qpos = translated_render(env, anchor["sim_state"], joint, dx, dy)
            masks = segmentation_masks(env, raw, subject)
            integrity = render_integrity(clean_raw, raw, clean_masks, masks,
                                         anchor["sim_state"], env.get_sim_state(), addr)
            truth, masses = mask_relative(masks)
            true_delta = truth - truth_clean
            require(float(np.linalg.norm(true_delta)) >= 2.0,
                    f"Oracle relative shift <2 tokens for {name}: {true_delta}")
            altered = observation(raw, cfg)
            # Isolate Wrist-DINO route: Front/proprio/language stay exactly clean.
            hybrid = dict(anchor["obs"], wrist_image=altered["wrist_image"])
            require(np.array_equal(hybrid["full_image"], anchor["obs"]["full_image"])
                    and np.array_equal(hybrid["state"], anchor["obs"]["state"]),
                    "Non-Wrist policy input changed")
            with torch.no_grad():
                action, dino = differentiable(wrapper, hybrid, task.language, None, (0, 0))
                restored = action_with_dino_value(wrapper, hybrid, task.language,
                                                  anchor["clean_dino"])
            predicted = readout(dino[0].float().cpu(), prototypes, TEMPERATURE, EXCLUDE)
            pred = (predicted["target_object"] - predicted["gripper"]).numpy()
            pred_delta = pred - pred_clean
            action_change = float(torch.linalg.vector_norm(action.float() - anchor["clean_action"].float()))
            residual = float(torch.linalg.vector_norm(restored.float() - anchor["clean_action"].float()))
            results.append({"name": name, "world_delta_xy_m": [dx, dy],
                            "target_qpos": qpos.tolist(), "true_delta_xy_tokens": true_delta.tolist(),
                            "predicted_delta_xy_tokens": pred_delta.tolist(),
                            "oracle_shift_tokens": float(np.linalg.norm(true_delta)),
                            "dino_shift_tokens": float(np.linalg.norm(pred_delta)),
                            "signed_alignment": float(np.dot(true_delta, pred_delta) /
                                                      max(np.linalg.norm(true_delta), 1e-6)),
                            "action_l2_change": action_change,
                            "dino_restored_action_l2_residual": residual,
                            "restored_fraction": float(1.0 - residual / action_change)
                                                 if action_change > 1e-8 else None,
                            "visible_token_mass": masses, "integrity": integrity})
    finally:
        env.close()
    report = {"status": "OBJECT_ONLY_POSE_CF_PILOT_COMPLETE", "diagnostic_not_attack": True,
              "task_id": TASK_ID, "trajectory_rank": RANK, "decision_index": DECISION,
              "state_index": anchor["state_index"], "seed": anchor["seed"],
              "bank_sha256": sha256_file(bank_path),
              "source_config_sha256": sha256_file(CONFIG_PATH),
              "temperature": TEMPERATURE, "offsets": results,
              "limitation": "One discovery state only; not an independent mechanism confirmation."}
    OUTPUT_ROOT.mkdir(parents=True)
    (OUTPUT_ROOT / "STATUS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and RANK != holdout_rank(TASK_ID),
            "Pilot source or split mismatch")
    bank = torch.load(MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt", map_location="cpu", weights_only=False)
    records = task_records(bank, TASK_ID)[RANK]
    if args.check_only:
        print(json.dumps({"status": "CHECK_ONLY_PASS", "task_id": TASK_ID,
                          "trajectory_rank": RANK, "decision_index": DECISION,
                          "state_index": int(records[0]["state_index"]),
                          "offsets": OFFSETS, "gpu_job_submitted": False}, indent=2))
        return
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
