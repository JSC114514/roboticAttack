#!/usr/bin/env python3
"""Multi-task discovery audit for bounded object--gripper Jacobian stimuli.

One pre-contact state is selected deterministically from each non-heldout
trajectory. Altered renders are scored only with simulator masks and integrity
checks; they are never passed through DINO or the action head.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from collect_object_bank_spatial_masks import holdout_rank, task_records
from pilot_object_pose_counterfactual import (
    CONFIG_PATH, MGWP_ROOT, PROJECT_ROOT, load_json, load_wrapper,
    make_segmentation_env, mask_relative, render_integrity, require, require_gpu,
    rerender, segmentation_masks, target_goal_subject, translated_render,
)
from pilot_object_pose_jacobian_oracle import (
    MAX_CONDITION_NUMBER, MAX_WORLD_NORM_METERS, PROBE_METERS, TARGETS,
    candidate_metrics, central_jacobian, jacobian_diagnostics, solve_world_delta,
    symmetric_pair_gate,
)
from replay_object_bank_mask_pilot import (
    check_observation, observation, reset_wait, sha256_file,
)
from runtime import differentiable


OUTPUT_ROOT = PROJECT_ROOT / "outputs/exploring_oft_dino/object_pose_jacobian_discovery_v2"
LINE_SEARCH_SCALES = (1.0, 0.5, 0.25, 0.125)
MAX_CORRECTION_ROUNDS = 2
MIN_TARGET_VISIBLE_FRACTION = 0.25
MIN_TARGET_TOKEN_MASS = 1.0


def discovery_selections(bank: dict, task_id: int) -> list[tuple[int, int, list[dict]]]:
    """Four trajectory-level units; last pre-contact state chosen without masks."""
    groups = task_records(bank, task_id)
    heldout = holdout_rank(task_id)
    selected = []
    for rank, records in enumerate(groups):
        if rank == heldout:
            continue
        precontact = [int(row["decision_index"]) for row in records
                      if row["phase_proxy"] == "pre_contact_proxy"]
        require(precontact, f"Task {task_id} rank {rank} has no pre-contact state")
        selected.append((rank, max(precontact), records))
    require(len(selected) == 4 and len({rank for rank, _, _ in selected}) == 4,
            "Discovery trajectory selection drift")
    return selected


def bounded(delta: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= MAX_WORLD_NORM_METERS:
        return delta
    return delta * (MAX_WORLD_NORM_METERS / norm)


def replay_to_selected(cfg: dict, wrapper, env, task, initial_states,
                       task_id: int, decision_index: int, records: list[dict]) -> dict:
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action

    state_index = int(records[0]["state_index"])
    seed = int(cfg["seed"]) + task_id * 100 + state_index
    raw = reset_wait(env, initial_states[state_index], seed, int(cfg["num_steps_wait"]))
    queue = deque()
    decision = 0
    selected = [row for row in records if int(row["decision_index"]) == decision_index]
    require(len(selected) == 1 and selected[0]["phase_proxy"] == "pre_contact_proxy",
            "Selected discovery decision drift")
    frozen = selected[0]
    for _ in range(int(cfg["max_steps"])):
        if not queue:
            obs = observation(raw, cfg)
            if decision == decision_index:
                check_observation(obs, frozen["observation"])
                with torch.no_grad():
                    action, dino = differentiable(wrapper, obs, task.language, None, (0, 0))
                action_diff = float((action.cpu() - frozen["clean_action"].float().cpu()).abs().max())
                dino_diff = float((dino.cpu().float() - frozen["clean_dino"].float().cpu()).abs().max())
                require(action_diff <= 0.02 and dino_diff <= 1e-3,
                        f"Clean replay drift action={action_diff} dino={dino_diff}")
                return {"sim_state": env.get_sim_state().copy(), "state_index": state_index,
                        "seed": seed, "action_replay_max_abs": action_diff,
                        "dino_replay_max_abs": dino_diff}
            chunk = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(-1, 7)
            require(chunk.shape == (cfg["num_open_loop_steps"], 7), "OFT chunk shape drift")
            queue.extend(chunk)
            decision += 1
        raw, _, done, _ = env.step(process_libero_action(np.asarray(queue.popleft())).tolist())
        require(not done, "Episode ended before selected discovery decision")
    raise RuntimeError("Selected discovery decision not reached")


def audit_state(env, anchor: dict, joint: str, joint_addr: tuple[int, int], subject: str) -> dict:
    clean_raw = rerender(env, anchor["sim_state"])
    repeat_raw = rerender(env, anchor["sim_state"])
    require(all(np.array_equal(clean_raw[key], repeat_raw[key])
                for key in ("agentview_image", "robot0_eye_in_hand_image")),
            "Same-state renderer is not deterministic")
    clean_qpos = env.sim.data.qpos.copy()
    clean_masks = segmentation_masks(env, clean_raw, subject)
    clean_relative, clean_mass = mask_relative(clean_masks)
    minimum_mass = max(MIN_TARGET_TOKEN_MASS,
                       MIN_TARGET_VISIBLE_FRACTION * clean_mass["target_object"])

    probes, probe_rows = {}, []
    for name, (dx, dy) in (("x_plus", (PROBE_METERS, 0.0)),
                           ("x_minus", (-PROBE_METERS, 0.0)),
                           ("y_plus", (0.0, PROBE_METERS)),
                           ("y_minus", (0.0, -PROBE_METERS))):
        raw, _ = translated_render(env, anchor["sim_state"], joint, dx, dy)
        masks = segmentation_masks(env, raw, subject)
        integrity = render_integrity(clean_raw, raw, clean_masks, masks, clean_qpos,
                                     env.sim.data.qpos.copy(), joint_addr)
        relative, mass = mask_relative(masks)
        require(mass["target_object"] >= minimum_mass, "Jacobian probe target visibility too low")
        probes[name] = relative - clean_relative
        probe_rows.append({"name": name, "world_delta_xy_m": [dx, dy],
                           "token_delta_xy": probes[name].tolist(),
                           "visible_token_mass": mass, "integrity": integrity})
    jacobian = central_jacobian(probes, PROBE_METERS)
    diagnostics = jacobian_diagnostics(jacobian)
    candidates = []
    if diagnostics["valid"]:
        for name, desired_pair in TARGETS:
            desired = np.asarray(desired_pair, dtype=np.float64)
            current = bounded(solve_world_delta(jacobian, desired))
            row = {"name": name, "desired_token_delta_xy": desired.tolist(),
                   "accepted": False, "rejection_reason": None, "evaluations": []}

            def evaluate(delta: np.ndarray) -> dict | None:
                try:
                    raw, _ = translated_render(env, anchor["sim_state"], joint,
                                               float(delta[0]), float(delta[1]))
                    masks = segmentation_masks(env, raw, subject)
                    integrity = render_integrity(clean_raw, raw, clean_masks, masks,
                                                 clean_qpos, env.sim.data.qpos.copy(), joint_addr)
                    relative, mass = mask_relative(masks)
                    if mass["target_object"] < minimum_mass:
                        raise RuntimeError("target_visibility_below_frozen_fraction")
                    actual = relative - clean_relative
                    metrics = candidate_metrics(actual, desired)
                    result = {"world_delta_xy_m": delta.tolist(),
                              "world_norm_m": float(np.linalg.norm(delta)),
                              "actual_token_delta_xy": actual.tolist(), "metrics": metrics,
                              "visible_token_mass": mass, "integrity": integrity,
                              "valid": True}
                except RuntimeError as exc:
                    result = {"world_delta_xy_m": delta.tolist(),
                              "world_norm_m": float(np.linalg.norm(delta)),
                              "valid": False, "rejection_reason": str(exc)}
                row["evaluations"].append(result)
                return result if result["valid"] else None

            best = evaluate(current)
            for _ in range(MAX_CORRECTION_ROUNDS):
                if best is not None and best["metrics"]["accepted"]:
                    break
                if best is None:
                    break
                actual = np.asarray(best["actual_token_delta_xy"], dtype=np.float64)
                update = solve_world_delta(jacobian, desired - actual)
                trials = []
                seen = set()
                for scale in LINE_SEARCH_SCALES:
                    proposal = bounded(current + scale * update)
                    key = tuple(np.round(proposal, 10))
                    if key not in seen:
                        seen.add(key)
                        trial = evaluate(proposal)
                        if trial is not None:
                            trials.append((trial["metrics"]["target_error_tokens"], proposal, trial))
                if not trials:
                    break
                _, proposal, trial = min(trials, key=lambda item: item[0])
                if trial["metrics"]["target_error_tokens"] >= best["metrics"]["target_error_tokens"]:
                    break
                current, best = proposal, trial
            if best is not None:
                row["metrics"] = best["metrics"]
                row["selected_world_delta_xy_m"] = current.tolist()
                row["accepted"] = bool(best["metrics"]["accepted"])
            if not row["accepted"]:
                row["rejection_reason"] = "bounded_oracle_solver_did_not_reach_target"
            candidates.append(row)
    pair_gate = symmetric_pair_gate(candidates) if candidates else {
        "all_four_accepted": False, "reason": "invalid_local_jacobian"}
    return {"clean_relative_xy_tokens": clean_relative.tolist(),
            "clean_visible_token_mass": clean_mass,
            "minimum_target_token_mass": minimum_mass,
            "probes": probe_rows, "jacobian": diagnostics,
            "candidates": candidates, "symmetric_pair_gate": pair_gate}


def run_task(task_id: int) -> dict:
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object", "Discovery audit requires Object config")
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
            clean_raw = rerender(env, anchor["sim_state"])
            subject = target_goal_subject(env)
            target = env.env.objects_dict[subject]
            require(len(target.joints) == 1, "Target requires one free joint")
            joint = target.joints[0]
            joint_addr = env.sim.model.get_joint_qpos_addr(joint)
            require(isinstance(joint_addr, tuple) and joint_addr[1] - joint_addr[0] == 7,
                    "Target requires a 7D free joint")
            del clean_raw
            result = audit_state(env, anchor, joint, joint_addr, subject)
            row = {"trajectory_rank": rank, "decision_index": decision,
                   "state_index": anchor["state_index"], "seed": anchor["seed"],
                   "clean_replay": {key: anchor[key] for key in
                                    ("action_replay_max_abs", "dino_replay_max_abs")},
                   **result}
            rows.append(row)
            print(json.dumps({"task_id": task_id, "trajectory_rank": rank,
                              "decision_index": decision,
                              "all_four_accepted": result["symmetric_pair_gate"]["all_four_accepted"]}),
                  flush=True)
    finally:
        env.close()
    receipt = {"status": "OBJECT_POSE_JACOBIAN_DISCOVERY_TASK_COMPLETE",
               "diagnostic_not_attack": True, "no_counterfactual_model_inference": True,
               "task_id": task_id, "heldout_trajectory_rank_untouched": holdout_rank(task_id),
               "discovery_trajectory_units": 4, "bank_sha256": sha256_file(bank_path),
               "source_config_sha256": sha256_file(CONFIG_PATH),
               "solver": {"probe_m": PROBE_METERS,
                          "maximum_condition_number": MAX_CONDITION_NUMBER,
                          "maximum_world_norm_m": MAX_WORLD_NORM_METERS,
                          "line_search_scales": LINE_SEARCH_SCALES,
                          "maximum_correction_rounds": MAX_CORRECTION_ROUNDS,
                          "minimum_target_visible_fraction": MIN_TARGET_VISIBLE_FRACTION,
                          "minimum_target_token_mass": MIN_TARGET_TOKEN_MASS},
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
                          "heldout_rank": holdout_rank(args.task_id),
                          "discovery_rank_decisions": [[rank, decision]
                                                       for rank, decision, _ in selections],
                          "trajectory_units": len(selections),
                          "gpu_job_submitted": False}, indent=2))
        return
    print(json.dumps(run_task(args.task_id), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
