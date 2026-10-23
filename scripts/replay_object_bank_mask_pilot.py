#!/usr/bin/env python3
"""Replay one exact LIBERO-Object bank trajectory and audit simulator masks.

No masks are accepted unless task language, Wrist/Front RGB, proprioception,
clean action, and successful rollout match the frozen 300-state Object bank.
This pilot never trains a patch or submits a job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MGWP_ROOT = REPO_ROOT / "oft_mgwp_wrist_only"
sys.path.insert(0, str(MGWP_ROOT / "scripts"))

from common import CONFIG_PATH, load_json, require, require_gpu, sha256_file  # noqa: E402
from runtime import differentiable, load_wrapper, observation, reset_wait  # noqa: E402


def select_records(bank: dict[str, Any], task_id: int, trajectory_rank: int) -> list[dict[str, Any]]:
    require(bank["experiment_id"] == "OFT-MGWP-WRIST-STAGE1-01", "Unexpected source bank")
    require(bank["selected_xy"] == [26, 137], "Bank patch anchor drift")
    records = [row for row in bank["records"]
               if int(row["task_id"]) == task_id and int(row["trajectory_rank"]) == trajectory_rank]
    require(len(records) == 6, "Expected six sampled decisions in one clean trajectory")
    require(len({int(row["state_index"]) for row in records}) == 1, "Initial-state mismatch")
    require(len({str(row["task_name"]) for row in records}) == 1, "Task-language mismatch")
    require(len({int(row["decision_index"]) for row in records}) == 6, "Duplicate sampled decision")
    return sorted(records, key=lambda row: int(row["decision_index"]))


def make_segmentation_env(cfg: dict[str, Any], task_id: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    bench = benchmark.get_benchmark_dict()[cfg["suite"]]()
    task = bench.get_task(task_id)
    initial_states = bench.get_task_init_states(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = SegmentationRenderEnv(
        bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256, camera_widths=256, camera_segmentations="instance",
    )
    env.reset()
    return env, task, initial_states


def target_goal_subject(env: Any) -> str:
    inner = getattr(env, "env", env)
    goals = inner.parsed_problem["goal_state"]
    subjects = {str(goal[1]) for goal in goals
                if len(goal) >= 3 and str(goal[0]).lower() in ("in", "on")}
    require(len(subjects) == 1, f"Expected exactly one manipulated Object goal subject: {subjects}")
    subject = next(iter(subjects))
    require(subject in inner.objects_dict, f"Goal subject is not a simulator object: {subject}")
    return subject


def segmentation_masks(env: Any, raw: dict[str, Any], target_name: str) -> dict[str, np.ndarray]:
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    keys = [key for key in raw if key.startswith("robot0_eye_in_hand") and "segmentation" in key]
    require(len(keys) == 1, f"Expected one Wrist instance segmentation sensor: {keys}")
    ids = np.asarray(raw[keys[0]]).squeeze()[::-1, ::-1].astype(np.int32)
    inner = getattr(env, "env", env)
    names = list(inner.model.instances_to_ids.keys())
    require(ids.shape == (256, 256), f"Unexpected segmentation shape: {ids.shape}")
    require(int(ids.min()) >= 0 and int(ids.max()) <= len(names), "Instance ID out of mapping range")
    robot_names = [name for name in names
                   if any(marker in name.lower() for marker in ("panda", "robot", "mount", "gripper"))]
    target_names = [name for name in names if target_name in name or name in target_name]
    require(robot_names and target_names, f"Missing robot or target instance: {target_name}")
    arm = np.isin(ids, [names.index(name) + 1 for name in robot_names])
    target = np.isin(ids, [names.index(name) + 1 for name in target_names])
    require(bool(target.any()) and bool(arm.any()), "Target or robot invisible in Wrist")
    matrix = get_camera_transform_matrix(env.sim, "robot0_eye_in_hand", 256, 256)
    center = project_points_from_world_to_camera(
        np.asarray(raw["robot0_eef_pos"], dtype=np.float64)[None], matrix, 256, 256,
    )[0]
    yy, xx = np.ogrid[:256, :256]
    disk = (yy - float(center[0])) ** 2 + (xx - float(center[1])) ** 2 <= 18.0 ** 2
    gripper = arm & disk
    require(bool(gripper.any()), "Gripper invisible in Wrist")
    forbidden = arm | target
    candidates = np.flatnonzero(~forbidden)
    distance = ((yy - float(center[0])) ** 2 + (xx - float(center[1])) ** 2).reshape(-1)
    ordered = candidates[np.argsort(distance[candidates], kind="stable")]
    background = np.zeros(ids.size, dtype=bool)
    background[ordered[:int(max(target.sum(), gripper.sum()))]] = True
    require(bool(background.any()), "No matched background region")
    return {"target_object": target.astype(np.uint8), "gripper": gripper.astype(np.uint8),
            "matched_background": background.reshape(ids.shape).astype(np.uint8)}


def check_observation(observed: dict[str, Any], frozen: dict[str, Any]) -> dict[str, float]:
    report = {}
    for key in ("full_image", "wrist_image"):
        current = np.asarray(observed[key], dtype=np.int16)
        original = np.asarray(frozen[key], dtype=np.int16)
        require(current.shape == original.shape == (224, 224, 3), f"{key} shape drift")
        difference = np.abs(current - original)
        report[f"{key}_max_abs"] = float(difference.max())
        report[f"{key}_mae"] = float(difference.mean())
        require(report[f"{key}_max_abs"] <= 1 and report[f"{key}_mae"] <= 0.01,
                f"Exact-bank visual replay failed: {key} {report}")
    current_state = np.asarray(observed["state"], dtype=np.float32)
    frozen_state = np.asarray(frozen["state"], dtype=np.float32)
    require(current_state.shape == frozen_state.shape == (8,), "Proprioception shape drift")
    report["state_max_abs"] = float(np.max(np.abs(current_state - frozen_state)))
    require(report["state_max_abs"] <= 1e-5, f"Exact-bank proprioception replay failed: {report}")
    return report


def replay(cfg: dict[str, Any], records: list[dict[str, Any]], task_id: int) -> tuple[list[dict], dict]:
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action

    require("3090" in torch.cuda.get_device_name(0), "Pilot uses one RTX 3090")
    wrapper = load_wrapper(cfg)
    env, task, initial_states = make_segmentation_env(cfg, task_id)
    require(task.language == records[0]["task_name"], "Simulator task language differs from bank")
    initial_state_index = int(records[0]["state_index"])
    seed = int(cfg["seed"]) + task_id * 100 + initial_state_index
    expected = {int(row["decision_index"]): row for row in records}
    captured = []
    queue: deque[np.ndarray] = deque()
    decision_index = 0
    target_name = target_goal_subject(env)
    success = False
    try:
        raw = reset_wait(env, initial_states[initial_state_index], seed, int(cfg["num_steps_wait"]))
        for _ in range(int(cfg["max_steps"])):
            if not queue:
                obs = observation(raw, cfg)
                if decision_index in expected:
                    frozen = expected[decision_index]
                    metrics = check_observation(obs, frozen["observation"])
                    with torch.no_grad():
                        action, _dino = differentiable(wrapper, obs, task.language, None, (0, 0))
                    action_diff = (action.float().cpu() - frozen["clean_action"].float().cpu()).abs()
                    metrics["normalized_action_max_abs"] = float(action_diff.max())
                    require(metrics["normalized_action_max_abs"] <= 0.02,
                            f"Exact-bank action replay failed: {metrics}")
                    masks = segmentation_masks(env, raw, target_name)
                    captured.append({"decision_index": decision_index, "phase_proxy": frozen["phase_proxy"],
                                     "metrics": metrics, "wrist_rgb": np.asarray(raw["robot0_eye_in_hand_image"])[::-1, ::-1].copy(),
                                     "masks": masks})
                chunk = np.asarray(wrapper.predict_action(obs, task.language), dtype=np.float32).reshape(-1, 7)
                require(chunk.shape == (cfg["num_open_loop_steps"], 7), "Action chunk shape drift")
                queue.extend(chunk)
                decision_index += 1
            raw, _reward, done, _info = env.step(process_libero_action(np.asarray(queue.popleft())).tolist())
            if done:
                success = True
                break
    finally:
        env.close()
    require(success and len(captured) == 6, f"Replay incomplete: success={success} captured={len(captured)}")
    return captured, {"task_id": task_id, "task_language": task.language,
                      "target_object_name": target_name, "initial_state_index": initial_state_index,
                      "seed": seed, "decisions": decision_index, "clean_success": success}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), default=0)
    parser.add_argument("--trajectory-rank", type=int, choices=range(5), default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CONFIG_PATH)
    require(cfg["suite"] == "libero_object" and cfg["unnorm_key"] == "libero_object_no_noops",
            "Not a LIBERO-Object source")
    bank_path = MGWP_ROOT / "artifacts" / "GEOMETRY_TRAIN_BANK.pt"
    require(bank_path.is_file(), "Missing exact Object bank")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    require(len(bank["records"]) == 300, "Expected 300 exact Object states")
    records = select_records(bank, args.task_id, args.trajectory_rank)
    require(all(str(row["task_name"]).strip() for row in records), "Missing task language")
    summary = {"status": "CHECK_ONLY_PASS" if args.check_only else "REPLAY_PENDING",
               "source_suite": cfg["suite"], "source_checkpoint": cfg["paths"]["checkpoint"],
               "task_id": args.task_id, "trajectory_rank": args.trajectory_rank,
               "initial_state_index": int(records[0]["state_index"]),
               "decision_indices": [int(row["decision_index"]) for row in records],
               "task_language": records[0]["task_name"], "gpu_job_submitted_by_script": False}
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print(json.dumps(require_gpu()), flush=True)
    captured, replay_meta = replay(cfg, records, args.task_id)
    output = PROJECT_ROOT / "outputs/exploring_oft_dino/spatial_relation_object_bank_mask_pilot" / f"task_{args.task_id}"
    output = output / f"trajectory_rank_{args.trajectory_rank}"
    require(not output.exists(), f"Refusing to overwrite existing pilot output: {output}")
    output.mkdir(parents=True)
    receipt = {**summary, **replay_meta, "status": "EXACT_OBJECT_REPLAY_MASK_PILOT_COMPLETE",
               "bank_sha256": sha256_file(bank_path),
               "source_config_sha256": sha256_file(CONFIG_PATH), "samples": []}
    for sample in captured:
        decision_index = sample["decision_index"]
        path = output / f"decision_{decision_index:03d}.npz"
        np.savez_compressed(path, wrist_rgb=sample["wrist_rgb"],
                            **{f"wrist_pixel_mask_{role}": mask for role, mask in sample["masks"].items()})
        receipt["samples"].append({"decision_index": decision_index,
                                   "phase_proxy": sample["phase_proxy"],
                                   "metrics": sample["metrics"], "array_path": str(path),
                                   "array_sha256": sha256_file(path)})
    (output / "RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
