#!/usr/bin/env python3
"""Collect frozen-UADA2000 attacked Object trajectories and DINO restoration scores."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker")]
sys.path.append(str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from dino_replay_common import CONFIG, config, runtime_cfg, selected_decisions, source_records
from white_patch.UADA_OFT_DINO import atomic_json, atomic_torch, require, sha256_file
from runtime import (action_with_dino_value, differentiable_visual, load_wrapper,
                     make_env, observation, paste_numpy, renderer_metadata, reset_wait)


def rollout(env, wrapper, task, initial, seed, cfg, runtime, patch):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw = reset_wait(env, initial, seed, int(cfg["num_steps_wait"]))
    queue = deque()
    snapshots = []
    success = False
    steps = 0
    for _ in range(int(runtime["max_steps"])):
        if not queue:
            clean = observation(raw, runtime)
            snapshots.append({key: np.array(clean[key], copy=True)
                              for key in ("full_image", "wrist_image", "state")})
            attacked = paste_numpy(clean, patch, tuple(cfg["patch_xy"]))
            action = np.asarray(wrapper.predict_action(attacked, task.language), dtype=np.float32).reshape(-1, 7)
            require(action.shape == (8, 7) and bool(np.isfinite(action).all()), "Bad rollout action")
            queue.extend(action)
        raw, _reward, done, _info = env.step(process_libero_action(queue.popleft()).tolist())
        steps += 1
        if done:
            success = True
            break
    require(snapshots, "No decision block in attacked rollout")
    return snapshots, success, steps


def score_record(wrapper, obs, task_name, patch, cfg):
    xy = tuple(cfg["patch_xy"])
    low = torch.tensor(cfg["action_head_q01"], device=wrapper.device).reshape(1, 1, 7)
    high = torch.tensor(cfg["action_head_q99"], device=wrapper.device).reshape(1, 1, 7)
    with torch.no_grad():
        a0, d0, _ = differentiable_visual(wrapper, obs, task_name, None, xy)
        ap, dp, _ = differentiable_visual(wrapper, obs, task_name, patch, xy)
        ar = action_with_dino_value(wrapper, obs, task_name, d0, patch, xy)
        target = torch.where(a0 >= (low + high) / 2, low, high)
        field = cfg["maskidx"]
        lp = F.mse_loss(ap[..., field], target[..., field])
        lr = F.mse_loss(ar[..., field], target[..., field])
        score = max(0., float(lr - lp))
        dino_delta = float((dp.float() - d0.float()).square().mean().sqrt())
    require(np.isfinite([score, dino_delta, float(lp), float(lr)]).all(), "Nonfinite replay score")
    return {"dino_score": score, "dino_delta_rms": dino_delta,
            "patched_target_mse": float(lp), "restored_target_mse": float(lr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--task-id", type=int, choices=range(10))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    clean = source_records(cfg)
    runtime = runtime_cfg(cfg)
    starts = {task: sorted({int(r["state_index"]) for r in clean if int(r["task_id"]) == task})
              for task in range(10)}
    if args.preflight:
        print(json.dumps({"status": "PASS", "source_states": starts,
                          "collector_shards": 10, "gpu_jobs_submitted": False}, indent=2))
        return
    require(args.task_id is not None and os.environ.get("SLURM_JOB_ID"),
            "Collection requires a user-submitted Slurm task ID")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one GPU required")
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
            and os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None,
            "EGL GPU renderer environment missing; refusing to load model")
    out = Path(cfg["output_root"]) / "data"
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"task_{args.task_id:02d}.pt"
    receipt = out / f"task_{args.task_id:02d}.json"
    if shard.exists() or receipt.exists():
        require(shard.exists() and receipt.exists(), "Incomplete replay shard pair")
        old = json.loads(receipt.read_text())
        require(old["status"] == "COMPLETE" and old["shard_sha256"] == sha256_file(shard)
                and old["config_sha256"] == sha256_file(args.config), "Existing replay shard drift")
        print(json.dumps({"status": "ALREADY_COMPLETE", "task_id": args.task_id}))
        return
    started = time.monotonic()
    wrapper = load_wrapper(runtime)
    env, task, initial_states = make_env(runtime, args.task_id)
    renderer, vendor = renderer_metadata(runtime)
    require("nvidia" in vendor.lower(), "Replay renderer is not NVIDIA")
    task_names = {r["task_name"] for r in clean if int(r["task_id"]) == args.task_id}
    require(task_names == {task.language}, "Replay task language differs from source Object bank")
    patch = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"].float().to(wrapper.device)
    # paste_numpy rounds RGB to uint8. Score the image the policy actually saw.
    displayed_patch = patch.mul(255).round().div(255)
    rows, episodes = [], []
    try:
        require(max(starts[args.task_id]) < len(initial_states), "Frozen source initial state unavailable")
        trajectory = []
        for rank, state_index in enumerate(starts[args.task_id]):
            seed = int(cfg["rollout_seed"]) + 100 * args.task_id + state_index
            states, success, steps = rollout(env, wrapper, task, initial_states[state_index], seed, cfg, runtime, patch)
            episodes.append({"task_id": args.task_id, "state_index": state_index, "trajectory_rank": rank,
                             "rollout_seed": seed, "success": int(success), "steps": steps,
                             "decision_blocks": len(states)})
            trajectory.append((rank, state_index, seed, states, success))
            print(f"REPLAY_COLLECT task={args.task_id} state={state_index} success={int(success)} "
                  f"decisions={len(states)}", flush=True)
        survivors = [item for item in trajectory if item[-1]]
        candidates = survivors if survivors else trajectory
        for rank, state_index, seed, states, success in candidates:
            for decision in selected_decisions(len(states)):
                phase = ("early", "middle", "late")[min(2, 3 * decision // len(states))]
                obs = states[decision]
                score = score_record(wrapper, obs, task.language, displayed_patch, cfg)
                rows.append({"task_id": args.task_id, "task_name": task.language,
                             "state_index": state_index, "trajectory_rank": rank,
                             "decision_index": decision, "phase_proxy": phase,
                             "rollout_seed": seed, "attacked_episode_success": bool(success),
                             "observation": obs, **score})
        require(rows and len(rows) <= 30, "Replay per-task frame count drift")
        payload = {"records": rows, "episodes": episodes, "task_id": args.task_id,
                   "experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(args.config),
                   "baseline_patch_sha256": cfg["baseline_patch_sha256"]}
        atomic_torch(payload, shard)
        atomic_json({"status": "COMPLETE", "task_id": args.task_id, "shard_sha256": sha256_file(shard),
                     "config_sha256": sha256_file(args.config), "records": len(rows), "episodes": len(episodes),
                     "surviving_episodes": len(survivors), "renderer": renderer, "vendor": vendor,
                     "gpu_name": torch.cuda.get_device_name(0),
                     "elapsed_s": time.monotonic() - started}, receipt)
    finally:
        env.close()
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
