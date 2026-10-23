#!/usr/bin/env python3
"""Collect fresh attacked Object trajectories and scored three-decision clips."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from collect_exploring_oft_dino_replay import rollout, score_record
from dino_window_common import (CONFIG, clip_indices, config, runtime_cfg,
                                selected_decisions, source_records)
from white_patch.UADA_OFT_DINO import atomic_json, atomic_torch, require, sha256_file
from runtime import load_wrapper, make_env, renderer_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--task-id", type=int, choices=range(10))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    clean = source_records(cfg)
    starts = {task: sorted({int(r["state_index"]) for r in clean if int(r["task_id"]) == task})
              for task in range(10)}
    if args.preflight:
        print(json.dumps({"status": "PASS", "source_states": starts,
                          "collector_shards": 10, "gpu_jobs_submitted": False}, indent=2))
        return
    require(args.task_id is not None and bool(os.environ.get("SLURM_JOB_ID")),
            "Collection requires a user-submitted Slurm task")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one mapped GPU required")
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
            and os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None,
            "EGL renderer environment missing")
    out = Path(cfg["output_root"]) / "data"
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"task_{args.task_id:02d}.pt"
    receipt = shard.with_suffix(".json")
    if shard.exists() or receipt.exists():
        require(shard.exists() and receipt.exists(), "Incomplete window shard pair")
        old = json.loads(receipt.read_text())
        require(old["status"] == "COMPLETE" and old["shard_sha256"] == sha256_file(shard)
                and old["config_sha256"] == sha256_file(args.config),
                "Existing window shard drift")
        print(json.dumps({"status": "ALREADY_COMPLETE", "task_id": args.task_id}))
        return
    started = time.monotonic()
    runtime = runtime_cfg(cfg)
    wrapper = load_wrapper(runtime)
    env, task, initial_states = make_env(runtime, args.task_id)
    try:
        renderer, vendor = renderer_metadata(runtime)
        require("nvidia" in vendor.lower(), "Window collector renderer is not NVIDIA")
        task_names = {r["task_name"] for r in clean if int(r["task_id"]) == args.task_id}
        require(task_names == {task.language}, "Source Object task language drift")
        require(max(starts[args.task_id]) < len(initial_states), "Source initial state unavailable")
        patch = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"].float().to(wrapper.device)
        displayed = patch.mul(255).round().div(255)
        trajectories, episodes = [], []
        for rank, state_index in enumerate(starts[args.task_id]):
            seed = int(cfg["rollout_seed"]) + 100 * args.task_id + state_index
            states, success, steps = rollout(env, wrapper, task, initial_states[state_index],
                                             seed, cfg, runtime, patch)
            trajectories.append((rank, state_index, seed, states, success))
            episodes.append({"task_id": args.task_id, "state_index": state_index,
                             "trajectory_rank": rank, "rollout_seed": seed,
                             "success": int(success), "steps": steps,
                             "decision_blocks": len(states)})
            print(f"DINO_WINDOW_COLLECT task={args.task_id} state={state_index} "
                  f"success={int(success)} decisions={len(states)}", flush=True)
        survivors = [row for row in trajectories if row[-1]]
        rows = []
        for rank, state_index, seed, states, success in survivors or trajectories:
            for decision in selected_decisions(len(states), cfg["centers_per_trajectory"]):
                indices = clip_indices(decision, len(states))
                score = score_record(wrapper, states[decision], task.language, displayed, cfg)
                rows.append({"task_id": args.task_id, "task_name": task.language,
                             "state_index": state_index, "trajectory_rank": rank,
                             "decision_index": decision,
                             "window_indices": indices,
                             "observations": [states[i] for i in indices],
                             "phase_proxy": ("early", "middle", "late")[min(2, 3 * decision // len(states))],
                             "rollout_seed": seed, "attacked_episode_success": bool(success), **score})
        require(1 <= len(rows) <= 30, "Window records per task drift")
        atomic_torch({"records": rows, "episodes": episodes, "task_id": args.task_id,
                      "experiment_id": cfg["experiment_id"],
                      "config_sha256": sha256_file(args.config),
                      "baseline_patch_sha256": cfg["baseline_patch_sha256"]}, shard)
        atomic_json({"status": "COMPLETE", "task_id": args.task_id,
                     "shard_sha256": sha256_file(shard),
                     "config_sha256": sha256_file(args.config),
                     "records": len(rows), "episodes": len(episodes),
                     "surviving_episodes": len(survivors), "renderer": renderer,
                     "vendor": vendor, "gpu_name": torch.cuda.get_device_name(0),
                     "elapsed_s": time.monotonic() - started}, receipt)
    finally:
        env.close()
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
