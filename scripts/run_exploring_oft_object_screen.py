#!/usr/bin/env python3
"""Paired Object closed-loop screen for provenance-correct Exploring→OFT patches."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RAMP_ROOT = ROOT.parent / "oft_ramp_recovery_aware"
RAMP_SCRIPTS = RAMP_ROOT / "scripts"
sys.path.insert(0, str(RAMP_SCRIPTS))

from common import atomic_parquet, load_json, require, require_gpu, sha256_file  # noqa: E402
from run_continuous_cldev import run_episode  # noqa: E402
from runtime import load_wrapper, make_env, renderer_metadata  # noqa: E402


def frozen_clean(cfg: dict) -> pd.DataFrame:
    old = pd.read_parquet(cfg["paths"]["existing_final100_rollouts"])
    clean = old.loc[old.condition == "clean", [
        "task_id", "state_index", "success", "rollout_seed"]].copy()
    require(len(clean) == 100 and not clean.duplicated(["task_id", "state_index"]).any(),
            "Frozen Object Final-100 clean grid drift")
    require(set(clean.state_index.astype(int)) == set(map(int, cfg["state_indices"])),
            "Frozen clean state split drift")
    return clean


def load_patches(cfg: dict, device: torch.device) -> dict[str, tuple[torch.Tensor, str]]:
    status = load_json(cfg["training_status"])
    require(status["status"] == "READY_FOR_MANUAL_VISUAL_REVIEW", "Training structure report not ready")
    require(not status["missing_images"], "Training structure report has missing patch images")
    require(cfg["manual_visual_review"].startswith("PASS_USER_"), "Manual visual review has not passed")
    require(set(cfg["patches"]) == set(cfg["conditions"]), "Patch/condition mapping drift")
    result = {}
    for condition in cfg["conditions"]:
        path = Path(cfg["patches"][condition])
        require(path.is_file(), f"Missing trained patch: {path}")
        payload = torch.load(path, map_location=device, weights_only=False)
        require(payload.get("experiment_id") == "EXPLORING-OFT-DINO-RESTART-TIMEBOXED-01",
                f"Unexpected patch lineage: {condition}")
        patch = payload["patch"].float().clamp(0, 1)
        require(tuple(patch.shape) == tuple(cfg["patch_shape"]), f"Patch shape drift: {condition}")
        result[condition] = (patch, sha256_file(path))
    return result


def validate(rollouts: pd.DataFrame, traces: pd.DataFrame, task_id: int, cfg: dict) -> None:
    conditions = set(cfg["conditions"])
    states = set(map(int, cfg["state_indices"]))
    require(len(rollouts) == len(states) * len(conditions), "Object screen rollout count mismatch")
    require(set(rollouts.condition) == conditions, "Object screen condition drift")
    require(set(rollouts.task_id.astype(int)) == {task_id}, "Object screen task drift")
    require(set(rollouts.state_index.astype(int)) == states, "Object screen state drift")
    require(not rollouts.duplicated(["task_id", "state_index", "condition"]).any(),
            "Duplicate Object screen rollout")
    require(rollouts.condition_order.between(1, len(conditions)).all(), "Condition order drift")
    require(rollouts.renderer_vendor.str.lower().str.contains("nvidia").all(), "Non-NVIDIA renderer")
    require(not traces.empty and not traces.duplicated(
        ["task_id", "state_index", "condition", "decision_block"]).any(), "Trace grid drift")
    require(np.isfinite(traces.select_dtypes(include=[np.number]).to_numpy()).all(), "Non-finite trace")


def run_task(task_id: int, config: Path) -> None:
    cfg = load_json(config)
    require(cfg["suite"] == "libero_object", "Only the frozen Object screen is allowed")
    clean = frozen_clean(cfg)
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    final_rollouts = output / f"task_{task_id:02d}_rollouts.parquet"
    final_traces = output / f"task_{task_id:02d}_traces.parquet"
    if final_rollouts.is_file() and final_traces.is_file():
        validate(pd.read_parquet(final_rollouts), pd.read_parquet(final_traces), task_id, cfg)
        print(f"EXPLORING_OFT_OBJECT_SCREEN_ALREADY_COMPLETE task={task_id}")
        return
    partial_rollouts = output / f"task_{task_id:02d}_rollouts.partial.parquet"
    partial_traces = output / f"task_{task_id:02d}_traces.partial.parquet"
    rollout_rows = pd.read_parquet(partial_rollouts).to_dict("records") if partial_rollouts.is_file() else []
    trace_rows = pd.read_parquet(partial_traces).to_dict("records") if partial_traces.is_file() else []
    conditions = tuple(cfg["conditions"])
    complete = {int(state) for state, group in pd.DataFrame(rollout_rows).groupby("state_index")
                if len(group) == len(conditions) and set(group.condition) == set(conditions)} \
        if rollout_rows else set()
    rollout_rows = [row for row in rollout_rows if int(row["state_index"]) in complete]
    trace_rows = [row for row in trace_rows if int(row["state_index"]) in complete]

    wrapper = load_wrapper(cfg)
    env, task, init_states = make_env(cfg, task_id)
    renderer, vendor = renderer_metadata(cfg)
    patches = load_patches(cfg, wrapper.device)
    task_clean = clean.loc[clean.task_id.astype(int) == task_id]
    try:
        for state_index in map(int, cfg["state_indices"]):
            if state_index in complete:
                continue
            clean_row = task_clean.loc[task_clean.state_index.astype(int) == state_index]
            require(len(clean_row) == 1, f"Missing clean seed task={task_id} state={state_index}")
            seed = int(clean_row.iloc[0].rollout_seed)
            order_conditions = list(conditions)
            random.Random(int(cfg["order_seed"]) + task_id * 100 + state_index).shuffle(order_conditions)
            for order, condition in enumerate(order_conditions, start=1):
                patch, patch_hash = patches[condition]
                rollout, trace = run_episode(
                    env, wrapper, task, init_states[state_index], seed, condition,
                    patch, patch_hash, cfg, renderer, vendor, order)
                meta = {
                    "experiment_id": cfg["experiment_id"], "suite": cfg["suite"],
                    "task_id": task_id, "task_name": task.language, "state_index": state_index,
                    "frozen_clean_success": int(clean_row.iloc[0].success),
                }
                rollout_rows.append({**meta, **rollout})
                trace_rows.extend({**meta, **row} for row in trace)
                print(f"EXPLORING_OFT_OBJECT_SCREEN task={task_id} state={state_index} "
                      f"condition={condition} success={rollout['success']}", flush=True)
            atomic_parquet(pd.DataFrame(rollout_rows), partial_rollouts)
            atomic_parquet(pd.DataFrame(trace_rows), partial_traces)
        rollouts, traces = pd.DataFrame(rollout_rows), pd.DataFrame(trace_rows)
        validate(rollouts, traces, task_id, cfg)
        atomic_parquet(rollouts, final_rollouts)
        atomic_parquet(traces, final_traces)
        partial_rollouts.unlink(missing_ok=True)
        partial_traces.unlink(missing_ok=True)
    finally:
        env.close()
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    run_task(args.task_id, Path(args.config).resolve())


if __name__ == "__main__":
    main()

