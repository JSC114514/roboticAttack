#!/usr/bin/env python3
"""Paired, zero-shot four-suite closed-loop evaluation of five Scale-500 patches."""

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
RAMP_SCRIPTS = ROOT.parent / "oft_ramp_recovery_aware" / "scripts"
sys.path.insert(0, str(RAMP_SCRIPTS))

from common import atomic_parquet, load_json, require, require_gpu, sha256_file  # noqa: E402
from run_continuous_cldev import run_episode  # noqa: E402
from runtime import load_wrapper, make_env, renderer_metadata  # noqa: E402


def runtime_config(base: dict, suite: str) -> dict:
    require(suite in base["suites"], f"Unfrozen suite: {suite}")
    cfg = dict(base)
    cfg["suite"] = suite
    cfg["max_steps"] = int(base["max_steps_by_suite"][suite])
    cfg["unnorm_key"] = base["unnorm_keys"][suite]
    cfg["paths"] = {**base["paths"], "checkpoint": base["checkpoints"][suite]}
    return cfg


def frozen_clean(base: dict, suite: str) -> pd.DataFrame:
    path = base["frozen_object_clean"] if suite == "libero_object" else base["frozen_cross_suite_clean"]
    old = pd.read_parquet(path)
    clean = old.loc[(old.suite == suite) & (old.condition == "clean"),
                    ["suite", "task_id", "state_index", "success", "rollout_seed"]].copy()
    require(len(clean) == 100 and not clean.duplicated(["task_id", "state_index"]).any(),
            f"Frozen clean grid drift: {suite}")
    require(set(clean.task_id.astype(int)) == set(range(10))
            and set(clean.state_index.astype(int)) == set(map(int, base["state_indices"])),
            f"Frozen clean task/state coverage drift: {suite}")
    require(clean.success.isin([0, 1]).all(), f"Invalid clean success: {suite}")
    return clean


def load_patches(base: dict, device: torch.device) -> dict[str, tuple[torch.Tensor, str]]:
    if "patch_sources" in base:
        if "required_manual_visual_review" in base:
            require(base["manual_visual_review"] == base["required_manual_visual_review"],
                    "New patch awaits user visual approval")
        else:
            require(base["manual_visual_review"] == "PASS_USER_FULL_WORST500",
                    "New Full-WorstChunk patch awaits user visual approval")
        require(set(base["patch_sources"]) == set(base["conditions"]),
                "Control patch source coverage drift")
        patches = {}
        hashes = set()
        for method in base["conditions"]:
            source = base["patch_sources"][method]
            source_method = source.get("method", method)
            status = load_json(source["training_status"])
            require(status["status"] == "READY_FOR_MANUAL_VISUAL_REVIEW"
                    and not status["missing_images"] and not status["missing_mechanism_probes"]
                    and any(row.get("method") == source_method and row.get("status") == "COMPLETE"
                            for row in status["methods"]),
                    f"Incomplete training or structure report: {method}")
            path = Path(source["patch_path"])
            require(path.is_file(), f"Missing control patch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload.get("experiment_id") == source["experiment_id"]
                    and payload.get("method") == source_method
                    and int(payload.get("outer_iterations", -1)) == int(source.get("outer_iterations", 500)),
                    f"Control patch lineage drift: {method}")
            patch = payload["patch"].float()
            require(tuple(patch.shape) == tuple(base["patch_shape"])
                    and torch.isfinite(patch).all()
                    and bool(((patch >= 0) & (patch <= 1)).all()),
                    f"Invalid control patch tensor: {method}")
            digest = sha256_file(path)
            require(digest not in hashes, "Duplicate control patch content")
            hashes.add(digest)
            patches[method] = (patch.to(device), digest)
        return patches
    report = load_json(base["training_status"])
    require(report["status"] == "READY_FOR_MANUAL_VISUAL_REVIEW"
            and not report["missing_images"] and not report["missing_mechanism_probes"],
            "Scale-500 training or structure report not complete")
    require(base["manual_visual_review"] == "PASS_USER_20260921_SCALE500",
            "User visual approval missing")
    require({row["method"] for row in report["methods"]} == set(base["conditions"])
            and all(row["status"] == "COMPLETE" for row in report["methods"]),
            "Training method coverage drift")
    patches = {}
    hashes = set()
    for method in base["conditions"]:
        path = Path(base["patch_root"]) / method / "outer_0500.pt"
        require(path.is_file(), f"Missing Scale-500 patch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("experiment_id") == "EXPLORING-OFT-DINO-SCALE500-01"
                and payload.get("method") == method
                and int(payload.get("outer_iterations", -1)) == 500,
                f"Patch identity or training lineage drift: {path}")
        patch = payload["patch"].float()
        require(tuple(patch.shape) == tuple(base["patch_shape"])
                and torch.isfinite(patch).all() and bool(((patch >= 0) & (patch <= 1)).all()),
                f"Invalid patch tensor: {path}")
        digest = sha256_file(path)
        require(digest not in hashes, f"Duplicate patch file content: {path}")
        hashes.add(digest)
        patches[method] = (patch.to(device), digest)
    return patches


def validate(rollouts: pd.DataFrame, traces: pd.DataFrame, suite: str,
             task_id: int, base: dict) -> None:
    conditions = set(base["conditions"])
    states = set(map(int, base["state_indices"]))
    require(len(rollouts) == len(states) * len(conditions), "Cross-suite rollout count mismatch")
    require(set(rollouts.suite) == {suite} and set(rollouts.task_id.astype(int)) == {task_id},
            "Cross-suite shard identity drift")
    require(set(rollouts.state_index.astype(int)) == states and set(rollouts.condition) == conditions,
            "Cross-suite state/condition coverage drift")
    require(not rollouts.duplicated(["suite", "task_id", "state_index", "condition"]).any(),
            "Duplicate cross-suite rollout")
    require(rollouts.condition_order.between(1, len(conditions)).all(), "Condition order drift")
    require(rollouts.groupby("state_index").rollout_seed.nunique().eq(1).all(),
            "Paired rollout seed drift")
    require(rollouts.groupby("state_index").frozen_clean_success.nunique().eq(1).all(),
            "Paired clean eligibility drift")
    require(rollouts.frozen_clean_success.isin([0, 1]).all(), "Invalid clean eligibility")
    require(rollouts.renderer_vendor.str.lower().str.contains("nvidia").all(), "Non-NVIDIA renderer")
    require(not traces.empty and not traces.duplicated(
        ["suite", "task_id", "state_index", "condition", "decision_block"]).any(),
        "Cross-suite trace grid drift")
    require(np.isfinite(traces.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Non-finite trace values")


def run_shard(array_index: int, config: Path, patch_loader=None) -> None:
    base = load_json(config)
    log_tag = base.get("log_tag", "SCALE500_XSUITE")
    require(base["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"],
            "Frozen four-suite order drift")
    require(base["patch_xy"] == [26, 137], "Frozen Wrist placement drift")
    task_count = int(base["task_count"])
    suite = base["suites"][array_index // task_count]
    task_id = array_index % task_count
    cfg = runtime_config(base, suite)
    clean = frozen_clean(base, suite)
    task_clean = clean.loc[clean.task_id.astype(int) == task_id]
    output = Path(base["output_root"]) / suite
    output.mkdir(parents=True, exist_ok=True)
    final_rollouts = output / f"task_{task_id:02d}_rollouts.parquet"
    final_traces = output / f"task_{task_id:02d}_traces.parquet"
    if final_rollouts.is_file() and final_traces.is_file():
        validate(pd.read_parquet(final_rollouts), pd.read_parquet(final_traces), suite, task_id, base)
        print(f"{log_tag}_ALREADY_COMPLETE suite={suite} task={task_id}", flush=True)
        return
    partial_rollouts = output / f"task_{task_id:02d}_rollouts.partial.parquet"
    partial_traces = output / f"task_{task_id:02d}_traces.partial.parquet"
    rollout_rows = pd.read_parquet(partial_rollouts).to_dict("records") if partial_rollouts.is_file() else []
    trace_rows = pd.read_parquet(partial_traces).to_dict("records") if partial_traces.is_file() else []
    conditions = tuple(base["conditions"])
    complete = {int(state) for state, group in pd.DataFrame(rollout_rows).groupby("state_index")
                if len(group) == len(conditions) and set(group.condition) == set(conditions)} \
        if rollout_rows else set()
    rollout_rows = [row for row in rollout_rows if int(row["state_index"]) in complete]
    trace_rows = [row for row in trace_rows if int(row["state_index"]) in complete]

    wrapper = load_wrapper(cfg)
    env, task, init_states = make_env(cfg, task_id)
    require(max(map(int, base["state_indices"])) < len(init_states),
            f"State 49 unavailable: {suite} task={task_id}")
    renderer, vendor = renderer_metadata(cfg)
    patches = (patch_loader or load_patches)(base, wrapper.device)
    try:
        for state_index in map(int, base["state_indices"]):
            if state_index in complete:
                continue
            match = task_clean.loc[task_clean.state_index.astype(int) == state_index]
            require(len(match) == 1, f"Missing frozen clean seed: {suite} {task_id} {state_index}")
            seed = int(match.iloc[0].rollout_seed)
            clean_success = int(match.iloc[0].success)
            order = list(conditions)
            order_seed = int(base["order_seed"]) + 10_000 * base["suites"].index(suite) + 100 * task_id + state_index
            random.Random(order_seed).shuffle(order)
            for order_index, method in enumerate(order, start=1):
                patch, digest = patches[method]
                rollout, trace = run_episode(env, wrapper, task, init_states[state_index],
                                             seed, method, patch, digest, cfg, renderer, vendor, order_index)
                metadata = {
                    "experiment_id": base["experiment_id"], "training_suite": base["training_suite"],
                    "suite": suite, "task_id": task_id, "task_name": task.language,
                    "state_index": state_index, "frozen_clean_success": clean_success,
                    "checkpoint": cfg["paths"]["checkpoint"], "unnorm_key": cfg["unnorm_key"],
                }
                rollout_rows.append({**metadata, **rollout})
                trace_rows.extend({**metadata, **row} for row in trace)
                print(f"{log_tag} suite={suite} task={task_id} state={state_index} "
                      f"method={method} success={rollout['success']}", flush=True)
            atomic_parquet(pd.DataFrame(rollout_rows), partial_rollouts)
            atomic_parquet(pd.DataFrame(trace_rows), partial_traces)
        rollouts, traces = pd.DataFrame(rollout_rows), pd.DataFrame(trace_rows)
        validate(rollouts, traces, suite, task_id, base)
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
    parser.add_argument("--array-index", type=int, choices=range(40), required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    run_shard(args.array_index, Path(args.config).resolve())


if __name__ == "__main__":
    main()
