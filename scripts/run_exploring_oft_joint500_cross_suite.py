#!/usr/bin/env python3
"""Seven-patch paired closed-loop evaluation on the frozen four-suite grid."""

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
from run_exploring_oft_scale500_cross_suite import frozen_clean, runtime_config, validate  # noqa: E402

NEW_METHODS = (
    "exploring_oft_dino_feature_uniform", "exploring_oft_siglip_feature_uniform",
    "exploring_oft_dino_signed_geometry", "exploring_oft_dino_signed_geometry_sham",
    "exploring_oft_full_persistent6d", "exploring_oft_dino_persistent6d",
)
BASELINE = "exploring_oft_uada_o0500"


def patch_path(base: dict, method: str) -> Path:
    return (Path(base["baseline_patch"]) if method == BASELINE else
            Path(base["training_root"]) / method / "outer_0500.pt")


def validate_sources(base: dict) -> dict[str, Path]:
    require(base["experiment_id"] == "EXPLORING-OFT-JOINT500-V1-XSUITE", "Experiment ID drift")
    require(base["manual_visual_review"] == "PASS_USER_20260924_JOINT500",
            "Manual visual approval missing")
    require(base["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"],
            "Suite order drift")
    require(base["task_count"] == 10 and base["state_indices"] == list(range(40, 50)),
            "Frozen task/state grid drift")
    require(base["conditions"] == [BASELINE, *NEW_METHODS] and
            set(base["patch_sha256"]) == set(base["conditions"]), "Condition contract drift")
    require(base["patch_xy"] == [26, 137] and base["patch_shape"] == [3, 50, 50],
            "Wrist patch placement/shape drift")
    require(Path(base["output_root"]).resolve() != Path(base["training_root"]).resolve(),
            "Evaluation output overlaps training root")
    paths = {}
    for method in base["conditions"]:
        path = patch_path(base, method)
        require(path.is_file() and sha256_file(path) == base["patch_sha256"][method],
                f"Patch missing or SHA256 drift: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_id = ("EXPLORING-OFT-DINO-SCALE500-01" if method == BASELINE else
                       "EXPLORING-OFT-JOINT500-V1")
        expected_method = "exploring_oft_uada" if method == BASELINE else method
        require(payload.get("experiment_id") == expected_id and payload.get("method") == expected_method
                and int(payload.get("outer_iterations", -1)) == 500,
                f"Patch lineage drift: {method}")
        patch = payload["patch"]
        require(tuple(patch.shape) == (3, 50, 50) and torch.isfinite(patch).all()
                and bool(((patch >= 0) & (patch <= 1)).all()), f"Invalid RGB patch: {method}")
        if method != BASELINE:
            directory = path.parent
            status = load_json(directory / "status.json")
            require(status.get("status") == "COMPLETE" and status.get("method") == method,
                    f"Incomplete training status: {method}")
            require(status.get("baseline_patch_sha256") == base["patch_sha256"][BASELINE],
                    f"Training baseline identity drift: {method}")
            require((directory / "outer_0500.png").is_file(), f"Missing final visual: {method}")
            for outer in (100, 200, 500):
                gate = load_json(directory / f"structure_gate_{outer:04d}.json")
                require(gate.get("outer_iteration") == outer and gate.get("passed") is True,
                        f"Structure gate failed: {method} outer={outer}")
        paths[method] = path
    require(len(set(base["patch_sha256"].values())) == len(paths), "Duplicate patch digest")
    return paths


def preflight(base: dict) -> dict:
    paths = validate_sources(base)
    denominator = {}
    for suite in base["suites"]:
        clean = frozen_clean(base, suite)
        denominator[suite] = int(clean.success.sum())
        runtime = runtime_config(base, suite)
        require(Path(runtime["paths"]["checkpoint"], "config.json").is_file(),
                f"Missing target checkpoint: {suite}")
        require(runtime["unnorm_key"] == base["unnorm_keys"][suite],
                f"Normalization key drift: {suite}")
    require(denominator == {"libero_object": 99, "libero_spatial": 90,
                            "libero_goal": 98, "libero_10": 95}, "Clean denominator drift")
    return {"status": "JOINT500_XSUITE_PREFLIGHT_PASS", "conditions": list(paths),
            "frozen_clean_episodes": 400, "new_attack_episodes": 2800,
            "eligible_attack_rows_expected": len(paths) * sum(denominator.values()),
            "clean_success_denominators": denominator}


def load_patches(base: dict, device: torch.device) -> dict[str, tuple[torch.Tensor, str]]:
    paths = validate_sources(base)
    patches = {}
    for method, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        patches[method] = (payload["patch"].float().to(device), base["patch_sha256"][method])
    return patches


def run_shard(array_index: int, base: dict) -> None:
    require(0 <= array_index < 40, "Array index outside frozen task grid")
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
        print(f"JOINT500_XSUITE_ALREADY_COMPLETE suite={suite} task={task_id}", flush=True)
        return
    partial_rollouts = output / f"task_{task_id:02d}_rollouts.partial.parquet"
    partial_traces = output / f"task_{task_id:02d}_traces.partial.parquet"
    rollout_rows = pd.read_parquet(partial_rollouts).to_dict("records") if partial_rollouts.is_file() else []
    trace_rows = pd.read_parquet(partial_traces).to_dict("records") if partial_traces.is_file() else []
    conditions = tuple(base["conditions"])
    complete = ({int(state) for state, group in pd.DataFrame(rollout_rows).groupby("state_index")
                 if len(group) == len(conditions) and set(group.condition) == set(conditions)}
                if rollout_rows else set())
    rollout_rows = [row for row in rollout_rows if int(row["state_index"]) in complete]
    trace_rows = [row for row in trace_rows if int(row["state_index"]) in complete]

    wrapper = load_wrapper(cfg)
    env, task, init_states = make_env(cfg, task_id)
    require(max(map(int, base["state_indices"])) < len(init_states),
            f"Frozen state unavailable: {suite} task={task_id}")
    renderer, vendor = renderer_metadata(cfg)
    patches = load_patches(base, wrapper.device)
    try:
        for state_index in map(int, base["state_indices"]):
            if state_index in complete:
                continue
            match = task_clean.loc[task_clean.state_index.astype(int) == state_index]
            require(len(match) == 1, f"Missing frozen clean seed: {suite} {task_id} {state_index}")
            seed = int(match.iloc[0].rollout_seed)
            clean_success = int(match.iloc[0].success)
            order = list(conditions)
            order_seed = (int(base["order_seed"]) + 10_000 * base["suites"].index(suite)
                          + 100 * task_id + state_index)
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
                print(f"JOINT500_XSUITE suite={suite} task={task_id} state={state_index} "
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
    parser.add_argument("--array-index", type=int, choices=range(40))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    require(args.preflight != (args.array_index is not None),
            "Choose exactly one of --preflight or --array-index")
    base = load_json(Path(args.config).resolve())
    if args.preflight:
        print(json.dumps(preflight(base), ensure_ascii=False, indent=2))
        return
    print(json.dumps(require_gpu()), flush=True)
    run_shard(args.array_index, base)


if __name__ == "__main__":
    main()
