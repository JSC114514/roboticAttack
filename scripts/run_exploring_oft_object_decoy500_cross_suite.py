#!/usr/bin/env python3
"""Paired four-suite rollout of UADA-500 and the two Object-Decoy-500 patches."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware" / "scripts"))

from common import atomic_parquet, require, require_gpu, sha256_file  # noqa: E402
from run_continuous_cldev import run_episode  # noqa: E402
from runtime import load_wrapper, make_env, renderer_metadata  # noqa: E402
from run_exploring_oft_scale500_cross_suite import (  # noqa: E402
    frozen_clean, runtime_config, validate,
)

BASELINE = "exploring_oft_uada_o0500"
R03 = "exploring_oft_dino_object_decoy_r03"
R10 = "exploring_oft_dino_object_decoy_r10"
EXPERIMENT_ID = "EXPLORING-OFT-OBJECT-DECOY500-XSUITE-V1"


def load_config(path: Path) -> dict:
    overlay = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"experiment_id", "base_config", "training_suite", "conditions",
               "patch_sources", "output_root", "order_seed", "bootstrap_seed"}
    require(set(overlay) == allowed, "Object-Decoy evaluation config keys drift")
    base_path = Path(overlay["base_config"])
    require(base_path.resolve() == ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json",
            "Frozen Scale-500 evaluation source drift")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base.update({key: value for key, value in overlay.items() if key != "base_config"})
    base["base_config"] = str(base_path)
    return base


def validate_sources(cfg: dict) -> dict[str, Path]:
    require(cfg["experiment_id"] == EXPERIMENT_ID, "Evaluation ID drift")
    require(cfg["training_suite"] == "libero_object", "Training suite drift")
    require(cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"],
            "Suite order drift")
    require(cfg["task_count"] == 10 and cfg["state_indices"] == list(range(40, 50)),
            "Frozen task/state grid drift")
    require(cfg["conditions"] == [BASELINE, R03, R10]
            and set(cfg["patch_sources"]) == set(cfg["conditions"]),
            "Condition/source drift")
    require(cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50],
            "Wrist placement/shape drift")
    require(Path(cfg["output_root"]).resolve() != Path(cfg["patch_sources"][R03]["path"]).parent.parent.resolve(),
            "Evaluation output overlaps training root")
    require(int(cfg["num_open_loop_steps"]) == 8 and cfg["dtype"] == "bfloat16",
            "OFT rollout contract drift")
    paths: dict[str, Path] = {}
    for method in cfg["conditions"]:
        source = cfg["patch_sources"][method]
        path = Path(source["path"])
        require(path.is_file() and sha256_file(path) == source["sha256"],
                f"Patch missing or SHA256 drift: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("experiment_id") == source["experiment_id"]
                and payload.get("method") == source["method"]
                and int(payload.get("outer_iterations", -1)) == 500,
                f"Patch lineage drift: {method}")
        patch = payload["patch"]
        require(tuple(patch.shape) == (3, 50, 50) and torch.isfinite(patch).all()
                and bool(((patch >= 0) & (patch <= 1)).all()),
                f"Invalid RGB patch: {method}")
        status_path = path.parent / "status.json"
        require(status_path.is_file(), f"Missing training status: {method}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        require(status.get("status") == "COMPLETE" and status.get("method") == source["method"]
                and status.get("experiment_id") == source["experiment_id"],
                f"Incomplete training: {method}")
        if method != BASELINE:
            require(status.get("baseline_patch_sha256") == cfg["patch_sources"][BASELINE]["sha256"],
                    f"Training baseline drift: {method}")
            require((path.parent / "outer_0500.png").is_file(), f"Missing final image: {method}")
            for outer in (100, 200, 500):
                gate = json.loads((path.parent / f"structure_gate_{outer:04d}.json").read_text())
                require(gate.get("outer_iteration") == outer and gate.get("passed") is True,
                        f"Structure gate failed: {method} outer={outer}")
        paths[method] = path
    require(len({cfg["patch_sources"][method]["sha256"] for method in paths}) == len(paths),
            "Duplicate patch digest")
    return paths


def preflight(cfg: dict) -> dict:
    validate_sources(cfg)
    denominators = {}
    for suite in cfg["suites"]:
        clean = frozen_clean(cfg, suite)
        denominators[suite] = int(clean.success.sum())
        runtime = runtime_config(cfg, suite)
        require(Path(runtime["paths"]["checkpoint"], "config.json").is_file(),
                f"Missing target checkpoint: {suite}")
        require(runtime["unnorm_key"] == cfg["unnorm_keys"][suite],
                f"Normalization key drift: {suite}")
    require(denominators == {"libero_object": 99, "libero_spatial": 90,
                             "libero_goal": 98, "libero_10": 95},
            "Frozen clean denominator drift")
    return {"status": "OBJECT_DECOY500_XSUITE_PREFLIGHT_PASS",
            "conditions": cfg["conditions"], "clean_rollouts_reused": 400,
            "new_attack_rollouts": 1200,
            "eligible_attack_rows_expected": len(cfg["conditions"]) * sum(denominators.values()),
            "clean_success_denominators": denominators}


def validate_result(rollouts: pd.DataFrame, traces: pd.DataFrame,
                    suite: str, task_id: int, cfg: dict) -> None:
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all(), "Shard experiment ID drift")
    for method in cfg["conditions"]:
        rows = rollouts.loc[rollouts.condition == method]
        require(rows.patch_sha256.eq(cfg["patch_sources"][method]["sha256"]).all(),
                f"Shard patch SHA256 drift: {method}")


def run_shard(array_index: int, cfg: dict) -> None:
    require(0 <= array_index < 40, "Array index outside frozen task grid")
    validate_sources(cfg)
    task_count = int(cfg["task_count"])
    suite = cfg["suites"][array_index // task_count]
    task_id = array_index % task_count
    runtime = runtime_config(cfg, suite)
    clean = frozen_clean(cfg, suite)
    task_clean = clean.loc[clean.task_id.astype(int) == task_id]
    output = Path(cfg["output_root"]) / suite
    output.mkdir(parents=True, exist_ok=True)
    final_rollouts = output / f"task_{task_id:02d}_rollouts.parquet"
    final_traces = output / f"task_{task_id:02d}_traces.parquet"
    if final_rollouts.is_file() and final_traces.is_file():
        validate_result(pd.read_parquet(final_rollouts), pd.read_parquet(final_traces), suite, task_id, cfg)
        print(f"OBJECT_DECOY500_XSUITE_ALREADY_COMPLETE suite={suite} task={task_id}", flush=True)
        return
    partial_rollouts = output / f"task_{task_id:02d}_rollouts.partial.parquet"
    partial_traces = output / f"task_{task_id:02d}_traces.partial.parquet"
    rollout_rows = pd.read_parquet(partial_rollouts).to_dict("records") if partial_rollouts.is_file() else []
    trace_rows = pd.read_parquet(partial_traces).to_dict("records") if partial_traces.is_file() else []
    if rollout_rows:
        partial = pd.DataFrame(rollout_rows)
        for method in cfg["conditions"]:
            rows = partial.loc[partial.condition == method]
            require(rows.experiment_id.eq(EXPERIMENT_ID).all()
                    and rows.patch_sha256.eq(cfg["patch_sources"][method]["sha256"]).all(),
                    f"Partial rollout identity drift: {method}")
    complete = ({int(state) for state, group in pd.DataFrame(rollout_rows).groupby("state_index")
                 if len(group) == len(cfg["conditions"]) and set(group.condition) == set(cfg["conditions"])}
                if rollout_rows else set())
    rollout_rows = [row for row in rollout_rows if int(row["state_index"]) in complete]
    trace_rows = [row for row in trace_rows if int(row["state_index"]) in complete]

    wrapper = load_wrapper(runtime)
    env, task, init_states = make_env(runtime, task_id)
    require(max(cfg["state_indices"]) < len(init_states),
            f"Frozen state unavailable: {suite} task={task_id}")
    renderer, vendor = renderer_metadata(runtime)
    patches = {method: (torch.load(path, map_location="cpu", weights_only=False)["patch"].float()
                        .to(wrapper.device), cfg["patch_sources"][method]["sha256"])
               for method, path in validate_sources(cfg).items()}
    try:
        for state_index in map(int, cfg["state_indices"]):
            if state_index in complete:
                continue
            match = task_clean.loc[task_clean.state_index.astype(int) == state_index]
            require(len(match) == 1, f"Missing frozen clean seed: {suite} {task_id} {state_index}")
            seed = int(match.iloc[0].rollout_seed)
            clean_success = int(match.iloc[0].success)
            order = list(cfg["conditions"])
            order_seed = (int(cfg["order_seed"]) + 10_000 * cfg["suites"].index(suite)
                          + 100 * task_id + state_index)
            random.Random(order_seed).shuffle(order)
            for order_index, method in enumerate(order, start=1):
                patch, digest = patches[method]
                rollout, trace = run_episode(env, wrapper, task, init_states[state_index],
                                             seed, method, patch, digest, runtime,
                                             renderer, vendor, order_index)
                metadata = {"experiment_id": EXPERIMENT_ID, "training_suite": cfg["training_suite"],
                            "suite": suite, "task_id": task_id, "task_name": task.language,
                            "state_index": state_index, "frozen_clean_success": clean_success,
                            "checkpoint": runtime["paths"]["checkpoint"],
                            "unnorm_key": runtime["unnorm_key"]}
                rollout_rows.append({**metadata, **rollout})
                trace_rows.extend({**metadata, **row} for row in trace)
                print(f"OBJECT_DECOY500_XSUITE suite={suite} task={task_id} state={state_index} "
                      f"method={method} success={rollout['success']}", flush=True)
            atomic_parquet(pd.DataFrame(rollout_rows), partial_rollouts)
            atomic_parquet(pd.DataFrame(trace_rows), partial_traces)
        rollouts, traces = pd.DataFrame(rollout_rows), pd.DataFrame(trace_rows)
        validate_result(rollouts, traces, suite, task_id, cfg)
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
    cfg = load_config(Path(args.config).resolve())
    if args.preflight:
        print(json.dumps(preflight(cfg), ensure_ascii=False, indent=2))
        return
    print(json.dumps(require_gpu()), flush=True)
    run_shard(args.array_index, cfg)


if __name__ == "__main__":
    main()
