#!/usr/bin/env python3
"""Paired four-suite evaluation with old UADA, two site-matched arms and gray occlusion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import atomic_parquet, require, require_gpu, sha256_file  # noqa: E402
from run_continuous_cldev import run_episode  # noqa: E402
from runtime import load_wrapper, make_env, renderer_metadata  # noqa: E402
from run_exploring_oft_scale500_cross_suite import (  # noqa: E402
    frozen_clean, runtime_config, validate,
)

EXPERIMENT_ID = "EXPLORING-OFT-DINO-SAMPLING500-XSUITE-V1"
BASE = "exploring_oft_uada_o0500"
CONTROL = "exploring_oft_site_uada"
GUIDED = "exploring_oft_site_dino_sample"
GRAY = "gray_same_site"
CONDITIONS = [BASE, CONTROL, GUIDED, GRAY]


def load_config(path: Path) -> dict:
    overlay = json.loads(path.read_text())
    allowed = {"experiment_id", "base_config", "training_suite", "conditions",
               "manual_visual_review", "patch_sources", "patch_xy", "output_root",
               "order_seed", "bootstrap_seed"}
    require(set(overlay) == allowed, "DINO sampling evaluation config keys drift")
    base_path = Path(overlay["base_config"])
    require(base_path.resolve() == ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json",
            "Frozen Scale500 source drift")
    base = json.loads(base_path.read_text())
    base.update({k: v for k, v in overlay.items() if k != "base_config"})
    return base


def gray_patch() -> tuple[torch.Tensor, str]:
    patch = torch.full((3, 50, 50), 0.5, dtype=torch.float32)
    return patch, hashlib.sha256(patch.numpy().tobytes()).hexdigest()


def validate_sources(cfg: dict) -> dict[str, tuple[Path | None, str]]:
    require(cfg["experiment_id"] == EXPERIMENT_ID and cfg["conditions"] == CONDITIONS,
            "Evaluation identity/conditions drift")
    require(cfg["training_suite"] == "libero_object" and cfg["patch_xy"] == [91, 140]
            and cfg["patch_shape"] == [3, 50, 50], "Training suite/site/shape drift")
    require(cfg["manual_visual_review"] == "PASS_USER_DINO_SAMPLING500",
            "New patches require explicit user visual approval")
    require(set(cfg["patch_sources"]) == {BASE, CONTROL, GUIDED}, "Patch source coverage drift")
    paths = {}
    for method, source in cfg["patch_sources"].items():
        path = Path(source["path"])
        require(path.is_file(), f"Missing patch: {path}")
        digest = sha256_file(path)
        if method == BASE:
            require(digest == source["sha256"], "Frozen UADA500 hash drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("experiment_id") == source["experiment_id"]
                and payload.get("method") == source["method"]
                and int(payload.get("outer_iterations", -1)) == 500,
                f"Patch lineage drift: {method}")
        patch = payload["patch"]
        require(tuple(patch.shape) == (3, 50, 50) and torch.isfinite(patch).all()
                and bool(((patch >= 0) & (patch <= 1)).all()), f"Invalid RGB patch: {method}")
        if method != BASE:
            require(payload.get("site_xy") == [91, 140], f"Training site drift: {method}")
            status = json.loads((path.parent / "status.json").read_text())
            require(status.get("status") == "COMPLETE" and status.get("method") == method
                    and status.get("experiment_id") == source["experiment_id"],
                    f"Incomplete training: {method}")
            require(path.with_suffix(".png").is_file(), f"Missing user review image: {method}")
            for outer in (100, 200, 500):
                gate = json.loads((path.parent / f"structure_gate_{outer:04d}.json").read_text())
                require(gate.get("passed") is True, f"Structure gate failed: {method} {outer}")
        paths[method] = (path, digest)
    paths[GRAY] = (None, gray_patch()[1])
    require(len({digest for _, digest in paths.values()}) == len(CONDITIONS),
            "Duplicate patch digest")
    return paths


def preflight(cfg: dict) -> dict:
    sources = validate_sources(cfg)
    require(cfg["suites"] == ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
            and cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10,
            "Frozen four-suite task/state grid drift")
    denominators = {}
    for suite in cfg["suites"]:
        denominators[suite] = int(frozen_clean(cfg, suite).success.sum())
        runtime = runtime_config(cfg, suite)
        require(Path(runtime["paths"]["checkpoint"], "config.json").is_file(),
                f"Missing OFT suite checkpoint: {suite}")
    require(denominators == {"libero_object": 99, "libero_spatial": 90,
                             "libero_goal": 98, "libero_10": 95},
            "Frozen clean denominators drift")
    return {"status": "DINO_SAMPLING500_XSUITE_PREFLIGHT_PASS", "conditions": CONDITIONS,
            "patch_sha256": {name: digest for name, (_, digest) in sources.items()},
            "new_attack_rollouts": 1600, "eligible_attack_rows_expected": 4 * sum(denominators.values()),
            "clean_success_denominators": denominators}


def validate_result(rollouts: pd.DataFrame, traces: pd.DataFrame,
                    suite: str, task_id: int, cfg: dict,
                    sources: dict[str, tuple[Path | None, str]]) -> None:
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all(), "Shard experiment ID drift")
    for method, (_, digest) in sources.items():
        require(rollouts.loc[rollouts.condition == method, "patch_sha256"].eq(digest).all(),
                f"Shard patch digest drift: {method}")


def run_shard(array_index: int, cfg: dict) -> None:
    require(0 <= array_index < 40, "Array index outside frozen grid")
    sources = validate_sources(cfg)
    suite = cfg["suites"][array_index // 10]
    task_id = array_index % 10
    runtime = runtime_config(cfg, suite)
    clean = frozen_clean(cfg, suite)
    task_clean = clean.loc[clean.task_id.astype(int) == task_id]
    output = Path(cfg["output_root"]) / suite
    output.mkdir(parents=True, exist_ok=True)
    final_rollouts = output / f"task_{task_id:02d}_rollouts.parquet"
    final_traces = output / f"task_{task_id:02d}_traces.parquet"
    if final_rollouts.is_file() and final_traces.is_file():
        validate_result(pd.read_parquet(final_rollouts), pd.read_parquet(final_traces),
                        suite, task_id, cfg, sources)
        print(f"DINO_SAMPLING500_XSUITE_ALREADY_COMPLETE suite={suite} task={task_id}", flush=True)
        return
    partial_rollouts = output / f"task_{task_id:02d}_rollouts.partial.parquet"
    partial_traces = output / f"task_{task_id:02d}_traces.partial.parquet"
    rollout_rows = pd.read_parquet(partial_rollouts).to_dict("records") if partial_rollouts.is_file() else []
    trace_rows = pd.read_parquet(partial_traces).to_dict("records") if partial_traces.is_file() else []
    if rollout_rows:
        partial = pd.DataFrame(rollout_rows)
        for method, (_, digest) in sources.items():
            rows = partial.loc[partial.condition == method]
            require(rows.experiment_id.eq(EXPERIMENT_ID).all()
                    and rows.patch_sha256.eq(digest).all(),
                    f"Partial rollout identity drift: {method}")
    complete = ({int(state) for state, group in pd.DataFrame(rollout_rows).groupby("state_index")
                 if len(group) == len(CONDITIONS) and set(group.condition) == set(CONDITIONS)}
                if rollout_rows else set())
    rollout_rows = [row for row in rollout_rows if int(row["state_index"]) in complete]
    trace_rows = [row for row in trace_rows if int(row["state_index"]) in complete]

    wrapper = load_wrapper(runtime)
    env, task, init_states = make_env(runtime, task_id)
    require(max(cfg["state_indices"]) < len(init_states), f"State 49 unavailable: {suite} {task_id}")
    renderer, vendor = renderer_metadata(runtime)
    patches = {method: ((torch.load(path, map_location="cpu", weights_only=False)["patch"].float()
                         if path is not None else gray_patch()[0]).to(wrapper.device), digest)
               for method, (path, digest) in sources.items()}
    try:
        for state_index in map(int, cfg["state_indices"]):
            if state_index in complete:
                continue
            match = task_clean.loc[task_clean.state_index.astype(int) == state_index]
            require(len(match) == 1, f"Missing frozen clean seed: {suite} {task_id} {state_index}")
            seed = int(match.iloc[0].rollout_seed)
            clean_success = int(match.iloc[0].success)
            order = list(CONDITIONS)
            random.Random(int(cfg["order_seed"]) + 10_000 * cfg["suites"].index(suite)
                          + 100 * task_id + state_index).shuffle(order)
            for order_index, method in enumerate(order, start=1):
                patch, digest = patches[method]
                rollout, trace = run_episode(env, wrapper, task, init_states[state_index], seed,
                                             method, patch, digest, runtime, renderer, vendor, order_index)
                metadata = {"experiment_id": EXPERIMENT_ID, "training_suite": cfg["training_suite"],
                            "suite": suite, "task_id": task_id, "task_name": task.language,
                            "state_index": state_index, "frozen_clean_success": clean_success,
                            "checkpoint": runtime["paths"]["checkpoint"],
                            "unnorm_key": runtime["unnorm_key"]}
                rollout_rows.append({**metadata, **rollout})
                trace_rows.extend({**metadata, **row} for row in trace)
                print(f"DINO_SAMPLING500_XSUITE suite={suite} task={task_id} state={state_index} "
                      f"method={method} success={rollout['success']}", flush=True)
            atomic_parquet(pd.DataFrame(rollout_rows), partial_rollouts)
            atomic_parquet(pd.DataFrame(trace_rows), partial_traces)
        rollouts, traces = pd.DataFrame(rollout_rows), pd.DataFrame(trace_rows)
        validate_result(rollouts, traces, suite, task_id, cfg, sources)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--array-index", type=int, choices=range(40))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    require(args.preflight != (args.array_index is not None),
            "Choose exactly one of --preflight or --array-index")
    cfg = load_config(args.config.resolve())
    if args.preflight:
        print(json.dumps(preflight(cfg), ensure_ascii=False, indent=2))
        return
    print(json.dumps(require_gpu()), flush=True)
    run_shard(args.array_index, cfg)


if __name__ == "__main__":
    main()
