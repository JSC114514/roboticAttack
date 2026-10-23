#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import ROOT, atomic_parquet, load_json, require, require_gpu, sha256_file
from eval_open_loop import phase_indices
from run_cid_cf_stage_c import action_l2, causal_api, clean_trajectory, runtime_config
from runtime import make_env, paste_numpy, renderer_metadata
from train_ramp import load_patch_payload


CONFIG = ROOT / "configs" / "H1_H4_FOLLOWUP.json"
METRICS = (
    "dino_rms_displacement", "siglip_rms_displacement", "full_action_l2",
    "dino_only_action_l2", "siglip_only_action_l2", "dino_restore_reduction",
    "siglip_restore_reduction", "dino_restore_fraction", "dino_only_fraction_of_full",
    "restore_both_action_l2",
)


def frozen_states(cfg: dict[str, Any], task_id: int) -> list[dict[str, int]]:
    raw = pd.read_parquet(cfg["paths"]["stage_c_raw"])
    rows = (raw.loc[(raw.suite == cfg["mechanism_suite"]) &
                    (raw.task_id.astype(int) == task_id), ["state_index", "rollout_seed"]]
            .drop_duplicates().sort_values("state_index"))
    require(len(rows) == int(cfg["mechanism_states_per_task"]),
            f"Frozen Stage-C Object states drift: task {task_id}")
    return [{"state_index": int(row.state_index), "rollout_seed": int(row.rollout_seed)}
            for row in rows.itertuples(index=False)]


def load_patches(cfg: dict[str, Any], device: torch.device):
    result = {}
    for method in cfg["mechanism_methods"]:
        path = Path(cfg["patches"][method])
        require(path.is_file(), f"Missing UADA mechanism patch: {path}")
        patch = load_patch_payload(path).to(device).float().clamp(0, 1)
        require(tuple(patch.shape) == tuple(cfg["patch_shape"]), f"Patch shape drift: {method}")
        result[method] = (patch, sha256_file(path))
    return result


def validate(frame: pd.DataFrame, task_id: int, cfg: dict[str, Any]) -> None:
    expected = int(cfg["mechanism_states_per_task"]) * int(cfg["phase_count"]) * len(cfg["mechanism_methods"])
    require(len(frame) == expected, f"Expected {expected} UADA mechanism rows, got {len(frame)}")
    require(set(frame.task_id.astype(int)) == {task_id}, "UADA mechanism task drift")
    require(set(frame.method) == set(cfg["mechanism_methods"]), "UADA mechanism method drift")
    require(not frame.duplicated(["task_id", "state_index", "phase_order", "method"]).any(),
            "Duplicate UADA mechanism row")
    require(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all(),
            "Non-finite UADA mechanism metric")
    require(frame.renderer_vendor.str.lower().str.contains("nvidia").all(),
            "Non-NVIDIA UADA mechanism renderer")


def run_task(task_id: int) -> None:
    base = load_json(CONFIG)
    require(base["mechanism_suite"] == "libero_object" and base["long_required"] is False,
            "UADA mechanism boundary drift")
    output = ROOT / "outputs" / "h1_h4" / "mechanism" / f"task_{task_id:02d}.parquet"
    if output.is_file():
        validate(pd.read_parquet(output), task_id, base)
        print(f"UADA_MECHANISM_ALREADY_COMPLETE task={task_id}")
        return
    cfg = runtime_config(base, base["mechanism_suite"])
    load_runtime, prepare, run, visual_spec = causal_api(base)
    runtime = load_runtime(cfg, cfg["suite"])
    env, task, init_states = make_env(cfg, task_id)
    renderer, vendor = renderer_metadata(cfg)
    patches = load_patches(base, torch.device("cuda"))
    scale = np.asarray(load_json(base["paths"]["action_scale"])["scale"], dtype=np.float64)
    if scale.shape == (7,):
        scale = scale.reshape(1, 7)
    specs = {site: visual_spec(site, "wrist") for site in ("dino", "siglip")}
    rows = []
    try:
        for record in frozen_states(base, task_id):
            state_index, seed = record["state_index"], record["rollout_seed"]
            success, trajectory = clean_trajectory(
                env, runtime, prepare, run, task, init_states[state_index], seed, cfg)
            require(success, f"Frozen Object clean state did not reproduce: task{task_id}/state{state_index}")
            chosen = phase_indices([row["gripper"] for row in trajectory])
            require(len(chosen) == int(base["phase_count"]), "UADA mechanism phase drift")
            for phase_order, (decision_index, phase_proxy) in enumerate(chosen):
                obs = trajectory[decision_index]["observation"]
                clean_prepared = prepare(runtime, obs, task.language)
                clean_action, clean_values = run(runtime, clean_prepared, list(specs.values()))
                clean_features = dict(zip(specs, clean_values))
                for method in base["mechanism_methods"]:
                    patch, patch_hash = patches[method]
                    attacked_obs = paste_numpy(obs, patch, tuple(base["patch_xy"]))
                    attacked_prepared = prepare(runtime, attacked_obs, task.language)
                    attacked_action, attacked_values = run(runtime, attacked_prepared, list(specs.values()))
                    attacked_features = dict(zip(specs, attacked_values))
                    dino_restored_action, _ = run(
                        runtime, attacked_prepared, [], replace_specs=[specs["dino"]],
                        replace_values=[clean_features["dino"]])
                    siglip_restored_action, _ = run(
                        runtime, attacked_prepared, [], replace_specs=[specs["siglip"]],
                        replace_values=[clean_features["siglip"]])
                    both_restored_action, _ = run(
                        runtime, attacked_prepared, [],
                        replace_specs=[specs["dino"], specs["siglip"]],
                        replace_values=[clean_features["dino"], clean_features["siglip"]])
                    full = action_l2(clean_action, attacked_action, scale)
                    dino_only = action_l2(clean_action, siglip_restored_action, scale)
                    siglip_only = action_l2(clean_action, dino_restored_action, scale)
                    both = action_l2(clean_action, both_restored_action, scale)
                    dino_delta = attacked_features["dino"].float() - clean_features["dino"].float()
                    siglip_delta = attacked_features["siglip"].float() - clean_features["siglip"].float()
                    rows.append({
                        "suite": cfg["suite"], "task_id": task_id, "task_name": task.language,
                        "state_index": state_index, "rollout_seed": seed,
                        "decision_index": int(decision_index), "phase_order": int(phase_order),
                        "phase_proxy": phase_proxy, "method": method, "patch_sha256": patch_hash,
                        "dino_rms_displacement": float(torch.sqrt(torch.mean(dino_delta.square()))),
                        "siglip_rms_displacement": float(torch.sqrt(torch.mean(siglip_delta.square()))),
                        "full_action_l2": full, "dino_only_action_l2": dino_only,
                        "siglip_only_action_l2": siglip_only,
                        "dino_restore_reduction": full - siglip_only,
                        "siglip_restore_reduction": full - dino_only,
                        "dino_restore_fraction": (full - siglip_only) / max(full, 1e-8),
                        "dino_only_fraction_of_full": dino_only / max(full, 1e-8),
                        "restore_both_action_l2": both,
                        "renderer": renderer, "renderer_vendor": vendor,
                    })
            print(f"UADA_MECHANISM task={task_id} state={state_index} rows={len(rows)}", flush=True)
        frame = pd.DataFrame(rows)
        validate(frame, task_id, base)
        atomic_parquet(frame, output)
    finally:
        env.close()
        runtime.release()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    print(json.dumps(require_gpu()), flush=True)
    run_task(args.task_id)


if __name__ == "__main__":
    main()
