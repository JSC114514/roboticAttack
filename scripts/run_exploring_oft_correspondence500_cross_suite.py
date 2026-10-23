#!/usr/bin/env python3
"""Five approved full-RGB patches; reuse the established paired rollout runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import atomic_json, load_json, require, require_gpu, sha256_file
from run_exploring_oft_scale500_cross_suite import frozen_clean, runtime_config, run_shard, validate

EXPERIMENT_ID = "OFT-DINO-CORRESPONDENCE500-XSUITE-V1"
CONDITIONS = ("exploring_oft_uada", "uada_tv", "correspondence", "correspondence_tv", "feature_tv")
VISUAL_REVIEW = "PASS_USER_20260927_CORRESPONDENCE500"
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}


def load_config(path: Path) -> dict:
    overlay = load_json(path)
    require(set(overlay) == {"experiment_id", "base_config", "base_config_sha256", "conditions",
                             "manual_visual_review", "output_root", "patch_sources"}, "Overlay keys drift")
    base_path = Path(overlay["base_config"])
    require(base_path.resolve() == ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json",
            "Wrong evaluation reference")
    require(sha256_file(base_path) == overlay["base_config_sha256"], "Evaluation reference changed")
    cfg = load_json(base_path)
    cfg.update(overlay)
    cfg["log_tag"] = "CORRESPONDENCE500_XSUITE"
    require(cfg["experiment_id"] == EXPERIMENT_ID and tuple(cfg["conditions"]) == CONDITIONS,
            "Five-arm experiment identity drift")
    require(cfg["manual_visual_review"] == VISUAL_REVIEW, "User visual approval missing")
    require(cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
            and cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10,
            "Patch/state grid drift")
    return cfg


def load_correspondence_patches(cfg: dict, device: torch.device):
    require(set(cfg["patch_sources"]) == set(CONDITIONS), "Patch source coverage drift")
    patches = {}
    for method in CONDITIONS:
        source = cfg["patch_sources"][method]
        path = Path(source["patch_path"])
        require(path.is_file() and sha256_file(path) == source["sha256"], f"Patch hash mismatch: {method}")
        status = load_json(source["training_status"])
        require(status["status"] == "COMPLETE" and status["method"] == method, f"Training incomplete: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if method == CONDITIONS[0]:
            require(status["experiment_id"] == payload["experiment_id"] == source["experiment_id"]
                    and payload["method"] == method and int(payload["outer_iterations"]) == 500,
                    "Baseline lineage mismatch")
        else:
            manifest = load_json(source["training_manifest"])
            require(status["signature"] == manifest["signature"] == payload["signature"] == source["signature"],
                    f"Training signature mismatch: {method}")
            require(manifest["experiment_id"] == source["experiment_id"] == "OFT-DINO-CORRESPONDENCE500-V1"
                    and manifest["method"] == payload["method"] == method
                    and status["outer"] == 500 and status["optimizer_updates"] == 25000
                    and status["final_patch_sha256"] == source["sha256"], f"Training identity mismatch: {method}")
            train = manifest["config"]
            require(train["suite"] == "libero_object" and train["outer_iterations"] == 500
                    and train["inner_loop"] == 50 and train["baseline_patch_sha256"] == cfg["patch_sources"][CONDITIONS[0]]["sha256"],
                    f"Training budget/reference drift: {method}")
            require(path.with_suffix(".png").is_file(), f"Missing visually reviewed PNG: {method}")
        patch = payload["patch"].float()
        require(tuple(patch.shape) == (3, 50, 50) and bool(torch.isfinite(patch).all())
                and bool(((patch >= 0) & (patch <= 1)).all()), f"Invalid RGB tensor: {method}")
        require(Path(cfg["output_root"]).resolve() != path.parent.parent.resolve(), "Evaluation overlaps training")
        patches[method] = patch.to(device), source["sha256"]
    require(len({digest for _, digest in patches.values()}) == 5, "Duplicate patch hashes")
    return patches


def validate_identity(rollouts: pd.DataFrame, traces: pd.DataFrame, cfg: dict):
    keys = ["suite", "task_id", "state_index", "condition"]
    require(not rollouts.empty and not rollouts.duplicated(keys).any(), "Duplicate/empty rollout rows")
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all() and traces.experiment_id.eq(EXPERIMENT_ID).all(),
            "Wrong experiment in existing outputs")
    require(rollouts.condition.isin(CONDITIONS).all() and traces.condition.isin(CONDITIONS).all(),
            "Unknown condition in outputs")
    require(rollouts.success.isin([0, 1]).all() and rollouts.failure.isin([0, 1]).all()
            and (rollouts.success + rollouts.failure).eq(1).all(), "Invalid success/failure labels")
    for method in CONDITIONS:
        rows = rollouts.loc[rollouts.condition == method]
        require(rows.patch_sha256.eq(cfg["patch_sources"][method]["sha256"]).all(),
                f"Rollout patch hash mismatch: {method}")
    require(traces.merge(rollouts[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").all(),
            "Trace without matching rollout")


def validate_result(rollouts, traces, suite, task_id, cfg):
    validate_identity(rollouts, traces, cfg)
    validate(rollouts, traces, suite, task_id, cfg)
    expected = frozen_clean(cfg, suite)
    linked = rollouts.merge(expected, on=["suite", "task_id", "state_index"], suffixes=("", "_clean"), validate="many_to_one")
    require(len(linked) == len(rollouts) and linked.rollout_seed.eq(linked.rollout_seed_clean).all()
            and linked.frozen_clean_success.eq(linked.success_clean).all(), "Frozen clean linkage mismatch")


def preflight(cfg):
    patches = load_correspondence_patches(cfg, torch.device("cpu"))
    denominators = {}
    for suite in cfg["suites"]:
        denominators[suite] = int(frozen_clean(cfg, suite).success.sum())
        runtime = runtime_config(cfg, suite)
        require(Path(runtime["paths"]["checkpoint"], "config.json").is_file(), f"Missing checkpoint: {suite}")
    require(denominators == DENOMINATORS, "Frozen clean denominator drift")
    return {"status": "CORRESPONDENCE500_XSUITE_PREFLIGHT_PASS", "conditions": list(CONDITIONS),
            "patch_sha256": {m: digest for m, (_, digest) in patches.items()},
            "clean_rollouts_reused": 400, "new_attack_rollouts": 2000,
            "eligible_attack_rows_expected": 1910, "clean_success_denominators": denominators,
            "manual_visual_review": cfg["manual_visual_review"], "model_loaded": False}


def run_checked_shard(array_index, config: Path):
    cfg = load_config(config)
    require(0 <= array_index < 40, "Array index outside frozen grid")
    suite, task_id = cfg["suites"][array_index // 10], array_index % 10
    output = Path(cfg["output_root"])
    manifest = {"config": cfg, "frozen_clean_sha256": {p: sha256_file(p) for p in
                (cfg["frozen_object_clean"], cfg["frozen_cross_suite_clean"])},
                "source_sha256": {str(p): sha256_file(p) for p in (Path(__file__),
                ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py",
                ROOT.parent / "oft_ramp_recovery_aware/scripts/run_continuous_cldev.py",
                ROOT.parent / "oft_ramp_recovery_aware/scripts/runtime.py")}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "EVAL_MANIFEST.json"
    if manifest_path.exists():
        require(load_json(manifest_path)["signature"] == signature, "Evaluation config/source changed during resume")
    else:
        atomic_json({"signature": signature, **manifest}, manifest_path)
    folder = output / suite
    for suffix in ("", ".partial"):
        rpath = folder / f"task_{task_id:02d}_rollouts{suffix}.parquet"
        tpath = folder / f"task_{task_id:02d}_traces{suffix}.parquet"
        require(rpath.exists() == tpath.exists(), "Incomplete rollout/trace file pair")
        if rpath.exists():
            validate_identity(pd.read_parquet(rpath), pd.read_parquet(tpath), cfg)
    # The shared runner reads a complete config, so pass the expanded object
    # through its existing load_json seam; no files or evaluator formulas changed.
    import run_exploring_oft_scale500_cross_suite as runner
    original_loader = runner.load_json
    runner.load_json = lambda path: cfg if Path(path).resolve() == config.resolve() else original_loader(path)
    try:
        run_shard(array_index, config, patch_loader=load_correspondence_patches)
    finally:
        runner.load_json = original_loader
    validate_result(pd.read_parquet(folder / f"task_{task_id:02d}_rollouts.parquet"),
                    pd.read_parquet(folder / f"task_{task_id:02d}_traces.parquet"), suite, task_id, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--array-index", type=int, choices=range(40))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    require(args.preflight != (args.array_index is not None), "Choose preflight or array index")
    if args.preflight:
        print(json.dumps(preflight(load_config(args.config)), indent=2))
        return
    require(bool(os.environ.get("SLURM_JOB_ID")), "Evaluation requires a user-submitted Slurm job")
    print(json.dumps(require_gpu()), flush=True)
    run_checked_shard(args.array_index, args.config.resolve())


if __name__ == "__main__":
    main()
