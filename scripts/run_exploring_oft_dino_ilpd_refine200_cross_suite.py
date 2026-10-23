#!/usr/bin/env python3
"""Evaluate the four frozen ILPD-V2 Wrist RGB patches with the established rollout runner."""
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
from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, runtime_config, validate

EXPERIMENT_ID = "OFT-DINO-ILPD-REFINE200-XSUITE-V2"
TRAIN_EXPERIMENT = "OFT-DINO-ILPD-REFINE200-V2"
CONDITIONS = ("uada_bounded", "dino_gradscale_bounded", "dino_ilpd_bounded", "siglip_ilpd_bounded")
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}


def load_config(path: Path) -> dict:
    overlay = load_json(path)
    require(set(overlay) == {"experiment_id", "base_config", "base_config_sha256", "conditions",
                             "manual_test_request", "training_root", "output_root", "patch_sha256",
                             "evidence_sha256"}, "Evaluation overlay drift")
    base_path = Path(overlay["base_config"])
    require(base_path.resolve() == ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json"
            and sha256_file(base_path) == overlay["base_config_sha256"], "Frozen evaluation reference drift")
    cfg = load_json(base_path)
    cfg.update(overlay)
    cfg["log_tag"] = "DINO_ILPD_REFINE200_XSUITE"
    require(cfg["experiment_id"] == EXPERIMENT_ID and tuple(cfg["conditions"]) == CONDITIONS,
            "Evaluation identity drift")
    require(cfg["manual_test_request"] == "USER_REQUEST_20260928_TEST_FOUR_FINAL_PATCHES",
            "Four-patch test request missing")
    require(cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
            and cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10,
            "Frozen Wrist placement or paired state grid drift")
    require(Path(cfg["output_root"]).resolve().is_relative_to(ROOT / "research_runs/wrist_dino_ilpd_v1/evaluation")
            and not Path(cfg["output_root"]).resolve().is_relative_to(Path(cfg["training_root"]).resolve()),
            "Evaluation output must be separate from training")
    return cfg


def load_ilpd_patches(cfg: dict, device: torch.device) -> dict:
    require(set(cfg["patch_sha256"]) == set(CONDITIONS)
            and set(cfg["evidence_sha256"]) == set(CONDITIONS), "Patch source coverage drift")
    original = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(original.stat().st_mode & 0o222 == 0
            and sha256_file(original) == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
            "Original Exploring source changed or writable")
    patches = {}
    reference_config = None
    reference_initial = None
    for method in CONDITIONS:
        folder = Path(cfg["training_root"]) / method
        paths = [folder / "status.json", folder / "RUN_MANIFEST.json", folder / "RUNTIME_ALIGNMENT.json"]
        require([sha256_file(path) for path in paths] == cfg["evidence_sha256"][method],
                f"Training evidence hash mismatch: {method}")
        status, manifest, alignment = (load_json(path) for path in paths)
        patch_path = folder / "outer_0200.pt"
        require(sha256_file(patch_path) == cfg["patch_sha256"][method]
                == status["final_patch_sha256"], f"Final patch hash mismatch: {method}")
        payload = torch.load(patch_path, map_location="cpu", weights_only=False)
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == 200 and status["total_outer"] == 700
                and status["optimizer_updates"] == 10000 and status["total_optimizer_updates"] == 35000,
                f"Incomplete training or budget drift: {method}")
        require(manifest["experiment_id"] == TRAIN_EXPERIMENT and manifest["method"] == method
                and status["signature"] == manifest["signature"] == payload["signature"]
                and payload["method"] == method and payload["total_outer"] == 700
                and payload["total_optimizer_updates"] == 35000,
                f"Training identity drift: {method}")
        train = manifest["config"]
        require(train["suite"] == "libero_object" and train["outer_iterations"] == 200
                and train["base_outer_iterations"] == 500 and train["inner_loop"] == 50
                and train["baseline_patch_sha256"] == "c7ebb69204290fc193c3f694e25ba02c656de9e2df0443a74470d8ad587db136"
                and train["maskidx"] == [0] and train["patch_shape"] == [3, 50, 50]
                and train["num_open_loop_steps"] == 8 and train["feature_alpha"] == 0.5
                and manifest["parameter_count"] == 7500 and manifest["optimizer_restarted"] is True,
                f"Training protocol drift: {method}")
        if reference_config is None:
            reference_config = train
        require(train == reference_config, "Matched training configuration drift")
        require(alignment["status"] == "PASS" and len(alignment["records"]) == 2
                and {r["condition"] for r in alignment["records"]} == {"clean", "reference_eot"}
                and all(r["pixel_exact"] and r["feature_close"] for r in alignment["records"]),
                f"Full-model runtime alignment missing: {method}")
        initial = torch.load(folder / "outer_0000.pt", map_location="cpu", weights_only=False)["patch"].float()
        if reference_initial is None:
            baseline_path = Path(train["baseline_patch"])
            require(sha256_file(baseline_path) == train["baseline_patch_sha256"], "UADA500 anchor changed")
            reference_initial = torch.load(baseline_path, map_location="cpu", weights_only=False)["patch"].float()
        require(torch.equal(initial, reference_initial), f"Warm-start mismatch: {method}")
        require(patch_path.with_suffix(".png").is_file(), f"Missing final PNG: {method}")
        patch = payload["patch"].float()
        require(tuple(patch.shape) == (3, 50, 50) and bool(torch.isfinite(patch).all())
                and bool(((patch >= 0) & (patch <= 1)).all()), f"Invalid RGB tensor: {method}")
        patches[method] = (patch.to(device), cfg["patch_sha256"][method])
    require(len(set(cfg["patch_sha256"].values())) == 4, "Duplicate patches")
    return patches


def validate_identity(rollouts: pd.DataFrame, traces: pd.DataFrame, cfg: dict) -> None:
    keys = ["suite", "task_id", "state_index", "condition"]
    require(not rollouts.empty and not rollouts.duplicated(keys).any(), "Duplicate or empty rollouts")
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all() and traces.experiment_id.eq(EXPERIMENT_ID).all(),
            "Wrong experiment in evaluation outputs")
    require(rollouts.condition.isin(CONDITIONS).all() and traces.condition.isin(CONDITIONS).all(),
            "Unknown evaluation condition")
    require(rollouts.success.isin([0, 1]).all() and rollouts.failure.isin([0, 1]).all()
            and (rollouts.success + rollouts.failure).eq(1).all(), "Invalid success/failure labels")
    for method in CONDITIONS:
        rows = rollouts.loc[rollouts.condition == method]
        require(rows.patch_sha256.eq(cfg["patch_sha256"][method]).all(), f"Rollout patch mismatch: {method}")
    require(traces.merge(rollouts[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").all(),
            "Trace without matching rollout")
    require(rollouts[keys].merge(traces[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            ["_merge"].eq("both").all(), "Rollout without matching trace")


def validate_result(rollouts: pd.DataFrame, traces: pd.DataFrame, suite: str, task_id: int, cfg: dict) -> None:
    validate_identity(rollouts, traces, cfg)
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.groupby("state_index").condition_order.nunique().eq(len(CONDITIONS)).all(),
            "Condition order not a permutation")
    expected = frozen_clean(cfg, suite)
    linked = rollouts.merge(expected, on=["suite", "task_id", "state_index"],
                            suffixes=("", "_clean"), validate="many_to_one")
    require(len(linked) == len(rollouts) and linked.rollout_seed.eq(linked.rollout_seed_clean).all()
            and linked.frozen_clean_success.eq(linked.success_clean).all(), "Frozen clean linkage mismatch")


def preflight(cfg: dict) -> dict:
    patches = load_ilpd_patches(cfg, torch.device("cpu"))
    denominators = {}
    for suite in cfg["suites"]:
        denominators[suite] = int(frozen_clean(cfg, suite).success.sum())
        runtime = runtime_config(cfg, suite)
        require((Path(runtime["paths"]["checkpoint"]) / "config.json").is_file(),
                f"Missing model checkpoint: {suite}")
    require(denominators == DENOMINATORS, "Frozen clean denominator drift")
    return {"status": "PASS", "conditions": list(CONDITIONS), "patch_sha256": cfg["patch_sha256"],
            "frozen_clean_success_denominators": denominators, "reused_clean_episodes": 400,
            "new_attack_episodes": 1600, "eligible_attack_rows_expected": 1528,
            "model_loaded": False, "gpu_jobs_submitted": False}


def run_checked_shard(index: int, config: Path) -> None:
    cfg = load_config(config)
    require(0 <= index < 40, "Array index outside frozen grid")
    preflight(cfg)
    suite, task_id = cfg["suites"][index // 10], index % 10
    output = Path(cfg["output_root"])
    manifest = {"config": cfg,
                "frozen_clean_sha256": {str(path): sha256_file(path) for path in
                                        (Path(cfg["frozen_object_clean"]), Path(cfg["frozen_cross_suite_clean"]))},
                "source_sha256": {str(path): sha256_file(path) for path in
                                  (Path(__file__), ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py",
                                   ROOT.parent / "oft_ramp_recovery_aware/scripts/run_continuous_cldev.py",
                                   ROOT.parent / "oft_ramp_recovery_aware/scripts/runtime.py")}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "EVAL_MANIFEST.json"
    if manifest_path.exists():
        require(load_json(manifest_path)["signature"] == signature, "Evaluation changed during resume")
    else:
        atomic_json({"signature": signature, **manifest}, manifest_path)
    folder = output / suite
    for suffix in ("", ".partial"):
        rpath = folder / f"task_{task_id:02d}_rollouts{suffix}.parquet"
        tpath = folder / f"task_{task_id:02d}_traces{suffix}.parquet"
        require(rpath.exists() == tpath.exists(), "Incomplete rollout/trace pair")
        if rpath.exists():
            validate_identity(pd.read_parquet(rpath), pd.read_parquet(tpath), cfg)
    expanded = output / "EVAL_CONFIG.json"
    atomic_json(cfg, expanded)
    run_shard(index, expanded, patch_loader=load_ilpd_patches)
    validate_result(pd.read_parquet(folder / f"task_{task_id:02d}_rollouts.parquet"),
                    pd.read_parquet(folder / f"task_{task_id:02d}_traces.parquet"), suite, task_id, cfg)


def main() -> None:
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
