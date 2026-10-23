#!/usr/bin/env python3
"""Two new wrist patches; noisy rolling patch is authorized for testing, not approved."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_exploring_oft_dino_ila_refine200_cross_suite as previous
from run_exploring_oft_dino_ila_refine200_cross_suite import (
    ROOT, DENOMINATORS, atomic_json, frozen_clean, load_json, require, sha256_file,
    runtime_config, run_shard, validate, require_gpu,
)

EXPERIMENT_ID = "OFT-DINO-ALTERNATING-REFINE200-XSUITE-V1"
CONDITIONS = ("alternating_fixed_tv", "alternating_rolling_tv")
CONTROLS = ("exploring_oft_uada", "uada_continue_tv", "dino_ila_tv")
ALL_CONDITIONS = CONTROLS + CONDITIONS
AUTHORIZATION = "USER_REQUESTED_TEST_DESPITE_ROLLING_NOISE_20260927"
APPEARANCE = {CONDITIONS[0]: "STRUCTURE_RETAINED_NOT_FORMALLY_APPROVED",
              CONDITIONS[1]: "NOISY_USER_REPORTED_NOT_ACCEPTED"}
RUNTIME_KEYS = ("training_suite", "suites", "task_count", "state_indices", "order_seed",
                "bootstrap_seed", "num_steps_wait", "max_steps_by_suite", "num_open_loop_steps",
                "image_resolution", "patch_shape", "patch_xy", "dtype", "checkpoints", "unnorm_keys",
                "paths", "frozen_object_clean", "frozen_cross_suite_clean")


def load_config(path: Path) -> dict:
    overlay = load_json(path)
    require(set(overlay) == {"experiment_id", "base_config", "base_config_sha256", "conditions",
                             "evaluation_authorization", "appearance_assessment", "output_root",
                             "patch_sources", "reused_controls"}, "Overlay keys drift")
    base_path = Path(overlay["base_config"])
    require(base_path.resolve() == ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json"
            and sha256_file(base_path) == overlay["base_config_sha256"], "Evaluation reference changed")
    cfg = load_json(base_path)
    cfg.update(overlay)
    # Do not inherit an unrelated baseline's visual PASS into the new evaluation.
    cfg.pop("manual_visual_review", None)
    cfg["log_tag"] = "DINO_ALTERNATING200_XSUITE"
    require(cfg["experiment_id"] == EXPERIMENT_ID and tuple(cfg["conditions"]) == CONDITIONS,
            "Two-arm experiment identity drift")
    require(cfg["evaluation_authorization"] == AUTHORIZATION, "User evaluation authorization missing")
    require(cfg["appearance_assessment"] == APPEARANCE, "Appearance assessment drift")
    require(tuple(cfg["reused_controls"]["conditions"]) == CONTROLS, "Reused control coverage drift")
    require(cfg["patch_xy"] == [26, 137] and cfg["patch_shape"] == [3, 50, 50]
            and cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10,
            "Patch/state grid drift")
    return cfg


def checked_previous_config(cfg):
    ref = cfg["reused_controls"]
    require(sha256_file(ref["config"]) == ref["config_sha256"], "Reused config hash mismatch")
    old = previous.load_config(Path(ref["config"]))
    require(all(cfg[k] == old[k] for k in RUNTIME_KEYS), "Reused runtime/model/state grid drift")
    require(Path(cfg["output_root"]).resolve() != Path(old["output_root"]).resolve(),
            "Evaluation overlaps previous outputs")
    return old


def load_refinement_patches(cfg, device):
    require(set(cfg["patch_sources"]) == set(CONDITIONS), "Patch source coverage drift")
    original = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(original.stat().st_mode & 0o222 == 0 and sha256_file(original) ==
            "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
            "Original UADA source changed or writable")
    old = checked_previous_config(cfg)
    base = old["patch_sources"][CONTROLS[0]]
    require(sha256_file(base["patch_path"]) == base["sha256"], "Baseline patch hash mismatch")
    initial = torch.load(base["patch_path"], map_location="cpu", weights_only=False)["patch"].float()
    old_train = load_json(old["patch_sources"][CONTROLS[1]]["training_manifest"])
    patches, shared_train = {}, None
    for method in CONDITIONS:
        source = cfg["patch_sources"][method]
        path = Path(source["patch_path"])
        require(sha256_file(path) == source["sha256"], f"Patch hash mismatch: {method}")
        for key in ("training_status", "training_manifest", "runtime_alignment", "patch_png"):
            require(sha256_file(source[key]) == source[key + "_sha256"],
                    f"Training evidence hash mismatch: {method} {key}")
        status, manifest = load_json(source["training_status"]), load_json(source["training_manifest"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(status["status"] == "COMPLETE" and status["method"] == manifest["method"] == payload["method"] == method,
                f"Training incomplete/identity drift: {method}")
        require(status["signature"] == manifest["signature"] == payload["signature"] == source["signature"]
                and manifest["experiment_id"] == source["experiment_id"] == "OFT-DINO-ALTERNATING-REFINE200-V1",
                f"Training signature mismatch: {method}")
        require(status["outer"] == 200 and status["total_outer"] == payload["total_outer"] == 700
                and status["optimizer_updates"] == 10000
                and status["action_optimizer_updates"] == status["feature_optimizer_updates"] == 5000
                and status["total_optimizer_updates"] == payload["total_optimizer_updates"] == 35000
                and status["final_patch_sha256"] == source["sha256"], f"Training budget drift: {method}")
        train = manifest["config"]
        require(train["suite"] == "libero_object" and train["outer_iterations"] == 200
                and train["base_outer_iterations"] == 500 and train["inner_loop"] == 50
                and train["action_steps_per_outer"] == train["feature_steps_per_outer"] == 25
                and train["baseline_patch_sha256"] == base["sha256"] and manifest["parameter_count"] == 7500
                and manifest["optimizer_restarted"] is True and manifest["feature_main_objective"] is True,
                f"Training reference/schedule drift: {method}")
        for key in ("training_bank_sha256", "baseline_patch_sha256", "seed", "data_seed", "learning_rate",
                    "weight_decay", "warmup_outer_iterations", "maskidx", "geometry", "tv_threshold", "tv_weight"):
            require(train[key] == old_train["config"][key], f"Matched training setup drift: {key}")
        require(manifest["packages"] == old_train["packages"], "Training package identity drift")
        if shared_train is None:
            shared_train = train
        require(train == shared_train, "Two-arm training config drift")
        alignment = load_json(source["runtime_alignment"])
        require(alignment["status"] == "PASS" and len(alignment["records"]) == 2
                and {r["condition"] for r in alignment["records"]} == {"clean", "reference_eot"}
                and all(r["pixel_exact"] and r["feature_close"] for r in alignment["records"]),
                f"Actual full-OFT/DINO alignment missing: {method}")
        start = torch.load(path.parent / "outer_0000.pt", map_location="cpu", weights_only=False)["patch"].float()
        require(torch.equal(start, initial), f"Warm-start mismatch: {method}")
        patch = payload["patch"].float()
        require(tuple(patch.shape) == (3, 50, 50) and bool(torch.isfinite(patch).all())
                and bool(((patch >= 0) & (patch <= 1)).all()), f"Invalid RGB tensor: {method}")
        require(not Path(cfg["output_root"]).resolve().is_relative_to(path.parent.parent.resolve()),
                "Evaluation overlaps training")
        patches[method] = patch.to(device), source["sha256"]
    require(len({digest for _, digest in patches.values()}) == 2, "Duplicate patch hashes")
    return patches


def validate_identity(rollouts, traces, cfg):
    previous.validate_identity(rollouts, traces, {**cfg, "patch_sources": cfg["patch_sources"]}) if False else None
    keys = ["suite", "task_id", "state_index", "condition"]
    require(not rollouts.empty and not rollouts.duplicated(keys).any(), "Duplicate/empty rollout rows")
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all() and traces.experiment_id.eq(EXPERIMENT_ID).all(),
            "Wrong experiment in existing outputs")
    require(rollouts.condition.isin(CONDITIONS).all() and traces.condition.isin(CONDITIONS).all(),
            "Unknown condition in outputs")
    require(rollouts.success.isin([0, 1]).all() and rollouts.failure.isin([0, 1]).all()
            and (rollouts.success + rollouts.failure).eq(1).all(), "Invalid success/failure labels")
    for method in CONDITIONS:
        require(rollouts.loc[rollouts.condition == method, "patch_sha256"].eq(
            cfg["patch_sources"][method]["sha256"]).all(), f"Rollout patch hash mismatch: {method}")
    require(not traces.duplicated(keys + ["decision_block"]).any()
            and np.isfinite(traces.select_dtypes(include=[np.number]).to_numpy()).all(), "Invalid trace rows")
    require(traces.merge(rollouts[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").all()
            and rollouts[keys].merge(traces[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            ["_merge"].eq("both").all(), "Rollout/trace linkage mismatch")
    for suite, rows in rollouts.groupby("suite"):
        expected = frozen_clean(cfg, suite)
        linked = rows.merge(expected, on=keys[:3], suffixes=("", "_clean"), validate="many_to_one")
        require(len(linked) == len(rows) and linked.rollout_seed.eq(linked.rollout_seed_clean).all()
                and linked.frozen_clean_success.eq(linked.success_clean).all(), "Frozen clean linkage mismatch")
        require(rows.checkpoint.eq(cfg["checkpoints"][suite]).all()
                and rows.unnorm_key.eq(cfg["unnorm_keys"][suite]).all()
                and rows.training_suite.eq(cfg["training_suite"]).all(), "Rollout model/runtime metadata drift")


def validate_result(rollouts, traces, suite, task_id, cfg):
    validate_identity(rollouts, traces, cfg)
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.groupby("state_index").condition_order.nunique().eq(2).all(),
            "Condition order not a permutation")


def frozen_controls(cfg):
    ref = cfg["reused_controls"]
    old = checked_previous_config(cfg)
    for key in ("manifest", "status", "rollouts", "traces"):
        require(sha256_file(ref[key]) == ref[key + "_sha256"], f"Reused evidence hash mismatch: {key}")
    manifest, status = load_json(ref["manifest"]), load_json(ref["status"])
    require(manifest["config"] == old and status["status"] == "COMPLETE"
            and status["experiment_id"] == previous.EXPERIMENT_ID and status["new_attack_episodes"] == 1600,
            "Reused evaluation identity drift")
    for path, digest in {**manifest["source_sha256"], **manifest["frozen_clean_sha256"]}.items():
        require(sha256_file(path) == digest, f"Reused runner/runtime/clean changed: {path}")
    previous.load_refinement_patches(old, torch.device("cpu"))
    rows, traces = pd.read_parquet(ref["rollouts"]), pd.read_parquet(ref["traces"])
    require(len(rows) == 1600, "Reused rollout count drift")
    for suite in cfg["suites"]:
        for task in range(10):
            r = rows.loc[(rows.suite == suite) & (rows.task_id == task)]
            t = traces.loc[(traces.suite == suite) & (traces.task_id == task)]
            previous.validate_result(r, t, suite, task, old)
            require(r.checkpoint.eq(cfg["checkpoints"][suite]).all()
                    and r.unnorm_key.eq(cfg["unnorm_keys"][suite]).all(), "Reused rollout model metadata drift")
    rows, traces = rows.loc[rows.condition.isin(CONTROLS)].copy(), traces.loc[traces.condition.isin(CONTROLS)].copy()
    require(len(rows) == 1200, "Selected control count drift")
    return rows, traces


def training_costs(cfg):
    old = checked_previous_config(cfg)
    rows = [r for r in previous.training_costs(old) if r["condition"] in CONTROLS]
    for method in CONDITIONS:
        status = load_json(cfg["patch_sources"][method]["training_status"])
        rows.append({"condition": method, "total_outer_iterations": 700, "total_optimizer_updates": 35000,
                     "refinement_outer_iterations": 200, "logged_training_s": float(status["elapsed_s"]),
                     "refinement_model_load_s": float(status["costs"]["model_load_s"]),
                     "known_cost_including_base_s": float(status["cost_including_base_s"]),
                     "cost_boundary": status["cost_boundary"]})
    return rows


def evaluation_manifest(cfg, create=False):
    source = [Path(__file__), ROOT / "scripts/analyze_exploring_oft_dino_alternating_refine200_cross_suite.py",
              ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py",
              ROOT / "scripts/run_exploring_oft_dino_ila_refine200_cross_suite.py",
              ROOT.parent / "oft_ramp_recovery_aware/scripts/run_continuous_cldev.py",
              ROOT.parent / "oft_ramp_recovery_aware/scripts/runtime.py",
              ROOT.parent / "oft_ramp_recovery_aware/scripts/common.py",
              ROOT.parent / "oft_ramp_recovery_aware/scripts/statistics.py"]
    data = {"config": cfg, "source_sha256": {str(p): sha256_file(p) for p in source},
            "frozen_clean_sha256": {p: sha256_file(p) for p in
                                    (cfg["frozen_object_clean"], cfg["frozen_cross_suite_clean"])}}
    signature = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    path = Path(cfg["output_root"]) / "EVAL_MANIFEST.json"
    if path.exists():
        require(load_json(path)["signature"] == signature, "Evaluation config/source changed during resume")
    else:
        require(create, "Missing evaluation manifest")
        atomic_json({"signature": signature, **data}, path)
    return signature


def preflight(cfg):
    patches = load_refinement_patches(cfg, torch.device("cpu"))
    controls, _ = frozen_controls(cfg)
    denominators = {s: int(frozen_clean(cfg, s).success.sum()) for s in cfg["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominator drift")
    for suite in cfg["suites"]:
        require(Path(runtime_config(cfg, suite)["paths"]["checkpoint"], "config.json").is_file(),
                f"Missing checkpoint: {suite}")
    return {"status": "DINO_ALTERNATING200_XSUITE_PREFLIGHT_PASS", "conditions": list(CONDITIONS),
            "patch_sha256": {m: h for m, (_, h) in patches.items()}, "new_attack_rollouts": 800,
            "reused_attack_rollouts": len(controls), "combined_attack_rollouts": 2000,
            "new_eligible_rows_expected": 764, "combined_eligible_rows_expected": 1910,
            "clean_rollouts_reused": 400, "clean_success_denominators": denominators,
            "evaluation_authorization": cfg["evaluation_authorization"],
            "appearance_assessment": cfg["appearance_assessment"], "training_costs": training_costs(cfg),
            "model_loaded": False, "sbatch_submitted": False}


def run_checked_shard(array_index, config):
    cfg = load_config(config)
    require(0 <= array_index < 40, "Array index outside frozen grid")
    output = Path(cfg["output_root"])
    suite, task = cfg["suites"][array_index // 10], array_index % 10
    folder = output / suite
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"task_{task:02d}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Evaluation shard already running") from error
        evaluation_manifest(cfg, create=True)
        for suffix in ("", ".partial"):
            rp, tp = (folder / f"task_{task:02d}_{kind}{suffix}.parquet" for kind in ("rollouts", "traces"))
            require(rp.exists() == tp.exists(), "Incomplete rollout/trace file pair")
            if rp.exists():
                r, t = pd.read_parquet(rp), pd.read_parquet(tp)
                validate_identity(r, t, cfg)
                require(r.suite.eq(suite).all() and r.task_id.eq(task).all(), "Existing shard identity drift")
        expanded = output / "EVAL_CONFIG.json"
        atomic_json(cfg, expanded)
        run_shard(array_index, expanded, patch_loader=load_refinement_patches)
        validate_result(pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet"),
                        pd.read_parquet(folder / f"task_{task:02d}_traces.parquet"), suite, task, cfg)


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
