#!/usr/bin/env python3
"""Paired evaluation of three methods at 500/750/1000 total outer iterations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

from dino_scratch_common import ROOT, CONFIG, METHODS, CHECKPOINTS, config, prefix_state, replay_records
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file
from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, validate
sys.path.append(str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import require_gpu

EXPERIMENT_ID = ("OFT-DINO-SCRATCH1000-XSUITE-V2" if CONFIG.name.endswith("_v2.json")
                 else "OFT-DINO-SCRATCH1000-XSUITE-V1")
CONDITIONS = tuple(f"{method}_{step}" for step in CHECKPOINTS for method in METHODS)
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}


def load_config():
    train = config()
    cfg = json.loads(Path(train["eval_reference"]).read_text())
    cfg.update(experiment_id=EXPERIMENT_ID, conditions=list(CONDITIONS),
               output_root=str(Path(train["output_root"]) / "evaluation/four_suite"),
               visual_review=str(Path(train["output_root"]) / "VISUAL_REVIEW.json"),
               log_tag="DINO_SCRATCH_XSUITE", train_config=str(CONFIG))
    require(cfg["patch_xy"] == train["patch_xy"] and cfg["state_indices"] == list(range(40, 50)),
            "Frozen test grid/site changed")
    return cfg


def training_patches(cfg, device, *, require_visual=True):
    train = config()
    prefix, prefix_sha, _ = prefix_state(train)
    _, bank_sha = replay_records(train)
    patches, hashes = {}, {}
    for method in METHODS:
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
        require(status["status"] == "COMPLETE" and status["outer"] == 1000
                and status["optimizer_updates"] == 50000 and status["method"] == method
                and manifest["config"] == train and manifest["signature"] == status["signature"]
                and manifest["data"]["prefix_resume"] == prefix_sha
                and manifest["data"]["window_bank"] == bank_sha
                and alignment["status"] == "PASS"
                and all(r["pixel_exact"] and r["feature_close"] for r in alignment["records"]),
                f"Invalid training lineage: {method}")
        require(sha256_file(folder / "outer_1000.pt") == status["final_patch_sha256"]
                and sha256_file(folder / "outer_1000.png") == status["final_png_sha256"]
                and sha256_file(folder / "resume.pt") == status["resume_sha256"],
                f"Final training artifacts changed: {method}")
        initial = torch.load(folder / "outer_0000.pt", map_location="cpu", weights_only=False)["patch"]
        start = torch.load(folder / "outer_0250.pt", map_location="cpu", weights_only=False)["patch"]
        require(torch.equal(initial, prefix["initial"]) and torch.equal(start, prefix["patch"]),
                f"Common random initialization/prefix changed: {method}")
        for step in CHECKPOINTS:
            path = folder / f"outer_{step:04d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            patch = payload["patch"].float()
            require(payload["signature"] == status["signature"] and payload["method"] == method
                    and payload["total_outer"] == step and payload["total_optimizer_updates"] == step * 50
                    and tuple(patch.shape) == (3, 50, 50) and bool(torch.isfinite(patch).all())
                    and bool(((patch >= 0) & (patch <= 1)).all()), "Invalid checkpoint RGB/budget")
            name = f"{method}_{step}"
            hashes[name] = {"pt": sha256_file(path), "png": sha256_file(path.with_suffix(".png"))}
            patches[name] = (patch.to(device), hashes[name]["pt"])
    if require_visual:
        marker = Path(cfg["visual_review"])
        approval = ("AUTO_VALIDATED" if train.get("auto_eval_after_training", False)
                    else "APPROVED_BY_USER")
        require(marker.is_file() and json.loads(marker.read_text()) ==
                {"status": approval, "patches": hashes}, "Nine PNGs await validation")
    return {name: patches[name] for name in CONDITIONS}, hashes


def auto_validate_for_evaluation(cfg):
    train = config()
    require(train.get("auto_eval_after_training", False), "Automatic validation is only for V2")
    _, hashes = training_patches(cfg, torch.device("cpu"), require_visual=False)
    root = Path(train["output_root"])
    appearance = {method: {key: json.loads((root / "training" / method / "status.json").read_text())[key]
                           for key in ("tv", "saturation_fraction", "init_correlation", "init_rms_delta")}
                  for method in METHODS}
    report = {"status": "COMPLETE", "policy": "automatic artifact and lineage validation; not human visual approval",
              "experiment_id": train["experiment_id"], "patch_count": len(hashes),
              "patches": hashes, "appearance": appearance}
    report_path = root / "AUTO_VALIDATION.json"
    if report_path.exists():
        require(json.loads(report_path.read_text()) == report, "Automatic validation report changed")
    else:
        atomic_json(report, report_path)
    marker = Path(cfg["visual_review"])
    expected = {"status": "AUTO_VALIDATED", "patches": hashes}
    if marker.exists():
        require(json.loads(marker.read_text()) == expected, "Automatic validation marker changed")
    else:
        atomic_json(expected, marker)
    return report


def preflight(cfg):
    denominators = {s: int(frozen_clean(cfg, s).success.sum()) for s in cfg["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators drift")
    train = config()
    status_paths = [Path(train["output_root"]) / "training" / m / "status.json" for m in METHODS]
    if not all(p.is_file() and json.loads(p.read_text()).get("status") == "COMPLETE" for p in status_paths):
        return {"status": "WAITING_FOR_TRAINING", "conditions": CONDITIONS,
                "new_attack_episodes": 3600, "gpu_jobs_submitted": False}
    training_patches(cfg, torch.device("cpu"), require_visual=False)
    if not Path(cfg["visual_review"]).is_file():
        waiting = ("WAITING_FOR_AUTO_VALIDATION" if train.get("auto_eval_after_training", False)
                   else "WAITING_FOR_VISUAL_REVIEW")
        return {"status": waiting, "gpu_jobs_submitted": False}
    training_patches(cfg, torch.device("cpu"))
    return {"status": "PASS", "conditions": CONDITIONS, "denominators": denominators,
            "new_attack_episodes": 3600, "eligible_rows_expected": 3438, "gpu_jobs_submitted": False}


def validate_result(rollouts, traces, suite, task_id, cfg):
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all() and traces.experiment_id.eq(EXPERIMENT_ID).all(),
            "Wrong experiment identity")
    paired = rollouts.merge(frozen_clean(cfg, suite), on=["suite", "task_id", "state_index"],
                           suffixes=("", "_clean"), validate="many_to_one")
    require(len(paired) == len(rollouts) and paired.rollout_seed.eq(paired.rollout_seed_clean).all()
            and paired.frozen_clean_success.eq(paired.success_clean).all(), "Broken clean pairing")


def run_checked_shard(index):
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Evaluation is not ready")
    output = Path(cfg["output_root"])
    sources = {str(p): sha256_file(p) for p in (Path(__file__), CONFIG,
        ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py", Path(cfg["visual_review"]))}
    signature = hashlib.sha256(json.dumps({"config": cfg, "source": sources}, sort_keys=True).encode()).hexdigest()
    manifest = output / "EVAL_MANIFEST.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())["signature"] == signature, "Evaluation changed during run")
    else:
        atomic_json({"signature": signature, "config": cfg, "source": sources}, manifest)
    expanded = output / "EVAL_CONFIG.json"
    atomic_json(cfg, expanded)
    run_shard(index, expanded, patch_loader=lambda base, device: training_patches(base, device)[0])
    suite, task = cfg["suites"][index // 10], index % 10
    folder = output / suite
    validate_result(pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet"),
                    pd.read_parquet(folder / f"task_{task:02d}_traces.parquet"), suite, task, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--require-ready", action="store_true")
    group.add_argument("--approve-reviewed-nine-pngs", action="store_true")
    group.add_argument("--auto-validate-nine-pngs", action="store_true")
    group.add_argument("--array-index", type=int, choices=range(40))
    args = parser.parse_args()
    cfg = load_config()
    if args.preflight or args.require_ready:
        report = preflight(cfg)
        if args.require_ready:
            require(report["status"] == "PASS", f"Not ready: {report['status']}")
        print(json.dumps(report, indent=2))
    elif args.approve_reviewed_nine_pngs:
        require(not config().get("auto_eval_after_training", False), "V2 uses automatic validation")
        _, hashes = training_patches(cfg, torch.device("cpu"), require_visual=False)
        marker = Path(cfg["visual_review"])
        require(not marker.exists(), "Existing visual approval is immutable")
        atomic_json({"status": "APPROVED_BY_USER", "patches": hashes}, marker)
        print(json.dumps({"status": "APPROVED_BY_USER", "marker": str(marker)}, indent=2))
    elif args.auto_validate_nine_pngs:
        print(json.dumps(auto_validate_for_evaluation(cfg), indent=2))
    else:
        require(bool(os.environ.get("SLURM_JOB_ID")), "User-submitted Slurm job required")
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None, "EGL environment missing")
        print(json.dumps(require_gpu()), flush=True)
        run_checked_shard(args.array_index)
