#!/usr/bin/env python3
"""Four-suite evaluation of fresh original Exploring→OFT UADA checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, validate, require, sha256_file
from common import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research_runs/uada_original1000_v1"
TRAIN_CONFIG = ROOT / "configs/exploring_oft_dino/uada_original1000_v1.json"
BASE_CONFIG = ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json"
STEPS = (500, 750, 1000)
CONDITIONS = tuple(f"original_uada_{step}" for step in STEPS)
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
EXPERIMENT_ID = "EXPLORING-OFT-UADA-ORIGINAL1000-XSUITE-V1"


def config():
    base = json.loads(BASE_CONFIG.read_text())
    base.update(experiment_id=EXPERIMENT_ID, conditions=list(CONDITIONS),
                output_root=str(RUN / "evaluation/four_suite"), log_tag="UADA_ORIGINAL1000_XSUITE",
                order_seed=20260929, bootstrap_seed=20260929)
    require(base["suites"] == list(DENOMINATORS) and base["state_indices"] == list(range(40, 50))
            and base["patch_xy"] == [26, 137], "Frozen evaluation grid changed")
    return base


def patches(cfg, device):
    train = json.loads(TRAIN_CONFIG.read_text())
    folder = Path(train["output_root"]) / "exploring_oft_uada"
    status = json.loads((folder / "status.json").read_text())
    require(status["status"] == "COMPLETE" and status["experiment_id"] == train["experiment_id"]
            and status["method"] == "exploring_oft_uada" and status["outer_iterations"] == 1000
            and status["seed"] == 42 and status["data_seed"] == 20260920
            and status["source_uada_sha256"] == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
            and status["training_bank_sha256"] == "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8",
            "Original UADA training lineage incomplete or changed")
    loaded = {}
    for step, name in zip(STEPS, CONDITIONS):
        path = folder / f"outer_{step:04d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload["patch"].float()
        require(payload["experiment_id"] == train["experiment_id"]
                and payload["method"] == "exploring_oft_uada"
                and payload["outer_iterations"] == 1000
                and payload["inner_loop"] == 50 and payload["maskidx"] == [0]
                and payload["seed"] == 42 and payload["data_seed"] == 20260920
                and tuple(value.shape) == (3, 50, 50) and bool(torch.isfinite(value).all())
                and bool(((value >= 0) & (value <= 1)).all())
                and path.with_suffix(".png").is_file(), f"Invalid baseline checkpoint {step}")
        with Image.open(path.with_suffix(".png")) as picture:
            require(picture.size == (50, 50) and picture.mode == "RGB",
                    f"Invalid baseline PNG {step}")
            picture.verify()
        loaded[name] = (value.to(device), sha256_file(path))
    return loaded


def preflight(cfg):
    den = {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]}
    require(den == DENOMINATORS, "Frozen clean denominators changed")
    folder = RUN / "training/exploring_oft_uada"
    if not (folder / "status.json").is_file():
        return {"status": "WAITING_FOR_TRAINING", "denominators": den,
                "new_attack_episodes": 1200, "submitted": False}
    patches(cfg, torch.device("cpu"))
    return {"status": "PASS", "denominators": den,
            "new_attack_episodes": 1200, "eligible_rows_expected": 1146, "submitted": False}


def run_checked(index):
    cfg = config()
    require(preflight(cfg)["status"] == "PASS", "Baseline not ready")
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    sources = {str(p): sha256_file(p) for p in (Path(__file__), BASE_CONFIG, TRAIN_CONFIG,
                                                ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py")}
    identity = {"config": cfg, "source": sources,
                "patches": {name: digest for name, (_, digest) in patches(cfg, torch.device("cpu")).items()}}
    manifest = output / "EVAL_MANIFEST.json"
    if manifest.is_file():
        require(json.loads(manifest.read_text()) == identity, "Evaluation manifest changed")
    else:
        atomic_json(identity, manifest)
    expanded = output / "EVAL_CONFIG.json"
    atomic_json(cfg, expanded)
    run_shard(index, expanded, patch_loader=patches)
    suite, task = cfg["suites"][index // 10], index % 10
    folder = output / suite
    rows = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
    traces = pd.read_parquet(folder / f"task_{task:02d}_traces.parquet")
    validate(rows, traces, suite, task, cfg)
    require(rows.experiment_id.eq(EXPERIMENT_ID).all()
            and traces.experiment_id.eq(EXPERIMENT_ID).all(), "Evaluation identity changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--require-ready", action="store_true")
    group.add_argument("--array-index", type=int, choices=range(40))
    args = parser.parse_args()
    if args.preflight or args.require_ready:
        result = preflight(config())
        if args.require_ready:
            require(result["status"] == "PASS", "Baseline not ready")
        print(json.dumps(result, indent=2))
    else:
        require(os.environ.get("SLURM_JOB_ID"), "User-submitted Slurm job required")
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "EGL environment missing")
        run_checked(args.array_index)
