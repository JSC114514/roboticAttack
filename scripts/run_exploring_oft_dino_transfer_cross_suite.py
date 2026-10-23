#!/usr/bin/env python3
"""Paired four-suite evaluation of five from-scratch 2000-outer training curves."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from dino_transfer_common import CONFIG, METHODS, CHECKPOINTS, config, prefix_state, replay_records
from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, validate
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/exploring_oft_dino/scale500_cross_suite.json"
ORIGINAL_CONFIG = ROOT / "configs/exploring_oft_dino/uada_original2000_transfer_v1.json"
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
ARMS = ("original_uada", *METHODS)
CONDITIONS = tuple(f"{arm}_{step}" for step in CHECKPOINTS for arm in ARMS)
EXPERIMENT_ID = "OFT-DINO-TRANSFER2000-XSUITE-V1"


def load_config():
    train = config()
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg.update(experiment_id=EXPERIMENT_ID, conditions=list(CONDITIONS),
               output_root=str(Path(train["output_root"]) / "evaluation/four_suite"),
               log_tag="DINO_TRANSFER2000_XSUITE", order_seed=20260930,
               bootstrap_seed=20260930, training_suite="libero_object")
    require(cfg["suites"] == list(DENOMINATORS) and cfg["state_indices"] == list(range(40, 50))
            and cfg["patch_xy"] == train["patch_xy"], "Frozen test grid/site changed")
    return cfg


def patches(cfg, device):
    train = config()
    _, prefix_digest, _ = prefix_state(train)
    _, bank_digest = replay_records(train)
    original = json.loads(ORIGINAL_CONFIG.read_text())
    require(original["outer_iterations"] == 2000 and original["inner_loop"] == 50
            and original["training_bank"] == train["training_bank"], "Original baseline budget/bank drift")
    result, hashes, movement = {}, {}, {}
    for arm in ARMS:
        folder = (Path(original["output_root"]) / "exploring_oft_uada" if arm == "original_uada"
                  else Path(train["output_root"]) / "training" / arm)
        status = json.loads((folder / "status.json").read_text())
        require(status["status"] == "COMPLETE", f"Incomplete training: {arm}")
        if arm == "original_uada":
            require(status["outer_iterations"] == 2000 and status["inner_loop"] == 50
                    and status["method"] == "exploring_oft_uada"
                    and status["training_bank_sha256"] == train["training_bank_sha256"],
                    "Original baseline lineage changed")
        else:
            manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
            alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
            require(status["outer"] == 2000 and status["optimizer_updates"] == 100000
                    and status["method"] == arm and manifest["config"] == train
                    and manifest["data"]["prefix_resume"] == prefix_digest
                    and manifest["data"]["window_bank"] == bank_digest
                    and alignment["status"] == "PASS"
                    and all(row["pixel_exact"] and row["feature_close"]
                            for row in alignment["records"]), f"Adapted training lineage changed: {arm}")
            require(sha256_file(folder / "outer_2000.pt") == status["final_patch_sha256"]
                    and sha256_file(folder / "outer_2000.png") == status["final_png_sha256"],
                    f"Final checkpoint changed: {arm}")
        values = {}
        for step in CHECKPOINTS:
            path = folder / f"outer_{step:04d}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            value = payload["patch"].float()
            require(tuple(value.shape) == (3, 50, 50) and bool(torch.isfinite(value).all())
                    and bool(((value >= 0) & (value <= 1)).all()), f"Invalid RGB checkpoint: {arm}/{step}")
            if arm == "original_uada":
                require(payload["experiment_id"] == original["experiment_id"]
                        and payload["method"] == "exploring_oft_uada"
                        and payload["outer_iterations"] == 2000
                        and payload["inner_loop"] == 50, "Original checkpoint identity drift")
            else:
                require(payload["signature"] == status["signature"] and payload["method"] == arm
                        and payload["total_outer"] == step
                        and payload["total_optimizer_updates"] == step * 50,
                        "Adapted checkpoint identity drift")
            png = path.with_suffix(".png")
            with Image.open(png) as picture:
                require(picture.size == (50, 50) and picture.mode == "RGB", "Bad patch PNG")
                picture.verify()
            name = f"{arm}_{step}"
            result[name] = (value.to(device), sha256_file(path))
            hashes[name] = {"pt": sha256_file(path), "png": sha256_file(png)}
            values[step] = value
        movement[arm] = {"mae_1000_to_2000": float((values[1000] - values[2000]).abs().mean()),
                         "rms_1000_to_2000": float(((values[1000] - values[2000]) ** 2).mean().sqrt())}
        require(movement[arm]["mae_1000_to_2000"] > 1e-7,
                f"Patch did not change between checkpoints: {arm}")
    return result, hashes, movement


def preflight(cfg):
    denominators = {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators changed")
    train = config()
    original = json.loads(ORIGINAL_CONFIG.read_text())
    folders = [Path(original["output_root"]) / "exploring_oft_uada"] + [
        Path(train["output_root"]) / "training" / arm for arm in METHODS]
    if not all((folder / "status.json").is_file()
               and json.loads((folder / "status.json").read_text()).get("status") == "COMPLETE"
               for folder in folders):
        return {"status": "WAITING_FOR_TRAINING", "conditions": CONDITIONS,
                "new_attack_episodes": 4000, "submitted": False}
    patches(cfg, torch.device("cpu"))
    return {"status": "PASS", "conditions": CONDITIONS, "denominators": denominators,
            "new_attack_episodes": 4000, "eligible_rows_expected": 3820, "submitted": False}


def auto_validate():
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS", "Training incomplete")
    _, hashes, movement = patches(cfg, torch.device("cpu"))
    root = Path(config()["output_root"])
    report = {"status": "COMPLETE", "policy": "automatic tensor/lineage/change check, not visual approval",
              "checkpoints": hashes, "checkpoint_movement": movement,
              "visual_review_pending": True, "evaluation_ready": True}
    atomic_json(report, root / "AUTO_VALIDATION.json")
    return report


def run_checked(index):
    cfg = load_config()
    require(preflight(cfg)["status"] == "PASS" and
            (Path(config()["output_root"]) / "AUTO_VALIDATION.json").is_file(),
            "Automatic validation not complete")
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    expanded = output / "EVAL_CONFIG.json"
    atomic_json(cfg, expanded)
    run_shard(index, expanded, patch_loader=lambda base, device: patches(base, device)[0])
    suite, task = cfg["suites"][index // 10], index % 10
    folder = output / suite
    rollouts = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
    traces = pd.read_parquet(folder / f"task_{task:02d}_traces.parquet")
    validate(rollouts, traces, suite, task, cfg)
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all()
            and traces.experiment_id.eq(EXPERIMENT_ID).all(), "Evaluation identity drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--auto-validate", action="store_true")
    group.add_argument("--array-index", type=int, choices=range(40))
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(load_config()), indent=2))
    elif args.auto_validate:
        print(json.dumps(auto_validate(), indent=2))
    else:
        require(os.environ.get("SLURM_JOB_ID"), "User-submitted Slurm job required")
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID"), "EGL environment missing")
        run_checked(args.array_index)
