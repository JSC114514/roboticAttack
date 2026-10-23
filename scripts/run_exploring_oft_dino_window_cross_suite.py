#!/usr/bin/env python3
"""Nine-condition, paired four-suite closed-loop evaluation after RGB review."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts")]
from dino_window_common import METHODS, SEEDS, config as train_config
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file
sys.path.append(str(ROOT.parent / "oft_ramp_recovery_aware/scripts"))
from common import require_gpu
from run_exploring_oft_scale500_cross_suite import frozen_clean, run_shard, validate

CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_window_refine200_cross_suite_v1.json"
EXPERIMENT_ID = "OFT-DINO-WINDOW-REFINE200-XSUITE-V1"
CONDITIONS = ("uniform_2200_frozen", *(f"{method}_seed{seed}"
              for seed in SEEDS for method in METHODS))
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90,
                "libero_goal": 98, "libero_10": 95}


def load_config(path: Path = CONFIG):
    overlay = json.loads(Path(path).read_text())
    require(set(overlay) == {"experiment_id", "train_config", "base_config",
                            "base_config_sha256", "conditions", "output_root",
                            "visual_review"}, "Evaluation overlay drift")
    require(overlay["experiment_id"] == EXPERIMENT_ID
            and tuple(overlay["conditions"]) == CONDITIONS, "Evaluation identity drift")
    train = train_config(Path(overlay["train_config"]))
    require(Path(overlay["base_config"]).resolve() == Path(train["eval_reference"]).resolve()
            and sha256_file(overlay["base_config"]) == overlay["base_config_sha256"],
            "Frozen cross-suite reference changed")
    cfg = json.loads(Path(overlay["base_config"]).read_text())
    cfg.update(overlay)
    cfg["log_tag"] = "DINO_WINDOW_XSUITE"
    require(cfg["patch_xy"] == train["patch_xy"] and cfg["patch_shape"] == train["patch_shape"]
            and cfg["state_indices"] == list(range(40, 50)) and cfg["task_count"] == 10,
            "Frozen evaluation grid/placement drift")
    return cfg


def training_patches(cfg, device, *, require_visual=True):
    train = train_config(Path(cfg["train_config"]))
    bank = Path(train["replay_bank"])
    bank_status = json.loads(bank.with_name("STATUS.json").read_text())
    require(bank_status["status"] == "COMPLETE" and bank_status["bank_sha256"] == sha256_file(bank),
            "Fresh clip bank changed")
    baseline = Path(train["baseline_patch"])
    patches = {CONDITIONS[0]:
               (torch.load(baseline, map_location="cpu", weights_only=False)["patch"].float().to(device),
                train["baseline_patch_sha256"])}
    visual_hashes = {}
    for method in CONDITIONS[1:]:
        kind, seed_text = method.rsplit("_seed", 1)
        seed = int(seed_text)
        require(kind in METHODS and seed in SEEDS, "Unrecognized training arm")
        folder = Path(train["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
        path = folder / "outer_0200.pt"
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == 200 and status["total_outer"] == 2400
                and status["optimizer_updates"] == 10000
                and status["total_optimizer_updates"] == 120000
                and status["final_patch_sha256"] == sha256_file(path),
                f"Incomplete/changed trained RGB patch: {method}")
        run_config = {**train, "seed": seed}
        require(manifest["experiment_id"] == train["experiment_id"]
                and manifest["method"] == method and manifest["signature"] == status["signature"]
                and manifest["config"] == run_config
                and manifest["data"]["baseline_patch"] == train["baseline_patch_sha256"]
                and manifest["data"]["window_bank"] == bank_status["bank_sha256"]
                and alignment["status"] == "PASS"
                and all(row["pixel_exact"] and row["feature_close"] for row in alignment["records"]),
                f"Training provenance or pixel alignment invalid: {method}")
        initial = torch.load(folder / "outer_0000.pt", map_location="cpu", weights_only=False)["patch"].float()
        require(torch.equal(initial, patches[CONDITIONS[0]][0].cpu()),
                f"Wrong common RGB start: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        patch = payload["patch"].float()
        require(payload["method"] == method and payload["signature"] == status["signature"]
                and payload["total_outer"] == 2400 and tuple(patch.shape) == (3, 50, 50)
                and bool(torch.isfinite(patch).all())
                and bool(((patch >= 0) & (patch <= 1)).all()),
                f"Invalid final RGB patch: {method}")
        png = path.with_suffix(".png")
        require(png.is_file() and sha256_file(png) == status["final_png_sha256"],
                f"Final PNG missing or changed: {method}")
        visual_hashes[method] = {"pt": sha256_file(path), "png": sha256_file(png)}
        patches[method] = (patch.to(device), sha256_file(path))
    require(len({digest for _, digest in patches.values()}) == len(CONDITIONS),
            "Two compared RGB patches are identical")
    if require_visual:
        marker = Path(cfg["visual_review"])
        require(marker.is_file() and json.loads(marker.read_text()) ==
                {"status": "APPROVED_BY_USER", "patches": visual_hashes},
                "Eight final PNGs await user visual review")
    return patches, visual_hashes


def validate_result(rollouts, traces, suite, task_id, cfg):
    validate(rollouts, traces, suite, task_id, cfg)
    require(rollouts.experiment_id.eq(EXPERIMENT_ID).all()
            and traces.experiment_id.eq(EXPERIMENT_ID).all(), "Wrong evaluation experiment")
    clean = frozen_clean(cfg, suite)
    linked = rollouts.merge(clean, on=["suite", "task_id", "state_index"],
                            suffixes=("", "_clean"), validate="many_to_one")
    require(len(linked) == len(rollouts)
            and linked.rollout_seed.eq(linked.rollout_seed_clean).all()
            and linked.frozen_clean_success.eq(linked.success_clean).all(),
            "Frozen clean pair linkage drift")


def preflight(cfg):
    denominators = {suite: int(frozen_clean(cfg, suite).success.sum()) for suite in cfg["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators drift")
    train = train_config(Path(cfg["train_config"]))
    folders = [Path(train["output_root"]) / "training" / method for method in CONDITIONS[1:]]
    if not all((folder / "status.json").is_file() and
               json.loads((folder / "status.json").read_text()).get("status") == "COMPLETE"
               for folder in folders):
        return {"status": "WAITING_FOR_TRAINING", "conditions": CONDITIONS,
                "new_attack_episodes": 3600, "gpu_jobs_submitted": False}
    patches, _ = training_patches(cfg, torch.device("cpu"), require_visual=False)
    if not Path(cfg["visual_review"]).is_file():
        return {"status": "WAITING_FOR_VISUAL_REVIEW", "patch_sha256":
                {key: digest for key, (_, digest) in patches.items()},
                "gpu_jobs_submitted": False}
    training_patches(cfg, torch.device("cpu"), require_visual=True)
    return {"status": "PASS", "conditions": CONDITIONS, "denominators": denominators,
            "new_attack_episodes": 3600, "eligible_rows_expected": 3438,
            "gpu_jobs_submitted": False}


def run_checked_shard(index, config_path):
    cfg = load_config(config_path)
    require(preflight(cfg)["status"] == "PASS" and 0 <= index < 40,
            "Evaluation preflight not approved")
    output = Path(cfg["output_root"])
    sources = {str(path): sha256_file(path) for path in
               (Path(__file__), ROOT / "scripts/run_exploring_oft_scale500_cross_suite.py",
                Path(cfg["train_config"]), Path(cfg["visual_review"]))}
    signature = hashlib.sha256(json.dumps({"cfg": cfg, "source": sources}, sort_keys=True).encode()).hexdigest()
    manifest = output / "EVAL_MANIFEST.json"
    if manifest.exists():
        require(json.loads(manifest.read_text())["signature"] == signature,
                "Evaluation config/source changed during run")
    else:
        atomic_json({"signature": signature, "source": sources, "config": cfg}, manifest)
    expanded = output / "EVAL_CONFIG.json"
    atomic_json(cfg, expanded)
    run_shard(index, expanded, patch_loader=lambda base, device: training_patches(base, device)[0])
    suite, task_id = cfg["suites"][index // 10], index % 10
    folder = output / suite
    validate_result(pd.read_parquet(folder / f"task_{task_id:02d}_rollouts.parquet"),
                    pd.read_parquet(folder / f"task_{task_id:02d}_traces.parquet"), suite, task_id, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--array-index", type=int, choices=range(40))
    args = parser.parse_args()
    require(args.preflight != (args.array_index is not None), "Choose preflight or array index")
    if args.preflight:
        print(json.dumps(preflight(load_config(args.config)), indent=2))
    else:
        require(bool(os.environ.get("SLURM_JOB_ID")), "Evaluation needs a user-submitted Slurm task")
        require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl"
                and os.environ.get("MUJOCO_EGL_DEVICE_ID") is not None,
                "EGL renderer environment missing")
        print(json.dumps(require_gpu()), flush=True)
        run_checked_shard(args.array_index, args.config)
