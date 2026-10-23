#!/usr/bin/env python3
"""Independent entry point for Exploring-OFT branch-feature patch training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "VLAAttacker", PROJECT_ROOT / "VLAAttacker" / "white_patch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper, require
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_FEATURE import METHOD_BRANCH, ExploringOFTFeatureAttacker


def validate_config(cfg: dict, cfg_path: Path, method: str) -> None:
    require(cfg["experiment_id"] == "EXPLORING-OFT-FEATURE500-01", "Experiment ID drift")
    require(cfg["methods"] == list(METHOD_BRANCH) and method in cfg["methods"], "Method list drift")
    require(cfg["feature_gradient_ratio"] == 0.3, "Frozen gradient ratio drift")
    require(cfg["outer_iterations"] == 500 and cfg["inner_loop"] == 50,
            "Frozen 500 outer x 50 inner budget drift")
    require(cfg["maskidx"] == [0] and cfg["batch_size"] == 1, "UADA action target/batch drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["suite"] == "libero_object",
            "Wrist patch shape or source suite drift")
    require(cfg["checkpoint_outer_iterations"][-1] == 500, "Final checkpoint drift")
    require(Path(cfg["training_bank"]).is_file(), "Missing frozen training bank")
    require(Path(cfg["checkpoint"], "config.json").is_file(), "Missing OFT checkpoint")
    model_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text(encoding="utf-8"))
    timm_ids = model_cfg.get("timm_model_ids", [])
    require(len(timm_ids) == 2 and "dinov2" in timm_ids[0].lower()
            and "siglip" in timm_ids[1].lower(), "Visual branch identity/order drift")
    baseline_cfg = json.loads((cfg_path.parent / "object_scale500.json").read_text(encoding="utf-8"))
    matched = ("seed", "data_seed", "suite", "unnorm_key", "checkpoint", "training_bank",
               "patch_shape", "maskidx", "outer_iterations", "inner_loop", "batch_size",
               "learning_rate", "weight_decay", "warmup_outer_iterations", "geometry",
               "resize_patch", "checkpoint_outer_iterations", "mechanism_probe_outer_iterations",
               "resume_every_outer_iterations", "action_head_q01", "action_head_q99",
               "structure_gates", "num_open_loop_steps", "dtype")
    for key in matched:
        require(cfg[key] == baseline_cfg[key], f"500-round baseline contract drift: {key}")
    baseline_patch = Path(cfg["baseline_patch"])
    require(baseline_patch.is_file() and sha256_file(baseline_patch) == cfg["baseline_patch_sha256"],
            "UADA-500 baseline patch identity drift")
    require(not Path(cfg["output_root"]).resolve().is_relative_to(Path(baseline_cfg["output_root"]).resolve()),
            "New output must not overwrite baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/exploring_oft_dino/object_feature500.json"))
    parser.add_argument("--method", choices=tuple(METHOD_BRANCH), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg, cfg_path, args.method)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "method": args.method,
                          "source": "white_patch/UADA.py via UADA_OFT_DINO.py inheritance",
                          "branch": METHOD_BRANCH[args.method],
                          "optimizer_updates": 500 * 50, "old_patch_warm_start": False,
                          "baseline_sha256": cfg["baseline_patch_sha256"]}, indent=2))
        return
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped CUDA GPU required")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}), flush=True)
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == 300, "Expected 300 Object training records")
    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTFeatureAttacker(wrapper, cfg["output_root"], resize_patch=bool(cfg["resize_patch"]))
    attacker.train_feature_patch(records, cfg, args.method)


if __name__ == "__main__":
    main()
