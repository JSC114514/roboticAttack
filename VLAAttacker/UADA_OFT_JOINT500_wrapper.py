#!/usr/bin/env python3
"""Manual-only entry point for the six-arm Exploring-OFT comparison."""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "VLAAttacker", PROJECT_ROOT / "VLAAttacker" / "white_patch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper, require
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_JOINT500 import METHODS, ExploringOFTJoint500Attacker


SOURCE_UADA_SHA256 = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
DIRECTION_SHA256 = "58ddb6df5c279c89213d40b11c7b5e5f88b27d54abc01b73ceacad2d788faf31"


def validate_config(cfg: dict, cfg_path: Path, method: str) -> None:
    require(cfg["experiment_id"] == "EXPLORING-OFT-JOINT500-V1", "Experiment ID drift")
    require(tuple(cfg["methods"]) == METHODS and method in METHODS, "Method list drift")
    require(cfg["outer_iterations"] == 500 and cfg["inner_loop"] == 50,
            "Frozen 500 outer x 50 inner budget drift")
    require(cfg["maskidx"] == [0] and cfg["batch_size"] == 1, "Legacy UADA target/batch drift")
    require(cfg["auxiliary_gradient_ratio"] == 0.3 and cfg["persistent_beta"] == 1.0,
            "Objective weight drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["suite"] == "libero_object",
            "Patch/source suite drift")
    source = PROJECT_ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_UADA_SHA256, "Read-only UADA.py content drift")
    require(not (source.stat().st_mode & stat.S_IWUSR), "UADA.py must remain owner-read-only")
    require(sha256_file(cfg["signed_direction_artifact"]) == DIRECTION_SHA256,
            "Signed-direction artifact drift")
    require(Path(cfg["training_bank"]).is_file(), "Missing frozen training bank")
    require(Path(cfg["baseline_patch"]).is_file()
            and sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"],
            "Frozen UADA-500 baseline drift")
    baseline_cfg = json.loads((cfg_path.parent / "object_scale500.json").read_text(encoding="utf-8"))
    matched = ("seed", "data_seed", "suite", "unnorm_key", "checkpoint", "training_bank",
               "patch_shape", "outer_iterations", "inner_loop", "batch_size", "learning_rate",
               "weight_decay", "warmup_outer_iterations", "geometry", "resize_patch",
               "checkpoint_outer_iterations", "resume_every_outer_iterations",
               "action_head_q01", "action_head_q99", "structure_gates", "dtype")
    for key in matched:
        require(cfg[key] == baseline_cfg[key], f"Frozen UADA-500 contract drift: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        PROJECT_ROOT / "configs/exploring_oft_dino/object_joint500_v1.json"))
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    validate_config(cfg, cfg_path, args.method)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "method": args.method,
                          "source_uada": "read-only", "optimizer_updates": 25_000,
                          "old_patch_warm_start": False}, indent=2))
        return
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped CUDA GPU required")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}), flush=True)
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == 300, "Expected 300 Object training records")
    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTJoint500Attacker(
        wrapper, cfg["output_root"], resize_patch=bool(cfg["resize_patch"]))
    attacker.train_joint_patch(records, cfg, args.method)


if __name__ == "__main__":
    main()
