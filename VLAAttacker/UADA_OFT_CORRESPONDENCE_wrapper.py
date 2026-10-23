#!/usr/bin/env python3
"""Manual Slurm entry. --dry-run loads tensors/metadata only, never an OFT model."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "VLAAttacker", PROJECT_ROOT / "VLAAttacker/white_patch"):
    sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.UADA_OFT_CORRESPONDENCE import (
    METHODS, ExploringOFTCorrespondenceAttacker, load_masks, source_manifest,
)
from white_patch.oft_correspondence_objectives import total_variation

SOURCE_SHA = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"


def validate_config(cfg):
    require(cfg["experiment_id"] == "OFT-DINO-CORRESPONDENCE500-V1", "Experiment ID drift")
    require(tuple(cfg["methods"]) == METHODS, "Method list drift")
    source = PROJECT_ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_SHA and source.stat().st_mode & 0o222 == 0,
            "Original UADA must retain frozen contents and read-only permissions")
    baseline_cfg = json.loads((PROJECT_ROOT / "configs/exploring_oft_dino/object_scale500.json").read_text())
    for key in ("seed", "data_seed", "suite", "unnorm_key", "checkpoint", "training_bank", "patch_shape",
                "maskidx", "outer_iterations", "inner_loop", "batch_size", "learning_rate", "weight_decay",
                "warmup_outer_iterations", "geometry", "resize_patch", "checkpoint_outer_iterations",
                "resume_every_outer_iterations", "action_head_q01", "action_head_q99", "num_open_loop_steps", "dtype"):
        require(cfg[key] == baseline_cfg[key], f"UADA500 pairing contract drift: {key}")
    constants = {"max_queries": 16, "min_object_coverage": 0.02, "neighborhood_radius": 2,
                 "shift_tokens": 1, "temperature": 0.07, "aux_initial_gradient_ratio": 0.3, "tv_weight": 1.0}
    require(all(cfg[k] == v for k, v in constants.items()), "Frozen objective contract drift")
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"], "Training bank drift")
    require(sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"], "UADA500 reference drift")
    reference = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"]
    require(abs(float(total_variation(reference)) - cfg["tv_threshold"]) < 1e-8, "TV reference drift")
    model_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text())
    ids = model_cfg.get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower() and "siglip" in ids[1].lower(),
            "Checkpoint DINO/SigLIP branch order drift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/exploring_oft_dino/object_correspondence500_v1.json"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    records = bank["records"]
    masks, mask_hashes = load_masks(cfg, records)
    sources = source_manifest(cfg)
    if args.dry_run:
        print(json.dumps({"status": "CORRESPONDENCE500_DRY_RUN_PASS", "methods": list(METHODS),
                          "training_records": len(records), "verified_masks": len(masks),
                          "parameter_count": 7500, "updates_per_patch": 25000,
                          "source_files": len(sources), "model_loaded": False, "sbatch_submitted": False,
                          "tv_threshold": cfg["tv_threshold"]}, indent=2))
        return
    require(args.method is not None, "Training requires --method")
    require(bool(os.environ.get("SLURM_JOB_ID")), "Real-model training must run inside a user-submitted Slurm job")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one Slurm-mapped CUDA GPU required")
    out = Path(cfg["output_root"]) / args.method
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = out / "status.json"
        require(not status.exists() or json.loads(status.read_text()).get("status") != "COMPLETE",
                "Already COMPLETE; refusing to overwrite")
        print(json.dumps({"gpu": torch.cuda.get_device_name(0), "method": args.method,
                          "slurm_job_id": os.environ["SLURM_JOB_ID"]}), flush=True)
        wrapper = load_wrapper(cfg)
        attacker = ExploringOFTCorrespondenceAttacker(wrapper, cfg["output_root"], resize_patch=False)
        attacker.train(records, masks, cfg, args.method, sources,
                       {"training_bank": cfg["training_bank_sha256"], "reference_patch": cfg["baseline_patch_sha256"],
                        "mask_files": mask_hashes})


if __name__ == "__main__":
    main()
