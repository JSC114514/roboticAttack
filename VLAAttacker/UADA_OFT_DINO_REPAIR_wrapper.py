#!/usr/bin/env python3
"""User-submitted GPU calibration or repair training with a frozen counted prefix."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]
from dino_repair_common import CONFIG, METHODS, config, prefix_state, replay_records, source_records
from white_patch.UADA_OFT_DINO import require, sha256_file


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--method", choices=METHODS)
    group.add_argument("--calibrate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config()
    clean = source_records(cfg)
    saved, digest, _ = prefix_state(cfg)
    replay, bank_digest = replay_records(cfg)
    data = {"clean_bank": cfg["training_bank_sha256"], "config": sha256_file(CONFIG),
            "prefix_resume": digest, "window_bank": bank_digest}
    if args.dry_run:
        print(json.dumps({"status": "PASS", "clean_records": len(clean), "replay_records": len(replay),
            "methods": METHODS, "total_outer": 1000, "scheduler_total_outer": 2000,
            "shared_counted_prefix": saved["outer"], "inherited_optimizer_updates": 12500,
            "final_optimizer_updates": 50000, "training_gpu_limit": 4, "evaluation_gpu_limit": 8,
            "model_loaded": False, "sbatch_submitted": False}, indent=2))
        return
    require((args.method or args.calibrate) and os.environ.get("SLURM_JOB_ID"),
            "A user-submitted Slurm GPU job is required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA GPU required")
    from run_exploring_oft_dino_repair_cross_suite import source_preflight
    source_preflight(cfg)
    from UADA_OFT_DINO_wrapper import load_wrapper
    from white_patch.UADA_OFT_DINO_REPAIR import ExploringOFTDINORepairAttacker, source_manifest
    folder = Path(cfg["output_root"]) / ("reports" if args.calibrate else f"training/{args.method}")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        load_s = time.monotonic() - started
        require(not any(p.requires_grad for p in wrapper.vla.parameters()), "VLA is not frozen")
        attacker = ExploringOFTDINORepairAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        sources = source_manifest(cfg, wrapper)
        if args.calibrate:
            report = attacker.calibrate_appearance(clean, cfg, sources, data, load_s)
            print(json.dumps(report, indent=2))
        else:
            attacker.train_repair(clean, replay, cfg, args.method, sources, data, load_s)


if __name__ == "__main__":
    main()
