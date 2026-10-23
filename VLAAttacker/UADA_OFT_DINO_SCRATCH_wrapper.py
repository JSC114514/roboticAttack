#!/usr/bin/env python3
"""User-submitted 4090 training; prefix then matched learning curves."""
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
from dino_scratch_common import CONFIG, METHODS, config, prefix_state, replay_records, source_records
from UADA_OFT_DINO_wrapper import load_wrapper
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.UADA_OFT_DINO_SCRATCH import ExploringOFTDINOScratchAttacker, source_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("warmup", *METHODS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config()
    clean = source_records(cfg)
    data = {"clean_bank": cfg["training_bank_sha256"], "config": sha256_file(CONFIG)}
    if args.dry_run:
        prefix_ready = (Path(cfg["output_root"]) / "training/warmup/status.json").is_file()
        print(json.dumps({"status": "PASS", "clean_records": len(clean), "methods": METHODS,
                          "initialization": "seed42 random RGB", "historical_patch_loaded": False,
                          "total_outer": 1000, "shared_counted_prefix": 250,
                          "checkpoints": [500, 750, 1000], "checkpoint_count": 9,
                          "optimizer_updates_at_checkpoints": [25000, 37500, 50000],
                          "prefix_status_exists": prefix_ready,
                          "model_loaded": False, "sbatch_submitted": False}, indent=2))
        return
    require(args.method is not None and bool(os.environ.get("SLURM_JOB_ID")),
            "Training requires a user-submitted Slurm job")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA GPU required")
    replay = []
    if args.method != "warmup":
        _, digest, _ = prefix_state(cfg)
        data["prefix_resume"] = digest
        replay, data["window_bank"] = replay_records(cfg)
    folder = Path(cfg["output_root"]) / "training" / args.method
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        load_s = time.monotonic() - started
        require(not any(p.requires_grad for p in wrapper.vla.parameters()), "VLA is not frozen")
        attacker = ExploringOFTDINOScratchAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        attacker.train_scratch(clean, replay, cfg, args.method, source_manifest(cfg, wrapper), data, load_s)


if __name__ == "__main__":
    main()
