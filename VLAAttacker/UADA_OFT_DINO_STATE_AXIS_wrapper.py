#!/usr/bin/env python3
"""CPU preflight or explicit user-submitted single-GPU fresh RGB training."""
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
from dino_state_axis_common import (
    CONFIG, METHODS, config, pipeline_preflight, replay_records,
    require_preparation, source_records,
)
from white_patch.UADA_OFT_DINO import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--preflight", "--dry-run", dest="preflight", action="store_true")
    args = parser.parse_args()
    require(args.config.resolve() == Path(CONFIG).resolve(),
            "--config must match OFT_DINO_STATE_AXIS_CONFIG so all source/data fingerprints use the same file")
    cfg = config(args.config)
    clean = source_records(cfg)
    replay, _ = replay_records(cfg)
    preflight = pipeline_preflight(cfg)
    if args.preflight:
        print(json.dumps({"status": "PASS", "config": str(args.config), "method": args.method,
            "methods": METHODS, "clean_records": len(clean), "replay_records": len(replay),
            "total_outer": cfg["outer_iterations"], "optimizer_updates": cfg["outer_iterations"] * cfg["inner_loop"],
            "fresh_start_outer": 0, "initial_optimizer_state_empty": True,
            "historical_patch_used_for_initialization": False,
            "historical_reference_patches_checked": True, "shared_trained_prefix": False,
            "model_loaded": False, "cuda_initialized": False, "sbatch_submitted": False}, indent=2))
        return
    require(args.method and os.environ.get("SLURM_JOB_ID"), "A user-submitted Slurm GPU job is required")
    require_preparation(cfg, preflight)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped CUDA GPU required")
    from UADA_OFT_DINO_wrapper import load_wrapper
    from white_patch.UADA_OFT_DINO_STATE_AXIS import ExploringOFTDINOStateAxisAttacker, source_manifest
    out = Path(cfg["output_root"]) / "training" / args.method
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        load_s = time.monotonic() - started
        require(not any(parameter.requires_grad for parameter in wrapper.vla.parameters()), "VLA must be frozen")
        sources = source_manifest(cfg, wrapper)
        require(all(sources.get(name) == digest for name, digest in preflight["source"].items()),
                "Actual runtime source changed after CPU preflight")
        attacker = ExploringOFTDINOStateAxisAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        attacker.train_state_axis(clean, replay, cfg, args.method, sources, preflight["data"], load_s)


if __name__ == "__main__":
    main()
