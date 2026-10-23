#!/usr/bin/env python3
"""Read-only CPU preflight or user-submitted single-GPU action-counter training."""
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
from dino_action_counter_common import (
    CONFIG, METHODS, config, pipeline_preflight, replay_records, schedule_records,
    require_preparation, source_preflight, source_records,
)
from white_patch.UADA_OFT_DINO import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config()
    clean = source_records(cfg)
    replay, _ = replay_records(cfg)
    schedule = schedule_records(cfg)
    preflight = source_preflight(cfg)
    pipeline_preflight(cfg)
    if args.dry_run:
        print(json.dumps({"status": "PASS", "config": str(CONFIG), "methods": METHODS,
            "method": args.method, "clean_records": len(clean), "replay_records": len(replay),
            "schedule_records": len(schedule), "total_outer": cfg["outer_iterations"],
            "optimizer_updates": cfg["outer_iterations"] * cfg["inner_loop"],
            "fresh_start_outer": 0, "initial_optimizer_state_empty": True,
            "scheduler_total_outer": cfg["scheduler_total_outer"], "model_loaded": False,
            "sbatch_submitted": False}, indent=2))
        return
    require(args.method and os.environ.get("SLURM_JOB_ID"), "A user-submitted Slurm GPU job is required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one Slurm-mapped CUDA GPU required")
    require_preparation(cfg, preflight)
    from UADA_OFT_DINO_wrapper import load_wrapper
    from white_patch.UADA_OFT_DINO_ACTION_COUNTER import ExploringOFTDINOActionCounterAttacker, source_manifest
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
                "Source changed between CPU preflight and model load")
        attacker = ExploringOFTDINOActionCounterAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        attacker.train_action_counter(clean, replay, cfg, args.method, sources, preflight["data"], load_s)


if __name__ == "__main__":
    main()
