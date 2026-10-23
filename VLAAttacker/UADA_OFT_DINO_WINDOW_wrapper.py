#!/usr/bin/env python3
"""User-submitted Slurm entry for matched Exploring RGB clip training."""
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
from dino_window_common import CONFIG, METHODS, SEEDS, config, source_records
from UADA_OFT_DINO_wrapper import load_wrapper
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.UADA_OFT_DINO_WINDOW import ExploringOFTDINOWindowAttacker, source_manifest


def replay_records(cfg):
    bank = Path(cfg["replay_bank"])
    status_path = bank.with_name("STATUS.json")
    require(bank.is_file() and status_path.is_file(), "Complete window bank required before training")
    status = json.loads(status_path.read_text())
    require(status["status"] == "COMPLETE" and status["bank_sha256"] == sha256_file(bank)
            and status["baseline_patch_sha256"] == cfg["baseline_patch_sha256"]
            and status["source_bank_sha256"] == cfg["training_bank_sha256"],
            "Window bank identity drift")
    payload = torch.load(bank, map_location="cpu", weights_only=False)
    records = payload["records"]
    require(len(records) == status["records"] and
            {int(r["task_id"]) for r in records} == set(range(10)),
            "Window bank task coverage drift")
    return records, sha256_file(bank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    clean = source_records(cfg)
    if args.dry_run:
        try:
            replay, digest = replay_records(cfg)
        except RuntimeError as error:
            if "Complete window bank required" not in str(error):
                raise
            replay, digest = [], None
        print(json.dumps({"status": "PASS" if digest else "WAITING_FOR_WINDOW_BANK",
                          "clean_records": len(clean), "clip_records": len(replay),
                          "methods": METHODS, "seeds": SEEDS,
                          "additional_optimizer_updates_per_arm": 10000,
                          "source_immutable": True, "model_loaded": False,
                          "sbatch_submitted": False}, indent=2))
        return
    require(args.method is not None and args.seed is not None and bool(os.environ.get("SLURM_JOB_ID")),
            "Training requires a user-submitted Slurm job, method and seed")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one mapped CUDA GPU required")
    cfg["seed"] = args.seed
    run_id = f"{args.method}_seed{args.seed}"
    replay, digest = replay_records(cfg)
    out = Path(cfg["output_root"]) / "training" / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = out / "status.json"
        require(not status_path.exists() or json.loads(status_path.read_text())["status"] != "COMPLETE",
                "Completed arm is immutable")
        print(json.dumps({"method": args.method, "seed": args.seed,
                          "gpu": torch.cuda.get_device_name(0),
                          "job": os.environ["SLURM_JOB_ID"]}), flush=True)
        started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        load_s = time.monotonic() - started
        require(not any(p.requires_grad for p in wrapper.vla.parameters()), "VLA weights not frozen")
        sources = source_manifest(cfg, wrapper)
        attacker = ExploringOFTDINOWindowAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        attacker.train(clean, replay, cfg, run_id, sources,
                       {"training_bank": cfg["training_bank_sha256"],
                        "window_bank": digest, "baseline_patch": cfg["baseline_patch_sha256"],
                        "config": sha256_file(args.config)}, model_load_s=load_s)


if __name__ == "__main__":
    main()
