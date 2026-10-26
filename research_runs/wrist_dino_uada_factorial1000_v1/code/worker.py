"""Explicit arm dispatch; process/file locks prevent duplicate GPU work."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import time

def main():
    from common import ROOT, METHODS, SUITES, atomic_json, require
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "eval"), required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    require(0 <= args.index < (3 if args.stage == "train" else 12), "Worker index out of range")
    arm = METHODS[args.index if args.stage == "train" else args.index // 4]
    suite = None if args.stage == "train" else SUITES[args.index % 4]
    os.environ["OFT_FACTORIAL_ARM"] = arm
    lock_path = ROOT / "reports/locks" / f"{args.stage}_{arm}_{suite or 'rgb'}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        outcome = "FAILED"
        try:
            if args.stage == "train":
                from train import train
                train(arm)
            else:
                from evaluate import run_suite
                run_suite(arm, suite)
            outcome = "COMPLETE"
        except Exception as error:
            atomic_json({"stage": args.stage, "arm": arm, "suite": suite, "error": repr(error),
                         "job_id": os.environ.get("SLURM_JOB_ID"), "task_id": os.environ.get("SLURM_ARRAY_TASK_ID")},
                        ROOT / "reports/FAILURES" / f"{time.time_ns()}.json")
            raise
        finally:
            atomic_json({"stage": args.stage, "arm": arm, "suite": suite, "status": outcome,
                         "elapsed_s": time.monotonic() - started, "job_id": os.environ.get("SLURM_JOB_ID")},
                        ROOT / "reports/WORKER_COSTS" / f"{time.time_ns()}.json")

if __name__ == "__main__":
    main()
