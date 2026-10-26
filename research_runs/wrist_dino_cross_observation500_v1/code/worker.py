"""Stage locks, append-only failure receipts, failed-attempt resource costs."""
from __future__ import annotations
import argparse
import fcntl
import os
import time
from common import ROOT, CONDITIONS, SUITES, atomic_json, require
from resources import snapshot

def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage", required=True, choices=("prepare", "train", "validate", "eval", "stats"))
    p.add_argument("--index", type=int); a = p.parse_args()
    require(a.stage != "eval" or a.index is not None and 0 <= a.index < 8, "Invalid evaluation worker index")
    require(bool(os.environ.get("SLURM_JOB_ID")), "Run workload stages in the user's Slurm allocation")
    label = a.stage+("_"+str(a.index) if a.index is not None else "")
    path = ROOT / "reports/locks" / (label+".lock"); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); status = "FAILED"
        try:
            if a.stage == "prepare":
                from data import prepare
                prepare()
            elif a.stage == "train":
                from train import train
                train()
            elif a.stage == "eval":
                from evaluate import run_suite
                run_suite(CONDITIONS[a.index//4], SUITES[a.index%4])
            else:
                from analyze import validate_training, stats
                (validate_training if a.stage == "validate" else stats)()
            status = "COMPLETE"
        except Exception as error:
            atomic_json({"stage": a.stage, "index": a.index, "error": repr(error), "job_id": os.environ.get("SLURM_JOB_ID"), **snapshot()},
                        ROOT / "reports/FAILURES" / f"{time.time_ns()}.json")
            raise
        finally:
            atomic_json({"stage": a.stage, "index": a.index, "status": status, "elapsed_s": time.monotonic()-start,
                         "job_id": os.environ.get("SLURM_JOB_ID"), **snapshot()}, ROOT / "reports/WORKER_COSTS" / f"{time.time_ns()}.json")

if __name__ == "__main__": main()
