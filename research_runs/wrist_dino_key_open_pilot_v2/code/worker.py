"""Slurm-only workloads with stage locks and per-attempt failure/resource receipts."""
from __future__ import annotations
import argparse
import fcntl
import os
import signal
import time
from common import ROOT, SUITES, require, atomic_json, load_json
from resources import snapshot

def main():
    p = argparse.ArgumentParser(); p.add_argument("--stage", required=True, choices=("prepare", "train", "validate", "eval", "stats"))
    p.add_argument("--index", type=int); a = p.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Workload stages require the user's Slurm allocation")
    require(a.stage != "eval" or a.index is not None and 0 <= a.index < 4, "Invalid suite worker")
    label = a.stage+("_"+str(a.index) if a.index is not None else "")
    lock_path = ROOT / "reports/locks" / (label+".lock"); lock_path.parent.mkdir(parents=True, exist_ok=True)
    def stopped(signum, _frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    with lock_path.open("a") as lock:
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
                run_suite(SUITES[a.index])
            else:
                from analyze import validate_training, stats
                (validate_training if a.stage == "validate" else stats)()
                if a.stage == "validate" and not load_json(ROOT / "reports/AUTO_VALIDATION.json")["evaluation_allowed"]:
                    from analyze import stop_report
                    stop_report()
                    status = "SCIENTIFIC_STOP"
                    atomic_json({"status": status, "exit_code": 3, "evaluation_GPU_allocation_allowed": False,
                                 "reason": "No validated changed deployment PNG; afterok blocks evaluation"}, ROOT / "reports/SCIENTIFIC_STOP.json")
                    print("SCIENTIFIC_STOP; exit3 gates afterok evaluation; no GPU evaluation and no job cancellation", flush=True)
                    return 3
            status = "COMPLETE"
        except BaseException as error:
            atomic_json({"stage": a.stage, "index": a.index, "error": repr(error), "job_id": os.environ["SLURM_JOB_ID"], **snapshot()},
                        ROOT / "reports/FAILURES" / f"{time.time_ns()}.json")
            raise
        finally:
            atomic_json({"stage": a.stage, "index": a.index, "status": status, "elapsed_s": time.monotonic()-start,
                         "job_id": os.environ["SLURM_JOB_ID"], **snapshot()}, ROOT / "reports/WORKER_COSTS" / f"{time.time_ns()}.json")

if __name__ == "__main__": raise SystemExit(main())
