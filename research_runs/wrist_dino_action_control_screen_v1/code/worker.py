"""Slurm-only workers with locks, failure receipts and measured resource peaks."""
from __future__ import annotations
import argparse
import fcntl
import os
import signal
import time
from common import ROOT, atomic_json, require
from resources import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("screen", "analyze"))
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "Workers require the user's Slurm allocation")
    lock_path = ROOT / "reports/locks" / f"{args.stage}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def stopped(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopped)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, status = time.monotonic(), "FAILED"
        try:
            if args.stage == "screen":
                from screen import screen
                screen()
            else:
                from analyze import analyze
                analyze()
            status = "COMPLETE"
        except BaseException as error:
            atomic_json({"stage": args.stage, "status": "FAILED", "error": repr(error),
                         "job_id": os.environ["SLURM_JOB_ID"], **snapshot()},
                        ROOT / "reports/FAILURES" / f"{time.time_ns()}.json")
            raise
        finally:
            atomic_json({"stage": args.stage, "status": status, "elapsed_s": time.monotonic() - started,
                         "job_id": os.environ["SLURM_JOB_ID"], **snapshot()},
                        ROOT / "reports/WORKER_COSTS" / f"{time.time_ns()}.json")


if __name__ == "__main__":
    main()
