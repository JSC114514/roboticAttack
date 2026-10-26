"""Worker routing with durable failure evidence; no resource submission here."""
from __future__ import annotations
import argparse
import os
import time
import traceback
from run_core import ARMS, ROOT, SUITES, read_json, signature, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["teacher", "head", "train", "eval", "stats"])
    p.add_argument("arm", nargs="?", choices=ARMS)
    args = p.parse_args()
    started, completed = time.monotonic(), False
    try:
        if args.stage == "teacher":
            from teacher import build_cache
            build_cache()
        elif args.stage == "head":
            from fit_head import fit
            fit()
        elif args.stage == "train":
            from train_patch import train
            train(args.arm)
        elif args.stage == "eval":
            from rollout import evaluate
            evaluate(args.arm, SUITES[int(os.environ["SLURM_ARRAY_TASK_ID"])])
        else:
            from summarize import summarize
            summarize()
        completed = True
    except BaseException as error:
        identity = "_".join([args.stage, args.arm or "shared", os.environ.get("SLURM_ARRAY_TASK_ID", "single")])
        write_json({"status": "FAILED", "signature": signature(), "stage": args.stage, "arm": args.arm,
                    "error": repr(error), "traceback": traceback.format_exc(),
                    "job_id": os.environ.get("SLURM_JOB_ID")}, ROOT / "reports/FAILURES" / (identity + f"_{time.time_ns()}.json"))
        statuses = []
        if args.stage in ["teacher", "head"]:
            statuses = [ROOT / ("teacher" if args.stage == "teacher" else "readout") / "STATUS.json"]
        elif args.stage == "train":
            statuses = [ROOT / "training" / args.arm / "STATUS.json"]
        elif args.stage == "eval":
            suite = SUITES[int(os.environ["SLURM_ARRAY_TASK_ID"])]
            statuses = list((ROOT / "evaluation" / args.arm / suite).glob("task_*_STATUS.json"))
        elif args.stage == "stats":
            statuses = [ROOT / "results/STATUS.json"]
        for path in statuses:
            previous = read_json(path) if path.exists() else {}
            if previous.get("status") != "COMPLETE":
                write_json({**previous, "status": "FAILED", "signature": signature(), "error": repr(error),
                            "job_id": os.environ.get("SLURM_JOB_ID")}, path)
        raise
    finally:
        # Includes preparation/model/cache loading and failed attempts, not queue wait.
        write_json({"signature": signature(), "stage": args.stage, "arm": args.arm,
                    "elapsed_s": time.monotonic() - started, "completed": completed,
                    "gpu_worker": args.stage != "stats", "job_id": os.environ.get("SLURM_JOB_ID"),
                    "array_index": os.environ.get("SLURM_ARRAY_TASK_ID")},
                   ROOT / "reports/WORKER_COSTS" / f"{args.stage}_{args.arm or 'shared'}_{time.time_ns()}.json")


if __name__ == "__main__":
    main()
