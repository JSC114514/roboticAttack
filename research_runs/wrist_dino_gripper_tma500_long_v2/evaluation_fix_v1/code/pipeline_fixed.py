#!/usr/bin/env python3
"""Manual evaluation-only submission. Preflight neither queries nor submits."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess

from eval_common import (ROOT, ORIGIN_ROOT, CONFIG, REPAIR_ID, atomic_json, config, digest,
                         load_json, preflight, require, sha256_file)
from pipeline import parse_queue  # Frozen queue guard, not its submit function.


def is_complete(path):
    return path.is_file() and load_json(path).get("status") == "COMPLETE"


def smoke_complete():
    path = ROOT / "reports/GPU_SMOKE.json"
    if not path.exists():
        return False
    from evaluate_fixed import load_patches, require_smoke
    from eval_common import preparation
    cfg, prep = config(), preparation()
    _, hashes = load_patches(cfg)
    require_smoke(cfg, prep, hashes)
    return True


def stages_to_submit():
    cfg = config()
    tasks = [task for task in range(10) if not all(
        is_complete(ROOT / "evaluation" / suite / f"task_{task:02d}_STATUS.json")
        for suite in cfg["evaluation"]["suites"])]
    stages = [("prepare", "154_prepare_cpu.sbatch", None)]
    if tasks:
        if not smoke_complete():
            stages.append(("smoke", "155_smoke_4090.sbatch", None))
        stages.append(("eval", "156_eval_4090.sbatch",
                       ",".join(map(str, tasks)) + "%5"))
    stages.append(("stats", "157_stats_cpu.sbatch", None))
    return stages


def submit():
    """Only the USER may invoke this branch; no automatic assistant submission."""
    report = preflight()
    if is_complete(ROOT / "evaluation/STATUS.json"):
        status = load_json(ROOT / "evaluation/STATUS.json")
        require(status["repair_id"] == REPAIR_ID and status["config_sha256"] == sha256_file(CONFIG)
                and all(sha256_file(path) == checksum for path, checksum in status["shards"].items()),
                "Finished evaluation changed; no resubmission")
        print("ALREADY_COMPLETE; no new jobs", flush=True)
        return
    for folder in (ROOT / "logs", ROOT / "reports"):
        folder.mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%b|%R"],
                                  capture_output=True, text=True, check=False, timeout=30)
        require(response.returncode == 0,
                "squeue failed; no submission because occupancy is unknown: " + response.stderr.strip())
        waits = parse_queue(response.stdout)
        receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
        receipt = {"status": "SUBMITTING", "repair_id": REPAIR_ID,
                   "original_experiment_root": str(ORIGIN_ROOT),
                   "config_sha256": sha256_file(CONFIG), "preflight_signature": digest(report),
                   "external_wait_jobs": waits, "jobs": [], "training_restarted": False,
                   "gpu_limits_by_serial_phase": {"smoke_4090": 1, "eval_4090": 5}}
        if receipt_path.exists():
            receipt["previous_receipt"] = load_json(receipt_path)
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, script, array in stages_to_submit():
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--nodes=1", "--ntasks=1"]
            if stage in {"smoke", "eval"}:
                command.extend(["--partition=rtx", "--gres=gpu:4090:1"])
            else:
                command.extend(["--partition=cpu", "--gres=none"])
            if dependency:
                command.append("--dependency=afterok:" + dependency)
            elif waits:
                command.append("--dependency=afterany:" + ":".join(waits))
            if array:
                command.append("--array=" + array)
            command.append(str(ROOT / "slurm" / script))
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
            job = result.stdout.strip().split(";")[0]
            if result.returncode or not re.fullmatch(r"\d+", job):
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                               error=result.stderr.strip() or result.stdout.strip())
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Submission failed at {stage}; earlier jobs retained: {receipt['error']}")
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            dependency = job
            atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("EVALUATION ONLY: CPU prepare -> 1x4090 smoke -> <=5x4090 evaluation -> CPU stats; "
              "original patches/data/training unchanged", flush=True)
        if waits:
            print("Waiting for your existing GPU/OFT jobs: " + ",".join(waits), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--submit", action="store_true")
    flags.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit()
    else:
        preflight()
        print(f"CPU_PREFLIGHT_PASS; 3 trained500 patches reused; 2000 episodes; "
              f"smoke1x4090 -> eval<=5x4090; stages={stages_to_submit()}; "
              "NO_GPU_STARTED; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION", flush=True)
