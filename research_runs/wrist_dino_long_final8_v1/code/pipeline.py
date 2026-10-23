"""USER invocation only: CPU preflight or serial afterok submission, no smoke job."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess
from common import ROOT, atomic_json, load_json, preflight, require, run_signature


STAGES = (("prepare", "01_prepare_cpu.sbatch"), ("prefix", "02_prefix_4090.sbatch"),
          ("collect", "03_collect_4090.sbatch"), ("bank", "04_bank_cpu.sbatch"),
          ("teachers", "05_teachers_4090.sbatch"), ("teacher_bank", "06_teacher_bank_cpu.sbatch"),
          ("train", "07_train_4090.sbatch"), ("validate", "08_validate_cpu.sbatch"),
          ("eval", "09_eval_3090.sbatch"), ("stats", "10_stats_cpu.sbatch"))


def existing_gpu_jobs(queue):
    jobs = set()
    for line in queue.splitlines():
        columns = line.split("|")
        require(len(columns) == 4, "Unknown squeue format; do not guess GPU occupancy")
        job, state, tres, reason = columns
        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"} or "gpu" not in tres.lower():
            continue
        require("DependencyNeverSatisfied" not in reason,
                f"Existing GPU job {job} has an invalid dependency. Repair/cancel it manually before submission; no new jobs started")
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Cannot safely resolve existing GPU array job")
        jobs.add(match.group(1))
    return sorted(jobs, key=int)


def command_for(stage, filename, dependency=None, external=()):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
    if dependency:
        command.append("--dependency=afterok:" + dependency)
    elif external:
        command.append("--dependency=afterany:" + ":".join(external))
    command.append(str(ROOT / "slurm" / filename))
    return command


def submit():
    report = preflight()
    status = ROOT / "results/STATUS.json"
    if status.exists() and load_json(status).get("status") == "COMPLETE":
        require(load_json(status)["signature"] == run_signature(), "Completed batch identity changed")
        print("ALREADY_COMPLETE; no submission")
        return
    for name in ("logs", "reports"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Existing active own jobs must not receive another batch accidentally.
        receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
        old = load_json(receipt_path) if receipt_path.exists() else None
        result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"],
                                capture_output=True, text=True, check=False, timeout=30)
        require(result.returncode == 0, "squeue unavailable; no jobs submitted: " + result.stderr.strip())
        existing = existing_gpu_jobs(result.stdout)
        if old:
            own = {j["job_id"] for j in old["jobs"]}
            all_active = {line.split("|", 1)[0].split("_", 1)[0] for line in result.stdout.splitlines()}
            require(not (own & all_active), "This batch has active jobs already; inspect receipt, do not duplicate submit")
        receipt = {"status": "SUBMITTING", "signature": report["signature"], "jobs": [],
                   "waiting_for_existing_gpu_jobs": existing, "caps_by_serial_phase": {"4090": 8, "3090": 8}}
        if old:
            receipt["previous_receipt"] = old
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, filename in STAGES:
            command = command_for(stage, filename, dependency, existing)
            try:
                response = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout)
            except Exception as error:
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Submission failed at {stage}. Earlier jobs retained in {receipt_path}: {error}") from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            dependency = job
            atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("Fresh Long8: <=8x4090 stages THEN <=8x3090 automatic test; no independent smoke; no auto-cancel", flush=True)
        if existing:
            print("Wait for existing own GPU jobs: " + ",".join(existing), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit()
    else:
        report = preflight()
        print(f"CPU_PREFLIGHT_PASS: Long clean300; 8 methods; 24 deployment snapshots; "
              f"10000 episodes; caps4090=8/3090=8; signature={report['signature']}")
        print("NO_GPU_STARTED; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION")
