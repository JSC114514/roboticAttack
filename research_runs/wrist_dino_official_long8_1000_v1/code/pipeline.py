"""User-run submission only. CPU preflight never probes or submits Slurm/GPU."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
from common import ROOT, atomic_json, load_json, preflight, require, run_signature

STAGES = (("prepare", "01_prepare_cpu.sbatch"), ("train", "02_train_4090.sbatch"),
          ("validate", "03_validate_cpu.sbatch"), ("eval", "04_eval_4090.sbatch"), ("stats", "05_stats_cpu.sbatch"))

def existing_gpu_jobs(queue):
    jobs = set()
    for line in queue.splitlines():
        columns = line.split("|")
        require(len(columns) == 4, "Unknown squeue columns; refusing resource guess")
        job, state, tres, reason = columns
        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"} or "gpu" not in tres.lower():
            continue
        require("DependencyNeverSatisfied" not in reason, f"Existing GPU job {job} has an invalid dependency; resolve it before another batch")
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Unrecognized GPU array ID")
        jobs.add(match.group(1))
    return sorted(jobs, key=int)

def command_for(filename, dependency=None, external=()):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
    if dependency:
        command.append("--dependency=afterok:" + dependency)
    elif external:
        command.append("--dependency=afterany:" + ":".join(external))
    command.append(str(ROOT / "slurm" / filename))
    return command

def submit():
    report = preflight()
    final = ROOT / "results/STATUS.json"
    if final.exists() and load_json(final).get("status") == "COMPLETE":
        require(load_json(final)["signature"] == run_signature(), "Completed experiment identity changed")
        print("ALREADY_COMPLETE; no submission")
        return
    for name in ("logs", "reports"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
        old = load_json(receipt_path) if receipt_path.exists() else None
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"], capture_output=True, text=True, timeout=30)
        require(queue.returncode == 0, "squeue unavailable; no jobs submitted: " + queue.stderr.strip())
        existing = existing_gpu_jobs(queue.stdout)
        if old:
            active = {line.split("|", 1)[0].split("_", 1)[0] for line in queue.stdout.splitlines()}
            own = {job["job_id"] for job in old["jobs"]}
            require(not (own & active), "This batch already has active jobs; do not duplicate submit")
        receipt = {"status": "SUBMITTING", "signature": report["signature"], "jobs": [],
                   "waiting_for_existing_gpu_jobs": existing, "serial_phase_cap": 8}
        if old:
            receipt["previous_receipt"] = old
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, filename in STAGES:
            command = command_for(filename, dependency, existing)
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout)
            except Exception as error:
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Submission failed at {stage}; earlier jobs retained in {receipt_path}: {error}") from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            dependency = job
            atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("Official Long: eight fresh1000 methods; <=8x4090 training THEN <=8x4090 automatic evaluation; 3600 episodes; no smoke", flush=True)
        if existing:
            print("Waiting for existing own GPU jobs: " + ",".join(existing), flush=True)

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
        print(f"CPU_PREFLIGHT_PASS: official379 demos,8 methods,1000x50,B1,3600 episodes; signature={report['signature']}")
        print("NO_GPU_STARTED; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION")
