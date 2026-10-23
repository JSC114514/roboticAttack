"""User-run submission only. CPU preflight never probes or submits Slurm/GPU."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
from common import ROOT, SUITES, atomic_json, load_json, preflight, require, run_signature, worker_schedule

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
        # A permanently invalid pending dependency cannot run or occupy a GPU.
        # Report it, but do not cancel it or wait forever on it.
        if state == "PENDING" and "DependencyNeverSatisfied" in reason:
            continue
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Unrecognized GPU array ID")
        jobs.add(match.group(1))
    return sorted(jobs, key=int)


def ignored_invalid_jobs(queue):
    return [line.split("|", 1)[0] for line in queue.splitlines()
            if len(line.split("|")) == 4 and line.split("|")[1] == "PENDING"
            and "gpu" in line.split("|")[2].lower() and "DependencyNeverSatisfied" in line.split("|")[3]]

def command_for(filename, dependency=None, external=(), array=None):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
    if dependency:
        command.append("--dependency=afterok:" + dependency)
    elif external:
        command.append("--dependency=afterany:" + ":".join(external))
    if array is not None:
        command.append("--array=" + array)
    command.append(str(ROOT / "slurm" / filename))
    return command

def pending_arrays():
    signature = run_signature()
    pending_workers, pending_tasks = [], []
    for index, methods in enumerate(worker_schedule()):
        complete = True
        for method in methods:
            folder = ROOT / "training" / method
            status = folder / "status.json"
            if status.exists():
                value = load_json(status)
                require(value["signature"] == signature, "Existing training identity changed")
            else:
                value = {}
            if not (value.get("status") == "COMPLETE" and value.get("outer") == 500
                    and (folder / "resume.pt").exists() and (folder / "outer_0500.png").exists()):
                complete = False
        if not complete:
            pending_workers.append(index)
    for task in range(10):
        complete = True
        for suite in SUITES:
            folder = ROOT / "evaluation" / suite
            status = folder / f"task_{task:02d}_STATUS.json"
            value = load_json(status) if status.exists() else {}
            if value:
                require(value["signature"] == signature, "Existing evaluation identity changed")
            if not (value.get("status") == "COMPLETE" and value.get("episodes") == 330
                    and (folder / f"task_{task:02d}_rollouts.parquet").exists()
                    and (folder / f"task_{task:02d}_traces.parquet").exists()):
                complete = False
        if not complete:
            pending_tasks.append(task)
    return {"train": pending_workers, "eval": pending_tasks}


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
        if old:
            require(old["signature"] == report["signature"], "Previously submitted experiment changed; do not mix results")
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"], capture_output=True, text=True, timeout=30)
        require(queue.returncode == 0, "squeue unavailable; no jobs submitted: " + queue.stderr.strip())
        existing = existing_gpu_jobs(queue.stdout)
        if old:
            active = set()
            for line in queue.stdout.splitlines():
                job, state, _tres, reason = line.split("|")
                if state == "PENDING" and "DependencyNeverSatisfied" in reason:
                    continue
                active.add(job.split("_", 1)[0])
            own = {job["job_id"] for job in old["jobs"]}
            require(not (own & active), "This batch already has active jobs; do not duplicate submit")
        receipt = {"status": "SUBMITTING", "signature": report["signature"], "jobs": [],
                   "waiting_for_existing_gpu_jobs": existing, "serial_phase_cap": 8,
                   "training_gpu_cap": 6, "evaluation_gpu_cap": 8,
                   "ignored_invalid_pending_gpu_jobs": ignored_invalid_jobs(queue.stdout)}
        if old:
            receipt["previous_receipt"] = old
        atomic_json(receipt, receipt_path)
        dependency = None
        pending = pending_arrays()
        for stage, filename in STAGES:
            if stage in pending and not pending[stage]:
                print(f"{stage}: all complete, no GPU array submitted", flush=True)
                continue
            array = None
            if stage in pending:
                cap = 6 if stage == "train" else 8
                array = ",".join(map(str, pending[stage])) + f"%{cap}"
            command = command_for(filename, dependency, existing, array)
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
        print("BATCH500: 32 fresh patches / 500x50; <=6x4090 training THEN <=8x4090 automatic evaluation; 13200 episodes", flush=True)
        if existing:
            print("Waiting for existing own GPU jobs: " + ",".join(existing), flush=True)
        if receipt["ignored_invalid_pending_gpu_jobs"]:
            print("Not waiting on permanently invalid pending dependencies (not cancelled): " +
                  ",".join(receipt["ignored_invalid_pending_gpu_jobs"]), flush=True)

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
        atomic_json(report, ROOT / "reports/CPU_PREFLIGHT.json")
        print(f"CPU_PREFLIGHT_PASS: official379 demos,same250 pool,32 arms,500x50,B1/B2/B4/B8,13200 episodes,6 train / 8 eval; signature={report['signature']}")
        print("NO_GPU_STARTED; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION")
