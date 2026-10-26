"""Manual CPU preparation -> <=8x4090 full evaluation -> CPU statistics."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import getpass
import re
import subprocess
from eval_common import ROOT, SOURCE_ROOT, atomic_json, load_json, preflight, require

STAGES = (("prepare", "01_prepare_cpu.sbatch"), ("eval", "02_eval_4090.sbatch"),
          ("stats", "03_stats_cpu.sbatch"))


def queue_dependencies(output, own):
    gpu_jobs, active = set(), set()
    for line in output.splitlines():
        columns = line.split("|")
        require(len(columns) == 5, "Unknown squeue format; no submission")
        job, state, tres, reason, name = (value.strip() for value in columns)
        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"}:
            continue
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job)
        require(match is not None, "Cannot resolve active array parent ID")
        parent = match.group(1)
        active.add(parent)
        require(not name.startswith(("oft_long8v2_", "oft_long8_4090_")),
                f"Old/new Long Final-8 job {job} is still active. Cancel only the obsolete pending 3090 eval/stats before replacing; no duplicate submitted")
        if "gpu" in tres.lower():
            require("DependencyNeverSatisfied" not in reason,
                    f"GPU job {job} has an invalid dependency; repair it manually before submitting")
            gpu_jobs.add(parent)
    require(not (active & own), "This evaluation already has active jobs; no duplicate submission")
    return sorted(gpu_jobs, key=int)


def command_for(filename, dependency=None, existing=()):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
    if dependency:
        command.append("--dependency=afterok:" + dependency)
    elif existing:
        command.append("--dependency=afterany:" + ":".join(existing))
    command.append(str(ROOT / "slurm" / filename))
    return command


def receipt_jobs(value):
    if not value:
        return set()
    return ({str(job["job_id"]) for job in value.get("jobs", [])}
            | receipt_jobs(value.get("previous_receipt")))


def submit():
    # Use the same lock as the parent full/recovery pipelines.
    with (SOURCE_ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = preflight()
        final = ROOT / "results/STATUS.json"
        if final.exists() and load_json(final).get("status") == "COMPLETE":
            require(load_json(final)["signature"] == report["signature"], "Completed evaluation identity changed")
            print("ALREADY_COMPLETE; no submission")
            return
        receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
        old = load_json(receipt_path) if receipt_path.exists() else None
        query = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R|%j"],
                               capture_output=True, text=True, check=False, timeout=20)
        require(query.returncode == 0, "squeue unavailable; no submission: " + query.stderr.strip())
        existing = queue_dependencies(query.stdout, receipt_jobs(old))
        for directory in ("logs", "reports"):
            (ROOT / directory).mkdir(parents=True, exist_ok=True)
        receipt = {"status": "SUBMITTING", "signature": report["signature"], "jobs": [],
                   "source_training_signature": report["source_training_signature"],
                   "waiting_for_existing_gpu_jobs": existing, "gpu_type": "4090", "max_concurrent": 8,
                   "time": datetime.datetime.now().astimezone().isoformat(), "training_restarted": False}
        if old:
            receipt["previous_receipt"] = old
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, filename in STAGES:
            command = command_for(filename, dependency, existing)
            try:
                response = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout)
            except Exception as error:
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage, failed_command=command, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Failed at {stage}; earlier IDs kept in {receipt_path}: {error}") from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            dependency = job
            atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("EVALUATION_ONLY; <=8x4090; 24 frozen patches + fresh clean; no training/smoke/auto-cancel", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--preflight", action="store_true")
    flags.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.submit:
        submit()
    else:
        report = preflight()
        print("EVAL4090_CPU_PREFLIGHT_PASS; 24 frozen PNGs; 10000 episodes; max8x4090; signature=" + report["signature"])
        print("NO_GPU_STARTED; NO_RENDERER; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION")


if __name__ == "__main__":
    main()
