"""User-run submission entry; workers never submit or cancel jobs."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time

from common import ROOT, atomic_json, config, load_json, preflight, require, run_signature, sha256_file

SBATCH = {"screen": ROOT / "slurm/01_screen_4090.sbatch",
          "analyze": ROOT / "slurm/02_analyze_cpu.sbatch"}


def ensure_runtime_directories(root=ROOT):
    """Create every path Slurm or a worker must open before the job starts.

    Slurm opens stdout/stderr before executing the batch script.  In particular,
    ``logs`` cannot be left for the worker's atomic writers to create.
    """
    paths = (
        root / "logs",
        root / "raw",
        root / "results",
        root / "reports" / "FAILURES",
        root / "reports" / "WORKER_COSTS",
        root / "reports" / "locks",
        root / "reports" / "SUBMISSION_ARCHIVE",
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _screen_complete(signature):
    path = ROOT / "reports/SCREEN_STATUS.json"
    if not path.exists():
        return False
    value = load_json(path)
    if value.get("status") != "COMPLETE":
        return False
    require(value["signature"] == signature and value["completed_arms"] == value["planned_arms"] == 54,
            "Completed screen identity/grid changed")
    for relative, expected in value["artifacts"].items():
        require(sha256_file(ROOT / relative) == expected, f"Completed screen artifact changed: {relative}")
    return True


def _analysis_complete(signature):
    path = ROOT / "results/STATUS.json"
    if not path.exists():
        return False
    value = load_json(path)
    if value.get("status") != "COMPLETE":
        return False
    require(value["signature"] == signature and value["rows"] == 2160, "Completed analysis identity/grid changed")
    for name, expected in value["artifacts"].items():
        require(sha256_file(ROOT / "results" / name) == expected, f"Completed analysis artifact changed: {name}")
    return True


def gpu_dependencies(output):
    jobs = set()
    for line in output.splitlines():
        fields = line.split("|")
        require(len(fields) == 5, "Unrecognized Slurm queue output")
        job, state, tres, reason, _name = fields
        if "gpu" not in tres.lower():
            continue
        require("DependencyNeverSatisfied" not in reason,
                "An existing GPU chain has failed dependencies; inspect it before submitting a new GPU job")
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Unknown Slurm array identifier")
        jobs.add(match.group(1))
    return sorted(jobs, key=int)


def command_for(stage, signature, dependency=None, existing=()):
    command = ["sbatch", "--parsable", f"--job-name=acs1_{signature[:8]}_{stage}"]
    dependencies = []
    if dependency:
        dependencies.append("afterok:" + dependency)
    if stage == "screen" and existing:
        dependencies.append("afterany:" + ":".join(existing))
    if dependencies:
        command.append("--dependency=" + ",".join(dependencies))
    return command + [str(SBATCH[stage])]


def dry_run():
    signature = run_signature()
    print("DRY_RUN_ONLY; no scheduler query or submission")
    print("screen", command_for("screen", signature, existing=("EXISTING_GPU_JOBS",)))
    print("analyze", command_for("analyze", signature, dependency="SCREEN_JOB"))
    print("resources", config()["resources"])


def status():
    for relative in ("reports/FULL_SUBMISSION.json", "reports/LAST_LAUNCH_FAILURE.json",
                     "reports/GPU_INTERFACE.json",
                     "reports/SCREEN_STATUS.json", "results/STATUS.json"):
        path = ROOT / relative
        if path.exists():
            value = load_json(path)
            print(relative, {key: item for key, item in value.items()
                             if key not in ("artifacts", "sites", "sources", "trace_summary")})
        else:
            print(relative, "NOT_STARTED")
    failures = sorted((ROOT / "reports/FAILURES").glob("*.json"))
    if failures:
        print("LATEST_FAILURE", load_json(failures[-1]))


def submit():
    preflight()
    # This must happen before sbatch: slurmd opens the configured log files
    # before the Python worker gets a chance to create any parent directory.
    ensure_runtime_directories()
    signature = run_signature()
    if _analysis_complete(signature):
        print("ALREADY_COMPLETE; no submission")
        return
    stages = ([] if _screen_complete(signature) else ["screen"]) + ["analyze"]
    receipt_path = ROOT / "reports/FULL_SUBMISSION.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "reports/SUBMISSION.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R|%j"],
                               capture_output=True, text=True, timeout=30)
        require(queue.returncode == 0, "Cannot read Slurm queue; no jobs submitted: " + queue.stderr.strip())
        active = {line.split("|", 1)[0].split("_", 1)[0] for line in queue.stdout.splitlines()}
        old = load_json(receipt_path) if receipt_path.exists() else None
        if old:
            require(old["signature"] == signature, "Prior submission used a different source signature")
            require(old.get("status") not in ("SUBMITTING", "AMBIGUOUS_SBATCH_RESPONSE"),
                    "Prior submission response is unresolved; inspect its receipt before retrying")
            require(not active.intersection(str(job["job_id"]) for job in old.get("jobs", [])),
                    "This screen already has active jobs; use --status instead of resubmitting")
        existing = gpu_dependencies(queue.stdout) if "screen" in stages else []
        attempt = time.time_ns()
        if old:
            atomic_json(old, ROOT / "reports/SUBMISSION_ARCHIVE" / f"{attempt}.json")
        receipt = {"status": "SUBMITTING", "signature": signature, "attempt": attempt,
                   "stages": stages, "jobs": [], "existing_gpu_dependencies": existing,
                   "resources": config()["resources"]}
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage in stages:
            command = command_for(stage, signature, dependency, existing)
            receipt["pending_command"] = command
            atomic_json(receipt, receipt_path)
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job_id = result.stdout.strip().split(";")[0]
                require(result.returncode == 0 and re.fullmatch(r"\d+", job_id),
                        result.stderr.strip() or result.stdout.strip() or "Empty sbatch response")
            except BaseException as error:
                receipt.update(status="AMBIGUOUS_SBATCH_RESPONSE", failed_stage=stage, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError("Inspect the submission receipt before retrying; no automatic cancellation or resubmission") from error
            receipt["jobs"].append({"stage": stage, "job_id": job_id, "command": command})
            receipt.pop("pending_command", None)
            dependency = job_id
            atomic_json(receipt, receipt_path)
            print(stage, job_id, flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("Full screen->analysis chain submitted by the user-run entry. Use --status; do not submit a second copy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("preflight", "dry-run", "status", "submit"))
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
        print("CPU_PREFLIGHT_PASS", result["signature"],
              "no model/GPU/scheduler/submission", result["resources"])
    elif args.action == "dry-run":
        dry_run()
    elif args.action == "status":
        status()
    else:
        submit()


if __name__ == "__main__":
    main()
