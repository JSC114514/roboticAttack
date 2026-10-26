"""Manual submission only; local preflight/dry-run/status never query Slurm."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time
from common import (ROOT, METHODS, SUITES, atomic_json, load_json, preflight, prepare,
                    experiment, require, run_signature, sha256_file)

STAGES = (("prepare", "01_prepare_cpu.sbatch"), ("train", "02_train_4090.sbatch"),
          ("validate", "03_validate_cpu.sbatch"), ("eval", "04_eval_4090.sbatch"), ("stats", "05_stats_cpu.sbatch"))

def existing_gpu_jobs(queue):
    jobs = set()
    for line in queue.splitlines():
        parts = line.split("|")
        require(len(parts) == 4, "Unknown squeue format; no submission")
        job, state, tres, reason = parts
        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"} or "gpu" not in tres.lower():
            continue
        require("DependencyNeverSatisfied" not in reason, "Existing GPU dependency is invalid; no new batch")
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Unknown array identifier")
        jobs.add(match.group(1))
    return sorted(jobs, key=int)

def pending_indices(signature):
    train = []
    for i, arm in enumerate(METHODS):
        folder = ROOT / "training" / arm
        path = folder / "status.json"
        if not path.exists():
            train.append(i); continue
        saved = load_json(path)
        require(saved["signature"] == signature, "Training signature drift; do not overwrite")
        if saved["status"] != "COMPLETE":
            train.append(i); continue
        require(saved["outer"] == 1000 and saved["optimizer_updates"] == 50000
                and (folder / "outer_1000.pt").exists() and (folder / "outer_1000.png").exists(), "Completed training artifacts missing")
    todo_eval = list(range(12))
    validation = ROOT / "reports/AUTO_VALIDATION.json"
    if not train and validation.exists():
        from evaluate import task_complete
        saved = load_json(validation)
        require(saved["signature"] == signature, "Validation identity changed")
        if saved["status"] == "COMPLETE":
            todo_eval = []
            for i in range(12):
                arm, suite = METHODS[i // 4], SUITES[i % 4]
                folder = ROOT / "evaluation" / arm / suite
                if not all(task_complete(folder, task, signature, saved["patches"][arm]["png"]) for task in range(10)):
                    todo_eval.append(i)
    return train, todo_eval

def command_for(stage, filename, signature, dependency=None, existing=(), indices=None):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name=fd1000_" + signature[:8] + "_" + stage]
    if dependency:
        command.append("--dependency=afterok:" + dependency)
    elif existing:
        command.append("--dependency=afterany:" + ":".join(existing))
    if indices is not None:
        require(indices, "Empty GPU array cannot be submitted")
        cap = experiment()["training_gpu_cap" if stage == "train" else "evaluation_gpu_cap"]
        command.append("--array=" + ",".join(map(str, indices)) + "%" + str(cap))
    command.append(str(ROOT / "slurm" / filename))
    return command

def submit():
    report = preflight(banks=True); signature = report["signature"]
    final = ROOT / "results/STATUS.json"
    if final.exists() and load_json(final).get("status") == "COMPLETE":
        saved = load_json(final)
        require(saved["signature"] == signature and all(sha256_file(ROOT / "results" / p) == h for p, h in saved["artifacts"].items()),
                "Completed result drift; refusing overwrite")
        print("ALREADY_COMPLETE; no jobs submitted"); return
    for name in ("logs", "reports"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = load_json(receipt_path) if receipt_path.exists() else None
        if old:
            require(old["signature"] == signature, "Previous submission identity drift")
            require(old["status"] not in {"SUBMITTING", "AMBIGUOUS_SBATCH_RESPONSE"},
                    "Prior sbatch response is unresolved; refusing possible duplicate submission")
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"],
                               capture_output=True, text=True, timeout=30)
        require(queue.returncode == 0, "squeue unavailable; no jobs submitted: " + queue.stderr.strip())
        if old:
            active = {line.split("|", 1)[0].split("_", 1)[0] for line in queue.stdout.splitlines()}
            require(not (active & {j["job_id"] for j in old["jobs"]}), "This batch has active jobs; do not duplicate")
        existing = existing_gpu_jobs(queue.stdout)
        train, evaluate = pending_indices(signature)
        receipt = {"status": "SUBMITTING", "signature": signature, "jobs": [], "waiting_for_existing_gpu_jobs": existing,
                   "train_indices": train, "eval_indices": evaluate, "training_gpu_cap": 3, "evaluation_gpu_cap": 4,
                   "future_seed_submitted": False, "attempt": time.time_ns()}
        if old:
            archive = ROOT / "reports/SUBMISSION_ARCHIVE" / f"{receipt['attempt']}.json"
            atomic_json(old, archive); receipt["previous_receipt"] = str(archive)
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, filename in STAGES:
            indices = train if stage == "train" else evaluate if stage == "eval" else None
            if indices == []:
                continue
            command = command_for(stage, filename, signature, dependency, existing, indices)
            receipt["pending_command"] = command
            atomic_json(receipt, receipt_path)
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout or "empty sbatch response")
            except Exception as error:
                receipt.update(status="AMBIGUOUS_SBATCH_RESPONSE", failed_stage=stage, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError("sbatch result unresolved; earlier jobs retained, no cancellation/retry: " + str(receipt_path)) from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            receipt.pop("pending_command", None); dependency = job
            atomic_json(receipt, receipt_path); print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"; atomic_json(receipt, receipt_path)
        print("3 missing patches; training <=3x4090 THEN evaluation <=4x4090; new1200 episodes; no automatic follow-up", flush=True)

def status():
    for arm in METHODS:
        path = ROOT / "training" / arm / "status.json"
        row = load_json(path) if path.exists() else {"status": "NOT_STARTED"}
        done = sum(1 for suite in SUITES for task in range(10)
                   if (p := ROOT / "evaluation" / arm / suite / f"task_{task:02d}_STATUS.json").exists()
                   and load_json(p).get("status") == "COMPLETE")
        print(arm, row.get("status"), "outer=" + str(row.get("outer", 0)), "evaluated_tasks=" + str(done) + "/40")
    final = ROOT / "results/DECISION.json"
    if final.exists():
        print(load_json(final)["recommendation"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("preflight", "prepare", "submit", "dry-run", "validate", "stats", "status"), required=True)
    action = parser.parse_args().action
    if action == "submit": submit()
    elif action == "prepare": prepare()
    elif action == "status": status()
    elif action == "preflight":
        report = preflight(banks=True)
        print("CPU_PREFLIGHT_PASS " + report["signature"] + "; no GPU/model/queue/submission")
    elif action == "dry-run":
        previous = None
        for stage, filename in STAGES:
            print(stage, command_for(stage, filename, run_signature(), dependency=previous,
                                     indices=list(range(3 if stage == "train" else 12)) if stage in ("train", "eval") else None))
            previous = "JOB_" + stage
        print("DRY_RUN_ONLY; no queue query or submission")
    else:
        from analyze import validate_training, stats
        (validate_training if action == "validate" else stats)()

if __name__ == "__main__":
    main()
