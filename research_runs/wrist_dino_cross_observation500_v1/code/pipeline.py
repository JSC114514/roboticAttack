"""User-only submission; preflight/dry-run/status never query a scheduler."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time
from common import (ROOT, METHOD, CONDITIONS, SUITES, config, load_json, atomic_json,
                    require, preflight, run_signature, sha256_file)

STAGES = (("prepare", "01_prepare_cpu.sbatch"), ("train", "02_train_4090.sbatch"),
          ("validate", "03_validate_cpu.sbatch"), ("eval", "04_eval_4090.sbatch"), ("stats", "05_stats_cpu.sbatch"))

def existing_gpu_jobs(text):
    result = set()
    for line in text.splitlines():
        fields = line.split("|"); require(len(fields) == 4, "Unknown Slurm queue format")
        job, state, tres, reason = fields
        if state in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT") or "gpu" not in tres.lower(): continue
        require("DependencyNeverSatisfied" not in reason, "Existing GPU chain has failed dependencies; no new submission")
        match = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(match is not None, "Unknown Slurm array identifier")
        result.add(match.group(1))
    return sorted(result, key=int)

def pending(signature):
    folder = ROOT / "training" / METHOD; path = folder / "status.json"; train = True
    if path.exists():
        v = load_json(path); require(v["signature"] == signature, "Previous training identity drift")
        train = v["status"] != "COMPLETE"
        if not train:
            require(v["outer"] == 500 and v["optimizer_updates"] == 25000
                    and (folder / "resume.pt").exists() and (folder / "outer_0500.png").exists(), "Completed training missing artifacts")
    indices = list(range(8)); validation = ROOT / "reports/AUTO_VALIDATION.json"
    if not train and validation.exists():
        v = load_json(validation); require(v["signature"] == signature and v["status"] == "COMPLETE", "Validation identity drift")
        from evaluate import task_complete
        indices = [i for i in range(8) if not all(task_complete(ROOT / "evaluation" / CONDITIONS[i//4] / SUITES[i%4],
                    task, signature, v["patches"][CONDITIONS[i//4]]["png"]) for task in range(10))]
    return train, indices

def command_for(stage, filename, signature, dependency=None, existing=(), indices=None):
    cmd = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name=dxo500_"+signature[:8]+"_"+stage]
    deps = []
    if dependency: deps.append("afterok:"+dependency)
    # CPU cache preparation can overlap old GPU work. New training must wait
    # for both its cache AND all pre-existing GPU jobs, including queued eval.
    if stage == "train" and existing: deps.append("afterany:"+":".join(existing))
    if deps: cmd.append("--dependency="+",".join(deps))
    if stage == "eval":
        require(indices, "Cannot submit empty evaluation array")
        cmd.append("--array="+",".join(map(str, indices))+"%4")
    return cmd+[str(ROOT / "slurm" / filename)]

def submit():
    report = preflight(bank=True); signature = report["signature"]
    final = ROOT / "results/STATUS.json"
    if final.exists() and load_json(final).get("status") == "COMPLETE":
        f = load_json(final)
        require(f["signature"] == signature and all(sha256_file(ROOT / "results" / p) == h for p, h in f["artifacts"].items()), "Completed result drift")
        print("ALREADY_COMPLETE; no jobs submitted"); return
    for n in ("logs", "reports"): (ROOT / n).mkdir(exist_ok=True)
    receipt_path = ROOT / "reports/SUBMISSION_RECEIPT.json"
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = load_json(receipt_path) if receipt_path.exists() else None
        if old:
            require(old["signature"] == signature, "Previous submission identity changed")
            require(old["status"] not in ("SUBMITTING", "AMBIGUOUS_SBATCH_RESPONSE"), "Prior submission unresolved; no duplicate allowed")
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"], capture_output=True, text=True, timeout=30)
        require(queue.returncode == 0, "Cannot read queue; no submission: "+queue.stderr.strip())
        if old:
            active = {l.split("|", 1)[0].split("_", 1)[0] for l in queue.stdout.splitlines()}
            require(not active.intersection(j["job_id"] for j in old["jobs"]), "This batch already has active jobs")
        existing = existing_gpu_jobs(queue.stdout); train, indices = pending(signature)
        receipt = {"status": "SUBMITTING", "signature": signature, "jobs": [], "existing_gpu_dependencies": existing,
                   "train_needed": train, "eval_indices": indices, "attempt": time.time_ns(),
                   "resources": config()["resources"], "future_seed_submitted": False}
        if old:
            path = ROOT / "reports/SUBMISSION_ARCHIVE" / f"{receipt['attempt']}.json"
            atomic_json(old, path); receipt["previous_receipt"] = str(path)
        atomic_json(receipt, receipt_path); dependency = None
        for stage, filename in STAGES:
            if stage == "train" and not train: continue
            if stage == "eval" and not indices: continue
            # If training is already done, eval still waits for old GPU work.
            command = command_for(stage, filename, signature, dependency, existing if train else (), indices)
            if stage == "eval" and not train and existing:
                dep = next((p for p in command if p.startswith("--dependency=")), None)
                extra = "afterany:"+":".join(existing)
                if dep: command[command.index(dep)] = dep+","+extra
                else: command.insert(-1, "--dependency="+extra)
            receipt["pending_command"] = command; atomic_json(receipt, receipt_path)
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout or "empty sbatch response")
            except Exception as error:
                receipt.update(status="AMBIGUOUS_SBATCH_RESPONSE", failed_stage=stage, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError("Submission unresolved; no automatic retry/cancellation. Inspect "+str(receipt_path)) from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            receipt.pop("pending_command", None); dependency = job; atomic_json(receipt, receipt_path)
            print(stage, job, flush=True)
        receipt["status"] = "SUBMITTED"; atomic_json(receipt, receipt_path)
        print("One new training patch; 160 new episodes; GPU training waits for existing GPU chain; no automatic follow-up", flush=True)

def status():
    path = ROOT / "training" / METHOD / "status.json"
    print(load_json(path) if path.exists() else {"training": "NOT_STARTED"})
    for c in CONDITIONS:
        done = sum(1 for suite in SUITES for task in range(10)
            if (p := ROOT / "evaluation" / c / suite / f"task_{task:02d}_STATUS.json").exists() and load_json(p).get("status") == "COMPLETE")
        print(c, "evaluated_tasks", done, "/40")
    p = ROOT / "results/DECISION.json"
    if p.exists(): print(load_json(p))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--action", required=True, choices=("preflight", "dry-run", "status", "submit")); a = p.parse_args()
    if a.action == "submit": submit()
    elif a.action == "status": status()
    elif a.action == "preflight":
        r = preflight(bank=True); print("CPU_PREFLIGHT_PASS", r["signature"], "no model/GPU/queue/submission")
    else:
        previous = None
        for stage, filename in STAGES:
            print(stage, command_for(stage, filename, run_signature(), previous, indices=list(range(8)) if stage == "eval" else None))
            previous = "JOB_"+stage
        print("DRY_RUN_ONLY", config()["resources"], "160 new episodes; no scheduler query")

if __name__ == "__main__": main()
