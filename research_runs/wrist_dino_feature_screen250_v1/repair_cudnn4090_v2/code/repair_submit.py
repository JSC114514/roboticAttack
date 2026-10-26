"""Manual-only replacement chain; never resubmit original training array."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
from repair_common import REPAIR_ROOT, BASE_ROOT, base, preflight, identity
import pipeline as original_pipeline


def queue_plan(queue, jobs, uda_status):
    rows = [line.split("|") for line in queue.splitlines()]
    base.require(all(len(r) == 4 for r in rows), "Unknown queue columns")
    original_base = jobs["train"] + "_0"
    obsolete = {jobs[k] for k in ("validate", "eval", "stats")}
    remaining, baseline_active = [], False
    for job, state, tres, reason in rows:
        parent = job.split("_", 1)[0]
        if parent in obsolete:
            base.require(state in {"PENDING", "COMPLETED", "CANCELLED", "FAILED"}, "Old downstream chain is running; do not duplicate it")
            continue  # Its afterok dependency on the failed array cannot pass.
        if parent == jobs["train"]:
            base.require(job == original_base and state in {"RUNNING", "PENDING"}, "Unexpected active original training task")
            base.require("DependencyNeverSatisfied" not in reason, "Original UADA dependency failed")
            baseline_active = True
            continue  # One UADA GPU + one generator GPU <= the four-GPU cap.
        remaining.append("|".join((job, state, tres, reason)))
    if uda_status == "RUNNING":
        base.require(baseline_active, "UADA status stale but task not active; inspect before submitting")
    return original_pipeline.existing_gpu_jobs("\n".join(remaining)), original_base if baseline_active else None


def submit():
    report = preflight()
    current = report["repair_signature"]
    final = REPAIR_ROOT / "reports/RESULT_LINEAGE.json"
    if final.exists() and base.load_json(final).get("status") == "COMPLETE":
        base.require(base.load_json(final)["repair_signature"] == current, "Completed repair changed")
        print("ALREADY_COMPLETE; no submission")
        return
    for name in ("reports", "logs"):
        (REPAIR_ROOT / name).mkdir(parents=True, exist_ok=True)
    with (REPAIR_ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = REPAIR_ROOT / "reports/SUBMISSION_RECEIPT.json"
        previous = base.load_json(receipt_path) if receipt_path.exists() else None
        if previous:
            base.require(previous["repair_signature"] == current, "Repair source changed after submission")
        response = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"],
                                  capture_output=True, text=True, timeout=20)
        base.require(response.returncode == 0, "squeue failed; no jobs submitted: " + response.stderr.strip())
        active = {line.split("|", 1)[0].split("_", 1)[0] for line in response.stdout.splitlines()}
        if previous:
            base.require(not ({j["job_id"] for j in previous["jobs"]} & active), "Repair already active; do not duplicate")
        # Status may have advanced to COMPLETE during the queue query.
        uda = base.load_json(BASE_ROOT / "training/uada_original/status.json")
        base.require(uda["signature"] == report["base_signature"], "UADA identity changed during submission")
        external, baseline = queue_plan(response.stdout, report["original_jobs"], uda["status"])
        receipt = {"status": "SUBMITTING", "repair_signature": current, "base_signature": report["base_signature"],
                   "jobs": [], "waiting_for_other_gpu_jobs": external, "uda_child_dependency": baseline,
                   "generator_gpu_cap": 1, "evaluation_gpu_cap": 4, "retrained_methods": ["advencoder_cos"]}
        if previous:
            receipt["previous_receipt"] = previous
        base.atomic_json(receipt, receipt_path)
        stages = [("generator", REPAIR_ROOT / "slurm/01_generator_4090.sbatch"),
                  ("validate", REPAIR_ROOT / "slurm/02_validate_cpu.sbatch"),
                  ("eval", BASE_ROOT / "slurm/04_eval_4090.sbatch"),
                  ("stats", REPAIR_ROOT / "slurm/04_stats_cpu.sbatch")]
        prior = None
        for stage, path in stages:
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
            if prior:
                dependencies = [prior] + ([baseline] if stage == "validate" and baseline else [])
                command.append("--dependency=afterok:" + ":".join(dependencies))
            elif external:
                command.append("--dependency=afterany:" + ":".join(external))
            command.append(str(path))
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=20)
                job = response.stdout.strip().split(";")[0]
                base.require(response.returncode == 0 and re.fullmatch(r"\d+", job), response.stderr or response.stdout)
            except Exception as error:
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage, error=repr(error))
                base.atomic_json(receipt, receipt_path)
                raise
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            prior = job
            base.atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        base.atomic_json(receipt, receipt_path)
        print("GENERATOR_ONLY_REPAIR; reuse UADA/Key/dense and Long cache; then automatic400-episode evaluation <=4x4090")


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
        base.atomic_json(report, REPAIR_ROOT / "reports/CPU_PREFLIGHT.json")
        print("CPU_PREFLIGHT_PASS; reuse3 methods+cache; generator1GPU/evaluation4GPU; NO_GPU/QUEUE/SUBMISSION")
