"""Submission requires explicit --submit; active jobs and finished stages are skipped safely."""
from __future__ import annotations
import argparse
import getpass
import re
import subprocess
import time
from run_core import ARMS, ROOT, SUITES, locked, preflight, read_json, require, signature, stage_done, write_json

STAGES = [
    ("B_train", "01_B_train.sbatch", []),
    ("teacher_head", "02_teacher_head.sbatch", []),
    ("C_train", "03_C_train.sbatch", ["teacher_head"]),
    ("B_eval", "04_B_eval.sbatch", ["B_train"]),
    ("C_eval", "05_C_eval.sbatch", ["C_train", "B_eval"]),
    ("summary", "06_summary.sbatch", ["B_eval", "C_eval"]),
]


def complete(stage):
    if stage in ["B_train", "C_train"]:
        arm = ARMS[0 if stage.startswith("B") else 1]
        return stage_done(ROOT / "training" / arm / "STATUS.json")
    if stage == "teacher_head":
        return stage_done(ROOT / "teacher/STATUS.json") and stage_done(ROOT / "readout/STATUS.json")
    if stage.endswith("eval"):
        arm = ARMS[0 if stage.startswith("B") else 1]
        return all(stage_done(ROOT / "evaluation" / arm / suite / f"task_{task:02d}_STATUS.json") for suite in SUITES for task in range(10))
    return stage_done(ROOT / "results/STATUS.json")


def job_command(stage, file, dependencies, prefix):
    cmd = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name=" + prefix + stage]
    if dependencies:
        cmd.append("--dependency=afterok:" + ":".join(dependencies))
    return cmd + [str(ROOT / "slurm" / file)]


def ensure_no_active(queue_text, prefix):
    for line in queue_text.splitlines():
        name = line.split("|", 1)[0].strip()
        require(not name.startswith(prefix), "This experiment already has active jobs; no duplicate submission")


def submit():
    with locked(ROOT / "reports/SUBMISSION.lock"):
        report = preflight()
        if complete("summary"):
            print("ALREADY_COMPLETE; no jobs submitted")
            return
        prefix = "dar1_" + signature()[:8] + "_"
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%100j|%i|%T"], capture_output=True, text=True, timeout=20)
        require(queue.returncode == 0, "Queue unavailable; no jobs submitted: " + queue.stderr.strip())
        ensure_no_active(queue.stdout, prefix)
        receipt = {"signature": signature(), "status": "SUBMITTING", "jobs": [], "skipped_completed": [],
                   "new_rollouts": 160, "reused_uada": True, "max_own_pipeline_gpus": 3,
                   "second_seed_auto_submission": False}
        archive = ROOT / "reports/submissions" / f"{time.time_ns()}.json"
        def persist():
            write_json(receipt, ROOT / "reports/SUBMISSION_RECEIPT.json")
            write_json(receipt, archive)
        persist()
        jobs = {}
        for stage, file, upstream in STAGES:
            if complete(stage):
                receipt["skipped_completed"].append(stage); persist()
                continue
            cmd = job_command(stage, file, [jobs[x] for x in upstream if x in jobs], prefix)
            receipt["pending_stage"] = stage; persist()
            try:
                response = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                job_id = response.stdout.strip().split(";", 1)[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job_id), "sbatch failed: " + (response.stderr or response.stdout))
            except BaseException as error:
                receipt.update(status="SUBMISSION_FAILED", error=repr(error)); persist()
                raise
            jobs[stage] = job_id
            receipt["jobs"].append({"stage": stage, "job_id": job_id, "command": cmd}); persist()
            print(stage, job_id, flush=True)
        receipt.pop("pending_stage", None)
        receipt["status"] = "SUBMITTED"; persist()


def status():
    paths = {"teacher": ROOT / "teacher/STATUS.json", "head": ROOT / "readout/STATUS.json",
             **{arm: ROOT / "training" / arm / "STATUS.json" for arm in ARMS}, "results": ROOT / "results/STATUS.json"}
    for name, path in paths.items():
        print(name, read_json(path) if path.exists() else "NOT_STARTED")
    for arm in ARMS:
        paths = list((ROOT / "evaluation" / arm).glob("*/task_*_STATUS.json"))
        values = [read_json(p) for p in paths]
        print("evaluation", arm, "complete_tasks", sum(v["status"] == "COMPLETE" for v in values), "/40")
    for path in sorted((ROOT / "reports/FAILURES").glob("*.json")):
        print("failure_evidence", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--submit", action="store_true"); group.add_argument("--status", action="store_true")
    args = p.parse_args()
    if args.submit:
        submit()
    elif args.status:
        status()
    else:
        report = preflight()
        print("CPU_PREFLIGHT_PASS", report["signature"], "train/val windows", report["head_train_windows"], report["head_validation_windows"])
        print("No GPU experiments or submissions. Planned DAG:", STAGES)
