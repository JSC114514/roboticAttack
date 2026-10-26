"""Explicit manual submission only; fail closed on unknown queue / submission state."""
from __future__ import annotations
import argparse
import getpass
import math
import re
import subprocess
import time
from loop_core import (CONDITIONS, ROOT, SUITES, cfg, locked, native, preflight, read_json,
                       require, signature, stage_done, write_json)
from qa_reuse import qa_done, metadata_path

STAGES = (("qa", "01_qa.sbatch", ()), ("eval", "02_eval.sbatch", ("qa",)),
          ("stats", "03_stats.sbatch", ("eval",)))


def complete(stage):
    if stage == "qa":
        return all(qa_done(s) for s in SUITES)
    if stage == "eval":
        return all(stage_done(ROOT / "evaluation" / s / f"task_{t:02d}" / c / "STATUS.json")
                   for s in SUITES for t in cfg()["tasks"] for c in CONDITIONS)
    return stage_done(ROOT / "results/STATUS.json")


def ensure_no_active(queue_text, prefix):
    for line in queue_text.splitlines():
        name = line.split("|", 1)[0].strip()
        require(not name.startswith(prefix), "This experiment already has active jobs; no duplicate submission")


def pending_qa_indices():
    return [i for i, suite in enumerate(SUITES) if not qa_done(suite)]


def job_command(stage, file, dependencies, prefix):
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--job-name=" + prefix + stage]
    if stage == "qa":
        indices = pending_qa_indices()
        require(bool(indices), "No unfinished QA suites to submit")
        command.append("--array=" + ",".join(map(str, indices)) + "%" + str(min(4, len(indices))))
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    return command + [str(ROOT / "slurm" / file)]


def submit():
    with locked(ROOT / "reports/SUBMISSION.lock"):
        preflight()
        if complete("stats"):
            print("ALREADY_COMPLETE: no jobs submitted"); return
        prefix = "dcgf1_" + signature()[:8] + "_"
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%100j|%i|%T"],
                               capture_output=True, text=True, timeout=20)
        require(queue.returncode == 0, "Queue unavailable; no jobs submitted: " + queue.stderr.strip())
        ensure_no_active(queue.stdout, prefix)
        ensure_no_active(queue.stdout, "dcg1_429f3668_")
        path = ROOT / "reports/SUBMISSION_RECEIPT.json"
        if path.exists():
            previous = read_json(path)
            require(not previous.get("submission_uncertain", False),
                    "Previous sbatch response was ambiguous; inspect Slurm accounting before another submission")
        receipt = {"signature": signature(), "status": "SUBMITTING", "jobs": [], "skipped_completed": [],
                   "max_own_pipeline_gpus": 4, "new_episodes": 160, "new_training_jobs": 0,
                   "automatic_followup": False, "qa_indices_to_run": pending_qa_indices(),
                   "reused_qa_suites": [s for s in SUITES if s not in [SUITES[i] for i in pending_qa_indices()]]}
        archive = ROOT / "reports/submissions" / f"{time.time_ns()}.json"
        def persist():
            write_json(receipt, path); write_json(receipt, archive)
        persist()
        jobs = {}
        for stage, file, upstream in STAGES:
            if complete(stage):
                receipt["skipped_completed"].append(stage); persist(); continue
            command = job_command(stage, file, [jobs[u] for u in upstream if u in jobs], prefix)
            receipt.update(pending_stage=stage, submission_uncertain=True); persist()
            try:
                response = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = response.stdout.strip().split(";", 1)[0]
                if response.returncode != 0:
                    # A normal sbatch rejection is known; a timeout / undecodable response is not.
                    receipt["submission_uncertain"] = False
                require(response.returncode == 0 and re.fullmatch(r"\d+", job),
                        "sbatch failed or returned an ambiguous response: " + (response.stderr or response.stdout))
            except BaseException as error:
                receipt.update(status="SUBMISSION_FAILED", error=repr(error)); persist(); raise
            jobs[stage] = job
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            receipt["submission_uncertain"] = False
            persist(); print(stage, job, flush=True)
        receipt.pop("pending_stage", None)
        receipt["status"] = "SUBMITTED"; persist()


def estimate():
    completed = []
    for suite in SUITES:
        if not qa_done(suite):
            print(suite, "GPU QA not run; no measured timing yet"); continue
        qa = read_json(metadata_path(suite))
        limit = native().config()["evaluation"]["max_steps_by_suite"][suite]
        plans = math.ceil(limit / 16)
        native_call = qa["native_policy_seconds"] / qa["policy_calls"]
        # All episodes run to their step limit: empirical projection, queue excluded.
        seconds = 10 * (2 * plans * qa["plan_receipt"]["elapsed_s"] + 2 * math.ceil(limit / 8) * native_call) * 1.25 + 600
        completed.append(seconds)
        print(suite, f"projected suite worker <=about {seconds / 3600:.2f} h if every episode reaches its limit")
    if completed:
        print("4-GPU parallel empirical projection, excluding queue:", round(max(completed) / 3600, 2), "h")


def status():
    for suite in SUITES:
        path = ROOT / "qa" / suite / "STATUS.json"
        state = read_json(path).get("status") if path.exists() else ("REUSED_COMPLETE" if qa_done(suite) else "NOT_STARTED")
        print("qa", suite, state)
        count = 0
        for task in cfg()["tasks"]:
            for condition in CONDITIONS:
                path = ROOT / "evaluation" / suite / f"task_{task:02d}" / condition / "STATUS.json"
                if not path.exists():
                    continue
                record = read_json(path)
                count += record.get("status") == "COMPLETE"
                if record.get("status") != "COMPLETE":
                    print("episode", suite, task, condition, record.get("status"), record.get("error", ""))
                    journal = ROOT / record.get("attempt", "missing") / "LIVE_PLANS.jsonl"
                    if journal.exists():
                        with journal.open() as stream:
                            last = ""
                            for line in stream:
                                last = line
                        if last:
                            print("latest_plan", last.strip())
        print("formal_episodes", suite, count, "/40")
    path = ROOT / "results/STATUS.json"
    print("results", read_json(path) if path.exists() else "NOT_STARTED")
    for path in sorted((ROOT / "reports/FAILURES").glob("*.json")):
        print("failure_receipt", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--submit", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit()
    elif args.status:
        status()
    elif args.estimate:
        estimate()
    else:
        result = preflight()
        print("CPU_PREFLIGHT_PASS", result["signature"], "160 episodes, zero training, peak 4 GPUs")
        print("Verified reused QA:", list(result["repair_reuse"]["qa"]), "; pending QA:", result["pending_qa_suites"])
        print("No jobs submitted. DAG:", STAGES)
