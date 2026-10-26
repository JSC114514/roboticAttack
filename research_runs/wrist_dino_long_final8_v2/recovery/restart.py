"""Manual recovery only; do not change the frozen v2 code/config signature.

Reuse completed replay/teacher banks. The existing train.py restores each
student's own RGB, Adam, scheduler, RNG and calibration state. No fresh start,
model loading, collection, teacher retraining, cancellation or auto-submission.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import getpass
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import (METHODS, atomic_json, config, load_json, preflight, prepared,
                    require, sha256_file)
from pipeline import command_for, existing_gpu_jobs

RECEIPT_NAME = "RECOVERY_SUBMISSION_RECEIPT.json"
STAGES = (("train", "07_train_4090.sbatch"), ("validate", "08_validate_cpu.sbatch"),
          ("eval", "09_eval_3090.sbatch"), ("stats", "10_stats_cpu.sbatch"))


def recovery_preflight():
    import torch
    from train import validate_resume

    report, prep, cfg = preflight(), prepared(), config()
    signature = report["signature"]
    require(signature == prep["signature"], "Frozen experiment identity changed")
    bank = load_json(ROOT / "data/STATE_BANK_STATUS.json")
    require(bank["status"] == "COMPLETE" and bank["signature"] == signature
            and bank["sha256"] == sha256_file(ROOT / "data/STATE_BANK.pt"),
            "Replay bank is incomplete or changed; do not reconstruct it during recovery")
    for family in ("uada", "tma"):
        path = ROOT / "teachers" / f"{family}_BANK.pt"
        teacher = load_json(path.with_suffix(".json"))
        require(teacher["status"] == "COMPLETE" and teacher["signature"] == signature
                and teacher["sha256"] == sha256_file(path)
                and teacher["states"] == bank["distinct_teacher_states"],
                f"{family} teacher bank is incomplete or changed; do not retrain it")
    pending, checkpoints = [], {}
    for index, method in enumerate(METHODS):
        folder = ROOT / "training" / method
        # Missing/corrupt checkpoints must not silently become fresh training.
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(saved, method, cfg, signature)
        require(saved["patch"].shape == (3, 50, 50)
                and torch.isfinite(saved["patch"]).all()
                and saved["patch"].min() >= 0 and saved["patch"].max() <= 1,
                f"Invalid checkpoint RGB: {method}")
        for state in saved["optimizer"]["state"].values():
            for name in ("exp_avg", "exp_avg_sq"):
                require(state[name].shape == saved["patch"].shape
                        and torch.isfinite(state[name]).all(), f"Invalid Adam state: {method}")
        require(all(row["outer_iteration"] == n for n, row in enumerate(saved["rows"], 1)),
                f"Noncontiguous checkpoint history: {method}")
        status = load_json(folder / "status.json")
        require(status["signature"] == signature, f"Status identity changed: {method}")
        checkpoints[method] = saved["outer"]
        if saved["outer"] < cfg["outer_iterations"]:
            require(status["status"] != "COMPLETE", f"Completion/checkpoint conflict: {method}")
            pending.append(index)
        else:
            require(status["status"] == "COMPLETE" and status["outer"] == 2000
                    and all((folder / f"outer_{outer:04d}.pt").is_file()
                            and (folder / f"outer_{outer:04d}.png").is_file()
                            for outer in cfg["evaluation_checkpoints"]),
                    f"Final checkpoint/status conflict; inspect before resubmitting: {method}")
    final = ROOT / "results/STATUS.json"
    complete = final.exists() and load_json(final).get("status") == "COMPLETE"
    if complete:
        require(load_json(final)["signature"] == signature and not pending,
                "Final results/training identity conflict")
    return {"signature": signature, "pending": pending, "checkpoints": checkpoints,
            "already_complete": complete, "models_loaded": False, "slurm_submitted": False}


def receipt_jobs(receipt):
    if not receipt:
        return set()
    return ({str(job["job_id"]) for job in receipt.get("jobs", [])}
            | receipt_jobs(receipt.get("previous_receipt")))


def check_queue(output, owned):
    active, short_queue = set(), []
    for line in output.splitlines():
        parts = line.split("|")
        require(len(parts) == 5, "Unknown squeue format; no jobs submitted")
        job, state, tres, reason, name = (p.strip() for p in parts)
        if state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"}:
            continue
        parent = job.split("_", 1)[0]
        active.add(parent)
        require(not name.startswith("oft_long8v2_"),
                f"Long Final-8 already has active job {job}; do not duplicate submit")
        short_queue.append("|".join((job, state, tres, reason)))
    require(not (active & owned), "This batch has active jobs already; no duplicate submission")
    return existing_gpu_jobs("\n".join(short_queue))


def build_command(stage, filename, pending, dependency=None, external=()):
    command = command_for(stage, filename, dependency, external)
    if stage == "train":
        require(bool(pending), "No incomplete training tasks")
        command.insert(-1, "--array=" + ",".join(str(i) for i in pending) + "%8")
    return command


def submit():
    # Share the original lock with the full pipeline; never touch its receipt.
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = recovery_preflight()
        if report["already_complete"]:
            print("ALREADY_COMPLETE; no submission")
            return
        receipt_path = ROOT / "reports" / RECEIPT_NAME
        old = load_json(receipt_path) if receipt_path.exists() else None
        original = ROOT / "reports/SUBMISSION_RECEIPT.json"
        owned = receipt_jobs(old) | receipt_jobs(load_json(original) if original.exists() else None)
        result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R|%j"],
                                capture_output=True, text=True, check=False, timeout=20)
        require(result.returncode == 0, "squeue unavailable; no submission: " + result.stderr.strip())
        existing = check_queue(result.stdout, owned)
        receipt = {"status": "SUBMITTING", "signature": report["signature"], "jobs": [],
                   "checkpoints_at_submission": report["checkpoints"],
                   "pending_array_indices": report["pending"],
                   "waiting_for_existing_gpu_jobs": existing,
                   "caps_by_serial_phase": {"4090": 8, "3090": 8},
                   "completed_banks_reused": True,
                   "time": datetime.datetime.now().astimezone().isoformat(),
                   "recovery_source_sha256": sha256_file(Path(__file__)),
                   "entry_sha256": sha256_file(ROOT.parents[1] / "submit_exploring_oft_long_final8_2000_resume_v2.sh")}
        if old:
            receipt["previous_receipt"] = old
        atomic_json(receipt, receipt_path)
        dependency = None
        for stage, filename in STAGES:
            if stage == "train" and not report["pending"]:
                continue
            command = build_command(stage, filename, report["pending"], dependency, existing)
            try:
                response = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
                job = response.stdout.strip().split(";")[0]
                require(response.returncode == 0 and re.fullmatch(r"\d+", job),
                        response.stderr or response.stdout)
            except Exception as error:
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                               failed_command=command, error=repr(error))
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Failed at {stage}; earlier IDs retained in {receipt_path}: {error}") from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command})
            dependency = job
            atomic_json(receipt, receipt_path)
            print(f"{stage}: {job}", flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        print("RESUME_SAME_EXPERIMENT; reuse replay+teachers; <=8x4090 THEN <=8x3090 auto-test; no auto-cancel", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--preflight", action="store_true")
    flags.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.submit:
        submit()
    else:
        report = recovery_preflight()
        print("RECOVERY_CPU_PREFLIGHT_PASS: " + str(report["checkpoints"]))
        print("reuse replay+teachers; pending=" + str(report["pending"]))
        print("NO_MODEL_LOADED; NO_GPU_STARTED; NO_QUEUE_QUERY; NO_SLURM_SUBMISSION")


if __name__ == "__main__":
    main()
