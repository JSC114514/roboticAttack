"""Frozen input contracts for the counted-prefix repair experiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from dino_replay_common import ROOT, SOURCE_SHA, BANK_SHA, source_records
from dino_transfer_common import draw_record as old_draw
from white_patch.UADA_OFT_DINO import require, sha256_file

CONFIG = Path(os.environ.get("OFT_DINO_REPAIR_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_repair1000_v1.json"))
METHODS = ("clear", "uniform_clear", "dino_causal_clear", "dino_repair_clear")


def frozen_digest(cfg, path):
    path = Path(path)
    matches = [digest for name, digest in cfg["frozen_inputs"].items()
               if Path(name).resolve() == path.resolve()]
    require(len(matches) == 1 and sha256_file(path) == matches[0],
            f"Frozen input missing/changed: {path}")
    return matches[0]


def config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    require(cfg["experiment_id"] == "OFT-DINO-REPAIR1000-V1"
            and tuple(cfg["methods"]) == METHODS and cfg["seed"] == 42,
            "Repair identity/seed changed")
    require(cfg["base_outer_iterations"] == 0 and cfg["bootstrap_outer"] == 250
            and cfg["outer_iterations"] == 1000 and cfg["scheduler_total_outer"] == 2000
            and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500, 750, 1000]
            and cfg["refresh_after_outer"] == [500],
            "Counted prefix/budget/schedule drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0] and cfg["clip_length"] == 3
            and cfg["centers_per_trajectory"] == 6 and cfg["learning_rate"] == .002
            and cfg["weight_decay"] == 0 and cfg["warmup_outer_iterations"] == 20
            and cfg["geometry"] is True and cfg["batch_size"] == 1
            and cfg["tv_threshold"] == .4 and cfg["tv_weight"] == 1
            and cfg["num_open_loop_steps"] == 8 and cfg["dtype"] == "bfloat16",
            "Repair threat/optimizer drift")
    require(cfg["repair_steps"] == 2 and cfg["repair_epsilon"] == 4/255
            and cfg["repair_step_size"] == 2/255 and cfg["appearance_grad_ratio"] == .15
            and cfg["hf_sigma"] == .8 and cfg["hf_kernel"] == 5,
            "Repair/appearance parameters changed")
    require(Path(cfg["output_root"]).resolve() ==
            (ROOT / "research_runs/wrist_dino_repair_v1").resolve()
            and cfg["auto_eval_after_training"] is True
            and cfg["eval_hooks"] is False
            and cfg["eval_quantization"] == "round(clamp(P,0,1)*255).uint8",
            "Repair output/deployment drift")
    require(not any(k in cfg for k in ("baseline_patch", "baseline_patch_sha256",
                                       "baseline_elapsed_s")), "Trained warm start prohibited")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_SHA and source.stat().st_mode & 0o222 == 0,
            "Read-only Exploring source changed")
    require(frozen_digest(cfg, cfg["training_bank"]) == cfg["training_bank_sha256"] == BANK_SHA
            and frozen_digest(cfg, cfg["eval_reference"]) == cfg["eval_reference_sha256"],
            "Clean data/evaluation reference drift")
    return cfg


def prefix_state(cfg):
    old_path = Path(cfg["prefix_config"])
    frozen_digest(cfg, old_path)
    old = json.loads(old_path.read_text())
    transfer_root = ROOT / "research_runs/wrist_dino_transfer_v1"
    require(old["experiment_id"] == "OFT-DINO-TRANSFER2000-V1"
            and old["outer_iterations"] == cfg["scheduler_total_outer"] == 2000
            and old["bootstrap_outer"] == 250
            and Path(old["output_root"]).resolve() == transfer_root.resolve(),
            "Only the frozen random 250 prefix is allowed")
    same = ("seed", "data_seed", "rollout_seed", "training_bank", "training_bank_sha256",
            "patch_shape", "patch_xy", "maskidx", "inner_loop", "learning_rate", "weight_decay",
            "warmup_outer_iterations", "geometry", "resize_patch", "tv_threshold", "tv_weight",
            "action_head_q01", "action_head_q99", "clip_length", "centers_per_trajectory",
            "dtype", "checkpoint", "num_open_loop_steps")
    require(all(old[key] == cfg[key] for key in same), "Prefix shape/hyperparameter drift")
    resume, patch_path = Path(cfg["prefix_resume"]), Path(cfg["prefix_patch"])
    require(resume.resolve() == (transfer_root / "training/warmup/resume.pt").resolve()
            and patch_path.resolve() == (transfer_root / "training/warmup/outer_0250.pt").resolve(),
            "A 1000/2000 warm start is forbidden")
    digest = frozen_digest(cfg, resume)
    patch_digest = frozen_digest(cfg, patch_path)
    status_path = resume.with_name("status.json")
    frozen_digest(cfg, status_path)
    status = json.loads(status_path.read_text())
    require(status["status"] == "COMPLETE" and status["method"] == "warmup"
            and status["outer"] == 250 and status["optimizer_updates"] == 12500
            and status["resume_sha256"] == digest
            and status["final_patch_sha256"] == patch_digest, "Invalid completed prefix")
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    require(saved["config"] == old and saved["signature"] == status["signature"]
            and saved["method"] == "warmup" and saved["outer"] == 250
            and len(saved["rows"]) == 250 and saved["scheduler"]["last_epoch"] == 250,
            "Prefix full resume state changed")
    states = saved["optimizer"]["state"]
    require(len(states) == 1 and all(int(s["step"]) == 12500
            and "exp_avg" in s and "exp_avg_sq" in s for s in states.values())
            and saved.keys() >= {"torch_rng", "numpy_rng", "python_rng", "cuda_rng"},
            "Prefix optimizer/RNG reset or missing")
    require(torch.equal(saved["patch"], torch.load(patch_path, map_location="cpu",
            weights_only=False)["patch"]), "Prefix RGB mismatch")
    return saved, digest, patch_path


def replay_records(cfg):
    path = Path(cfg["replay_bank"])
    require(path.resolve() == (ROOT / "research_runs/wrist_dino_transfer_v1/data/WINDOW_BANK.pt").resolve(),
            "Replay source changed")
    digest = frozen_digest(cfg, path)
    status_path = path.with_name("STATUS.json")
    frozen_digest(cfg, status_path)
    status = json.loads(status_path.read_text())
    require(status["status"] == "COMPLETE" and status["bank_sha256"] == digest
            and status["config_sha256"] == frozen_digest(cfg, cfg["prefix_config"])
            and status["baseline_patch_sha256"] == frozen_digest(cfg, cfg["prefix_patch"])
            and status["source_bank_sha256"] == cfg["training_bank_sha256"],
            "Replay bank prefix lineage changed")
    records = torch.load(path, map_location="cpu", weights_only=False)["records"]
    require(len(records) == status["records"] and {int(r["task_id"]) for r in records} == set(range(10))
            and {r["phase_proxy"] for r in records} == {"early", "middle", "late"},
            "Replay coverage changed")
    return records, digest


def draw_record(clean, replay, method, outer, cfg):
    require(method in METHODS and 251 <= outer <= 1000, "Invalid repair method/outer")
    return old_draw(clean, replay, "uniform" if method == "uniform_clear" else "diverse", outer, cfg)


def load_calibration(cfg):
    path = Path(cfg["output_root"]) / "reports/APPEARANCE_CALIBRATION.json"
    report = json.loads(path.read_text())
    require(report["status"] == "COMPLETE" and report["config"] == cfg,
            "Calibration not complete/config changed")
    expected = {"clean_bank": cfg["training_bank_sha256"], "config": sha256_file(CONFIG),
                "prefix_resume": frozen_digest(cfg, cfg["prefix_resume"]),
                "window_bank": frozen_digest(cfg, cfg["replay_bank"])}
    require(report["data"] == expected, "Calibration data lineage changed")
    for name, digest in report["source"].items():
        require(sha256_file(name) == digest, f"Calibration source changed: {name}")
    for name in ("lambda_tv", "lambda_hf"):
        require(math.isfinite(float(report[name])) and float(report[name]) > 0,
                "Invalid calibrated appearance weight")
    return report

def submit_chain(cfg):
    """User-run submission only; every successful receipt is recorded before next stage."""
    import re
    import subprocess
    from white_patch.UADA_OFT_DINO import atomic_json
    root = Path(cfg["output_root"])
    final = root / "evaluation/four_suite/STATUS.json"
    if final.exists() and json.loads(final.read_text()).get("status") == "COMPLETE":
        print(f"ALREADY_COMPLETE: {root}")
        return {"status": "ALREADY_COMPLETE", "jobs": []}
    jobnames = "oft_repair_calibrate,oft_repair_train,oft_repair_validate,oft_repair_eval,oft_repair_stats"
    queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", jobnames],
                           check=False, capture_output=True, text=True)
    require(queue.returncode == 0,
            "squeue query failed; cannot safely determine duplicate chains: " + queue.stderr.strip())
    require(not queue.stdout.strip(),
            "An earlier repair chain remains queued. Failed afterok dependencies can show "
            "DependencyNeverSatisfied: inspect its receipt and failed logs; resolve the queued chain "
            "before rerunning. Completed methods/shards are reused on rerun.")
    receipt = {"status": "SUBMITTING", "experiment_id": cfg["experiment_id"],
               "config_sha256": sha256_file(CONFIG), "jobs": [],
               "dependency_failure_help": "afterok prevents evaluation after failed training. "
                "DependencyNeverSatisfied is not a GPU failure. Inspect failed upstream logs; "
                "after resolving that queued chain, rerun this helper to resume complete outer checkpoints."}
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    receipt_path = root / "reports/SUBMISSION_RECEIPT.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        receipt["previous_receipt"] = previous
    atomic_json(receipt, receipt_path)

    def submit(stage, script, dependency=None, array=None):
        command = ["sbatch", "--parsable"]
        if dependency:
            require(bool(re.fullmatch(r"[0-9]+", str(dependency))), "Invalid dependency job ID")
            command += [f"--dependency=afterok:{dependency}"]
        if array:
            command += [f"--array={array}"]
        command += [str(ROOT / "slurm/exploring_oft_dino" / script)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        raw = result.stdout.strip()
        job_id = raw.split(";")[0]
        if result.returncode or not re.fullmatch(r"[0-9]+", job_id):
            receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                           error=result.stderr.strip() or f"Invalid sbatch receipt: {raw}")
            atomic_json(receipt, receipt_path)
            raise RuntimeError(f"Submission failed at {stage}; no downstream job submitted. "
                               f"Successful IDs remain in {receipt_path}: {receipt['error']}")
        receipt["jobs"].append({"stage": stage, "job_id": job_id, "dependency": dependency,
                                "command": command})
        atomic_json(receipt, receipt_path)
        return job_id

    calibration = root / "reports/APPEARANCE_CALIBRATION.json"
    dependency = None
    if calibration.exists() and json.loads(calibration.read_text()).get("status") == "COMPLETE":
        load_calibration(cfg)
    else:
        dependency = submit("calibrate", "102_repair1000_calibrate_4090.sbatch")
    pending = [i for i, m in enumerate(METHODS)
               if not (root / "training" / m / "status.json").exists()
               or json.loads((root / "training" / m / "status.json").read_text()).get("status") != "COMPLETE"]
    if pending:
        dependency = submit("train", "103_repair1000_train_4090.sbatch", dependency,
                            ",".join(map(str, pending)) + "%4")
    dependency = submit("validate", "104_repair1000_validate_cpu.sbatch", dependency)
    dependency = submit("eval", "105_repair1000_eval_3090.sbatch", dependency)
    submit("stats", "106_repair1000_stats_cpu.sbatch", dependency)
    receipt["status"] = "SUBMITTED"
    atomic_json(receipt, receipt_path)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", required=True)
    parser.parse_args()
    submit_chain(config())
