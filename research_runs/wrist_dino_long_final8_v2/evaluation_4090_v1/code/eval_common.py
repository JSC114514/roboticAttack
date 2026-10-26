"""4090 evaluation-only override; all parent training/science assets stay frozen."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent
sys.path.insert(0, str(SOURCE_ROOT / "code"))
import common as original
sys.path.insert(0, str(ROOT / "code"))

PROJECT = original.PROJECT
METHODS, SUITES = original.METHODS, original.SUITES
config = original.config
runtime_config = original.runtime_config
deployment_conditions = original.deployment_conditions
atomic_json, atomic_parquet, atomic_torch = original.atomic_json, original.atomic_parquet, original.atomic_torch
digest, load_json, require, sha256_file = original.digest, original.load_json, original.require, original.sha256_file
source_signature = original.run_signature
ENTRY = PROJECT / "submit_exploring_oft_long_final8_eval4090_v1.sh"
EVALUATION_RESOURCES = {"partition": "rtx", "gpu_type": "4090", "gres": "gpu:4090:1",
                        "max_concurrent": 8, "override_scope": "evaluation_only"}


def sources():
    paths = list((ROOT / "code").glob("*.py")) + list((ROOT / "slurm").glob("*.sbatch"))
    paths += [ROOT / "slurm/environment.sh", ENTRY]
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def run_signature():
    return digest({"source_training_signature": source_signature(), "sources": sources(),
                   "evaluation_resources": EVALUATION_RESOURCES})


def require_gpu(stage):
    import torch
    require(stage == "evaluation", "This entry cannot run collection, teachers or training")
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, "One user-submitted Slurm CUDA GPU required")
    gpu = torch.cuda.get_device_name(0)
    require("4090" in gpu, "Actual 4090 required for this evaluation; no hidden 3090/CPU fallback")
    os.environ["OFT_STAGE_GPU"] = "4090"
    return gpu


def source_assets():
    original.prepared()
    signature = source_signature()
    path = SOURCE_ROOT / "reports/AUTO_VALIDATION.json"
    validation = load_json(path)
    expected = deployment_conditions()[1:]
    require(validation["status"] == "COMPLETE" and validation["signature"] == signature
            and set(validation["patches"]) == set(expected), "Parent deployment assets not complete")
    paths = [path, original.CONFIG, SOURCE_ROOT / "reports/PREPARATION.json"]
    for method in METHODS:
        folder = SOURCE_ROOT / "training" / method
        status = load_json(folder / "status.json")
        require(status["status"] == "COMPLETE" and status["outer"] == 2000
                and status["optimizer_updates"] == 100000 and status["signature"] == signature,
                f"Parent training incomplete: {method}")
        paths.append(folder / "status.json")
    for condition in expected:
        method, outer = condition.split("@")
        for suffix in ("pt", "png"):
            path = SOURCE_ROOT / "training" / method / f"outer_{int(outer):04d}.{suffix}"
            require(sha256_file(path) == validation["patches"][condition][suffix],
                    f"Parent snapshot changed: {condition}/{suffix}")
            paths.append(path)
    return {str(path.resolve()): sha256_file(path) for path in paths}


def preflight():
    assets = source_assets()
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(),
            "source_training_signature": source_signature(), "sources": sources(), "source_assets": assets,
            "evaluation_resources": dict(EVALUATION_RESOURCES), "evaluation_episodes": 10000,
            "evaluation_settings": config()["evaluation"], "training_restarted": False,
            "models_loaded": False, "renderer_started": False, "slurm_submitted": False}


def prepare():
    report = preflight()
    path = ROOT / "reports/PREPARATION.json"
    if path.exists():
        require(load_json(path) == report, "4090 preparation changed; do not overwrite provenance")
    else:
        atomic_json(report, path)
    print("EVAL4090_PREPARED; reuse all 24 parent PNGs; no training; 10000 episodes", flush=True)


def prepared():
    report = load_json(ROOT / "reports/PREPARATION.json")
    require(report == preflight(), "4090 runtime, parent assets or evaluation settings changed")
    return report


if __name__ == "__main__":
    prepare()
