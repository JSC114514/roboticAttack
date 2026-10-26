"""Separate runtime repair: preserve the running experiment's frozen code."""
from __future__ import annotations
import sys
import re
from pathlib import Path

REPAIR_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = REPAIR_ROOT.parent
PROJECT_ROOT = BASE_ROOT.parents[1]
BASE_SIGNATURE = "590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620"
GENERATOR = "advencoder_cos"
sys.path.insert(0, str(BASE_ROOT / "code"))
import common as base


def identity():
    files = [*sorted((REPAIR_ROOT / "code").glob("*.py")),
             *sorted((REPAIR_ROOT / "slurm").glob("*.sbatch")),
             PROJECT_ROOT / "submit_exploring_oft_dino_feature_screen250_cudnnfix_v2.sh"]
    hashes = {str(p.resolve()): base.sha256_file(p) for p in files}
    return {"base_signature": BASE_SIGNATURE, "repair_signature": base.digest({"base": BASE_SIGNATURE, "files": hashes}),
            "files": hashes, "method": GENERATOR, "change": "disable cuDNN BEFORE first generator forward",
            "scientific_config_changed": False, "data_or_model_modified": False}


def original_jobs():
    receipt = base.load_json(BASE_ROOT / "reports/SUBMISSION_RECEIPT.json")
    base.require(receipt["signature"] == BASE_SIGNATURE, "Original submission identity changed")
    jobs = {j["stage"]: j["job_id"] for j in receipt["jobs"]}
    base.require(set(jobs) == {"prepare", "train", "validate", "eval", "stats"}, "Unexpected original pipeline")
    base.require(all(re.fullmatch(r"\d+", job) for job in jobs.values()), "Invalid original Slurm IDs")
    return jobs


def preflight():
    report = base.preflight()
    base.require(report["signature"] == BASE_SIGNATURE, "Do not edit running/frozen v1 code")
    base.prepared()
    data = base.load_json(BASE_ROOT / "data/DATA_STATUS.json")
    base.require(data["status"] == "COMPLETE" and data["signature"] == BASE_SIGNATURE
                 and base.sha256_file(BASE_ROOT / "data/CANDIDATE_BANK.pt") == data["bank_sha256"], "Existing data cache changed")
    for method in ("dino_key", "dino_center"):
        status = base.load_json(BASE_ROOT / "training" / method / "status.json")
        base.require(status["status"] == "COMPLETE" and status["outer"] == 250
                     and status["signature"] == BASE_SIGNATURE, "Completed feature group must be reused intact")
    uda = base.load_json(BASE_ROOT / "training/uada_original/status.json")
    base.require(uda["signature"] == BASE_SIGNATURE and uda["status"] in {"RUNNING", "COMPLETE"},
                 "UADA stopped unexpectedly; do not silently retrain it")
    generator_folder = BASE_ROOT / "training" / GENERATOR
    if (generator_folder / "resume.pt").exists():
        base.require((generator_folder / "RUNTIME_REPAIR.json").exists()
                     and base.load_json(generator_folder / "RUNTIME_REPAIR.json") == identity(),
                     "Generator already trained outside this repair; do not overwrite")
    return {**identity(), "status": "CPU_PREFLIGHT_PASS", "original_jobs": original_jobs(),
            "uda_status": uda, "bank_sha256": data["bank_sha256"], "reused_methods": ["uada_original", "dino_key", "dino_center"],
            "generator_gpu_cap": 1, "evaluation_gpu_cap": 4, "evaluation_episodes": 400,
            "slurm_submitted": False, "models_loaded": False}


def verify_gpu_completion():
    expected = identity()
    status = base.load_json(REPAIR_ROOT / "reports/GENERATOR_STATUS.json")
    runtime = base.load_json(BASE_ROOT / "training" / GENERATOR / "RUNTIME_REPAIR.json")
    base.require(status["status"] == "COMPLETE" and status["repair_signature"] == expected["repair_signature"]
                 and runtime == expected, "Generator repair runtime incomplete or changed")
    return expected
