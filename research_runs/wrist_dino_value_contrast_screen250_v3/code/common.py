"""Final independent screen: previous sources and episode records stay read-only."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
REFERENCE_ROOT = ROOT.parent / "wrist_dino_objectives_screen250_v2"
BASE_ROOT = ROOT.parent / "wrist_dino_feature_screen250_v1"
BASE_SIGNATURE = "590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620"
REFERENCE_SIGNATURE = "35abc0c93dad8534d5f3876c87d8d4c934d55223ba408090473e8aedf3b27542"
METHODS = ("dino_value", "advencoder_cos_b8", "advencoder_nce_b8")
REFERENCE_METHODS = ("uada_original", "dino_key", "advencoder_cos", "dino_affinity")
ALL_METHODS = (*REFERENCE_METHODS, *METHODS)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
COPIED_FILES = ("engine.py", "preprocessing.py", "source_transform.py", "model_runtime.py",
                "readonly_model.py", "deployment.py", "objectives.py", "UADA_SOURCE_COPY.py", "evaluate.py")
_spec = importlib.util.spec_from_file_location("screen250_completed_objectives_common", REFERENCE_ROOT / "code/common.py")
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
base = previous.base
require, load_json, sha256_file, digest = base.require, base.load_json, base.sha256_file, base.digest
atomic_json, atomic_torch, atomic_parquet = base.atomic_json, base.atomic_torch, base.atomic_parquet


def config():
    extra = load_json(ROOT / "config.json")
    expected = {"experiment_id": "OFT-DINO-VALUE-CONTRAST-SCREEN250-V3", "base_run": BASE_ROOT.name,
                "base_signature": BASE_SIGNATURE, "reference_run": REFERENCE_ROOT.name,
                "reference_signature": REFERENCE_SIGNATURE, "methods": list(METHODS),
                "comparison_methods": list(REFERENCE_METHODS), "value_layers": [8, 16],
                "batch_sizes": {"dino_value": 1, "advencoder_cos_b8": 8, "advencoder_nce_b8": 8},
                "nce_temperature": .1, "gradient_cache": "exact_two_pass_microbatch1", "recompute_tolerance": 1e-5,
                "resources": {"training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 3},
                              "evaluation": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 4}}}
    require(extra == expected, "Frozen final-screen configuration changed")
    return {**base.config(), **extra, "batch_size": extra["batch_sizes"]}


def sources():
    files = [*sorted((ROOT / "code").glob("*.py")), *sorted((ROOT / "vendor").glob("*")),
             *sorted((ROOT / "slurm").glob("*.sbatch")), ROOT / "slurm/environment.sh", ROOT / "config.json",
             ROOT / "PLAN.md", ROOT / "APPROVED_REUSE.json", PROJECT / "submit_exploring_oft_dino_value_contrast_screen250_v3.sh"]
    return {str(p.resolve()): sha256_file(p) for p in files if p.is_file()}


def run_signature():
    return digest({"config": config(), "references": [BASE_SIGNATURE, REFERENCE_SIGNATURE], "sources": sources()})


def verify_references():
    require(previous.run_signature() == REFERENCE_SIGNATURE, "Completed v2 source/config changed")
    previous.prepared()
    receipt = load_json(ROOT / "APPROVED_REUSE.json")
    require(receipt["reference_signature"] == REFERENCE_SIGNATURE and receipt["base_signature"] == BASE_SIGNATURE,
            "Reference receipt changed")
    for path, expected in receipt["files"].items():
        require(sha256_file(path) == expected, f"Frozen reference changed: {Path(path).name}")
    for name in COPIED_FILES:
        require((ROOT / "code" / name).read_bytes() == (REFERENCE_ROOT / "code" / name).read_bytes(),
                f"Native runtime/EOT/deployment copy changed: {name}")
    for name in ("code/feature_objectives.py", "vendor/adv_gan.py", "vendor/THIRD_PARTY_NOTICES.txt"):
        require((ROOT / name).read_bytes() == (BASE_ROOT / name).read_bytes(), "Successful generator source copy changed")
    status = load_json(REFERENCE_ROOT / "results/STATUS.json")
    require(status["status"] == "COMPLETE" and status["signature"] == REFERENCE_SIGNATURE and status["episodes"] == 560,
            "Completed reference grid changed")
    for field, filename in (("raw_sha256", "OBJECTIVES250_ROLLOUTS.parquet"), ("summary_sha256", "OBJECTIVES250_SUMMARY.csv"),
                            ("traces_sha256", "OBJECTIVES250_TRACES.parquet")):
        require(status[field] == sha256_file(REFERENCE_ROOT / "results" / filename), "Reference results changed")
    return receipt


def preflight():
    cfg, refs = config(), verify_references()
    for rel, expected in cfg["original_sources"].items():
        source = PROJECT / rel
        require(source.stat().st_mode & 0o222 == 0 and sha256_file(source) == expected, "Protected Exploring source changed")
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "references": refs, "methods": list(METHODS),
            "training_updates_per_method": 12500, "batch_sizes": cfg["batch_sizes"], "new_evaluation_episodes": 240,
            "reused_evaluation_episodes": 400, "combined_evaluation_episodes": 640, "training_gpu_cap": 3,
            "evaluation_gpu_cap": 4, "models_loaded": False, "slurm_submitted": False, "data_or_model_modified": False}


def prepared():
    value = load_json(ROOT / "reports/PREPARATION.json")
    require(value["signature"] == run_signature(), "New source/config drift since preparation")
    verify_references()
    return value


def load_bank():
    prepared()
    return base.load_bank()


def batch_uids(schedule, outer, method):
    require(method in METHODS and 1 <= outer <= 250 and len(schedule) == 250, "Invalid expert-record schedule")
    size = config()["batch_sizes"][method]
    result = [schedule[(size * (outer - 1) + j) % len(schedule)][0] for j in range(size)]
    require(len(set(result)) == size, "Batch contains duplicate source records")
    return result


def runtime_config(suite="libero_10"):
    return base.runtime_config(suite)


def require_gpu(stage):
    return base.require_gpu(stage)


def deployment_conditions():
    return [f"{m}@250" for m in METHODS]


def analysis_conditions():
    return ["clean", *(f"{m}@250" for m in ALL_METHODS)]


def reference_frames():
    import pandas as pd
    verify_references()
    conditions = ["clean", *(f"{m}@250" for m in REFERENCE_METHODS)]
    frames = [pd.read_parquet(REFERENCE_ROOT / "results" / name) for name in
              ("OBJECTIVES250_ROLLOUTS.parquet", "OBJECTIVES250_TRACES.parquet")]
    raw, trace = [frame.loc[frame.condition.isin(conditions)].copy() for frame in frames]
    require(len(raw) == 400 and not raw.evaluation_training_hooks.any(), "Reference grid changed")
    # Preserve v1/v2 source identity and generator repair lineage; only reuse flag changes.
    for frame in (raw, trace):
        frame["reused"] = True
    expected = raw.condition.eq("dino_affinity@250")
    require(raw.signature.eq(expected.map({True: REFERENCE_SIGNATURE, False: BASE_SIGNATURE})).all(), "Mixed reference identity changed")
    return raw, trace
