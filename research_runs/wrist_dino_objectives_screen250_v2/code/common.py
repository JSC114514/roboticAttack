"""Independent objectives screen; frozen previous assets are read-only."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE_ROOT = ROOT.parent / "wrist_dino_feature_screen250_v1"
BASE_SIGNATURE = "590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620"
METHODS = ("dino_attention", "dino_multilayer", "dino_affinity")
REFERENCE_METHODS = ("uada_original", "dino_key", "advencoder_cos")
ALL_METHODS = (*REFERENCE_METHODS, *METHODS)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
COPIED_FILES = ("engine.py", "preprocessing.py", "source_transform.py", "model_runtime.py",
                "readonly_model.py", "deployment.py", "objectives.py", "UADA_SOURCE_COPY.py")

_spec = importlib.util.spec_from_file_location("screen250_frozen_base_common", BASE_ROOT / "code/common.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
require, load_json, sha256_file, digest = base.require, base.load_json, base.sha256_file, base.digest
atomic_json, atomic_torch, atomic_parquet = base.atomic_json, base.atomic_torch, base.atomic_parquet


def config():
    extra = load_json(ROOT / "config.json")
    expected = {"experiment_id": "OFT-DINO-OBJECTIVES-SCREEN250-V2", "base_run": BASE_ROOT.name,
                "base_signature": BASE_SIGNATURE, "methods": list(METHODS),
                "comparison_methods": list(REFERENCE_METHODS), "attention_layers": [8, 16],
                "multilayer_layers": [8, 16, 22], "affinity_topk": 8, "objective_epsilon": 1e-6,
                "resources": {"training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 3},
                              "evaluation": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 4}}}
    require(extra == expected, "Approved objective-screen settings changed")
    return {**base.config(), **extra}


def sources():
    files = [*sorted((ROOT / "code").glob("*.py")), *sorted((ROOT / "slurm").glob("*.sbatch")),
             ROOT / "slurm/environment.sh", ROOT / "config.json", ROOT / "PLAN.md",
             ROOT / "APPROVED_REUSE.json", PROJECT / "submit_exploring_oft_dino_objectives_screen250_v2.sh"]
    return {str(p.resolve()): sha256_file(p) for p in files}


def run_signature():
    return digest({"config": config(), "base_signature": BASE_SIGNATURE, "sources": sources()})


def verify_references():
    require(base.run_signature() == BASE_SIGNATURE, "Previous frozen source/config changed; cannot reuse")
    base.prepared()  # Same original data, model JSON/code and weight size/mtime.
    receipt = load_json(ROOT / "APPROVED_REUSE.json")
    require(receipt["base_signature"] == BASE_SIGNATURE, "Reference receipt identity changed")
    for path, expected in receipt["files"].items():
        require(sha256_file(path) == expected, f"Frozen reference changed: {Path(path).name}")
    for name in COPIED_FILES:
        require((ROOT / "code" / name).read_bytes() == (BASE_ROOT / "code" / name).read_bytes(),
                f"Native runtime/EOT copy changed: {name}")
    status = load_json(BASE_ROOT / "results/STATUS.json")
    require(status["status"] == "COMPLETE" and status["signature"] == BASE_SIGNATURE
            and status["episodes"] == 400, "Previous closed-loop screen incomplete")
    require(status["raw_sha256"] == sha256_file(BASE_ROOT / "results/SCREEN250_ROLLOUTS.parquet")
            and status["summary_sha256"] == sha256_file(BASE_ROOT / "results/SCREEN250_SUMMARY.csv"),
            "Previous result hashes changed")
    return receipt


def preflight():
    cfg, refs = config(), verify_references()
    # Same protected Exploring source files, not just the derivative's name.
    for rel, expected in cfg["original_sources"].items():
        original = PROJECT / rel
        require(original.stat().st_mode & 0o222 == 0 and sha256_file(original) == expected,
                "Protected Exploring source changed or became writable")
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "base_signature": BASE_SIGNATURE,
            "sources": sources(), "references": refs, "methods": list(METHODS),
            "training_updates_per_method": 12500, "new_evaluation_episodes": 240,
            "combined_evaluation_episodes": 560, "training_gpu_cap": 3, "evaluation_gpu_cap": 4,
            "models_loaded": False, "slurm_submitted": False, "data_or_model_modified": False}


def prepared():
    value = load_json(ROOT / "reports/PREPARATION.json")
    require(value["signature"] == run_signature(), "New source/config drift since preparation")
    verify_references()
    return value


def load_bank():
    prepared()
    return base.load_bank()  # Read-only original shared 250-record schedule.


def runtime_config(suite="libero_10"):
    return base.runtime_config(suite)


def require_gpu(stage):
    return base.require_gpu(stage)


def deployment_conditions():
    """NEW episodes only; reused controls are never rerun here."""
    return [f"{m}@250" for m in METHODS]


def analysis_conditions():
    return ["clean", *(f"{m}@250" for m in ALL_METHODS)]


def reference_frames():
    import pandas as pd
    verify_references()
    conditions = ["clean", *(f"{m}@250" for m in REFERENCE_METHODS)]
    raw = pd.read_parquet(BASE_ROOT / "results/SCREEN250_ROLLOUTS.parquet")
    trace = pd.read_parquet(BASE_ROOT / "results/SCREEN250_TRACES.parquet")
    raw = raw.loc[raw.condition.isin(conditions)].copy()
    trace = trace.loc[trace.condition.isin(conditions)].copy()
    require(len(raw) == 320 and raw.signature.eq(BASE_SIGNATURE).all()
            and not raw.evaluation_training_hooks.any(), "Reference coverage/runtime changed")
    for frame in (raw, trace):
        frame["source_run"] = BASE_ROOT.name
        frame["reused"] = True
    return raw, trace
