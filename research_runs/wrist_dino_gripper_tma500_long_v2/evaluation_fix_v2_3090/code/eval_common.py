"""Evaluation-only repair contracts; original trained experiment stays frozen."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGIN_ROOT = ROOT.parent
sys.path.insert(0, str(ORIGIN_ROOT / "code"))
import gripper_common as original
# Restore repair import precedence; never alias/patch the frozen module.
sys.path.insert(0, str(ROOT / "code"))

PROJECT = original.PROJECT
CONFIG = original.CONFIG
METHODS = original.METHODS
CONDITIONS = original.CONDITIONS
DENOMINATORS = original.DENOMINATORS
REPAIR_ID = "LONG-V2-EVALUATION-EGL-FIX2-3090"
ENTRY = PROJECT / "submit_exploring_oft_gripper_tma500_long_v2_evalfix2_3090.sh"
atomic_json = original.atomic_json
atomic_parquet = original.atomic_parquet
atomic_torch = original.atomic_torch
config = original.config
condition_metadata = original.condition_metadata
digest = original.digest
load_json = original.load_json
reference_patch = original.reference_patch
require = original.require
runtime_config = original.runtime_config
sha256_file = original.sha256_file
EVALUATION_RESOURCES = {
    "partition": "rtx", "gres": "gpu:3090:1",
    "device_name": "3090", "max_concurrent": 5,
    "override_scope": "evaluation_only",
}


def require_stage_gpu(stage):
    require(stage == "evaluation", "This repair cannot run collection or training")
    return original.require_gpu(EVALUATION_RESOURCES["device_name"])


def source_manifest(wrapper=None):
    paths = {ROOT / "README.md", ROOT / "slurm/environment.sh", ENTRY}
    for directory, pattern in ((ROOT / "code", "*.py"), (ROOT / "tests", "*.py"),
                               (ROOT / "slurm", "*.sbatch")):
        paths.update(directory.glob(pattern))
    installed = Path(sys.prefix) / "lib/python3.10/site-packages/robosuite"
    paths.update(installed / name for name in (
        "utils/binding_utils.py", "renderers/context/egl_context.py", "environments/base.py"))
    paths.add(Path(config()["evaluation"]["paths"]["libero_root"]) /
              "libero/libero/envs/env_wrapper.py")
    site = Path(sys.prefix) / "lib/python3.10/site-packages"
    paths.update(site / name for name in (
        "mujoco/egl/egl_ext.py", "mujoco/egl/__init__.py",
        "OpenGL/platform/egl.py", "OpenGL/platform/__init__.py"))
    return {**original.source_manifest(wrapper),
            **{str(path.resolve()): sha256_file(path) for path in sorted(paths)}}


def trained_inventory():
    paths = {ORIGIN_ROOT / "reports/PREPARATION.json", ORIGIN_ROOT / "reports/AUTO_VALIDATION.json",
             ORIGIN_ROOT / "data/POOL.json"}
    for method in METHODS:
        folder = ORIGIN_ROOT / "training" / method
        paths.update(folder / name for name in
                     ("RUN_MANIFEST.json", "GPU_INTERFACE.json", "status.json", "TRAIN_HISTORY.parquet"))
        paths.update(folder.glob("outer_*.pt"))
        paths.update(folder.glob("outer_*.png"))
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def preflight():
    """Read-only CPU validation; no renderer, CUDA model, scheduler or writes."""
    from manage import validated_training
    origin_preparation = original.preparation()
    expected = original.load_json(ORIGIN_ROOT / "reports/AUTO_VALIDATION.json")
    require(validated_training(write=False) == expected, "Original validated training changed")
    return {"status": "CPU_PREFLIGHT_PASS", "repair_id": REPAIR_ID,
            "config_sha256": sha256_file(CONFIG),
            "original_preparation_signature": digest(origin_preparation),
            "trained_inventory": trained_inventory(), "source": source_manifest(),
            "evaluation_episodes": 2000, "smoke_paired_episodes_reused": 20,
            "maximum_gpu_concurrency": 5, "gpu_type": "3090",
            "evaluation_resources": dict(EVALUATION_RESOURCES),
            "training_restarted": False, "gpu_started": False}


def prepare():
    report = preflight()
    path = ROOT / "reports/PREPARATION.json"
    if path.exists():
        require(load_json(path) == report, "Repair preparation changed; do not overwrite provenance")
    else:
        atomic_json(report, path)
    print("EVALFIX_PREPARED; preserved all three trained patches; no training", flush=True)


def preparation():
    report = load_json(ROOT / "reports/PREPARATION.json")
    require(report["status"] == "CPU_PREFLIGHT_PASS" and report["repair_id"] == REPAIR_ID
            and report["config_sha256"] == sha256_file(CONFIG)
            and report["evaluation_resources"] == EVALUATION_RESOURCES
            and report["gpu_type"] == "3090" and report["maximum_gpu_concurrency"] == 5
            and report["source"] == source_manifest()
            and report["original_preparation_signature"] == digest(original.preparation())
            and report["trained_inventory"] == trained_inventory(),
            "Evaluation repair preparation/source/patches changed")
    return report


if __name__ == "__main__":
    prepare()
