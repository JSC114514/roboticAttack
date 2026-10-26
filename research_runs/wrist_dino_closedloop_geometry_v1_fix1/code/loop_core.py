"""Isolated outputs, immutable source receipts, and explicit GPU execution boundary."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = ROOT.parent / "wrist_dino_spatial_b12_diag500_v1"
BASE_SIGNATURE = "ac44e284cedc1bd5b47ca02702c2eeed20cffca1d2ecc6ae4505cea3f27c0349"
BASE_CONDITION = "m01_uada_original_b1@500"
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
CONDITIONS = ("clean", "uada_static", "uada_geometry", "loop_geometry")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg():
    value = read_json(ROOT / "config.json")
    require(value["suites"] == list(SUITES) and value["conditions"] == list(CONDITIONS), "Comparison grid changed")
    require(value["states"] == [40] and value["tasks"] == list(range(10)), "Pilot grid changed")
    require(value["horizon_blocks"] == value["commit_blocks"] == 2 and value["chunk_length"] == 8,
            "Approved two-block planning budget changed")
    require(len(value["candidates"]) == 7 and value["candidates"][0]["name"] == "identity", "Candidate grid changed")
    return value


@lru_cache(maxsize=1)
def native():
    sys.path.insert(0, str(BASE / "code"))
    import common
    require(Path(common.__file__).resolve() == BASE / "code/common.py", "Wrong historical runtime imported")
    require(common.run_signature() == BASE_SIGNATURE, "Historical source/config signature changed")
    return common


def source_hashes():
    files = [ROOT / "config.json", ROOT / "PLAN.md", ROOT / "run.sh"]
    for folder in ("code", "tests", "slurm", "vendor"):
        files += [p for p in (ROOT / folder).glob("*") if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".json")]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


@lru_cache(maxsize=1)
def signature():
    return hashlib.sha256(json.dumps({"sources": source_hashes(), "native": BASE_SIGNATURE}, sort_keys=True).encode()).hexdigest()


def outpath(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT), "Refuse output outside the new experiment")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic(path, writer):
    p = outpath(path)
    temporary = p.with_name(p.name + f".{os.getpid()}.tmp")
    writer(temporary)
    os.replace(temporary, p)


def write_json(value, path):
    atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"))


def save_frame(frame, path):
    atomic(path, lambda p: frame.to_parquet(p, index=False))


def save_torch(value, path):
    import torch
    atomic(path, lambda p: torch.save(value, p))


@contextmanager
def locked(path):
    import fcntl
    with outpath(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def stage_done(path):
    path = Path(path)
    if not path.exists():
        return False
    receipt = read_json(path)
    require(receipt.get("signature") == signature(), "Existing output belongs to different sources: " + str(path))
    if receipt.get("status") != "COMPLETE":
        return False
    require(bool(receipt.get("artifacts")), "Completed stage lacks artifact hashes")
    for relative, expected in receipt["artifacts"].items():
        p = ROOT / relative
        require(p.is_file() and sha(p) == expected, "Completed artifact changed: " + relative)
    return True


def finish(path, artifacts, **extra):
    write_json({"status": "COMPLETE", "signature": signature(), "completed_unix": time.time(),
                "artifacts": {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}, **extra}, path)


def external_sources():
    n = native()
    evaluation = n.runtime_config(SUITES[0])
    paths = evaluation["paths"]
    roots = [Path(paths["libero_root"]) / "libero/libero/envs",
             Path(paths["libero_root"]) / "libero/libero/bddl_files"]
    files = [PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py",
             PROJECT / "VLAAttacker/multiview/openvla_oft_experiments.py",
             Path(paths["openvla_oft_root"]) / "experiments/robot/openvla_utils.py",
             Path(paths["openvla_oft_root"]) / "prismatic/extern/hf/modeling_prismatic.py"]
    for root in roots:
        files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".bddl"))
    controllers = Path(sys.executable).parent.parent / "lib/python3.10/site-packages/robosuite"
    for folder in ("controllers", "robots", "models/grippers", "utils"):
        files.extend((controllers / folder).rglob("*.py"))
    return {str(p.resolve()): sha(p) for p in sorted(set(files))}


def preflight():
    import pandas as pd
    from PIL import Image
    cfg(); n = native(); n.prepared()
    existing = ROOT / "reports/PREPARATION.json"
    if existing.exists() and read_json(existing).get("signature") == signature() and (
            (ROOT / "reports/SUBMISSION_RECEIPT.json").exists() or (ROOT / "evaluation").exists()):
        return prepared()
    png = BASE / "training/m01_uada_original_b1/outer_0500.png"
    require(sha(png) == cfg()["baseline_png_sha256"], "Frozen UADA PNG changed")
    with Image.open(png) as image:
        require(image.mode == "RGB" and image.size == (50, 50), "Invalid UADA reference image")
    old = BASE / "results/BATCH500_ROLLOUTS.parquet"
    rows = pd.read_parquet(old)
    rows = rows.loc[rows.state_index.eq(40) & rows.condition.isin(["clean", BASE_CONDITION])].copy()
    require(len(rows) == 80 and not rows.duplicated(["suite", "task_id", "state_index", "condition"]).any(),
            "Frozen initial-state/seed grid is incomplete")
    require(rows.signature.eq(BASE_SIGNATURE).all(), "Historical reference signature mismatch")
    require(all(len(x) == 20 for _, x in rows.groupby("suite")), "Reference suite grid incomplete")
    previous = ROOT / "reports/PREPARATION.json"
    if previous.exists() and read_json(previous).get("signature") != signature():
        require(not (ROOT / "reports/SUBMISSION_RECEIPT.json").exists()
                and not any((ROOT / p).exists() for p in ("evaluation", "qa", "results")),
                "Experiment already executed/submitted; use a new version")
    for name in ("logs", "reports"):
        (ROOT / name).mkdir(exist_ok=True)
    save_frame(rows, ROOT / "data/HISTORICAL_REFERENCE.parquet")
    from task_progress import audit_bddl
    goals = audit_bddl(Path(n.runtime_config(SUITES[0])["paths"]["libero_root"]) / "libero/libero/bddl_files")
    report = {"status": "CPU_PREFLIGHT_PASS", "signature": signature(), "native_signature": BASE_SIGNATURE,
              "sources": source_hashes(), "external_sources": external_sources(),
              "reference_sha256": sha(old), "copied_reference_sha256": sha(ROOT / "data/HISTORICAL_REFERENCE.parquet"),
              "baseline_png_sha256": sha(png), "goal_audit": goals,
              "new_episodes": 160, "adaptive_attack_episodes": 80,
              "fresh_clean_episodes": 40, "fresh_static_uada_episodes": 40,
              "new_training_jobs": 0, "gpu_experiments_executed": False, "max_concurrent_gpus": 4}
    from qa_reuse import inventory
    report["repair_reuse"] = inventory(report["external_sources"])
    report["pending_qa_suites"] = [s for s in SUITES if s not in report["repair_reuse"]["qa"]]
    write_json(report, previous)
    return report


def prepared():
    report = read_json(ROOT / "reports/PREPARATION.json")
    require(report["signature"] == signature(), "Sources changed after CPU preparation")
    native().prepared()
    require(sha(BASE / "training/m01_uada_original_b1/outer_0500.png") == report["baseline_png_sha256"], "UADA reference changed")
    require(sha(ROOT / "data/HISTORICAL_REFERENCE.parquet") == report["copied_reference_sha256"], "Reference seed grid changed")
    for p, expected in report["external_sources"].items():
        require(sha(p) == expected, "External runtime source changed: " + p)
    from qa_reuse import verify_inventory
    verify_inventory(report["repair_reuse"])
    return report


def require_gpu():
    require(bool(os.environ.get("SLURM_JOB_ID")), "GPU work must run in a user-submitted Slurm allocation")
    native().require_gpu("evaluation")


def load_snapshot_vendor():
    spec = importlib.util.spec_from_file_location("closedloop_frozen_snapshot", ROOT / "vendor/simulator_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
