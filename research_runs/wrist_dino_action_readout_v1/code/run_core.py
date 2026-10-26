"""Versioned experiment state; the signed native experiment is read-only."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "wrist_dino_spatial_b12_diag500_v1"
BASE_SIGNATURE = "ac44e284cedc1bd5b47ca02702c2eeed20cffca1d2ecc6ae4505cea3f27c0349"
BASE_METHOD = "m01_uada_original_b1"
BASE_CONDITION = BASE_METHOD + "@500"
ARMS = ("B_native_energy", "C_dino_readout")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
sys.path.insert(0, str(BASE / "code"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def output_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Refuse to write outside this new experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write(path, writer):
    path = output_path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    writer(temporary)
    os.replace(temporary, path)


def write_json(value, path):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"))


def save_torch(value, path):
    import torch
    atomic_write(path, lambda p: torch.save(value, p))


def save_frame(frame, path):
    atomic_write(path, lambda p: frame.to_parquet(p, index=False))


@lru_cache(maxsize=1)
def cfg():
    out = read_json(ROOT / "config.json")
    require(out["seed"] == 42 and out["outer_iterations"] == 500 and out["inner_loop"] == 50, "Approved budget changed")
    require(out["evaluation_states"] == [40, 41] and out["arms"] == list(ARMS), "Approved comparison changed")
    return out


@lru_cache(maxsize=1)
def native():
    import common
    require(Path(common.__file__).resolve() == BASE / "code/common.py", "Wrong native common module imported")
    require(common.run_signature() == BASE_SIGNATURE, "Signed native source/config changed")
    return common


def source_hashes():
    paths = [ROOT / "config.json", ROOT / "PLAN.md", ROOT / "run.sh"]
    paths += sorted((ROOT / "code").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    paths += sorted((ROOT / "slurm").glob("*"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}


@lru_cache(maxsize=1)
def signature():
    value = {"sources": source_hashes(), "native_signature": BASE_SIGNATURE}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def split_episodes(bank, seed, validation_count=81):
    import random
    candidates = [x[0] for x in bank["candidates"]]
    episodes = {r["episode_id"] for r in bank["records"].values()}
    protected = {bank["records"][u]["episode_id"] for u in candidates}
    eligible = sorted(episodes - protected)
    require(len(eligible) >= validation_count, "Not enough independent validation episodes")
    random.Random(seed).shuffle(eligible)
    validation = set(eligible[:validation_count])
    train_uids = sorted(u for u, r in bank["records"].items() if r["episode_id"] not in validation)
    validation_uids = sorted(u for u, r in bank["records"].items() if r["episode_id"] in validation)
    return {"train_uids": train_uids, "validation_uids": validation_uids,
            "training_episodes": sorted(episodes - validation), "validation_episodes": sorted(validation),
            "patch_schedule": candidates, "protected_patch_episodes": sorted(protected)}


def baseline_inventory():
    import pandas as pd
    method = BASE / "training" / BASE_METHOD
    status = read_json(method / "status.json")
    require(status["status"] == "COMPLETE" and status["outer"] == 500 and status["signature"] == BASE_SIGNATURE,
            "Baseline training incomplete or changed")
    patch_hash = sha(method / "outer_0500.png")
    require(patch_hash == cfg()["baseline_png_sha256"], "Baseline PNG changed")
    frames, files = [], {}
    for suite in SUITES:
        for task in range(10):
            folder = BASE / "evaluation" / suite
            sp = folder / f"task_{task:02d}_STATUS.json"
            rp = folder / f"task_{task:02d}_rollouts.parquet"
            tp = folder / f"task_{task:02d}_traces.parquet"
            state = read_json(sp)
            require(state["status"] == "COMPLETE" and state["signature"] == BASE_SIGNATURE
                    and state["patches"][BASE_CONDITION] == patch_hash, "Baseline evaluation mismatch")
            require(sha(rp) == state["rollouts_sha256"] and sha(tp) == state["traces_sha256"], "Baseline results changed")
            frame = pd.read_parquet(rp)
            frame = frame.loc[frame.condition.isin(["clean", BASE_CONDITION])].copy()
            require(len(frame) == 4 and set(frame.state_index) == {40, 41}
                    and not frame.duplicated(["state_index", "condition"]).any(), "Baseline paired grid incomplete")
            require(frame.suite.eq(suite).all() and frame.task_id.eq(task).all(), "Baseline task mismatch")
            frames.append(frame)
            files.update({str(p): sha(p) for p in (sp, rp, tp)})
    return pd.concat(frames, ignore_index=True), {"files": files, "png_sha256": patch_hash,
            "initial_pt_sha256": sha(method / "outer_0000.pt"), "final_pt_sha256": sha(method / "outer_0500.pt")}


def prepared():
    report = read_json(ROOT / "reports/PREPARATION.json")
    require(report["signature"] == signature(), "New experiment changed after preparation; do not mix runs")
    native().prepared()
    require(sha(ROOT / "data/SPLIT.json") == report["split_sha256"], "Readout split changed")
    require(sha(ROOT / "data/REUSED_BASELINE.parquet") == report["reused_baseline_sha256"], "Copied baseline/seeds changed")
    for path, expected in report["baseline"]["files"].items():
        require(sha(path) == expected, "Reused baseline file changed: " + path)
    require(sha(BASE / "training" / BASE_METHOD / "outer_0500.png") == report["baseline"]["png_sha256"], "Baseline PNG drift")
    require(sha(BASE / "training" / BASE_METHOD / "outer_0000.pt") == report["baseline"]["initial_pt_sha256"], "Initial RGB drift")
    require(sha(BASE / "training" / BASE_METHOD / "outer_0500.pt") == report["baseline"]["final_pt_sha256"], "Teacher UADA float RGB drift")
    for path, expected in report["teacher_library"].items():
        require(sha(path) == expected, "Teacher patch library changed")
    return report


def bank():
    return native().load_bank()


def preflight():
    import torch
    (ROOT / "logs").mkdir(exist_ok=True)
    native().prepared()
    frame, inventory = baseline_inventory()
    b = bank()
    split = split_episodes(b, cfg()["split_seed"])
    require(len(b["records"]) == 1259 and len(split["patch_schedule"]) == 250, "Source bank dimensions changed")
    initial = torch.load(BASE / "training" / BASE_METHOD / "outer_0000.pt", map_location="cpu", weights_only=False)
    require(torch.equal(initial["patch"], torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))), "Baseline initialization mismatch")
    from PIL import Image
    import numpy as np
    final = torch.load(BASE / "training" / BASE_METHOD / "outer_0500.pt", map_location="cpu", weights_only=False)
    require(final["signature"] == BASE_SIGNATURE and final["outer_iterations"] == 500, "Wrong final baseline float snapshot")
    rgb = final["patch"].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    with Image.open(BASE / "training" / BASE_METHOD / "outer_0500.png") as image:
        require(np.array_equal(rgb, np.asarray(image)), "Baseline float/PNG mismatch")
    for path in [ROOT / "data/SPLIT.json", ROOT / "reports/PREPARATION.json"]:
        if path.exists():
            old = read_json(path)
            if old.get("signature") != signature():
                # CPU-only implementation checks may be refreshed before any submission.
                untouched = not (ROOT / "reports/SUBMISSION_RECEIPT.json").exists() and not any(
                    (ROOT / name).exists() for name in ["teacher", "readout", "training", "evaluation", "results", "reports/FAILURES"])
                require(untouched, "Sources changed after submission/execution; use a new version")
                write_json(old, ROOT / "reports/preflight_history" / f"{time.time_ns()}_{path.name}")
    split["signature"] = signature()
    write_json(split, ROOT / "data/SPLIT.json")
    save_frame(frame, ROOT / "data/REUSED_BASELINE.parquet")
    report = {"status": "CPU_PREFLIGHT_PASS", "signature": signature(), "native_signature": BASE_SIGNATURE,
              "baseline": inventory, "source_hashes": source_hashes(), "split_sha256": sha(ROOT / "data/SPLIT.json"),
              "reused_baseline_sha256": sha(ROOT / "data/REUSED_BASELINE.parquet"),
              "teacher_library": {str(p): sha(p) for p in [BASE / "training" / BASE_METHOD / "outer_0500.pt",
                                   BASE / "training/m02_dino_key_b1/outer_0500.pt"]},
              "source_windows": len(b["records"]), "head_train_windows": len(split["train_uids"]),
              "head_validation_windows": len(split["validation_uids"]), "head_train_episodes": len(split["training_episodes"]),
              "head_validation_episodes": len(split["validation_episodes"]), "new_patch_training_jobs": 2,
              "new_rollouts": 160, "reused_clean_rollouts": 80, "reused_uada_rollouts": 80,
              "gpu_experiments_executed": False}
    write_json(report, ROOT / "reports/PREPARATION.json")
    return report


def require_gpu(stage):
    native().require_gpu("evaluation" if stage == "evaluation" else "training")


def stage_done(path, artifacts=()):
    path = Path(path)
    if not path.exists():
        return False
    status = read_json(path)
    require(status["signature"] == signature(), "Existing stage signature mismatch")
    if status["status"] != "COMPLETE":
        return False
    require(bool(status.get("artifacts")), "Completed stage has no artifact receipt")
    for rel, expected in status.get("artifacts", {}).items():
        require(sha(ROOT / rel) == expected, "Completed artifact changed: " + rel)
    require(all(Path(p).exists() for p in artifacts), "Completed stage lost an artifact")
    return True


def finish(path, artifacts, **extra):
    values = {str(Path(p).relative_to(ROOT)): sha(p) for p in artifacts}
    write_json({"status": "COMPLETE", "signature": signature(), "artifacts": values, **extra}, path)


def locked(path):
    import fcntl
    from contextlib import contextmanager
    @contextmanager
    def context():
        with output_path(path).open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
    return context()
