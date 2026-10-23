"""Isolated pilot contracts. Importing this module never queries Slurm or loads a model."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
METHODS = ("key_point", "key_direction")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")

def require(value, message):
    if not bool(value):
        raise RuntimeError(message)

def load_json(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)

def atomic_torch(value, path):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp); os.replace(tmp, path)

def atomic_parquet(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value.to_parquet(tmp, index=False); os.replace(tmp, path)

def config():
    c = load_json(ROOT / "config.json")
    require(c["methods"] == list(METHODS) and c["training_suite"] == "libero_spatial"
            and c["seed"] == 42 and c["universal_outer"] == 40 and c["inner_loop"] == 50
            and c["key_steps"] == 20 and c["backtracking"] == [1., .5, .25], "Pilot contract changed")
    expected = {"train_observations": 20, "validation_observations": 20, "single_observations": 4,
                "single_candidates": 200, "learning_rate": .002, "warmup_outer": 20, "cosine_outer": 2000,
                "key_lr": .02, "key_rms_limit": .2, "key_regularizer": .1, "open_target": .6}
    require(all(c[k] == v for k, v in expected.items()), "Approved pilot parameters changed")
    require(c["evaluation"] == load_json(ROOT / "source/spatial_config.json")["evaluation"], "Native evaluation protocol changed")
    require(c["resources"]["training"]["host_memory_gib"] == 16
            and c["resources"]["evaluation"]["host_memory_gib"] == 12
            and c["resources"]["evaluation"]["max_concurrent"] == 2, "Resource contract changed")
    return c

def lineage(): return load_json(ROOT / "SOURCE_LINEAGE.json")

def sources():
    paths = [ROOT / f for f in ["config.json", "PLAN.md", "SOURCE_LINEAGE.json", "RECOVERY_SOURCE.json", "source/spatial_config.json"]]
    paths += sorted((ROOT / "code").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    paths += sorted((ROOT / "slurm").glob("*"))
    paths += [PROJECT / "submit_wrist_dino_key_open_pilot_v2.sh"]
    return {str(p): sha256_file(p) for p in paths if p.is_file()}

def run_signature(): return digest({"config": config(), "sources": sources()})

def verify_lineage(bank=False):
    l = lineage()
    for p, checksum in load_json(ROOT / "RECOVERY_SOURCE.json")["source_hashes"].items():
        require(sha256_file(p) == checksum, f"Frozen v1 failure/source evidence changed: {p}")
    import importlib.metadata
    for name, version in l["software_versions"].items():
        require(importlib.metadata.version(name) == version, f"Frozen software version changed: {name}")
    for group in ("historical", "runtime_sources", "model_jsons"):
        for p, h in l[group].items(): require(sha256_file(p) == h, f"Frozen {group} changed: {p}")
    for p, s in l["model_weight_stat"].items():
        v = Path(p).stat()
        require(v.st_size == s["size"] and v.st_mtime_ns == s["mtime_ns"], f"Frozen weight changed: {p}")
    for rel, v in l["native_copies"].items():
        require(sha256_file(ROOT / rel) == v["sha256"] == sha256_file(v["source"]), f"Native helper drift: {rel}")
    if bank: require(sha256_file(l["bank_path"]) == l["bank_sha256"], "Spatial bank changed")
    return l

def runtime_config(suite="libero_spatial"):
    require(suite in SUITES, "Unknown suite")
    return {**config()["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}

def history_rows():
    import pandas as pd
    l = lineage(); sp = Path(l["spatial_root"]); lg = Path(l["long_root"])
    a = pd.read_parquet(sp / "results/BATCH500_ROLLOUTS.parquet")
    b = pd.read_parquet(lg / "results/LONG4_ROLLOUTS.parquet")
    b = b[b.state_index.isin([40, 41])]
    keys = ["suite", "task_id", "state_index"]
    ac = a[a.condition.eq("clean")]; bc = b[b.condition.eq("clean")]
    z = ac.merge(bc, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    require(len(z) == 80 and z.success_a.eq(z.success_b).all()
            and z.rollout_seed_a.eq(z.rollout_seed_b).all(), "Clean pairing changed")
    a = a[a.condition.isin(["clean", "m01_uada_original_b1@500"])].copy()
    b = b[b.condition.eq("uada_original@1000")].copy()
    for d, parent, png, oldname in [(a, sp, sp / "training/m01_uada_original_b1/outer_0500.png", "m01_uada_original_b1@500"),
                                   (b, lg, lg / "training/uada_original/outer_1000.png", "uada_original@1000")]:
        require(d.signature.eq(load_json(parent / "results/STATUS.json")["signature"]).all()
                and not d.evaluation_training_hooks.any() and d.evaluation_gpu_type.eq("4090").all(), "Historical native lineage mismatch")
        require(d.loc[d.condition.eq(oldname), "deployment_png_sha256"].eq(sha256_file(png)).all(), "Baseline PNG mismatch")
        d["origin_file"] = str(parent); d["reused"] = True
    a.condition = a.condition.replace({"m01_uada_original_b1@500": "spatial_uada500"})
    b.condition = "long_uada1000"
    out = pd.concat([a, b], ignore_index=True)
    validate_grid(out, ("clean", "spatial_uada500", "long_uada1000"))
    return out

def validate_grid(frame, conditions):
    expected = {(s, t, i, c) for s in SUITES for t in range(10) for i in (40, 41) for c in conditions}
    require(len(frame) == len(expected) and set(zip(frame.suite, frame.task_id, frame.state_index, frame.condition)) == expected,
            "Incomplete or duplicated fixed grid")
    require(frame.failure.eq(1-frame.success).all() and frame.success.isin([0, 1]).all(), "Invalid outcomes")
    require(frame.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Reset seed mismatch")

def preflight(bank=True):
    verify_lineage(bank)
    rows = history_rows()
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "historical_rows": len(rows),
            "new_episodes": 160, "source": "Spatial", "sources": sources(), "resources": config()["resources"],
            "models_loaded": False, "scheduler_queried": False, "scheduler_submitted": False}

def prepared():
    d = load_json(ROOT / "reports/PREPARATION.json")
    require(d["signature"] == run_signature(), "Prepared source/config drift")
    verify_lineage()
    for p, h in d["prepared_artifacts"].items(): require(sha256_file(ROOT / p) == h, f"Prepared artifact changed: {p}")
    return d

def require_gpu(stage):
    import torch
    require(os.environ.get("SLURM_JOB_ID") and stage in ("training", "evaluation"), "User's Slurm allocation required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name, f"4090 required, got {name}")
    os.environ["OFT_STAGE_GPU"] = "4090"
    return name
