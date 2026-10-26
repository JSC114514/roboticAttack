"""Frozen local contracts; imports never load a model, query or submit Slurm."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
METHOD = "dino_cross_observation"
CONDITIONS = ("uada500", METHOD)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")

def require(value, message):
    if not bool(value):
        raise RuntimeError(message)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
    require(c["experiment_id"] == "OFT-DINO-CROSS-OBSERVATION500-V1"
            and c["methods"] == [METHOD] and c["seed"] == 42 and c["pair_gap"] == 8
            and c["outer_iterations"] == 500 and c["inner_loop"] == 50
            and c["schedule_outer_iterations"] == 2000 and c["batch_size"] == 1
            and c["uada_maskidx"] == [0] and c["learning_rate"] == .002
            and c["weight_decay"] == 0 and c["warmup_outer_iterations"] == 20
            and c["checkpoint_outer_iterations"] == [0, 250, 500]
            and c["evaluation"]["state_indices"] == [40, 41]
            and c["evaluation"]["num_open_loop_steps"] == 8
            and c["resources"]["training"]["host_memory_gib"] == 16
            and c["resources"]["evaluation"]["host_memory_gib"] == 12
            and c["resources"]["training"]["max_concurrent"] == 1
            and c["resources"]["evaluation"]["max_concurrent"] == 4,
            "Approved algorithm/budget/resource contract changed")
    return c

def lineage():
    return load_json(ROOT / "SOURCE_LINEAGE.json")

def parent_config():
    return load_json(ROOT / "source/parent_config.json")

def runtime_config(suite="libero_10"):
    require(suite in SUITES, "Unknown native suite")
    return {**config()["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}

def dataset_files():
    paths = sorted(Path(config()["dataset_directory"]).glob("*.tfrecord*"))
    require(len(paths) == 32, "Official32 RLDS shards changed")
    return paths

def sources():
    paths = [ROOT / p for p in ("config.json", "PLAN.md", "SOURCE_LINEAGE.json", "source/parent_config.json")]
    paths += sorted((ROOT / "code").glob("*.py")) + sorted((ROOT / "slurm").glob("*"))
    paths += [PROJECT / "submit_wrist_dino_cross_observation500_v1.sh"]
    paths += [Path(p) for p in lineage()["parent"]["sources"]]
    paths += [Path(p) for p in lineage().get("runtime_sources", {})]
    require(all(p.is_file() for p in paths), "Missing immutable implementation source")
    return {str(p.resolve()): sha256_file(p) for p in sorted(set(paths))}

def run_signature():
    return digest({"config": config(), "sources": sources()})

def verify_parent(bank=False):
    line = lineage(); p = line["parent"]
    for path, checksum in p["sources"].items():
        require(sha256_file(path) == checksum, f"Parent/shared code drift: {path}")
    for path, checksum in line.get("runtime_sources", {}).items():
        require(sha256_file(path) == checksum, f"Native optimizer implementation drift: {path}")
    for path, checksum in p["artifacts"].items():
        if bank or path != p["bank_path"]:
            require(sha256_file(path) == checksum, f"Historical artifact drift: {path}")
    for path, checksum in line["native_copies"].items():
        require(sha256_file(ROOT / path) == checksum, f"Native helper drift: {path}")
    require(sha256_file(ROOT / "source/parent_config.json") == line["parent_config_sha256"], "Frozen parent config drift")
    prep = load_json(Path(p["root"]) / "reports/PREPARATION.json")
    require(prep["signature"] == p["signature"], "Parent identity changed")
    for group in ("model_jsons", "seed_files", "dataset_metadata"):
        for path, checksum in prep[group].items():
            require(sha256_file(path) == checksum, f"Parent {group} drift: {path}")
    for group in ("model_weight_stat", "dataset_stat"):
        for path, stat in prep[group].items():
            current = Path(path).stat()
            require(current.st_size == stat["size"] and current.st_mtime_ns == stat["mtime_ns"], f"Parent {group} drift: {path}")
    return p

def validate_grid(frame, conditions):
    expected = {(s, t, i, c) for s in SUITES for t in range(10) for i in (40, 41) for c in conditions}
    require(set(zip(frame.suite, frame.task_id, frame.state_index, frame.condition)) == expected
            and len(frame) == len(expected), "Missing/duplicated paired pilot grid")
    require(frame.success.isin([0, 1]).all() and frame.failure.eq(1 - frame.success).all(), "Invalid episode outcomes")
    require(frame.groupby(["suite", "task_id", "state_index"]).rollout_seed.nunique().eq(1).all(), "Unpaired reset seeds")
    for suite in SUITES:
        seeds = reset_seeds(suite)
        sub = frame.loc[frame.suite.eq(suite)].merge(seeds, on=["task_id", "state_index"], suffixes=("", "_frozen"), validate="many_to_one")
        require(sub.rollout_seed.eq(sub.rollout_seed_frozen).all(), "Reset grid differs from frozen baseline")

def reset_seeds(suite):
    import pandas as pd
    f = pd.read_parquet(parent_config()["evaluation"]["seed_sources"][suite])
    f = f.loc[f.suite.eq(suite) & f.condition.eq("clean") & f.state_index.isin((40, 41)), ["task_id", "state_index", "rollout_seed"]]
    require(len(f) == 20 and not f.duplicated(["task_id", "state_index"]).any(), "Frozen20 reset rows missing")
    return f

def history_rows():
    import pandas as pd
    p = lineage()["parent"]
    f = pd.read_parquet(p["raw_path"])
    f = f.loc[f.condition.isin(("clean", "uada_original@1000")) & f.state_index.isin((40, 41))].copy()
    require(f.signature.eq(p["signature"]).all() and not f.evaluation_training_hooks.any(), "Historical inference/provenance changed")
    expected = p["artifacts"][str(Path(p["root"]) / "training/uada_original/outer_1000.png")]
    require(f.loc[f.condition.eq("uada_original@1000"), "deployment_png_sha256"].eq(expected).all(), "Strong baseline PNG mismatch")
    f["origin_signature"] = f.signature; f["origin_file"] = p["raw_path"]; f["reused"] = True
    f["condition"] = f.condition.replace({"uada_original@1000": "uada1000"})
    validate_grid(f, ("clean", "uada1000"))
    return f

def preflight(bank=True):
    p = verify_parent(bank=bank)
    f = history_rows()
    import torch
    import numpy as np
    from PIL import Image
    initial = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))
    base = Path(p["root"]) / "training/uada_original"
    zero = torch.load(base / "outer_0000.pt", map_location="cpu", weights_only=False)
    asset = torch.load(base / "outer_0500.pt", map_location="cpu", weights_only=False)
    require(torch.equal(zero["patch"], initial) and torch.equal(asset["initial_patch"], initial)
            and asset["signature"] == p["signature"] and asset["config"] == parent_config()
            and asset["method"] == "uada_original" and asset["outer_iterations"] == 500
            and asset["optimizer_updates"] == 25000, "Saved UADA500 training provenance differs")
    with Image.open(base / "outer_0500.png") as im:
        require(im.mode == "RGB" and im.size == (50, 50)
                and np.array_equal(np.asarray(im), asset["patch"].permute(1, 2, 0).mul(255).round().byte().numpy()),
                "Saved UADA500 PNG differs from trained snapshot")
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "sources": sources(),
            "parent_signature": p["signature"], "historical_rows": len(f), "new_training_arms": 1,
            "new_episodes": 160, "reused_episodes": 160, "gap": 8, "independent_training_seeds": 1,
            "models_loaded": False, "slurm_queried": False, "slurm_submitted": False,
            "resource_contract": config()["resources"], "baseline500_provenance_checked": True}

def prepared():
    r = load_json(ROOT / "reports/PREPARATION.json")
    require(r["status"] == "CPU_PREFLIGHT_PASS" and r["signature"] == run_signature(), "Prepared implementation drift")
    verify_parent(bank=False)
    for rel, h in r["prepared_artifacts"].items():
        require(sha256_file(ROOT / rel) == h, f"Prepared artifact drift: {rel}")
    return r

def load_bank():
    import torch
    prepared()
    b = torch.load(ROOT / "data/PAIR_BANK.pt", map_location="cpu", weights_only=False)
    require(b["signature"] == run_signature() and len(b["pairs"]) == 500, "Pair bank identity changed")
    return b

def require_gpu(stage):
    import torch
    require(stage in ("training", "evaluation") and os.environ.get("SLURM_JOB_ID"), "User's Slurm allocation required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated CUDA GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name, f"4090 required, got {name}")
    return name
