"""Frozen contracts for the internal action-control screen."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
V2 = PROJECT / "research_runs/wrist_dino_key_open_pilot_v2"


def require(value, message):
    if not bool(value):
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def config():
    value = load_json(ROOT / "config.json")
    intervention = value["intervention"]
    require(value["experiment_id"] == "OFT-WRIST-INTERNAL-ACTION-CONTROL-SCREEN-V1", "Experiment identity changed")
    require(value["training_suite"] == "libero_spatial" and value["seed"] == 42, "Source suite/seed changed")
    require(value["layers"] == [8, 16, 22] and len(value["candidates"]) == 27
            and len(set(value["candidates"])) == 27, "Candidate screen changed")
    require(value["endpoints"] == ["wrong_open", "uada_opposite"], "Endpoint screen changed")
    require(intervention == {"steps": 12, "learning_rate": .02, "relative_rms_limit": .2,
            "shared_rms_candidates": [.05, .1, .2], "regularizer": .02,
            "open_margin": .6, "prefix_tokens": 5,
            "dense_tokens": 256, "open_loop_steps": 8, "action_dim": 7},
            "Bounded intervention contract changed")
    require(value["resources"]["screen"] == {"cpus": 4, "host_memory_gib": 16,
            "gpu_type": "4090", "gpus": 1, "walltime": "04:00:00"}, "GPU resource contract changed")
    return value


def source_path():
    return PROJECT / config()["source"]["artifact"]


def provenance_paths():
    return {
        V2 / "config.json": "185ab9e3dfa44228c9574c0edff3d79c13c96b1f189eff00716e46bdca276d67",
        V2 / "SOURCE_LINEAGE.json": "b9da23f2944e8acad22981e1568a88c6c9ac8a155bca19062f202fdbcbc5fcb5",
        V2 / "code/model_runtime.py": "d37d819caaf67b9b2039d9827e1257d4c9a700d2dd6138e445d7bbe44f4ca5da",
        V2 / "code/readonly_model.py": "398a9c97f7e47078b6c731fbfc96c72f508ca90340f71172ee1edfd35999f857",
        PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py": "f772c48f63b12b85194db8c2a3d175672e88a5d8a2ff5abb06069cd5d9ef0a24",
        Path("/home/caijiasheng25/Desktop/tex3D/openvla-oft/prismatic/extern/hf/modeling_prismatic.py"): "b5431a074c0025a12e46dc954a5e18d1d73477babb5ae42e3a12ab4b907f33a6",
        Path("/home/caijiasheng25/Desktop/tex3D/openvla-oft/prismatic/models/action_heads.py"): "8e42df65d6407d64d47457286d9be05466d25d8a0f094aa9dfb90de84a9aa7cc",
    }


def sources():
    paths = [ROOT / "config.json", ROOT / "PLAN.md", ROOT / "README.md"]
    paths += sorted((ROOT / "code").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    paths += sorted((ROOT / "slurm").glob("*"))
    paths += [PROJECT / "submit_wrist_dino_action_control_screen_v1.sh"]
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def run_signature():
    return digest({"config": config(), "sources": sources(),
                   "input_sha256": config()["source"]["sha256"],
                   "runtime": {str(path): value for path, value in provenance_paths().items()}})


def verify_provenance():
    cfg = config()
    require(sha256_file(source_path()) == cfg["source"]["sha256"], "Dual-close source artifact changed")
    for path, expected in provenance_paths().items():
        require(path.is_file() and sha256_file(path) == expected, f"Frozen runtime source changed: {path}")
    lineage = load_json(V2 / "SOURCE_LINEAGE.json")
    # Recheck the exact Spatial model metadata and large-weight file stats without
    # hashing multi-gigabyte shards on every submission.
    for path, expected in lineage["model_jsons"].items():
        if "libero-spatial" in path:
            require(sha256_file(path) == expected, f"Spatial model metadata changed: {path}")
    for path, expected in lineage["model_weight_stat"].items():
        if "libero-spatial" in path:
            stat = Path(path).stat()
            require(stat.st_size == expected["size"] and stat.st_mtime_ns == expected["mtime_ns"],
                    f"Spatial model weight changed: {path}")
    return lineage


def load_records():
    import torch
    payload = torch.load(source_path(), map_location="cpu", weights_only=False)
    cfg = config()
    require(payload["signature"] == cfg["source"]["signature"], "Dual-close source signature changed")
    records = payload["records"]
    require(len(records) == 40 and len({row["uid"] for row in records}) == 40, "Expected 40 unique records")
    counts = {(split, task): 0 for split in ("train", "validation") for task in range(10)}
    for row in records:
        key = (row["pilot_split"], int(row["task_id"]))
        require(key in counts, "Unexpected split/task in source")
        counts[key] += 1
        require(tuple(row["expert_actions"].shape) == (1, 8, 7)
                and tuple(row["clean_action"].shape) == (1, 8, 7)
                and tuple(row["close_slots"].shape) == (8,), "Source action layout changed")
        expected = (row["expert_actions"].reshape(8, 7)[:, 6] < .5) & (row["clean_action"].reshape(8, 7)[:, 6] < .5)
        require(bool(row["close_slots"].any()) and bool((expected == row["close_slots"]).all()), "Dual-close mask changed")
    require(set(counts.values()) == {2}, "Source is not task-balanced 20/20")
    return records


def runtime_config():
    base = load_json(V2 / "config.json")["evaluation"]
    return {**base, "env_image_resolution": 256, "policy_image_resolution": 224}


def require_gpu():
    import torch
    require(os.environ.get("SLURM_JOB_ID"), "Screen requires the user's Slurm allocation")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name, f"4090 required, got {name}")
    return name


def preflight():
    verify_provenance()
    records = load_records()
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "records": len(records),
            "train": sum(row["pilot_split"] == "train" for row in records),
            "validation": sum(row["pilot_split"] == "validation" for row in records),
            "candidates": len(config()["candidates"]), "arms": len(config()["candidates"]) * 2,
            "resources": config()["resources"], "model_loaded": False,
            "scheduler_queried": False, "scheduler_submitted": False}
