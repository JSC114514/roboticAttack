"""Final-eight contracts. Importing this module never loads a model or submits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
CONFIG = ROOT / "config.json"
METHODS = ("uada_original", "uada_dino", "uada_dino_freq", "tma_xyz", "tma_xyz_dino",
           "tma_xyz_dino_freq", "tma_xyz_dino_teacher", "uada_dino_teacher")
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def config():
    cfg = load_json(CONFIG)
    expected = {"experiment_id": "OFT-WRIST-DINO-LONG-FINAL8-V1", "methods": list(METHODS),
                "training_suite": "libero_10", "seed": 42, "outer_iterations": 2000,
                "inner_loop": 50, "batch_size": 1, "patch_shape": [3, 50, 50],
                "uada_maskidx": [0], "tma_maskidx": [0, 1, 2], "tma_normalized_target": [0, 0, 0],
                "learning_rate": .002, "weight_decay": 0, "bootstrap_outer": 250,
                "teacher_updates": 100, "teacher_shards_per_family": 4,
                "evaluation_checkpoints": [500, 1000, 2000], "distill_gradient_ratio": .1}
    require(all(cfg.get(k) == v for k, v in expected.items()), "Final-eight scientific contract changed")
    require(tuple(cfg["evaluation"]["suites"]) == SUITES
            and cfg["evaluation"]["state_indices"] == list(range(40, 50)), "Evaluation grid changed")
    require(cfg["resources"] == {"training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 8},
                                  "evaluation": {"partition": "rtx", "gpu_type": "3090", "max_concurrent": 8}},
            "Approved GPU limits changed")
    return cfg


def family(method):
    require(method in METHODS, "Unknown method")
    return "uada" if method.startswith("uada") else "tma"


def method_flags(method):
    return {"family": family(method), "replay": method != "uada_original",
            "weighted": method not in {"uada_original", "tma_xyz"},
            "frequency": method.endswith("_freq"), "teacher": method.endswith("_teacher"),
            "tv": method in {"uada_dino", "uada_dino_freq", "uada_dino_teacher"},
            "diverse": method != "uada_original"}


def sources():
    paths = list((ROOT / "code").glob("*.py")) + list((ROOT / "source").glob("*.py"))
    paths += list((ROOT / "slurm").glob("*.sbatch")) + [ROOT / "slurm/environment.sh", CONFIG]
    paths += [PROJECT / "submit_exploring_oft_long_final8_2000_repair_v2.sh"]
    paths += [PROJECT / name for name in config()["original_sources"]]
    paths += [PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py",
              PROJECT / "VLAAttacker/multiview/openvla_oft_experiments.py"]
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def run_signature():
    return digest({"config": config(), "sources": sources()})


def require_gpu(stage):
    import torch
    cfg = config()
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "One user-submitted Slurm CUDA GPU required; no CPU experiment fallback")
    expected = cfg["resources"][stage]["gpu_type"]
    require(expected in torch.cuda.get_device_name(0), f"{expected} required for {stage}")
    os.environ["OFT_STAGE_GPU"] = expected
    return torch.cuda.get_device_name(0)


def runtime_config(suite="libero_10"):
    require(suite in SUITES, "Unknown suite")
    return {**config()["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}


def clean_records():
    import numpy as np
    import torch
    cfg = config()
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"], "Frozen Long bank changed")
    records = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)["records"]
    require(len(records) == 300 and {int(r["task_id"]) for r in records} == set(range(10)), "Long clean300 coverage")
    require(len({(r["task_id"], r["state_index"], r["decision_index"]) for r in records}) == 300,
            "Duplicate clean source states")
    for r in records:
        require(r["source_suite"] == "libero_10" and r["episode_success"] is True
                and int(r["state_index"]) < 40, "Wrong source suite/qualification or test leakage")
        require(tuple(r["clean_action"].shape) == (1, 8, 7) and torch.isfinite(r["clean_action"]).all(),
                "Invalid cached Long action")
        for key in ("full_image", "wrist_image"):
            image = np.asarray(r["observation"][key])
            require(image.shape == (224, 224, 3) and image.dtype == np.uint8, "Invalid source RGB")
    return records


def observation_key(task, state, decision):
    return f"{int(task)}:{int(state)}:{int(decision)}"


def prepared():
    value = load_json(ROOT / "reports/PREPARATION.json")
    require(value["signature"] == run_signature(), "Source/config changed after preparation")
    require(all(sha256_file(p) == checksum for p, checksum in value["model_jsons"].items()),
            "Model/normalization JSON changed after preparation")
    require(all(sha256_file(p) == checksum for p, checksum in value["seed_files"].items()),
            "Evaluation seed-source files changed after preparation")
    return value


def preflight():
    import pandas as pd
    cfg = config()
    for name in ("UADA.py", "TMA.py", "appply_random_transform.py"):
        origin = PROJECT / "VLAAttacker/white_patch" / name
        copied = ROOT / "source" / name
        require(origin.stat().st_mode & 0o222 == 0 and copied.stat().st_mode & 0o222 == 0,
                "Original/snapshot must be read-only")
        require(sha256_file(origin) == cfg["original_sources"][str(origin.relative_to(PROJECT))]
                and origin.read_bytes().rstrip(b"\n") == copied.read_bytes().rstrip(b"\n"),
                "Protected source bytes changed (snapshot may normalize final newline only)")
    records = clean_records()
    model_jsons = {}
    for suite in SUITES:
        folder = Path(cfg["evaluation"]["checkpoints"][suite])
        statistics = load_json(folder / "dataset_statistics.json")
        require(cfg["evaluation"]["unnorm_keys"][suite] in statistics, "Missing suite normalization")
        require(any(folder.glob("*.safetensors")), "Missing local model weights")
        model_jsons.update({str(p.resolve()): sha256_file(p) for p in folder.glob("*.json")})
    require(len(records) == 300, "No source data")
    require(set(cfg["evaluation"]["seed_sources"]) == set(SUITES), "All four suite seed files are required")
    seed_grid = {}
    for path in set(cfg["evaluation"]["seed_sources"].values()):
        seed_grid[path] = pd.read_parquet(path, columns=["suite", "task_id", "state_index", "condition", "rollout_seed"])
    for suite in SUITES:
        grid = seed_grid[cfg["evaluation"]["seed_sources"][suite]]
        grid = grid.loc[grid.suite.eq(suite) & grid.condition.eq("clean")]
        expected = {(task, state) for task in range(10) for state in range(40, 50)}
        actual = set(zip(grid.task_id.astype(int), grid.state_index.astype(int)))
        require(len(grid) == 100 and actual == expected and not grid.rollout_seed.isna().any(),
                f"Invalid frozen evaluation seed grid: {suite}")
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "source_records": len(records),
            "model_jsons": model_jsons,
            "seed_files": {str(Path(p).resolve()): sha256_file(p)
                           for p in set(cfg["evaluation"]["seed_sources"].values())},
            "models_loaded": False, "renderer_started": False, "slurm_submitted": False,
            "gpu_caps": {"4090": 8, "3090": 8}, "evaluation_episodes": 10000}


def load_bank():
    import torch
    value = torch.load(ROOT / "data/STATE_BANK.pt", map_location="cpu", weights_only=False)
    require(value["signature"] == run_signature(), "Bank source identity changed")
    receipt = load_json(ROOT / "data/STATE_BANK_STATUS.json")
    require(receipt["sha256"] == sha256_file(ROOT / "data/STATE_BANK.pt") and receipt["signature"] == run_signature(),
            "Frozen replay bank bytes changed")
    return value


def deployment_conditions():
    return ["clean", *(f"{m}@{n}" for m in METHODS for n in config()["evaluation_checkpoints"])]
