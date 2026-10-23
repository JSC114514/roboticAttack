"""Official-demo experiment contracts; imports never load models or submit jobs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
METHODS = ("uada_original", "dino_key", "dino_center", "advencoder_cos")
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def atomic_torch(value, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)

def atomic_parquet(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def config():
    cfg = load_json(ROOT / "config.json")
    expected = {"experiment_id": "OFT-DINO-FEATURE-SCREEN250-V1", "methods": list(METHODS),
                "training_suite": "libero_10", "outer_iterations": 250, "inner_loop": 50,
                "schedule_outer_iterations": 2000, "center_reference_count": 1024, "key_layer": "penultimate",
                "generator": {"latent_dimension": 100, "learning_rate": .0002, "betas": [.5, .999],
                              "feature_weight": 5., "quality_weight": 1., "high_frequency_weight": 1.,
                              "high_frequency_radius": 8, "criterion": "cos", "network_output_resolution": 64},
                "batch_size": 1, "candidates_per_outer": 1, "seed": 42,
                "patch_shape": [3, 50, 50], "uada_maskidx": [0], "tma_maskidx": [0, 1, 2],
                "tma_normalized_target": [0, 0, 0], "evaluation_checkpoints": [250],
                "learning_rate": .002, "weight_decay": 0, "warmup_outer_iterations": 20,
                "dataset_episodes": 379, "auxiliary_gradient_ratio": .1, "data_seed": 20261013,
                "checkpoint_outer_iterations": [0, 125, 250], "resume_every_outer": 10,
                "frequency_strength": 8, "frequency_power": .75, "geometry_scales": [.9, 1.1],
                "geometry_lateral_pixels": [-4, 4], "attention_layers": [8, 16], "dino_grid": 16,
                "dino_prefix_tokens": 5, "eot": {"mode": "exploring_original", "fixed_inner_count": 0,
                                               "rotation_degrees": 30, "shear": .2,
                                               "identity_probability": .2, "resize_patch": False}}
    require(all(cfg.get(k) == v for k, v in expected.items()), "Approved feature-screen contract changed")
    require(tuple(cfg["evaluation"]["suites"]) == SUITES
            and cfg["evaluation"]["state_indices"] == [40, 41]
            and cfg["evaluation"]["patch_xy"] == [26, 137] and cfg["evaluation"]["num_open_loop_steps"] == 8
            and cfg["evaluation"]["image_resolution"] == 224 and cfg["evaluation"]["dtype"] == "bfloat16",
            "Evaluation grid/native interface changed")
    for stage, cap in (("training", 4), ("evaluation", 4)):
        require(cfg["resources"][stage] == {"partition": "rtx", "gpu_type": "4090", "max_concurrent": cap},
                "Approved serial GPU phase changed")
    return cfg

def flags(method):
    require(method in METHODS, "Unknown method")
    return {"family": "uada", "objective": method, "sampling": False, "frequency": False,
            "geometry": False, "attention": False, "generator": method == "advencoder_cos"}

def sources():
    files = list((ROOT / "code").glob("*.py")) + list((ROOT / "source").glob("*.py"))
    files += list((ROOT / "slurm").glob("*.sbatch")) + [ROOT / "slurm/environment.sh", ROOT / "config.json"]
    files += list((ROOT / "vendor").glob("*.py")) + list((ROOT / "vendor").glob("*.txt"))
    files += [PROJECT / "submit_exploring_oft_dino_feature_screen250_v1.sh"]
    files += [PROJECT / p for p in config()["original_sources"]]
    files += [PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py",
              PROJECT / "VLAAttacker/multiview/openvla_oft_experiments.py"]
    official = Path(config()["evaluation"]["paths"]["openvla_oft_root"])
    files += [official / rel for rel in ("experiments/robot/openvla_utils.py", "experiments/robot/robot_utils.py",
              "prismatic/extern/hf/modeling_prismatic.py", "prismatic/extern/hf/configuration_prismatic.py",
              "prismatic/extern/hf/processing_prismatic.py", "prismatic/vla/constants.py",
              "prismatic/vla/datasets/rlds/oxe/transforms.py", "prismatic/vla/datasets/rlds/utils/data_utils.py")]
    return {str(p.resolve()): sha256_file(p) for p in sorted(files)}

def run_signature():
    return digest({"config": config(), "sources": sources()})

def runtime_config(suite="libero_10"):
    require(suite in SUITES, "Unknown suite")
    return {**config()["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}

def require_gpu(stage):
    import torch
    require(stage in {"training", "evaluation"} and os.environ.get("SLURM_JOB_ID"), "User-submitted Slurm job required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated CUDA GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name, f"RTX4090 required, got {name}")
    os.environ["OFT_STAGE_GPU"] = "4090"
    return name

def deployment_conditions():
    return ["clean", *(f"{m}@{outer}" for outer in config()["evaluation_checkpoints"] for m in METHODS)]

def dataset_files():
    folder = Path(config()["dataset_directory"])
    files = sorted(folder.glob("*-train.tfrecord-*-of-00032"))
    require(len(files) == 32 and all(p.stat().st_size > 0 for p in files), "Official Long32 shards missing")
    return files

def preflight():
    import pandas as pd
    from readonly_model import verify_checkpoint
    cfg = config()
    for rel, expected in cfg["original_sources"].items():
        origin = PROJECT / rel
        copied = ROOT / "source" / origin.name
        require(origin.stat().st_mode & 0o222 == 0 and copied.stat().st_mode & 0o222 == 0, "Protected source writable")
        require(sha256_file(origin) == expected and origin.read_bytes().rstrip(b"\n") == copied.read_bytes().rstrip(b"\n"),
                f"Protected source/copy changed: {origin.name}")
    folder = Path(cfg["dataset_directory"])
    require(sha256_file(folder / "dataset_info.json") == cfg["dataset_info_sha256"], "Official dataset metadata changed")
    info = load_json(folder / "dataset_info.json")
    require(len(info["splits"]) == 1 and info["splits"][0]["name"] == "train"
            and sum(map(int, info["splits"][0]["shardLengths"])) == 379, "Official train split changed")
    shard_stats = {str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in dataset_files()}
    model_jsons, model_weights = {}, {}
    for suite in SUITES:
        path = Path(cfg["evaluation"]["checkpoints"][suite])
        stats = load_json(path / "dataset_statistics.json")[cfg["evaluation"]["unnorm_keys"][suite]]
        require(all(k in stats["action"] for k in ("q01", "q99", "min", "max")), "Missing official action bounds")
        weights = sorted([*path.glob("*.safetensors"), *path.glob("*checkpoint.pt")])
        require(weights and all(p.stat().st_size > 0 for p in weights)
                and len(list(path.glob("action_head--*checkpoint.pt"))) == 1
                and len(list(path.glob("proprio_projector--*checkpoint.pt"))) == 1, "Local OFT weights/head/proprio missing")
        model_jsons.update({str(p.resolve()): sha256_file(p) for p in path.glob("*.json")})
        model_jsons.update(verify_checkpoint(path, cfg["evaluation"]["paths"]["openvla_oft_root"]))
        model_jsons.update({str(p.resolve()): sha256_file(p) for p in path.glob("*.py")})
        model_weights.update({str(p.resolve()): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights})
    seed_files = {}
    for path in set(cfg["evaluation"]["seed_sources"].values()):
        grid = pd.read_parquet(path, columns=["suite", "task_id", "state_index", "condition", "rollout_seed"])
        for suite in SUITES:
            subset = grid.loc[grid.suite.eq(suite) & grid.condition.eq("clean")]
            require(len(subset) == 100 and not subset.duplicated(["task_id", "state_index"]).any()
                    and set(zip(subset.task_id, subset.state_index)) == {(t, s) for t in range(10) for s in range(40, 50)},
                    f"Frozen reset seed grid invalid: {suite}")
        seed_files[str(Path(path).resolve())] = sha256_file(path)
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "model_jsons": model_jsons,
            "model_weight_stat": model_weights, "seed_files": seed_files, "dataset_stat": shard_stats,
            "dataset_metadata": {str((folder / p).resolve()): sha256_file(folder / p) for p in ("dataset_info.json", "features.json")},
            "official_train_episodes": 379, "methods": len(METHODS),
            "evaluation_checkpoints": cfg["evaluation_checkpoints"], "evaluation_episodes": 40 * len(cfg["evaluation"]["state_indices"]) * len(deployment_conditions()),
            "models_loaded": False, "slurm_submitted": False}

def prepared():
    value = load_json(ROOT / "reports/PREPARATION.json")
    require(value["signature"] == run_signature(), "Source/config drift since preparation")
    for group in ("model_jsons", "seed_files", "dataset_metadata"):
        require(all(sha256_file(p) == h for p, h in value[group].items()), f"Prepared {group} changed")
    for group in ("dataset_stat", "model_weight_stat"):
        require(all(Path(p).stat().st_size == s["size"] and Path(p).stat().st_mtime_ns == s["mtime_ns"]
                    for p, s in value[group].items()), f"Prepared {group} changed")
    return value

def load_bank():
    import torch
    prepared()
    path = ROOT / "data/CANDIDATE_BANK.pt"
    receipt = load_json(ROOT / "data/DATA_STATUS.json")
    require(receipt["status"] == "COMPLETE" and receipt["signature"] == run_signature()
            and receipt["bank_sha256"] == sha256_file(path), "Official candidate cache incomplete/changed")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    require(bank["signature"] == run_signature() and len(bank["candidates"]) == config()["outer_iterations"]
            and all(len(v) == 1 for v in bank["candidates"])
            and len(bank["center_reference_uids"]) == config()["center_reference_count"], "Record table/center references changed")
    return bank
