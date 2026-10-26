"""Local contracts and recoverable outputs; importing this module submits nothing."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
CONFIG = ROOT / "config.json"
METHODS = ("plain_gripper_tma", "task_region_dino_tma", "background_region_dino_tma")
CONDITIONS = ("clean", "reference_uada500", *METHODS)
DENOMINATORS = {"libero_object": 99, "libero_spatial": 90, "libero_goal": 98, "libero_10": 95}
RESOURCES = {
    "labels": {"partition": "a800", "qos": "a800", "gpu_type": "a800", "device_name": "A800", "max_concurrent": 3},
    "training": {"partition": "a800", "qos": "a800", "gpu_type": "a800", "device_name": "A800", "max_concurrent": 3},
    "evaluation": {"partition": "rtx", "qos": None, "gpu_type": "4090", "device_name": "4090", "max_concurrent": 5},
}
ORIGINALS = {
    "TMA.py": (PROJECT / "VLAAttacker/white_patch/TMA.py", "7d622093579656ce59e8126521b7bd034317544618b094e537996d8db6f725d2"),
    "appply_random_transform.py": (PROJECT / "VLAAttacker/white_patch/appply_random_transform.py", "4091a03a5b55760d6e8ae5463679ab8b68a8ee9253735d610154e0c5f0f6d4bf"),
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    expected = {"experiment_id": "OFT-WRIST-DINO-GRIPPER-TMA500-V1", "methods": list(METHODS),
                "seed": 42, "data_seed": 20261008, "outer_iterations": 500, "inner_loop": 50,
                "batch_size": 1, "learning_rate": .002, "weight_decay": 0,
                "warmup_outer_iterations": 20, "geometry": True, "resize_patch": False,
                "patch_shape": [3, 50, 50], "checkpoint_outer_iterations": [0, 100, 250, 500],
                "close_threshold": .5, "gripper_target": 1, "region_token_threshold": 0,
                "patch_exclusion_threshold": 0, "max_eot_attempts": 32}
    for key, value in expected.items():
        require(cfg.get(key) == value, f"Frozen experiment setting changed: {key}")
    require(cfg.get("resources") == RESOURCES, "Approved GPU resources/concurrency changed")
    ev = cfg["evaluation"]
    require(ev["suites"] == list(DENOMINATORS) and ev["task_count"] == 10
            and ev["state_indices"] == list(range(40, 50)) and ev["patch_xy"] == [26, 137]
            and ev["patch_shape"] == [3, 50, 50] and ev["num_open_loop_steps"] == 8,
            "Frozen evaluation grid changed")
    return cfg


def runtime_config(cfg, suite):
    ev = cfg["evaluation"]
    return {**ev, "paths": ev["paths"], "env_image_resolution": 256,
            "policy_image_resolution": 224}


def source_manifest(wrapper=None):
    paths = {CONFIG, ROOT / "PLAN.md", PROJECT / "submit_exploring_oft_gripper_tma500_v1.sh"}
    for directory, pattern in ((ROOT / "source", "*.py"), (ROOT / "code", "*.py"),
                               (ROOT / "tests", "*.py"), (ROOT / "slurm", "*.sbatch")):
        paths.update(directory.glob(pattern))
    paths.add(ROOT / "slurm/environment.sh")
    paths.update(item[0] for item in ORIGINALS.values())
    paths.add(PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py")
    paths.add(PROJECT / "VLAAttacker/multiview/openvla_oft_experiments.py")
    if wrapper is not None:
        for obj in (wrapper, wrapper.vla, wrapper.processor, wrapper.processor.image_processor,
                    wrapper.action_head, wrapper.proprio_projector, wrapper.vla.vision_backbone,
                    wrapper.vla.vision_backbone.featurizer):
            if obj is not None:
                path = Path(inspect.getfile(type(obj))).resolve()
                paths.add(path)
                if "transformers_modules" in str(path):
                    paths.update(path.parent.glob("*.py"))
        paths.add(Path(wrapper._official_openvla_utils.__file__).resolve())
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def require_gpu(kind):
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped CUDA GPU required; no CPU model/renderer fallback")
    require(kind in torch.cuda.get_device_name(0), f"This stage requires {kind}")
    return {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


def require_stage_gpu(stage):
    return require_gpu(config()["resources"][stage]["device_name"])


def check_rgb(value):
    import torch
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == (3, 50, 50)
            and torch.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), "Invalid RGB patch")
    return value.float().cpu()


def reference_patch(cfg):
    import pandas as pd
    import torch
    path = Path(cfg["reference_patch"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    # Legacy saver writes the TOTAL planned 2000 schedule into every snapshot.
    # The actual checkpoint budget is proved by the frozen 500 row, not this field.
    require(path.name=="outer_0500.pt" and isinstance(saved,dict)
            and saved.get("method")=="exploring_oft_uada" and saved.get("outer_iterations")==2000
            and saved.get("inner_loop")==50 and saved.get("maskidx")==[0]
            and saved.get("route_objective")=="none", "Wrong original UADA500 lineage")
    manifest = load_json(cfg["reference_manifest"])
    old = manifest["config"]
    require(old["batch_size"]==1 and old["seed"]==42 and old["outer_iterations"]==2000
            and old["methods"]==["exploring_oft_uada"] and manifest["bank"]==cfg["training_bank_sha256"],
            "Reference manifest changed")
    row = pd.read_parquet(cfg["reference_history"]).loc[lambda f:f.outer_iteration.eq(500)]
    require(len(row)==1 and int(row.iloc[0].optimizer_updates)==25000, "Actual UADA500 update evidence missing")
    patch = check_rgb(saved["patch"])
    tv = float((patch[:,:,1:]-patch[:,:,:-1]).abs().mean()+(patch[:,1:,:]-patch[:,:-1,:]).abs().mean())
    require(abs(tv-float(row.iloc[0].tv))<1e-6 and abs(tv-float(saved["tv"]))<1e-6,
            "Reference RGB does not match the 500-row structure evidence")
    return patch, {"pt":sha256_file(path),"manifest":sha256_file(cfg["reference_manifest"]),
                   "history":sha256_file(cfg["reference_history"]),"actual_checkpoint_outer":500}


def preflight():
    """Read-only CPU assets check. Never load models, render, query Slurm or submit."""
    import numpy as np
    import pandas as pd
    import torch
    cfg = config()
    for name, (path, checksum) in ORIGINALS.items():
        require(path.stat().st_mode & 0o222 == 0, f"Protected source writable: {path}")
        require(sha256_file(path) == checksum, f"Protected original bytes changed: {name}")
        require(path.read_bytes().rstrip(b"\n") == (ROOT / "source" / name).read_bytes().rstrip(b"\n"),
                f"Full source snapshot changed (only EOF newline normalization permitted): {name}")
        require((ROOT / "source" / name).stat().st_mode & 0o222 == 0, "Snapshot must be read-only")
    assets = {}
    for key in ("training_bank", "eval_reference"):
        actual = sha256_file(cfg[key])
        require(actual == cfg[key + "_sha256"], f"Frozen {key} bytes changed")
        assets[key] = actual
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    require(bank["experiment_id"] == "OFT-MGWP-WRIST-STAGE1-01" and len(bank["records"]) == 300
            and bank["selected_xy"] == [26, 137], "Wrong source bank")
    keys = [(int(r["task_id"]), int(r["trajectory_rank"]), int(r["decision_index"])) for r in bank["records"]]
    require(len(set(keys)) == 300 and set(k[0] for k in keys) == set(range(10)), "Duplicate/incomplete source bank")
    for row in bank["records"]:
        for field in ("full_image", "wrist_image"):
            image = np.asarray(row["observation"][field])
            require(image.shape == (224, 224, 3) and image.dtype == np.uint8, "Source RGB shape/dtype")
        require(np.asarray(row["observation"]["state"]).shape == (8,), "Source proprio shape")
    _patch, ref = reference_patch(cfg)
    assets.update(reference=ref, mask_runtime_config=sha256_file(cfg["mask_runtime_config"]))
    ev = cfg["evaluation"]
    clean_metadata = {}
    for suite, denominator in DENOMINATORS.items():
        path = ev["frozen_object_clean"] if suite == "libero_object" else ev["frozen_cross_suite_clean"]
        clean = pd.read_parquet(path)
        clean = clean.loc[clean.suite.eq(suite) & clean.condition.eq("clean")]
        require(len(clean) == 100 and not clean.duplicated(["task_id", "state_index"]).any()
                and int(clean.success.sum()) == denominator, f"Frozen clean grid changed: {suite}")
        clean_metadata[suite] = sha256_file(path)
        checkpoint = Path(ev["checkpoints"][suite])
        require(checkpoint.is_dir(), f"Missing local model: {checkpoint}")
        model_cfg = load_json(checkpoint / "config.json")
        ids = model_cfg.get("timm_model_ids", [])
        require(len(ids) == 2 and "dino" in ids[0].lower() and "siglip" in ids[1].lower(),
                "DINO/SigLIP order differs from frozen OFT")
        assets[suite] = {str(p.resolve()): sha256_file(p) for p in checkpoint.glob("*.json")}
        assets[suite + "_weight_inventory"] = {str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns]
            for p in sorted(checkpoint.iterdir()) if p.is_file() and p.suffix in {".safetensors",".bin",".pt"}}
        require(assets[suite + "_weight_inventory"], "Checkpoint has no local weight files")
    assets["frozen_clean"] = clean_metadata
    return {"status": "CPU_PREFLIGHT_PASS", "config_sha256": sha256_file(CONFIG), "assets": assets,
            "source": source_manifest(), "source_records": 300, "training_updates_per_method": 25000,
            "maximum_gpu_concurrency_by_phase": {stage: spec["max_concurrent"]
                                                  for stage, spec in cfg["resources"].items()},
            "gpu_resources_by_phase": cfg["resources"],
            "evaluation_episodes": 2000, "gpu_started": False, "scientific_result": "NOT_TESTED"}


def preparation():
    report = load_json(ROOT / "reports/PREPARATION.json")
    require(report["status"] == "CPU_PREFLIGHT_PASS" and report["config_sha256"] == sha256_file(CONFIG)
            and report["source"] == source_manifest(), "Preparation missing or source/config changed")
    cfg, assets = config(), report["assets"]
    for key in ("training_bank","eval_reference","mask_runtime_config"):
        require(sha256_file(cfg[key])==assets[key], f"Prepared asset changed: {key}")
    for suite in DENOMINATORS:
        for path, checksum in assets[suite].items():
            require(sha256_file(path)==checksum, f"Model statistics/config changed: {path}")
        for path, inventory in assets[suite+"_weight_inventory"].items():
            stat = Path(path).stat()
            require([stat.st_size,stat.st_mtime_ns]==inventory, "Model weight inventory changed")
    _, reference = reference_patch(cfg)
    require(reference==assets["reference"], "Prepared reference patch changed")
    for suite,checksum in assets["frozen_clean"].items():
        key = "frozen_object_clean" if suite=="libero_object" else "frozen_cross_suite_clean"
        require(sha256_file(cfg["evaluation"][key])==checksum,"Frozen qualification bytes changed")
    return report


def read_training_pool():
    import torch
    report = preparation()
    receipt = load_json(ROOT / "data/POOL.json")
    require(receipt["preparation_signature"] == digest(report), "Pool preparation changed")
    path = ROOT / "data/TRAINING_POOL.pt"
    require(sha256_file(path) == receipt["pool_sha256"], "Training pool bytes changed")
    pool = torch.load(path, map_location="cpu", weights_only=False)
    require(len(pool["items"]) == receipt["accepted_records"] > 0, "Training pool is empty/incomplete")
    return pool, receipt
