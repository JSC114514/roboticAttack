"""Frozen 8x4 contract; only explicit user submission starts GPU work."""
from __future__ import annotations
import copy
import importlib.util
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE_ROOT = ROOT.parent / "wrist_dino_feature_screen250_v1"
BASE_SIGNATURE = "590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620"
FAMILIES = {1: "uada_original", 2: "dino_key", 3: "dino_center", 4: "advencoder_cos",
            6: "dino_multilayer", 7: "dino_affinity", 9: "advencoder_nce", 12: "tma_xyz"}
BATCHES = (1, 2, 4, 8)
ARMS = {f"m{i:02d}_{name}_b{b}": {"id": i, "objective": name, "batch": b}
        for i, name in FAMILIES.items() for b in BATCHES}
METHODS = tuple(ARMS)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
_spec = importlib.util.spec_from_file_location("batch500_frozen_base_common", BASE_ROOT / "code/common.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
require, load_json, sha256_file, digest = base.require, base.load_json, base.sha256_file, base.digest
atomic_json, atomic_torch, atomic_parquet = base.atomic_json, base.atomic_torch, base.atomic_parquet


def config():
    extra = load_json(ROOT / "config.json")
    expected = {"experiment_id": "OFT-DINO-BATCH500-32-V1", "selected_ids": list(FAMILIES),
                "batch_sizes": list(BATCHES), "outer_iterations": 500, "inner_loop": 50,
                "evaluation_checkpoints": [500], "checkpoint_outer_iterations": [0, 125, 250, 375, 500],
                "base_run": BASE_ROOT.name, "base_signature": BASE_SIGNATURE,
                "multilayer_layers": [8, 16, 22], "affinity_topk": 8, "objective_epsilon": 1e-6,
                "nce_temperature": .1, "recompute_tolerance": 1e-5,
                "nce_b1": "explicit_zero_feature_gradient_control",
                "evaluation_state_indices": list(range(40, 50)),
                "resources": {"training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 6},
                              "evaluation": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 8}}}
    require(extra == expected, "Approved 32-arm contract changed")
    cfg = copy.deepcopy(base.config())
    cfg.update(extra)
    cfg["methods"] = list(METHODS)
    cfg["batch_size"] = {m: ARMS[m]["batch"] for m in METHODS}
    cfg["generator"]["criterion"] = "specified_by_objective"
    cfg["generator_criteria"] = {"advencoder_cos": "cos", "advencoder_nce": "nce"}
    cfg["evaluation"]["state_indices"] = extra["evaluation_state_indices"]
    return cfg


def arm_info(method):
    require(method in ARMS, "Unknown selected method/batch")
    return ARMS[method]


def condition(method):
    arm_info(method)
    return f"{method}@500"


def deployment_conditions():
    return ["clean", *(condition(m) for m in METHODS)]


def sources():
    files = [*sorted((ROOT / "code").glob("*.py")), *sorted((ROOT / "vendor").glob("*")),
             *sorted((ROOT / "tests").glob("*.py")), *sorted((ROOT / "slurm").glob("*.sbatch")),
             ROOT / "slurm/environment.sh", ROOT / "config.json", ROOT / "PLAN.md", ROOT / "REUSE.json",
             PROJECT / "submit_exploring_oft_dino_batch500_32_v1.sh"]
    return {str(p.resolve()): sha256_file(p) for p in files if p.is_file()}


@lru_cache(maxsize=1)
def run_signature():
    return digest({"config": config(), "sources": sources(), "base_signature": BASE_SIGNATURE})


def preflight():
    cfg = config()
    require(base.run_signature() == BASE_SIGNATURE, "Frozen source experiment changed")
    inputs = base.preflight()
    reuse = load_json(ROOT / "REUSE.json")
    require(reuse["base_signature"] == BASE_SIGNATURE, "Source cache identity changed")
    require(all(sha256_file(p) == expected for p, expected in reuse["inputs"].items()), "Read-only bank/mean changed")
    for name, entry in reuse["copies"].items():
        require(sha256_file(entry["source"]) == entry["source_sha256"]
                and sha256_file(ROOT / name) == entry["copy_sha256"], f"Runtime/source copy changed: {name}")
    schedule = worker_schedule()
    require(sorted(m for worker in schedule for m in worker) == sorted(METHODS), "Worker grid misses/duplicates arms")
    report = {**inputs, "signature": run_signature(), "base_signature": BASE_SIGNATURE,
              "input_hashes": reuse["inputs"], "methods": len(METHODS), "selected_ids": list(FAMILIES),
              "optimizer_updates_per_arm": 25000, "evaluation_checkpoints": [500],
              "evaluation_episodes": 13200, "training_gpu_cap": 6, "evaluation_gpu_cap": 8,
              "worker_schedule": schedule, "old_patch_loaded": False, "models_loaded": False,
              "slurm_submitted": False, "nce_b1": cfg["nce_b1"]}
    return report


def prepared():
    value = load_json(ROOT / "reports/PREPARATION.json")
    require(value["signature"] == run_signature(), "Source/config changed after prepare")
    for group in ("model_jsons", "seed_files", "dataset_metadata", "input_hashes"):
        require(all(sha256_file(p) == h for p, h in value[group].items()), f"Prepared {group} changed")
    for group in ("dataset_stat", "model_weight_stat"):
        require(all(Path(p).stat().st_size == s["size"] and Path(p).stat().st_mtime_ns == s["mtime_ns"]
                    for p, s in value[group].items()), f"Prepared {group} changed")
    return value


def load_bank():
    import torch
    prepared()
    reuse = load_json(ROOT / "REUSE.json")
    bank = torch.load(reuse["bank_path"], map_location="cpu", weights_only=False)
    require(bank["signature"] == BASE_SIGNATURE and len(bank["candidates"]) == 250
            and len(bank["center_reference_uids"]) == 1024, "Frozen Long window pool changed")
    for method in METHODS:
        for outer in range(1, 501):
            batch_uids(bank["candidates"], outer, method)
    return bank


def batch_uids(schedule, outer, method):
    size = arm_info(method)["batch"]
    require(1 <= outer <= 500 and len(schedule) == 250, "Invalid approved outer/pool size")
    uids = [schedule[(size * (outer - 1) + j) % len(schedule)][0] for j in range(size)]
    require(len(set(uids)) == size, "Batch has duplicate records; do not fabricate NCE negatives")
    return uids


def eot_seeds(outer, inner, size):
    # B1 exactly preserves the source pixel/cos-B1 EOT sequence. For larger B,
    # j=0 also has that seed; j>0 is a disjoint deterministic sequence.
    initial = config()["data_seed"] * 1000000 + outer * 10000 + inner
    return [initial if j == 0 else initial + j * 100000000000000 for j in range(size)]


def runtime_config(suite="libero_10"):
    require(suite in SUITES, "Unknown suite")
    return {**config()["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}


def require_gpu(stage):
    return base.require_gpu(stage)


def estimated_hours(method):
    info = arm_info(method)
    per_b1 = {1: 100.81/60, 2: 18.05/60, 3: 19.51/60, 6: 18.53/60, 7: 17.79/60, 12: 101.76/60}
    i, b = info["id"], info["batch"]
    if i == 4:
        return 20.06/60 if b == 1 else 197.87/60*b/8
    if i == 9:
        return 210.75/60*b/8
    return per_b1[i]*b


def worker_schedule():
    # Fixed longest-first load balancing, six array children / six log pairs.
    workers, loads = [[] for _ in range(6)], [0.] * 6
    for method in sorted(METHODS, key=lambda m: (-estimated_hours(m), m)):
        index = min(range(6), key=lambda j: (loads[j], j))
        workers[index].append(method)
        loads[index] += estimated_hours(method)
    return workers
