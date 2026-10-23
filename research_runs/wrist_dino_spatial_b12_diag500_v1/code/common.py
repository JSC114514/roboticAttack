"""Spatial source, unchanged objectives/budgets, fixed four-suite pilot grid."""
from __future__ import annotations
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PARENT = ROOT.parent / "wrist_dino_batch500_32_v1"
FEATURE = ROOT.parent / "wrist_dino_feature_screen250_v1"
spec = importlib.util.spec_from_file_location("spatial_frozen_helpers", FEATURE / "code/common.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require, load_json, sha256_file, digest = base.require, base.load_json, base.sha256_file, base.digest
atomic_json, atomic_torch, atomic_parquet = base.atomic_json, base.atomic_torch, base.atomic_parquet
BASE_SIGNATURE = "f8f8bb030fd513c2fe244205bd0e833bf1a851fddbcc8e87f7970895d2ffd39b"
FAMILIES = {1:"uada_original", 2:"dino_key", 3:"dino_center", 4:"advencoder_cos",
            6:"dino_multilayer", 7:"dino_affinity", 9:"advencoder_nce", 12:"tma_xyz"}
BATCHES = (1, 2)
ARMS = {f"m{i:02d}_{n}_b{b}":{"id":i,"objective":n,"batch":b} for i,n in FAMILIES.items() for b in BATCHES}
METHODS = tuple(ARMS)
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EVAL_TASKS = tuple((s,t) for t in range(10) for s in ("libero_10","libero_goal","libero_object","libero_spatial"))

@lru_cache(maxsize=1)
def config():
    cfg = load_json(ROOT / "config.json")
    require(cfg["training_suite"] == "libero_spatial" and cfg["dataset_episodes"] == 432, "Wrong source")
    require(cfg["outer_iterations"] == 500 and cfg["inner_loop"] == 50 and cfg["bank_window_count"] == 250,
            "Training budget changed")
    require(cfg["batch_sizes"] == [1,2] and cfg["selected_ids"] == list(FAMILIES), "Arm grid changed")
    require(cfg["evaluation"]["state_indices"] == [40,41] and tuple(cfg["evaluation"]["suites"]) == SUITES,
            "Predeclared four-suite pilot grid changed")
    require(cfg["resources"]["training"]["max_concurrent"] == 6
            and cfg["resources"]["evaluation"]["max_concurrent"] == 8, "Resource cap changed")
    cfg["methods"] = list(METHODS)
    cfg["batch_size"] = {m:ARMS[m]["batch"] for m in METHODS}
    cfg["generator"]["criterion"] = "specified_by_objective"
    cfg["generator_criteria"] = {"advencoder_cos":"cos","advencoder_nce":"nce"}
    return cfg

def arm_info(method):
    require(method in ARMS, "Unknown arm")
    return ARMS[method]

def condition(method):
    arm_info(method)
    return method + "@500"

def deployment_conditions():
    return ["clean", *(condition(m) for m in METHODS)]

def sources():
    files = [*ROOT.glob("code/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("vendor/*"),
             *ROOT.glob("slurm/*"), ROOT/"config.json", ROOT/"PLAN.md", ROOT/"SOURCE_PROVENANCE.json"]
    return {str(p.resolve()):sha256_file(p) for p in sorted(files) if p.is_file()}

@lru_cache(maxsize=1)
def run_signature():
    return digest({"config":config(), "sources":sources(), "parent":BASE_SIGNATURE})

def dataset_files():
    directory=Path(config()["dataset_directory"])
    meta=load_json(directory/"dataset_info.json")
    expected=len(next(s for s in meta["splits"] if s["name"]=="train")["shardLengths"])
    files = sorted(directory.glob("*-train.tfrecord-*-of-*"))
    require(len(files) == expected == 16 and all(p.stat().st_size > 0 for p in files), "Spatial source shards incomplete")
    return files

def preflight():
    # Parent helper verifies all four local checkpoint interfaces and reset seeds.
    out = base.preflight()
    cfg = config()
    folder = Path(cfg["dataset_directory"])
    require(sha256_file(folder/"dataset_info.json") == cfg["dataset_info_sha256"], "Source metadata changed")
    meta = load_json(folder/"dataset_info.json")
    require(sum(map(int, meta["splits"][0]["shardLengths"])) == 432, "Spatial episode count changed")
    out.update(signature=run_signature(), official_train_episodes=432, methods=len(METHODS),
               dataset_metadata={str((folder/n).resolve()):sha256_file(folder/n) for n in ("dataset_info.json","features.json")},
               dataset_stat={str(p.resolve()):{"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} for p in dataset_files()},
               evaluation_episodes=1360, training_suite=cfg["training_suite"],
               source_code_hashes=sources(), worker_schedule=worker_schedule())
    return out

def prepared():
    out = load_json(ROOT / "reports/PREPARATION.json")
    require(out["signature"] == run_signature(), "Source/config drift after preparation")
    for group in ("model_jsons","seed_files","dataset_metadata"):
        require(all(sha256_file(p) == h for p,h in out[group].items()), f"Prepared {group} changed")
    for group in ("dataset_stat","model_weight_stat"):
        require(all(Path(p).stat().st_size == v["size"] and Path(p).stat().st_mtime_ns == v["mtime_ns"]
                    for p,v in out[group].items()), f"Prepared {group} changed")
    return out

def load_bank():
    import torch
    prepared()
    path = ROOT/"data/CANDIDATE_BANK.pt"
    receipt = load_json(ROOT/"data/DATA_STATUS.json")
    require(receipt["signature"] == run_signature() and receipt["bank_sha256"] == sha256_file(path), "Bank changed")
    out = torch.load(path,map_location="cpu",weights_only=False)
    require(out["signature"] == run_signature() and len(out["candidates"]) == 250
            and len(out["center_reference_uids"]) == 1024, "Wrong bank dimensions")
    require(all(r["source_suite"] == "libero_spatial" and r["source_split"] == "train" for r in out["records"].values()),
            "Wrong source records")
    return out

def batch_uids(schedule, outer, method):
    b = arm_info(method)["batch"]
    require(1 <= outer <= 500 and len(schedule) == 250, "Invalid batch schedule")
    ids = [schedule[(b*(outer-1)+j)%250][0] for j in range(b)]
    require(len(set(ids)) == b, "Duplicate NCE negatives")
    return ids

def eot_seeds(outer, inner, size):
    initial = config()["data_seed"]*1000000 + outer*10000 + inner
    return [initial if j==0 else initial+j*100000000000000 for j in range(size)]

def runtime_config(suite="libero_spatial"):
    require(suite in SUITES,"Unknown suite")
    return {**config()["evaluation"],"env_image_resolution":256,"policy_image_resolution":224}

def require_gpu(stage):
    import torch
    require(stage in {"training","evaluation"} and os.environ.get("SLURM_JOB_ID"),"Allocated Slurm GPU required")
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"Exactly one allocated GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name,"4090 required for comparable timing")
    os.environ["OFT_STAGE_GPU"]="4090"
    return name

def worker_schedule():
    import csv
    times={r["method"]:float(r["elapsed_h"]) for r in csv.DictReader((PARENT/"results/BATCH500_TRAINING_COST.csv").open())}
    workers,loads=[[] for _ in range(6)],[0.]*6
    for m in sorted(METHODS,key=lambda m:(-times[m],m)):
        i=min(range(6),key=lambda j:(loads[j],j));workers[i].append(m);loads[i]+=times[m]
    return workers
