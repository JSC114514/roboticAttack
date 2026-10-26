"""Frozen factorial contracts; no import performs GPU, queue or file mutations."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
METHODS = ("legacy_freq", "original_dino", "original_freq")
SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
CELLS = ("uada_original", "uada_dino", "uada_freq", "uada_dino_freq")

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

def experiment():
    value = load_json(ROOT / "config.json")
    expected = [("legacy_freq", "legacy", False, True, 1000),
                ("original_dino", "original", True, False, 2000),
                ("original_freq", "original", False, True, 2000)]
    actual = [(a["name"], a["protocol"], a["sampling"], a["frequency"], a["schedule_outer_iterations"])
              for a in value["arms"]]
    require(value["experiment_id"] == "OFT-UADA-FACTORIAL1000-V1" and actual == expected
            and value["seed"] == 42 and value["outer_iterations"] == 1000
            and value["inner_loop"] == 50 and value["new_episodes"] == 1200
            and value["combined_episodes"] == 4000 and value["training_gpu_cap"] == 3
            and value["evaluation_gpu_cap"] == 4 and not value["future_seed_submitted"], "Approved factorial contract changed")
    return value

def arm_info(name=None):
    name = name or os.environ.get("OFT_FACTORIAL_ARM")
    choices = [a for a in experiment()["arms"] if a["name"] == name]
    require(len(choices) == 1, f"Explicit factorial arm required: {name}")
    return choices[0]

def lineage():
    return load_json(ROOT / "SOURCE_LINEAGE.json")

def parent_config(protocol):
    require(protocol in ("legacy", "original"), "Unknown frozen protocol")
    return load_json(ROOT / "source" / protocol / "config.json")

def config(name=None):
    info = arm_info(name)
    cfg = parent_config(info["protocol"])
    cfg.update(experiment_id=experiment()["experiment_id"], methods=list(METHODS),
               outer_iterations=1000, schedule_outer_iterations=info["schedule_outer_iterations"],
               checkpoint_outer_iterations=[0, 250, 500, 750, 1000], evaluation_checkpoints=[1000],
               arm=info["name"], protocol=info["protocol"])
    cfg["resources"] = {"training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 3},
                        "evaluation": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 4}}
    require(cfg["uada_maskidx"] == [0] and cfg["inner_loop"] == 50 and cfg["batch_size"] == 1
            and cfg["learning_rate"] == .002 and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20 and cfg["seed"] == 42, "Native optimization contract changed")
    return cfg

def flags(method):
    info = arm_info(method)
    return dict(family="uada", sampling=info["sampling"], frequency=info["frequency"], geometry=False, attention=False)

def scoring_diverse(cfg):
    return cfg["protocol"] == "original"

def training_diverse(cfg, inner):
    return cfg["protocol"] == "original" or inner >= cfg["eot"]["fixed_inner_count"]

def runtime_config(suite="libero_10", protocol=None):
    require(suite in SUITES, "Unknown suite")
    protocol = protocol or arm_info()["protocol"]
    return {**parent_config(protocol)["evaluation"], "env_image_resolution": 256, "policy_image_resolution": 224}

def require_gpu(stage):
    import torch
    require(stage in {"training", "evaluation"} and os.environ.get("SLURM_JOB_ID"), "Manual Slurm allocation required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one allocated CUDA GPU required")
    name = torch.cuda.get_device_name(0)
    require("4090" in name, f"RTX4090 required, got {name}")
    os.environ["OFT_STAGE_GPU"] = "4090"
    return name

def sources():
    paths = [ROOT / n for n in ("config.json", "PLAN.md", "SOURCE_LINEAGE.json")]
    paths += list((ROOT / "code").glob("*.py")) + list((ROOT / "slurm").glob("*"))
    paths += [PROJECT / "submit_wrist_dino_uada_factorial1000_v1.sh"]
    paths += [ROOT / p for parent in lineage().values() for p in parent["snapshots"]]
    # Include shared victim code and every parent source, detecting external drift too.
    paths += [Path(p) for parent in lineage().values() for p in parent["sources"]]
    require(all(p.is_file() for p in paths), "Required source is missing")
    return {str(p.resolve()): sha256_file(p) for p in sorted(set(paths)) if p.is_file()}

def run_signature():
    return digest({"configuration": experiment(), "sources": sources()})

def verify_parent(protocol, banks=False):
    parent = lineage()[protocol]
    for name, checksum in parent["sources"].items():
        require(sha256_file(name) == checksum, f"Historical source drift: {name}")
    for name, checksum in parent["snapshots"].items():
        require(sha256_file(ROOT / name) == checksum, f"Frozen snapshot drift: {name}")
    for name, checksum in parent["artifacts"].items():
        if banks or name != parent["bank_path"]:
            require(sha256_file(name) == checksum, f"Historical artifact drift: {name}")
    prep = load_json(Path(parent["root"]) / "reports/PREPARATION.json")
    require(prep["signature"] == parent["signature"], "Parent preparation signature changed")
    for group in ("model_jsons", "seed_files", "dataset_metadata"):
        for name, checksum in prep[group].items():
            require(sha256_file(name) == checksum, f"Historical {group} drift: {name}")
    for group in ("model_weight_stat", "dataset_stat"):
        for name, stat in prep[group].items():
            actual = Path(name).stat()
            require(actual.st_size == stat["size"] and actual.st_mtime_ns == stat["mtime_ns"], f"Historical {group} drift: {name}")
    for name in ("model_runtime.py", "deployment.py", "readonly_model.py", "preprocessing.py",
                 "objectives.py", "UADA_SOURCE_COPY.py", "source_transform.py"):
        require(sha256_file(ROOT / "code" / name) == parent["snapshots"][f"source/{protocol}/code/{name}"],
                f"Native shared implementation differs: {name}")
    require(sha256_file(ROOT / "code" / f"engine_{protocol}.py") == parent["snapshots"][f"source/{protocol}/code/engine.py"],
            "Native EOT engine changed")
    return parent

def history_rows(protocol):
    import pandas as pd
    parent = lineage()[protocol]
    frame = pd.read_parquet(parent["raw_path"])
    selected = ["clean", *(f"{m}@1000" for m in parent["keep_methods"])]
    frame = frame.loc[frame.condition.isin(selected)].copy()
    require(frame.signature.eq(parent["signature"]).all(), "Historical rollout signature changed")
    frame["origin_signature"] = frame["signature"]
    frame["origin_file"] = parent["raw_path"]
    frame["protocol"] = protocol; frame["reused"] = True
    frame["condition"] = frame.condition.str.replace("@1000", "", regex=False)
    require(len(frame) == 400 * len(selected) and not frame.duplicated(["suite", "task_id", "state_index", "condition"]).any(),
            "Historical factorial grid incomplete/duplicated")
    for suite in SUITES:
        expected = {(t, s, c) for t in range(10) for s in range(40, 50) for c in [v.split("@")[0] for v in selected]}
        subset = frame.loc[frame.suite.eq(suite)]
        require(set(zip(subset.task_id, subset.state_index, subset.condition)) == expected, "Historical task/state coverage changed")
        seeds = pd.read_parquet(parent_config(protocol)["evaluation"]["seed_sources"][suite])
        seeds = seeds.loc[seeds.suite.eq(suite) & seeds.condition.eq("clean"), ["task_id", "state_index", "rollout_seed"]]
        checked = subset.merge(seeds, on=["task_id", "state_index"], suffixes=("", "_frozen"), validate="many_to_one")
        require(checked.rollout_seed.eq(checked.rollout_seed_frozen).all(), "Historical reset seeds differ")
    for method in parent["keep_methods"]:
        checksum = parent["artifacts"][str(Path(parent["root"]) / "training" / method / "outer_1000.png")]
        require(frame.loc[frame.condition.eq(method), "deployment_png_sha256"].eq(checksum).all(), "Historical deployed PNG mismatch")
    require(not frame.evaluation_training_hooks.any(), "Historical evaluation used training hooks")
    return frame

def preflight(banks=True):
    import torch
    all_initial = []
    for protocol in ("legacy", "original"):
        parent = verify_parent(protocol, banks=banks)
        history_rows(protocol)
        for method in parent["keep_methods"]:
            value = torch.load(Path(parent["root"]) / "training" / method / "outer_0000.pt", map_location="cpu", weights_only=False)
            expected = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))
            require(torch.equal(value["patch"], expected), "Historical initialization differs from fresh seed42")
            all_initial.append(digest(value["patch"].flatten().tolist()))
    require(len(set(all_initial)) == 1, "Factorial initialization is not matched")
    return {"status": "CPU_PREFLIGHT_PASS", "signature": run_signature(), "sources": sources(),
            "new_training_arms": 3, "new_episodes": 1200, "reused_episodes": 2800,
            "independent_training_seeds": 1, "parent_banks_verified": banks,
            "models_loaded": False, "slurm_queried": False, "slurm_submitted": False}

def prepare():
    path = ROOT / "reports/PREPARATION.json"
    if path.exists():
        prepared()
        print("FACTORIAL_PREPARATION_ALREADY_COMPLETE", flush=True)
        return
    report = preflight(banks=True)
    import pandas as pd
    rows = pd.concat([history_rows(p) for p in ("legacy", "original")], ignore_index=True)
    atomic_parquet(rows, ROOT / "data/HISTORICAL_ROLLOUTS.parquet")
    report["history_sha256"] = sha256_file(ROOT / "data/HISTORICAL_ROLLOUTS.parquet")
    atomic_json(report, path)
    print("FACTORIAL_PREPARED: historical2800 verified; no GPU/model/Slurm started", flush=True)

def prepared():
    report = load_json(ROOT / "reports/PREPARATION.json")
    require(report["status"] == "CPU_PREFLIGHT_PASS" and report["signature"] == run_signature()
            and report["parent_banks_verified"], "Factorial preparation drift/incomplete")
    for p in ("legacy", "original"):
        verify_parent(p, banks=False)
    require(sha256_file(ROOT / "data/HISTORICAL_ROLLOUTS.parquet") == report["history_sha256"], "Prepared historical rows changed")
    return report

def load_bank():
    import torch
    parent = lineage()[arm_info()["protocol"]]
    require(sha256_file(parent["bank_path"]) == parent["bank_sha256"], "Parent candidate bank changed")
    bank = torch.load(parent["bank_path"], map_location="cpu", weights_only=False)
    require(bank["signature"] == parent["signature"] and len(bank["candidates"]) == parent_config(arm_info()["protocol"])["outer_iterations"],
            "Parent bank identity/clock changed")
    bank["candidates"] = bank["candidates"][:1000]
    require(len(bank["candidates"]) == 1000 and all(len(x) == 4 for x in bank["candidates"]), "Candidate prefix malformed")
    wanted = {v for row in bank["candidates"] for v in row}
    bank["records"] = {k: bank["records"][k] for k in wanted}
    return bank
