#!/usr/bin/env python3
"""Manual entry; CPU dry-run validates lineage/data without loading a VLA."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "VLAAttacker", PROJECT_ROOT / "VLAAttacker/white_patch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.UADA_OFT_ILPD_V2 import METHODS, ExploringOFTILPDV2Attacker, source_manifest
from white_patch.oft_correspondence_objectives import total_variation

SOURCE_SHA = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
CONFIG = PROJECT_ROOT / "configs/exploring_oft_dino/object_dino_ilpd_refine200_v2.json"


def validate_config(cfg):
    require(cfg["experiment_id"] == "OFT-DINO-ILPD-REFINE200-V2", "Experiment ID drift")
    require(tuple(cfg["methods"]) == METHODS, "Refinement method list drift")
    v1 = json.loads((PROJECT_ROOT / "configs/exploring_oft_dino/object_dino_ilpd_refine200_v1.json").read_text())
    require({k: v for k, v in cfg.items() if k not in ("experiment_id", "output_root")}
            == {k: v for k, v in v1.items() if k not in ("experiment_id", "output_root")},
            "Engineering V2 may not change the V1 scientific settings")
    template = json.loads((PROJECT_ROOT / "configs/exploring_oft_dino/object_dino_ila_refine200_v2.json").read_text())
    require(set(cfg) == (set(template) - {"coverage_threshold", "min_reference_rms", "projection_eps"})
            | {"feature_alpha", "rgb_radius"}, "Config keys drift")
    original = PROJECT_ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(original) == SOURCE_SHA and original.stat().st_mode & 0o222 == 0,
            "Original UADA must retain frozen contents and read-only permissions")
    baseline = json.loads((PROJECT_ROOT / "configs/exploring_oft_dino/object_scale500.json").read_text())
    for key in ("seed", "data_seed", "suite", "unnorm_key", "checkpoint", "training_bank", "patch_shape",
                "maskidx", "inner_loop", "batch_size", "learning_rate", "weight_decay", "warmup_outer_iterations",
                "geometry", "resize_patch", "resume_every_outer_iterations", "action_head_q01", "action_head_q99",
                "num_open_loop_steps", "dtype", "openvla_oft_root", "libero_root", "attribution_root"):
        require(cfg[key] == baseline[key], f"UADA continuation contract drift: {key}")
    fixed = {"base_outer_iterations": 500, "outer_iterations": 200, "feature_alpha": 0.5, "rgb_radius": 16 / 255,
             "checkpoint_outer_iterations": [0, 25, 50, 100, 200], "tv_weight": 1.0,
             "baseline_patch_sha256": "c7ebb69204290fc193c3f694e25ba02c656de9e2df0443a74470d8ad587db136",
             "training_bank_sha256": "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8"}
    require(all(cfg[k] == v for k, v in fixed.items()), "Frozen refinement objective/budget drift")
    reference_path = Path(baseline["output_root"]) / "exploring_oft_uada/outer_0500.pt"
    require(Path(cfg["baseline_patch"]).resolve() == reference_path.resolve(), "Baseline lineage drift")
    require(sha256_file(reference_path) == cfg["baseline_patch_sha256"], "UADA500 patch changed")
    status = json.loads(reference_path.with_name("status.json").read_text())
    require(status["status"] == "COMPLETE" and status["method"] == "exploring_oft_uada"
            and status["outer_iterations"] == 500 and status["inner_loop"] == 50
            and status["training_bank_sha256"] == cfg["training_bank_sha256"], "Baseline completion/data drift")
    require(cfg["baseline_elapsed_s"] == status["elapsed_s"], "Baseline measured cost drift")
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"], "Training bank changed")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)["patch"]
    require(tuple(reference.shape) == (3, 50, 50) and bool(torch.isfinite(reference).all())
            and bool((reference >= 0).all() and (reference <= 1).all()), "Baseline RGB invalid")
    require(abs(float(total_variation(reference)) - cfg["tv_threshold"]) < 1e-8, "TV reference drift")
    ids = json.loads(Path(cfg["checkpoint"], "config.json").read_text()).get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower() and "siglip" in ids[1].lower(),
            "OFT checkpoint branch order drift")
    require(Path(cfg["output_root"]).resolve() != reference_path.parent.resolve(), "Output overlaps baseline")
    require(Path(cfg["output_root"]).resolve() == (PROJECT_ROOT / "research_runs/wrist_dino_ilpd_v1/training_v2").resolve(),
            "ILPD training must use a separate output directory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    records = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)["records"]
    keys = {(int(r["task_id"]), int(r["trajectory_rank"]), int(r["decision_index"])) for r in records}
    require(len(records) == len(keys) == 300 and {int(r["task_id"]) for r in records} == set(range(10)),
            "Expected 300 distinct Object training states across 10 tasks")
    sources = source_manifest(cfg)
    if args.dry_run:
        states = {}
        for method in METHODS:
            path = Path(cfg["output_root"]) / method / "status.json"
            states[method] = json.loads(path.read_text()).get("status") if path.exists() else "NOT_STARTED"
        print(json.dumps({"status": "DINO_ILPD_REFINE200_V2_DRY_RUN_PASS", "methods": list(METHODS),
                          "training_records": len(records), "parameter_count": 7500,
                          "warm_start": "accepted frozen UADA500", "additional_updates_per_patch": 10000,
                          "total_updates_per_patch": 35000, "action_updates_per_patch": 10000, "feature_updates_per_patch": 0,
                          "feature_alpha": cfg["feature_alpha"], "rgb_radius": cfg["rgb_radius"],
                          "output_root": cfg["output_root"], "source_files": len(sources), "run_states": states,
                          "engineering_revision": 2, "old_failure_directory_preserved": str(PROJECT_ROOT / "research_runs/wrist_dino_ilpd_v1/training"),
                          "model_loaded": False, "sbatch_submitted": False}, indent=2))
        return
    require(args.method is not None, "Training requires --method")
    require(bool(os.environ.get("SLURM_JOB_ID")), "Real-model training requires a user-submitted Slurm job")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one mapped CUDA GPU required")
    out = Path(cfg["output_root"]) / args.method
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = out / "status.json"
        require(not status_path.exists() or json.loads(status_path.read_text()).get("status") != "COMPLETE",
                "Already COMPLETE; refusing to overwrite")
        print(json.dumps({"method": args.method, "gpu": torch.cuda.get_device_name(0),
                          "slurm_job_id": os.environ["SLURM_JOB_ID"]}), flush=True)
        load_started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        model_load_s = time.monotonic() - load_started
        require(not any(p.requires_grad for p in wrapper.vla.parameters()), "VLA weights must stay frozen")
        sources = source_manifest(cfg, wrapper)
        attacker = ExploringOFTILPDV2Attacker(wrapper, cfg["output_root"], resize_patch=False)
        attacker.train(records, cfg, args.method, sources,
                       {"training_bank": cfg["training_bank_sha256"], "reference_patch": cfg["baseline_patch_sha256"],
                        "reference_status": sha256_file(Path(cfg["baseline_patch"]).with_name("status.json"))},
                       model_load_s=model_load_s)


if __name__ == "__main__":
    main()


