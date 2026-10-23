#!/usr/bin/env python3
"""Manual-only train entry for matched Wrist-site UADA and DINO state sampling."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "VLAAttacker", ROOT / "VLAAttacker/white_patch", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper, require  # noqa: E402
from white_patch.UADA_OFT_DINO import sha256_file  # noqa: E402
from white_patch.UADA_OFT_DINO_SAMPLING import (  # noqa: E402
    METHODS, ExploringOFTDinoSamplingAttacker, dino_sampling_scores,
)

SOURCE_SHA256 = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"


def validate(cfg: dict, cfg_path: Path, method: str) -> tuple[int, int]:
    require(cfg["experiment_id"] == "EXPLORING-OFT-DINO-SAMPLING500-V1", "Experiment ID drift")
    require(tuple(cfg["methods"]) == METHODS and method in METHODS, "Method coverage drift")
    require(cfg["outer_iterations"] == 500 and cfg["inner_loop"] == 50, "Training budget drift")
    require(cfg["suite"] == "libero_object" and cfg["patch_shape"] == [3, 50, 50]
            and cfg["maskidx"] == [0] and cfg["batch_size"] == 1, "UADA contract drift")
    require(cfg["geometry"] is True and cfg["resize_patch"] is False,
            "Matched geometry contract drift")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_SHA256 and not (source.stat().st_mode & stat.S_IWUSR),
            "Read-only Exploring source drift")
    require(Path(cfg["training_bank"]).is_file(), "Missing frozen bank")
    require(Path(cfg["baseline_patch"]).is_file()
            and sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"],
            "Frozen UADA500 patch drift")
    receipt = json.loads(Path(cfg["placement_receipt"]).read_text())
    require(receipt["status"] == "PLACEMENT_CPU_AUDIT_COMPLETE"
            and receipt["training_bank_sha256"] == sha256_file(cfg["training_bank"])
            and receipt["patch_shape"] == cfg["patch_shape"]
            and receipt["image_size"] == 224, "Placement source drift")
    site = tuple(map(int, receipt["dino_xy"]))
    require(site == (91, 140), "DINO site drift; review plan before training")
    model_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text())
    ids = model_cfg.get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower() and "siglip" in ids[1].lower(),
            "OFT visual branch order drift")
    baseline = json.loads((cfg_path.parent / "object_scale500.json").read_text())
    matched = ("seed", "data_seed", "suite", "unnorm_key", "checkpoint", "training_bank",
               "patch_shape", "maskidx", "outer_iterations", "inner_loop", "batch_size",
               "learning_rate", "weight_decay", "warmup_outer_iterations", "geometry",
               "resize_patch", "checkpoint_outer_iterations", "resume_every_outer_iterations",
               "action_head_q01", "action_head_q99", "structure_gates", "dtype")
    require(all(cfg[k] == baseline[k] for k in matched), "Frozen UADA500 parameter drift")
    return site


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exploring_oft_dino/dino_sampling500_v1.json")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = json.loads(cfg_path.read_text())
    site = validate(cfg, cfg_path, args.method)
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    records = bank["records"]
    scores = dino_sampling_scores(bank, site)
    if args.dry_run:
        values = list(scores.values())
        print(json.dumps({"status": "DRY_RUN_PASS", "method": args.method,
                          "site_xy": site, "records": len(records), "dino_score_min": min(values),
                          "dino_score_max": max(values), "updates": 25_000,
                          "source_uada": "read-only", "gpu_job_submitted": False}, indent=2))
        return
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped GPU required")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}), flush=True)
    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTDinoSamplingAttacker(wrapper, cfg["output_root"], site)
    attacker.train(records, cfg, args.method, scores)


if __name__ == "__main__":
    main()
