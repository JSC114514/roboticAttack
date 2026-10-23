#!/usr/bin/env python3
"""Fresh original-style Exploring→OFT UADA run; no v2 prefix or TV term."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "VLAAttacker", ROOT / "VLAAttacker" / "white_patch"):
    sys.path.insert(0, str(path))
from UADA_OFT_DINO_wrapper import load_wrapper, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = ROOT / "configs/exploring_oft_dino/uada_original1000_v1.json"
    cfg = json.loads(path.read_text())
    require(cfg["experiment_id"] == "EXPLORING-OFT-UADA-ORIGINAL1000-V1"
            and cfg["methods"] == ["exploring_oft_uada"], "Baseline identity changed")
    require(cfg["outer_iterations"] == 1000 and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 500, 750, 1000]
            and cfg["maskidx"] == [0] and cfg["structure_gates"] == {}, "Baseline budget/loss drift")
    require(Path(cfg["training_bank"]).is_file(), "Training bank missing")
    require(Path(cfg["checkpoint"], "config.json").is_file(), "OFT checkpoint missing")
    model_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text())
    ids = model_cfg.get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower() and "siglip" in ids[1].lower(),
            "OFT visual branch order changed")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "updates": 50000,
                          "checkpoints": cfg["checkpoint_outer_iterations"],
                          "source": "white_patch/UADA_OFT_DINO.py::exploring_oft_uada"}, indent=2))
        return
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One mapped GPU required")
    from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    require(len(bank["records"]) == 300, "Training bank size drift")
    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTOpenVLAAttacker(wrapper, cfg["output_root"], resize_patch=False)
    attacker.patchattack_unconstrained_oft(bank["records"], cfg, "exploring_oft_uada")


if __name__ == "__main__":
    main()
