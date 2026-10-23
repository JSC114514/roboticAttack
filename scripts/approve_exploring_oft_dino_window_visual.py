#!/usr/bin/env python3
"""Record user approval of eight final PNG/PT pairs after visual inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_dino_window_cross_suite import (
    CONFIG, CONDITIONS, atomic_json, load_config, require, training_patches,
)
from dino_window_common import config as train_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--approve-reviewed-eight-pngs", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _patches, hashes = training_patches(cfg, torch.device("cpu"), require_visual=False)
    root = Path(train_config(Path(cfg["train_config"]))["output_root"])
    paths = {method: str(root / "training" / method / "outer_0200.png")
             for method in CONDITIONS[1:]}
    if not args.approve_reviewed_eight_pngs:
        print(json.dumps({"status": "AWAITING_USER_REVIEW", "pngs": paths,
                          "patch_hashes": hashes, "approval_written": False}, indent=2))
    else:
        marker = Path(cfg["visual_review"])
        require(not marker.exists(), "Existing visual approval marker must not be overwritten")
        atomic_json({"status": "APPROVED_BY_USER", "patches": hashes}, marker)
        print(json.dumps({"status": "APPROVED_BY_USER", "marker": str(marker),
                          "pngs": paths}, indent=2))
