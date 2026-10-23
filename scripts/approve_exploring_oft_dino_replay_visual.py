#!/usr/bin/env python3
"""Explicit user-side approval of four frozen final PNG/PT pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_exploring_oft_dino_replay_cross_suite import (
    CONFIG, METHODS, atomic_json, load_config, require, training_patches,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--approve-reviewed-four-pngs", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    _patches, hashes = training_patches(cfg, torch.device("cpu"), require_visual=False)
    root = Path(json.loads(Path(cfg["train_config"]).read_text())["output_root"])
    paths = {method: str(root / "training" / method / "outer_0200.png") for method in METHODS}
    if not args.approve_reviewed_four_pngs:
        print(json.dumps({"status": "AWAITING_USER_REVIEW", "pngs": paths,
                          "patch_hashes": hashes, "approval_written": False}, indent=2))
    else:
        marker = Path(cfg["visual_review"])
        require(not marker.exists(), "Existing visual approval marker must not be overwritten")
        atomic_json({"status": "APPROVED_BY_USER", "patches": hashes}, marker)
        print(json.dumps({"status": "APPROVED_BY_USER", "marker": str(marker), "pngs": paths}, indent=2))
