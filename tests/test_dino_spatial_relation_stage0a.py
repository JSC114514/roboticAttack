"""CPU checks for the four-task spatial-readout pilot."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_dino_spatial_relation_stage0a", ROOT / "scripts/probe_dino_spatial_relation_stage0a.py")
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def config():
    return json.loads((ROOT / "configs/exploring_oft_dino/spatial_relation_stage0a.json").read_text())


def test_source_is_four_trajectory_separated_tasks():
    cfg = config()
    assert cfg["task_ids"] == [0, 1, 2, 4]
    frame = probe.load_records(cfg)
    assert len(frame) == 240
    assert frame.groupby(["task_id", "split"]).size().to_dict() == {
        (task, split): count for task in cfg["task_ids"]
        for split, count in (("discovery", 30), ("validation", 10), ("test", 20))
    }
    assert frame.groupby("trajectory_id").split.nunique().eq(1).all()


def test_oracle_mask_transform_and_centroid():
    mask = np.zeros((256, 256), np.uint8)
    mask[96:160, 96:160] = 1
    weights = probe.transform_mask(mask)
    assert weights.shape == (256,)
    assert 0 < float(weights.sum()) < 256
    assert torch.allclose(probe.token_centroid(weights), torch.tensor([8.0, 8.0]), atol=0.25)


def test_patch_exclusion_does_not_use_patch_tokens_as_oracle():
    cfg = config()
    excluded = probe.patch_exclusion(cfg)
    assert excluded.shape == (256,)
    assert 0 < int(excluded.sum()) < 256
    mask = torch.ones(256)
    visible = mask.masked_fill(excluded, 0)
    assert float(visible.sum()) == 256 - int(excluded.sum())
    assert torch.isfinite(probe.token_centroid(visible)).all()


def test_readout_exclusion_masks_scores():
    features = torch.zeros((256, 2))
    features[:, 0] = 1
    prototypes = {"target_object": torch.tensor([1.0, 0.0]),
                  "gripper": torch.tensor([1.0, 0.0]),
                  "matched_background": torch.tensor([0.0, 1.0])}
    excluded = torch.zeros(256, dtype=torch.bool)
    excluded[:128] = True
    centers = probe.readout(features, prototypes, 0.05, excluded)
    assert float(centers["target_object"][1]) > 8
    assert float(centers["gripper"][1]) > 8
