"""Lightweight contracts for the new transfer training arms."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts")]

from dino_transfer_common import METHODS, CHECKPOINTS, config, draw_record
from white_patch.UADA_OFT_DINO_TRANSFER import ExploringOFTDINOTransferAttacker
from white_patch.appply_random_transform import RandomPatchTransform


def test_transfer_is_from_scratch_with_matched_checkpoints():
    cfg = config()
    assert cfg["base_outer_iterations"] == 0
    assert cfg["bootstrap_outer"] == 250
    assert cfg["outer_iterations"] == 2000
    assert CHECKPOINTS == (1000, 2000)
    assert METHODS == ("uniform", "static", "dynamic", "diverse")
    assert cfg["refresh_after_outer"] == [500, 1000, 1500]


def test_diverse_paste_changes_rgb_and_preserves_gradient_and_site():
    attacker = ExploringOFTDINOTransferAttacker.__new__(ExploringOFTDINOTransferAttacker)
    attacker.wrapper = SimpleNamespace(device=torch.device("cpu"))
    attacker.randomPatchTransform = RandomPatchTransform(torch.device("cpu"), False)
    attacker.site = (26, 137)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    patch = torch.full((3, 50, 50), 0.5, requires_grad=True)
    attacker.transfer_diverse = True
    random.seed(123)
    np.random.seed(123)
    rgb, support = attacker._paste_with_support(image, patch, True)
    assert rgb.shape == (3, 224, 224)
    assert support.shape == (224, 224)
    assert attacker.site == (26, 137)
    rgb.sum().backward()
    assert patch.grad is not None and patch.grad.norm().item() > 0


def test_uniform_and_dino_weighting_use_same_task_and_phase():
    cfg = config()
    clean = [{"task_id": i % 10, "id": i} for i in range(300)]
    replay = [{"task_id": task, "phase_proxy": phase,
               "dino_weight": float(weight), "id": (task, phase, weight)}
              for task in range(10) for phase in ("early", "middle", "late")
              for weight in (1, 2)]
    for method in METHODS:
        record, origin, index, base_index = draw_record(clean, replay, method, 252, cfg)
        assert origin == "replay"
        assert record["task_id"] == clean[base_index]["task_id"]
        assert record["phase_proxy"] == "early"
        assert 0 <= index < len(replay)
