from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VLAAttacker"))

from VLAAttacker.UADA_OFT_JOINT500_wrapper import validate_config
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_JOINT500 import (
    METHODS, combine_gradients, feature_disruption_loss, persistent6d_loss,
    persistent6d_target, signed_direction_loss,
)

CONFIG = ROOT / "configs/exploring_oft_dino/object_joint500_v1.json"


def test_read_only_source_is_exact_and_experiments_use_copy() -> None:
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    assert sha256_file(source) == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
    assert not (source.stat().st_mode & stat.S_IWUSR)
    joint = (ROOT / "VLAAttacker/white_patch/UADA_OFT_JOINT500.py").read_text()
    assert "from white_patch.UADA_OFT_SOURCE_COPY import" in joint


def test_six_arm_config_and_frozen_contract() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert tuple(cfg["methods"]) == METHODS and len(METHODS) == 6
    for method in METHODS:
        validate_config(cfg, CONFIG, method)
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25_000
    assert cfg["output_root"].endswith("object_joint500_v1")


def test_feature_loss_and_gradient_budget() -> None:
    clean = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    attacked = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], requires_grad=True)
    feature_disruption_loss(attacked, clean).backward()
    assert attacked.grad is not None and attacked.grad.abs().sum() > 0
    merged, weight, cosine, ratio = combine_gradients(
        torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0]), 0.3, None)
    assert torch.allclose(merged, torch.tensor([3.0, 5.5]))
    assert weight == pytest.approx(0.75) and cosine == pytest.approx(0.8)
    assert ratio == pytest.approx(0.3)


def test_persistent6d_uses_one_direction_per_chunk_and_normalizes_dofs() -> None:
    clean = torch.zeros(1, 8, 7)
    clean[:, 1::2, :6] = 0.2
    low = torch.full((1, 1, 7), -1.0)
    high = torch.full((1, 1, 7), 1.0)
    target = persistent6d_target(clean, low, high)
    assert target.shape == (1, 8, 6)
    assert torch.equal(target[:, :1].expand_as(target), target)
    at_target = torch.cat((target, torch.zeros(1, 8, 1)), dim=-1)
    assert float(persistent6d_loss(at_target, target, low, high, 1.0)) == pytest.approx(0.0)


def test_signed_sham_preserves_scale_but_changes_token_assignment() -> None:
    clean = torch.zeros(1, 4, 3)
    direction = torch.arange(12, dtype=torch.float32)
    attacked = direction.reshape_as(clean)
    _, score = signed_direction_loss(attacked, clean, direction, 2.0)
    _, sham = signed_direction_loss(attacked, clean, direction, 2.0, sham_seed=7)
    assert score != sham


def test_slurm_is_manual_six_arm_3090_array() -> None:
    source = (ROOT / "slurm/exploring_oft_dino/31_train_joint500_v1_3090.sbatch").read_text()
    assert "#SBATCH --array=0-5%6" in source
    assert "#SBATCH --gres=gpu:3090:1" in source
    assert source.count("exploring_oft_") >= 7
    assert "sbatch " not in source
