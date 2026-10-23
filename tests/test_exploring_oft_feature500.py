from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VLAAttacker"))

from VLAAttacker.UADA_OFT_FEATURE_wrapper import validate_config
from white_patch.UADA_OFT_FEATURE import combine_gradients, feature_disruption_loss


CONFIG_PATH = ROOT / "configs/exploring_oft_dino/object_feature500.json"


def test_feature_loss_rewards_cosine_disruption_and_has_gradient() -> None:
    clean = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    same = clean.clone().requires_grad_()
    opposite = (-clean).requires_grad_()
    assert float(feature_disruption_loss(same, clean)) == pytest.approx(0.0)
    assert float(feature_disruption_loss(opposite, clean)) == pytest.approx(-2.0)
    trial = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]], requires_grad=True)
    feature_disruption_loss(trial, clean).backward()
    assert trial.grad is not None and trial.grad.abs().sum() > 0


def test_feature_loss_rejects_shape_drift() -> None:
    with pytest.raises(RuntimeError, match="Feature shape mismatch"):
        feature_disruption_loss(torch.ones(1, 2, 3), torch.ones(1, 3, 3))


def test_auxiliary_weight_freezes_and_matches_first_step_budget() -> None:
    action = torch.tensor([3.0, 4.0])
    feature = torch.tensor([0.0, 2.0])
    merged, weight, cosine, ratio = combine_gradients(action, feature, 0.3, None)
    assert weight == pytest.approx(0.75)
    assert ratio == pytest.approx(0.3)
    assert cosine == pytest.approx(0.8)
    assert torch.allclose(merged, torch.tensor([3.0, 5.5]))
    _, frozen, _, next_ratio = combine_gradients(action, feature * 2, 0.3, weight)
    assert frozen == weight and next_ratio == pytest.approx(0.6)


def test_config_matches_frozen_baseline_and_no_warm_start() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    for method in cfg["methods"]:
        validate_config(cfg, CONFIG_PATH, method)
    assert "initial_patch" not in cfg
    assert cfg["feature_gradient_ratio"] == 0.3
    assert cfg["maskidx"] == [0]
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25_000


def test_baseline_contract_drift_is_rejected() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["data_seed"] += 1
    with pytest.raises(RuntimeError, match="baseline contract drift: data_seed"):
        validate_config(cfg, CONFIG_PATH, cfg["methods"][0])


def test_new_subclass_keeps_exploring_update_and_one_forward() -> None:
    source = (ROOT / "VLAAttacker/white_patch/UADA_OFT_FEATURE.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ExploringOFTFeatureAttacker")
    assert any(isinstance(base, ast.Name) and base.id == "ExploringOFTOpenVLAAttacker" for base in cls.bases)
    for piece in ("init_uada_patch(", "build_uada_adamw(", "step_uada_patch(",
                  "self._training_index(", "self._forward(", "scheduler.step()"):
        assert piece in source
    assert "clean_siglip=clean_siglip" not in source
    assert "clean_dino=clean_dino" not in source
    assert 'capture_siglip=True)' in source
    assert 'feature_grad = torch.autograd.grad(feature_loss, patch)[0]' in source


def test_slurm_two_manual_3090_arms() -> None:
    source = (ROOT / "slurm/exploring_oft_dino/30_train_feature500_3090.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in source
    assert "#SBATCH --gres=gpu:3090:1" in source
    assert "METHODS=(exploring_oft_dino_feature exploring_oft_siglip_feature)" in source
    assert "sbatch " not in source
