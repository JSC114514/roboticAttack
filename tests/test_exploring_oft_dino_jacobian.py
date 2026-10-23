from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VLAAttacker"))

from VLAAttacker.UADA_OFT_DINO_JACOBIAN_wrapper import validate_config
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_DINO_JACOBIAN import (
    METHODS,
    dino_action_descent_direction,
    dino_directional_loss,
)


CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_jacobian500_v1.json"


def test_jacobian_config_uses_read_only_source_copy_and_frozen_budget() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert tuple(cfg["methods"]) == METHODS and len(METHODS) == 2
    for method in METHODS:
        validate_config(cfg, CONFIG, method)
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    assert sha256_file(source) == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
    assert not (source.stat().st_mode & stat.S_IWUSR)
    trainer = (ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO_JACOBIAN.py").read_text()
    assert "from white_patch.UADA_OFT_SOURCE_COPY import" in trainer
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25_000


def test_dino_action_direction_points_toward_action_target() -> None:
    dino = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    action = dino[:, :1, :1].repeat(1, 8, 7)
    target = torch.zeros_like(action)
    direction, token_norm, loss = dino_action_descent_direction(action, dino, target, [0])
    assert float(loss) > 0
    assert direction.shape == dino.shape
    assert token_norm.shape == dino.shape[:2]
    assert direction[0, 0, 0] < 0
    assert direction[0, 1].abs().sum() == 0


def test_dino_directional_loss_rewards_movement_along_jacobian_direction() -> None:
    clean = torch.zeros(1, 4, 3)
    direction = torch.zeros_like(clean)
    direction[:, :, 0] = 1.0
    patch_tokens = torch.ones(4)
    sensitivity = torch.tensor([[1.0, 2.0, 1.0, 2.0]])
    good = clean + direction
    bad = clean - direction
    good_loss, good_stats = dino_directional_loss(good, clean, direction, patch_tokens, sensitivity)
    bad_loss, bad_stats = dino_directional_loss(bad, clean, direction, patch_tokens, sensitivity)
    assert float(good_loss) < float(bad_loss)
    assert good_stats["dino_direction_score"] > bad_stats["dino_direction_score"]


def test_slurm_is_manual_two_arm_3090_array() -> None:
    source = (ROOT / "slurm/exploring_oft_dino/47_train_dino_jacobian500_v1_3090.sbatch").read_text()
    submit = (ROOT / "submit_exploring_oft_dino_jacobian500.sh").read_text()
    assert "#SBATCH --array=0-1%2" in source
    assert "#SBATCH --gres=gpu:3090:1" in source
    assert "UADA_OFT_DINO_JACOBIAN_wrapper.py" in source
    assert "sbatch " not in source
    assert "CHECK_ONLY" in submit
