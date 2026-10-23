from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "VLAAttacker"))

from VLAAttacker.UADA_OFT_OBJECT_DECOY_wrapper import validate_config
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_OBJECT_DECOY import (
    METHODS,
    load_object_masks,
    object_decoy_loss,
)


CONFIG = ROOT / "configs/exploring_oft_dino/object_decoy500_v1.json"


def test_decoy_config_uses_read_only_source_copy_and_frozen_budget() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert tuple(cfg["methods"]) == METHODS and len(METHODS) == 2
    for method in METHODS:
        validate_config(cfg, CONFIG, method)
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    assert sha256_file(source) == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
    assert not (source.stat().st_mode & stat.S_IWUSR)
    trainer = (ROOT / "VLAAttacker/white_patch/UADA_OFT_OBJECT_DECOY.py").read_text()
    assert "from white_patch.UADA_OFT_SOURCE_COPY import" in trainer
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25_000


def test_object_masks_cover_the_frozen_training_bank() -> None:
    cfg = json.loads(CONFIG.read_text())
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    masks = load_object_masks(cfg, bank["records"])
    assert len(bank["records"]) == 300
    assert len(masks) == 300
    for record in bank["records"][:5]:
        key = (int(record["task_id"]), int(record["trajectory_rank"]), int(record["decision_index"]))
        assert set(masks[key]) == {"target_object", "matched_background"}
        assert all(float(value.sum()) > 0 for value in masks[key].values())


def test_object_decoy_loss_rewards_patch_as_target_and_object_as_background() -> None:
    clean = torch.zeros(1, 4, 3)
    clean[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    clean[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    clean[0, 2] = torch.tensor([0.0, 1.0, 0.0])
    clean[0, 3] = torch.tensor([0.0, 1.0, 0.0])
    good = clean.clone()
    good[0, 0] = torch.tensor([0.0, 1.0, 0.0])
    good[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    good[0, 2] = torch.tensor([1.0, 0.0, 0.0])
    good[0, 3] = torch.tensor([1.0, 0.0, 0.0])
    neutral = clean.clone()
    masks = {
        "target_object": torch.tensor([1.0, 1.0, 0.0, 0.0]),
        "matched_background": torch.tensor([0.0, 0.0, 1.0, 1.0]),
    }
    patch_tokens = torch.tensor([0.0, 0.0, 1.0, 1.0])
    good_loss, good_stats = object_decoy_loss(
        good, clean, masks, patch_tokens, object_weight=1.0, patch_weight=1.0)
    neutral_loss, neutral_stats = object_decoy_loss(
        neutral, clean, masks, patch_tokens, object_weight=1.0, patch_weight=1.0)
    assert float(good_loss) < float(neutral_loss)
    assert good_stats["patch_target_cos"] > neutral_stats["patch_target_cos"]
    assert good_stats["object_background_cos"] > neutral_stats["object_background_cos"]


def test_slurm_is_manual_two_arm_3090_array() -> None:
    source = (ROOT / "slurm/exploring_oft_dino/41_train_object_decoy500_v1_3090.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in source
    assert "#SBATCH --gres=gpu:3090:1" in source
    assert "UADA_OFT_OBJECT_DECOY_wrapper.py" in source
    assert "sbatch " not in source
