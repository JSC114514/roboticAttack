from __future__ import annotations

import ast
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "exploring_oft_dino" / "object_timeboxed.json"
TRAINER = ROOT / "VLAAttacker" / "white_patch" / "UADA_OFT_DINO.py"


def test_frozen_timeboxed_contract() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["outer_iterations"] == 200
    assert cfg["inner_loop"] == 50
    assert cfg["batch_size"] == 1
    assert cfg["maskidx"] == [0]
    assert cfg["geometry"] is True
    assert cfg["checkpoint_outer_iterations"] == [0, 25, 50, 100, 200]
    assert cfg["dino_gradient_ratio"] == 0.3
    assert "initial_patch" not in cfg


def test_trainer_is_explicitly_derived_from_exploring_uada() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.module == "white_patch.UADA" and any(alias.name == "OpenVLAAttacker" for alias in node.names)
               for node in imports)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    target = next(node for node in classes if node.name == "ExploringOFTOpenVLAAttacker")
    assert any(isinstance(base, ast.Name) and base.id == "OpenVLAAttacker" for base in target.bases)
    assert "RandomPatchTransform" in source
    assert "for inner in range(int(cfg[\"inner_loop\"]))" in source
    assert "optimizer.step()" in source
    assert "scheduler.step()" in source
    assert "patch.clamp_(0, 1)" in source


def test_no_legacy_learned_patch_dependency() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    forbidden = ("dino_norm/step_1500", "oft_ramp_recovery_aware/artifacts", "initial_patch")
    assert all(token not in source for token in forbidden)


def test_training_bank_is_source_observations_not_patch_warm_start() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    assert len(bank["records"]) == 300
    record = bank["records"][0]
    assert {"observation", "task_name", "clean_action", "clean_dino"}.issubset(record)
    assert tuple(record["clean_action"].shape) == (1, 8, 7)
    assert tuple(record["clean_dino"].shape) == (1, 256, 1024)
    assert tuple(cfg["patch_shape"]) == (3, 50, 50)


def test_structure_gate_is_predeclared_for_both_checkpoints() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["structure_gates"] == {
        "100": {"max_init_correlation": 0.95, "max_tv_ratio": 0.97},
        "200": {"max_init_correlation": 0.85, "max_tv_ratio": 0.90},
    }
