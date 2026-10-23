"""CPU-only contract tests for the fresh original-style UADA baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_exploring_oft_uada_original1000_cross_suite as baseline


def test_training_and_evaluation_budget_are_the_requested_baseline():
    train = json.loads(baseline.TRAIN_CONFIG.read_text())
    evaluation = baseline.config()
    assert train["methods"] == ["exploring_oft_uada"]
    assert train["seed"] == 42 and train["data_seed"] == 20260920
    assert train["outer_iterations"] == 1000 and train["inner_loop"] == 50
    assert train["checkpoint_outer_iterations"] == [0, 500, 750, 1000]
    assert train["maskidx"] == [0] and train["structure_gates"] == {}
    assert baseline.CONDITIONS == ("original_uada_500", "original_uada_750", "original_uada_1000")
    assert evaluation["suites"] == list(baseline.DENOMINATORS)
    assert evaluation["state_indices"] == list(range(40, 50))
    assert evaluation["patch_xy"] == [26, 137]


@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    folder = tmp_path / "training/exploring_oft_uada"
    folder.mkdir(parents=True)
    train = json.loads(baseline.TRAIN_CONFIG.read_text())
    train["output_root"] = str(tmp_path / "training")
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(train))
    monkeypatch.setattr(baseline, "TRAIN_CONFIG", config_path)
    status = {
        "status": "COMPLETE", "experiment_id": train["experiment_id"],
        "method": "exploring_oft_uada", "outer_iterations": 1000,
        "seed": 42, "data_seed": 20260920,
        "source_uada_sha256": "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
        "training_bank_sha256": "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8",
    }
    (folder / "status.json").write_text(json.dumps(status))
    for step in baseline.STEPS:
        value = torch.full((3, 50, 50), step / 1000)
        payload = {"patch": value, "experiment_id": train["experiment_id"],
                   "method": "exploring_oft_uada", "outer_iterations": 1000,
                   "inner_loop": 50, "maskidx": [0], "seed": 42, "data_seed": 20260920}
        path = folder / f"outer_{step:04d}.pt"
        torch.save(payload, path)
        Image.new("RGB", (50, 50), (100, 100, 100)).save(path.with_suffix(".png"))
    return folder


def test_loader_accepts_all_three_original_checkpoints(fake_training):
    found = baseline.patches(baseline.config(), torch.device("cpu"))
    assert tuple(found) == baseline.CONDITIONS
    assert all(tuple(value.shape) == (3, 50, 50) and len(digest) == 64
               for value, digest in found.values())


def test_loader_rejects_a_v2_or_wrong_method_checkpoint(fake_training):
    path = fake_training / "outer_0750.pt"
    payload = torch.load(path, weights_only=False)
    payload["method"] = "uada"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Invalid baseline checkpoint"):
        baseline.patches(baseline.config(), torch.device("cpu"))
