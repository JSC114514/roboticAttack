"""CPU guards for exact-bank simulator mask pilot."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_object_bank_mask_pilot", ROOT / "scripts/replay_object_bank_mask_pilot.py")
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def sample_record(decision: int, *, task_id: int = 0, trajectory_rank: int = 0):
    return {"task_id": task_id, "trajectory_rank": trajectory_rank, "state_index": 10,
            "decision_index": decision, "task_name": "pick up the alphabet soup and place it in the basket"}


def test_select_records_enforces_object_trajectory_identity():
    bank = {"experiment_id": "OFT-MGWP-WRIST-STAGE1-01", "selected_xy": [26, 137],
            "records": [sample_record(value) for value in (17, 4, 5, 6, 7, 16)]}
    selected = pilot.select_records(bank, 0, 0)
    assert [row["decision_index"] for row in selected] == [4, 5, 6, 7, 16, 17]
    bank["records"][0]["task_id"] = 1
    with pytest.raises(RuntimeError, match="six sampled decisions"):
        pilot.select_records(bank, 0, 0)


def test_check_observation_rejects_visual_or_proprio_drift():
    obs = {"full_image": np.zeros((224, 224, 3), np.uint8),
           "wrist_image": np.zeros((224, 224, 3), np.uint8),
           "state": np.zeros(8, np.float32)}
    assert pilot.check_observation(obs, obs)["wrist_image_mae"] == 0
    bad = {**obs, "wrist_image": np.full((224, 224, 3), 2, np.uint8)}
    with pytest.raises(RuntimeError, match="visual replay failed"):
        pilot.check_observation(bad, obs)
    bad = {**obs, "state": np.full(8, 1e-3, np.float32)}
    with pytest.raises(RuntimeError, match="proprioception replay failed"):
        pilot.check_observation(bad, obs)
