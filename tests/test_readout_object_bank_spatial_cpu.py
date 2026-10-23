"""CPU guards for frozen Object spatial readout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import readout_object_bank_spatial_cpu as readout  # noqa: E402


def test_summary_respects_trajectory_grouping_and_predeclared_gate():
    rows = [{"task_id": task, "trajectory_rank": readout.holdout_rank(task),
             "object_error_tokens": 1.0, "gripper_error_tokens": 1.5,
             "relative_error_tokens": 2.0}
            for task in range(10) for _ in range(6)]
    result = readout.summarize(rows)
    assert result["heldout_trajectories"] == 10
    assert result["heldout_states"] == 60
    assert result["pass_predeclared_localization_gate"]
    rows[0]["trajectory_rank"] = (rows[0]["trajectory_rank"] + 1) % 5
    with pytest.raises(RuntimeError, match="Heldout trajectory structure drift"):
        readout.summarize(rows)


def test_high_error_fails_without_changing_threshold():
    rows = [{"task_id": task, "trajectory_rank": readout.holdout_rank(task),
             "object_error_tokens": 2.1, "gripper_error_tokens": 1.0,
             "relative_error_tokens": 1.0}
            for task in range(10) for _ in range(6)]
    assert not readout.summarize(rows)["pass_predeclared_localization_gate"]
