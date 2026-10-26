"""Precommitted candidate grid and oracle-only symmetric-pair selection."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_object_pose_oracle as calibration  # noqa: E402


def candidates():
    return [{"magnitude_m": magnitude, "direction": direction, "valid": False}
            for magnitude in calibration.MAGNITUDES_M
            for direction, _, _ in calibration.DIRECTIONS]


def test_selects_smallest_valid_symmetric_pair_not_single_direction():
    rows = candidates()
    for row in rows:
        if row["magnitude_m"] == 0.04 and row["direction"] == "x_plus":
            row["valid"] = True
        if row["magnitude_m"] == 0.05 and row["direction"].startswith("x_"):
            row["valid"] = True
        if row["magnitude_m"] == 0.06 and row["direction"].startswith("y_"):
            row["valid"] = True
    assert calibration.select_symmetric_pairs(rows) == {"x": 0.05, "y": 0.06}


def test_incomplete_grid_rejected():
    with pytest.raises(RuntimeError, match="Incomplete oracle candidate grid"):
        calibration.select_symmetric_pairs(candidates()[:-1])
