"""CPU tests for the oracle-only local-Jacobian pilot."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_object_pose_jacobian_oracle as pilot  # noqa: E402


def test_central_jacobian_and_inverse_recover_known_mapping():
    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])
    eps = 0.01
    probes = {"x_plus": matrix @ np.array([eps, 0.0]),
              "x_minus": matrix @ np.array([-eps, 0.0]),
              "y_plus": matrix @ np.array([0.0, eps]),
              "y_minus": matrix @ np.array([0.0, -eps])}
    estimate = pilot.central_jacobian(probes, eps)
    desired = np.array([2.5, 0.0])
    assert np.allclose(estimate, matrix)
    assert np.allclose(matrix @ pilot.solve_world_delta(estimate, desired), desired)
    assert pilot.jacobian_diagnostics(estimate)["valid"]


def test_acceptance_requires_magnitude_direction_and_target_error():
    desired = np.array([2.5, 0.0])
    assert pilot.candidate_metrics(np.array([2.2, 0.0]), desired)["accepted"]
    assert not pilot.candidate_metrics(np.array([1.99, 0.0]), desired)["accepted"]
    assert not pilot.candidate_metrics(np.array([0.0, 2.5]), desired)["accepted"]
    assert not pilot.candidate_metrics(np.array([3.1, 0.0]), desired)["accepted"]


def test_pair_gate_never_substitutes_one_sided_result():
    rows = []
    for name, _ in pilot.TARGETS:
        accepted = name != "image_y_minus"
        rows.append({"name": name, "accepted": accepted,
                     "metrics": {"actual_shift_tokens": 2.5}})
    gate = pilot.symmetric_pair_gate(rows)
    assert gate["image_x"]["both_directions_accepted"]
    assert not gate["image_y"]["both_directions_accepted"]
    assert not gate["all_four_accepted"]
