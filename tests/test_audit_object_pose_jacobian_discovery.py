"""CPU guards for discovery sampling, bounded solver, and stopping rule."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_object_pose_jacobian_discovery as audit  # noqa: E402
import summarize_object_pose_jacobian_discovery as summary  # noqa: E402


def test_selection_uses_four_discovery_trajectories_and_last_precontact():
    bank = torch.load(audit.MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt",
                      map_location="cpu", weights_only=False)
    for task_id in range(10):
        selected = audit.discovery_selections(bank, task_id)
        assert len(selected) == 4
        assert audit.holdout_rank(task_id) not in {rank for rank, _, _ in selected}
        for _, decision, records in selected:
            expected = max(int(row["decision_index"]) for row in records
                           if row["phase_proxy"] == "pre_contact_proxy")
            assert decision == expected


def test_bounded_projection_preserves_direction_and_cap():
    source = np.array([0.12, -0.16])
    result = audit.bounded(source)
    assert np.isclose(np.linalg.norm(result), audit.MAX_WORLD_NORM_METERS)
    assert np.allclose(result / np.linalg.norm(result), source / np.linalg.norm(source))


def make_states(passed: int, tasks_with_pass: int) -> list[dict]:
    rows = [{"task_id": task, "all_four_accepted": False}
            for task in range(10) for _ in range(4)]
    for task in range(tasks_with_pass):
        rows[task * 4]["all_four_accepted"] = True
    remaining = passed - tasks_with_pass
    for row in rows:
        if remaining and not row["all_four_accepted"]:
            row["all_four_accepted"] = True
            remaining -= 1
    return rows


def test_gate_requires_both_overall_fraction_and_task_coverage():
    assert summary.decision_gate(make_states(20, 8))["advance_to_independent_holdout"]
    assert not summary.decision_gate(make_states(19, 10))["advance_to_independent_holdout"]
    assert not summary.decision_gate(make_states(20, 7))["advance_to_independent_holdout"]
