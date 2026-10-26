"""Check the geometry audit's trajectory selection and frozen decision gate."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_object_gripper_image_geometry as audit  # noqa: E402
import summarize_object_gripper_image_geometry as summary  # noqa: E402


def test_discovery_selection_excludes_every_task_holdout():
    bank = torch.load(audit.MGWP_ROOT / "artifacts/GEOMETRY_TRAIN_BANK.pt",
                      map_location="cpu", weights_only=False)
    for task_id in range(10):
        selected = audit.discovery_selections(bank, task_id)
        assert len(selected) == 4
        assert audit.holdout_rank(task_id) not in {rank for rank, _, _ in selected}


def test_geometry_gate_requires_both_trajectory_and_task_coverage():
    rows = [{"task_id": task, "valid": False} for task in range(10) for _ in range(4)]
    for task in range(8):
        rows[task * 4]["valid"] = True
    for row in rows:
        if sum(item["valid"] for item in rows) >= 20:
            break
        row["valid"] = True
    assert summary.evaluate_gate(rows)["broad_diagnostic_feasible"]
    rows[7 * 4]["valid"] = False
    assert not summary.evaluate_gate(rows)["broad_diagnostic_feasible"]
