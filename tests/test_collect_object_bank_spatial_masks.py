"""CPU checks for exact Object-bank mask collection and trajectory split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_object_bank_spatial_masks as collector  # noqa: E402


def test_holdout_is_frozen_and_pilot_rank_is_discovery():
    ranks = [collector.holdout_rank(task) for task in range(10)]
    assert ranks == [collector.holdout_rank(task) for task in range(10)]
    assert 0 not in ranks[:1]
    assert all(0 <= rank < 5 for rank in ranks)


def test_task_records_rejects_duplicate_initial_states():
    records = []
    for rank in range(5):
        for decision in range(6):
            records.append({"task_id": 0, "trajectory_rank": rank,
                            "state_index": rank + 1, "decision_index": decision,
                            "task_name": "object task"})
    bank = {"experiment_id": "OFT-MGWP-WRIST-STAGE1-01",
            "selected_xy": [26, 137], "records": records * 10}
    # Fill the other nine tasks so the frozen bank has the expected total.
    for task in range(1, 10):
        for rank in range(5):
            for decision in range(6):
                bank["records"].append({"task_id": task, "trajectory_rank": rank,
                                        "state_index": rank + 1, "decision_index": decision,
                                        "task_name": f"object task {task}"})
    bank["records"] = bank["records"][:30] + bank["records"][300:]
    assert len(bank["records"]) == 300
    assert len(collector.task_records(bank, 0)) == 5
    bank["records"][6]["state_index"] = 1
    with pytest.raises(RuntimeError, match="Repeated initial state"):
        collector.task_records(bank, 0)


def test_sample_quality_rejects_overlap_and_reports_patch_exclusion():
    target = np.zeros((256, 256), np.uint8)
    gripper = np.zeros_like(target)
    background = np.zeros_like(target)
    target[30:70, 30:70] = 1
    gripper[150:190, 150:190] = 1
    background[200:230, 20:50] = 1
    masks = {"target_object": target, "gripper": gripper,
             "matched_background": background}
    sample = {"masks": masks, "wrist_rgb": np.zeros((256, 256, 3), np.uint8),
              "mask_validity": {role: True for role in masks}}
    exclusion = collector.patch_exclusion({"token_grid": 16, "image_resolution": 224,
                                           "patch_xy": [26, 137], "patch_shape": [50, 50],
                                           "patch_token_halo": 1})
    quality = collector.sample_quality(sample, exclusion)
    assert quality["usable_for_spatial_readout"]
    assert quality["target_object_outside_patch_token_mass"] > 0
    assert quality["gripper_outside_patch_token_mass"] > 0
    gripper[40:45, 40:45] = 1
    with pytest.raises(RuntimeError, match="overlap"):
        collector.sample_quality(sample, exclusion)
