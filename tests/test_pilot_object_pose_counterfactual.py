"""CPU guards for object-only simulator counterfactual pilot."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_object_pose_counterfactual as pilot  # noqa: E402


def make_masks():
    target = np.zeros((256, 256), np.uint8)
    gripper = np.zeros_like(target)
    background = np.zeros_like(target)
    target[60:90, 60:90] = 1
    gripper[125:160, 125:160] = 1
    background[200:220, 200:220] = 1
    return {"target_object": target, "gripper": gripper,
            "matched_background": background}


def test_pilot_is_discovery_and_offsets_are_symmetric():
    assert pilot.RANK != pilot.holdout_rank(pilot.TASK_ID)
    vectors = [np.array(value) for _, value in pilot.OFFSETS]
    assert np.array_equal(vectors[0], -vectors[1])
    assert np.array_equal(vectors[2], -vectors[3])


def test_integrity_rejects_non_target_qpos_or_pixels():
    masks = make_masks()
    rgb = np.zeros((256, 256, 3), np.uint8)
    raw = {"robot0_eye_in_hand_image": rgb}
    clean = np.zeros(10)
    altered = clean.copy()
    altered[2] = 0.04
    report = pilot.render_integrity(raw, raw, masks, masks, clean, altered, (2, 9))
    assert report["non_target_sim_state_exact"]
    bad = altered.copy()
    bad[1] = 0.01
    with pytest.raises(RuntimeError, match="Non-target MuJoCo qpos"):
        pilot.render_integrity(raw, raw, masks, masks, clean, bad, (2, 9))
    changed_rgb = rgb.copy()
    changed_rgb[:50, :50] = 255
    with pytest.raises(RuntimeError, match="Non-target Wrist pixels"):
        pilot.render_integrity(raw, {"robot0_eye_in_hand_image": changed_rgb},
                               masks, masks, clean, altered, (2, 9))
