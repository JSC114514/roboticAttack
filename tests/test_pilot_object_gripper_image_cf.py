"""CPU tests for exact image-space object counterfactual construction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pilot_object_gripper_image_cf as pilot  # noqa: E402


def fixture_arrays():
    yy, xx = np.mgrid[:256, :256]
    clean = np.stack((xx, yy, (xx + yy) % 256), axis=-1).astype(np.uint8)
    background = np.zeros_like(clean) + 17
    target = np.zeros((256, 256), np.uint8)
    gripper = np.zeros_like(target)
    target[90:120, 100:130] = 1
    gripper[170:190, 100:130] = 1
    return clean, background, target, gripper


def test_zero_shift_reconstructs_clean_exactly():
    clean, background, target, gripper = fixture_arrays()
    output, shifted, report = pilot.composite_shift(
        clean, background, target, gripper, 0, 0)
    assert np.array_equal(output, clean)
    assert np.array_equal(shifted, target)
    assert report["outside_union_exact"]


def test_signed_shift_preserves_area_and_outside_union():
    clean, background, target, gripper = fixture_arrays()
    output, shifted, report = pilot.composite_shift(
        clean, background, target, gripper, 38, 0)
    assert shifted.sum() == target.sum()
    assert report["target_gripper_overlap_pixels"] == 0
    support = pilot.dilate(target, pilot.SUPPORT_DILATION_PIXELS)
    shifted_support = pilot.translate(support.astype(np.uint8), 38, 0).astype(bool)
    assert np.array_equal(output[~(support | shifted_support)], clean[~(support | shifted_support)])


def test_overlap_with_gripper_is_rejected():
    clean, background, target, gripper = fixture_arrays()
    with pytest.raises(RuntimeError, match="overlaps the gripper"):
        pilot.composite_shift(clean, background, target, gripper, 0, 70)


def test_selected_state_is_discovery_and_four_raw_shifts_are_geometrically_valid():
    assert pilot.RANK != pilot.holdout_rank(pilot.TASK_ID)
    path = pilot.PROJECT_ROOT / "outputs/exploring_oft_dino/spatial_relation_object_bank_masks" \
        / f"task_{pilot.TASK_ID:02d}" / f"trajectory_{pilot.RANK}_decision_{pilot.DECISION:03d}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        target = arrays["wrist_pixel_mask_target_object"]
        gripper = arrays["wrist_pixel_mask_gripper"]
    clean_relative = pilot.relative(target, gripper)
    for _, dx, dy in pilot.SHIFTS:
        shifted = pilot.translate(target, dx, dy)
        assert shifted.sum() == target.sum()
        assert not np.any(shifted.astype(bool) & gripper.astype(bool))
        delta = pilot.relative(shifted, gripper) - clean_relative
        desired = np.asarray([np.sign(dx) if dx else 0, np.sign(dy) if dy else 0])
        metrics = pilot.direction_metrics(delta, desired)
        assert abs(metrics["magnitude_tokens"] - pilot.ORACLE_TARGET_TOKENS) \
            <= pilot.ORACLE_TOLERANCE_TOKENS
        assert metrics["direction_cosine"] >= pilot.MIN_ORACLE_DIRECTION_COSINE
