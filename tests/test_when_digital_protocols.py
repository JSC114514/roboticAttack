from __future__ import annotations

import numpy as np

from VLAAttacker.when.transforms import apply_digital_patch_to_observation, assert_digital_protocol_effect


def observation() -> dict[str, np.ndarray]:
    front = np.zeros((32, 32, 3), dtype=np.uint8)
    wrist = np.full((32, 32, 3), 25, dtype=np.uint8)
    return {"full_image": front, "wrist_image": wrist}


def patch() -> np.ndarray:
    return np.ones((3, 8, 8), dtype=np.float32)


def test_digital_front_only_changes_front() -> None:
    clean = observation()
    attacked = apply_digital_patch_to_observation(clean, patch(), attack_protocol="digital_front", suite="libero_object")
    assert_digital_protocol_effect(clean, attacked, attack_protocol="digital_front")
    assert not np.array_equal(clean["full_image"], attacked["full_image"])
    assert np.array_equal(clean["wrist_image"], attacked["wrist_image"])


def test_digital_wrist_only_changes_wrist() -> None:
    clean = observation()
    attacked = apply_digital_patch_to_observation(clean, patch(), attack_protocol="digital_wrist", suite="libero_object")
    assert_digital_protocol_effect(clean, attacked, attack_protocol="digital_wrist")
    assert np.array_equal(clean["full_image"], attacked["full_image"])
    assert not np.array_equal(clean["wrist_image"], attacked["wrist_image"])


def test_digital_both_changes_both() -> None:
    clean = observation()
    attacked = apply_digital_patch_to_observation(clean, patch(), attack_protocol="digital_both", suite="libero_object")
    assert_digital_protocol_effect(clean, attacked, attack_protocol="digital_both")
    assert not np.array_equal(clean["full_image"], attacked["full_image"])
    assert not np.array_equal(clean["wrist_image"], attacked["wrist_image"])
