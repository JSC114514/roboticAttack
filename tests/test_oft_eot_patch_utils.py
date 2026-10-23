from __future__ import annotations

import random

import torch

from VLAAttacker.multiview.openvla_oft_experiments import (
    apply_patch_to_pixel_values,
    sample_oft_eot_transforms,
    validate_oft_eot_config,
)


def base_config() -> dict[str, str]:
    return {
        "train_patch_position": "anchored_eot",
        "eval_patch_position": "top_left",
        "eot_enabled": "true",
        "eot_samples": "2",
        "eot_warmup_fraction": "0.35",
        "eot_fixed_anchor_probability": "0.60",
        "eot_local_jitter_probability": "0.30",
        "eot_global_random_probability": "0.10",
        "eot_local_jitter_x": "16",
        "eot_local_jitter_y": "16",
        "eot_rotation_degrees": "10.0",
        "eot_scale_min": "0.90",
        "eot_scale_max": "1.10",
        "eot_shear_degrees": "5.0",
        "eval_eot_enabled": "false",
        "multiview_eot_transform_coupling": "shared",
    }


def test_warmup_is_strict_top_left_identity() -> None:
    cfg = base_config()
    validate_oft_eot_config(cfg)
    random.seed(0)
    transforms = sample_oft_eot_transforms(
        cfg,
        step=0,
        max_steps=100,
        batch_size=8,
        image_h=64,
        image_w=64,
        patch_h=10,
        patch_w=10,
    )
    assert all(item["stage"] == "warmup" for item in transforms)
    assert all(item["x"] == 0 and item["y"] == 0 for item in transforms)
    assert all(item["rotation"] == 0.0 and item["scale"] == 1.0 and item["shear"] == 0.0 for item in transforms)


def test_post_warmup_position_distribution_is_anchored() -> None:
    cfg = base_config()
    random.seed(1)
    transforms = sample_oft_eot_transforms(
        cfg,
        step=50,
        max_steps=100,
        batch_size=2000,
        image_h=64,
        image_w=64,
        patch_h=10,
        patch_w=10,
    )
    counts = {mode: 0 for mode in ["fixed_anchor", "local_jitter", "global_random"]}
    for item in transforms:
        counts[str(item["position_mode"])] += 1
    total = float(len(transforms))
    assert abs(counts["fixed_anchor"] / total - 0.60) < 0.06
    assert abs(counts["local_jitter"] / total - 0.30) < 0.06
    assert abs(counts["global_random"] / total - 0.10) < 0.04


def test_local_jitter_never_leaves_anchor_bounds() -> None:
    cfg = base_config()
    cfg["eot_fixed_anchor_probability"] = "0.0"
    cfg["eot_local_jitter_probability"] = "1.0"
    cfg["eot_global_random_probability"] = "0.0"
    random.seed(2)
    transforms = sample_oft_eot_transforms(
        cfg,
        step=50,
        max_steps=100,
        batch_size=128,
        image_h=20,
        image_w=20,
        patch_h=10,
        patch_w=10,
    )
    assert all(0 <= int(item["x"]) <= 10 for item in transforms)
    assert all(0 <= int(item["y"]) <= 10 for item in transforms)


def test_eot_patch_application_is_differentiable_and_front_only() -> None:
    pixel_values = torch.zeros((2, 12, 64, 64), dtype=torch.float32)
    patch = torch.rand((3, 10, 10), dtype=torch.float32, requires_grad=True)
    transforms = [
        {"x": 0, "y": 0, "rotation": 5.0, "scale": 1.0, "shear": 2.0},
        {"x": 3, "y": 4, "rotation": -5.0, "scale": 1.05, "shear": -2.0},
    ]
    out = apply_patch_to_pixel_values(
        pixel_values,
        patch,
        input_views=["front", "wrist"],
        patch_views=["front"],
        position="anchored_eot",
        eot_transforms=transforms,
    )
    assert torch.equal(out[:, 6:12], pixel_values[:, 6:12])
    loss = out[:, :6].sum()
    loss.backward()
    assert patch.grad is not None
    assert torch.isfinite(patch.grad).all()
    assert float(patch.grad.abs().sum()) > 0.0


def test_multiview_shared_transform_changes_same_xy_region() -> None:
    pixel_values = torch.zeros((1, 12, 64, 64), dtype=torch.float32)
    patch = torch.ones((3, 10, 10), dtype=torch.float32, requires_grad=True)
    transforms = [{"x": 5, "y": 7, "rotation": 0.0, "scale": 1.0, "shear": 0.0}]
    out = apply_patch_to_pixel_values(
        pixel_values,
        patch,
        input_views=["front", "wrist"],
        patch_views=["front", "wrist"],
        position="anchored_eot",
        eot_transforms=transforms,
    )
    front_changed = out[0, 0].abs() > 1e-6
    wrist_changed = out[0, 6].abs() > 1e-6
    assert torch.equal(front_changed, wrist_changed)
    ys, xs = torch.where(front_changed)
    assert int(xs.min()) == 5
    assert int(ys.min()) == 7
