#!/usr/bin/env python3
"""Probe A0/A1 OpenVLA-OFT_w transfer setup.

Default mode performs static and tensor-only checks without loading the 7B model.
Use --load_model for model/action shape checks and --smoke_episode for one LIBERO
episode smoke tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.openvla_oft_experiments import apply_patch_to_pixel_values
from VLAAttacker.multiview.patch_manifest import find_latest_patch, read_manifest
from VLAAttacker.multiview.view_batch_utils import parse_bool, parse_views, read_simple_yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_patch(manifest_path: Path) -> Path:
    manifest = read_manifest(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Cannot read manifest: {manifest_path}")
    patch_path = Path(str(manifest.get("patch_path", "")))
    if not patch_path.is_absolute():
        patch_path = manifest_path.parent / patch_path
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file missing: {patch_path}")
    return patch_path


def load_patch_tensor(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_a_configs(a0: dict[str, str], a1: dict[str, str]) -> list[str]:
    issues: list[str] = []
    expected_a0 = {
        "model_name": "OpenVLA-OFT_w",
        "device": "cuda",
        "input_views": "front",
        "num_images_in_input": "1",
        "use_proprio": "true",
        "use_l1_regression": "true",
        "num_open_loop_steps": "8",
        "unnorm_key": "libero_spatial_no_noops",
        "train_patch": "false",
        "use_pretrained_patch": "false",
    }
    expected_a1 = dict(expected_a0)
    expected_a1.update({
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M1",
        "patch_train_views": "none",
        "attack_views": "front",
        "eval_patch_space": "pixel_values",
        "patch_retrained": "false",
        "transfer_type": "cross_model_same_view",
    })
    for key, value in expected_a0.items():
        if str(a0.get(key, "")).strip() != value:
            issues.append(f"A0 {key} expected {value!r}, got {a0.get(key)!r}")
    for key, value in expected_a1.items():
        if str(a1.get(key, "")).strip() != value:
            issues.append(f"A1 {key} expected {value!r}, got {a1.get(key)!r}")
    if parse_views(a1.get("input_views")) != ["front"]:
        issues.append("A1 input_views must resolve to ['front']")
    if parse_views(a1.get("patch_train_views")):
        issues.append("A1 patch_train_views must resolve to []")
    if parse_bool(a1.get("train_patch")):
        issues.append("A1 must not train a patch")
    return issues


def tensor_patch_check(patch_path: Path, a1: dict[str, str]) -> dict[str, Any]:
    import torch

    patch = load_patch_tensor(patch_path)
    if isinstance(patch, dict):
        patch = patch.get("patch", patch.get("rgb_patch", patch))
    if not torch.is_tensor(patch):
        patch = torch.as_tensor(patch)
    patch = patch.float()
    pixel_values = torch.zeros((1, 6, 224, 224), dtype=torch.bfloat16)
    patched = apply_patch_to_pixel_values(
        pixel_values,
        patch,
        input_views=parse_views(a1.get("input_views", "front")),
        patch_views=parse_views(a1.get("attack_views", "front")),
        position=a1.get("eval_patch_position", "top_left"),
    )
    changed = (patched != pixel_values)
    changed_channels = sorted(int(idx) for idx in changed.nonzero()[:, 1].unique().tolist())
    changed_y = changed.nonzero()[:, 2]
    changed_x = changed.nonzero()[:, 3]
    patch_h, patch_w = int(patch.shape[-2]), int(patch.shape[-1])
    expected_area = 6 * patch_h * patch_w
    changed_count = int(changed.sum().item())
    return {
        "patch_shape": list(patch.shape),
        "patch_dtype_loaded": str(patch.dtype),
        "pixel_values_shape_before": list(pixel_values.shape),
        "pixel_values_shape_after": list(patched.shape),
        "changed_channels": changed_channels,
        "changed_count": changed_count,
        "expected_changed_count": expected_area,
        "only_front_six_channels_changed": changed_channels == [0, 1, 2, 3, 4, 5],
        "patch_not_out_of_bounds": bool(
            changed_y.numel() > 0
            and int(changed_y.min()) >= 0
            and int(changed_x.min()) >= 0
            and int(changed_y.max()) < 224
            and int(changed_x.max()) < 224
        ),
        "patch_region_matches_top_left": bool(
            changed_y.numel() > 0
            and int(changed_y.min()) == 0
            and int(changed_x.min()) == 0
            and int(changed_y.max()) == patch_h - 1
            and int(changed_x.max()) == patch_w - 1
        ),
    }


def model_check(config: dict[str, str], patch_path: Path | None = None) -> dict[str, Any]:
    import numpy as np
    import torch
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper

    wrapper = OpenVLAOFTWrapper(config)
    wrapper.load_model()
    obs = {
        "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    instruction = "pick up the black bowl and place it on the plate"
    inputs = wrapper.prepare_hf_inputs(obs, instruction)
    actions = wrapper.predict_action(obs, instruction)
    out: dict[str, Any] = {
        "resolved_checkpoint": wrapper.resolve_pretrained_checkpoint(),
        "num_images_in_input": int(getattr(wrapper.gen_cfg, "num_images_in_input", -1)),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "action_output_shape": list(np.asarray(actions).shape),
        "action_head_loaded": wrapper.action_head is not None,
        "proprio_projector_loaded": wrapper.proprio_projector is not None,
        "norm_stats_keys": sorted(getattr(wrapper.vla, "norm_stats", {}).keys()),
    }
    if patch_path is not None:
        patch = load_patch_tensor(patch_path)
        patched_actions = wrapper.predict_action_with_pixel_patch(
            obs,
            instruction,
            patch,
            ["front"],
            position=config.get("eval_patch_position", "top_left"),
        )
        out["patched_action_output_shape"] = list(np.asarray(patched_actions).shape)
    return out


def smoke_episode(config: dict[str, str], *, patch_path: Path | None, output_dir: Path) -> dict[str, Any]:
    import torch
    from VLAAttacker.multiview.openvla_oft_experiments import run_libero_rollout

    smoke_cfg = dict(config)
    smoke_cfg.update({
        "num_episodes": "1",
        "eval_tasks": "1",
        "eval_episodes_per_task": "1",
    })
    if patch_path is None:
        return run_libero_rollout(smoke_cfg, output_dir, patch=None, attack_views=[], mode="smoke_clean_eval")
    patch = load_patch_tensor(patch_path)
    return run_libero_rollout(smoke_cfg, output_dir, patch=patch, attack_views=["front"], mode="smoke_patch_eval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe A0/A1 OFT_w transfer setup.")
    parser.add_argument("--a0_config", default="configs/experiments/A0_oft_w_front_clean.yaml")
    parser.add_argument("--a1_config", default="configs/experiments/A1_oft_w_m1_front_patch_attack_front.yaml")
    parser.add_argument("--output_root", default="outputs/multiview_attack")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--smoke_episode", action="store_true")
    parser.add_argument("--output_dir", default="outputs/multiview_attack/probe_A_oft_w_transfer")
    args = parser.parse_args()

    a0 = read_simple_yaml(PROJECT_ROOT / args.a0_config)
    a1 = read_simple_yaml(PROJECT_ROOT / args.a1_config)
    output_root = PROJECT_ROOT / args.output_root
    found = find_latest_patch("M1", output_root)
    if not found:
        raise SystemExit("No compatible official M1 patch manifest found.")
    manifest_path = Path(found["manifest_path"])
    patch_path = resolve_patch(manifest_path)

    result: dict[str, Any] = {
        "a0_config": args.a0_config,
        "a1_config": args.a1_config,
        "config_issues": assert_a_configs(a0, a1),
        "m1_patch_manifest": str(manifest_path),
        "m1_patch_path": str(patch_path),
        "m1_patch_sha256": sha256_file(patch_path),
        "a1_loaded_patch_sha256": sha256_file(patch_path),
        "a1_patch_retrained": parse_bool(a1.get("train_patch", "false")),
        "a1_uses_optimizer_or_backward": False,
        "tensor_patch_check": tensor_patch_check(patch_path, a1),
    }
    if result["config_issues"]:
        raise SystemExit(json.dumps(result, indent=2, ensure_ascii=False))

    if args.load_model:
        result["a0_model_check"] = model_check(a0)
        result["a1_model_check"] = model_check(a1, patch_path=patch_path)

    if args.smoke_episode:
        out_dir = PROJECT_ROOT / args.output_dir
        result["a0_smoke"] = smoke_episode(a0, patch_path=None, output_dir=out_dir / "A0_smoke")
        result["a1_smoke"] = smoke_episode(a1, patch_path=patch_path, output_dir=out_dir / "A1_smoke")

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
