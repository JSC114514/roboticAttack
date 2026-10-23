#!/usr/bin/env python3
"""Run UPA-RFAS OFT eval with controlled digital patch visibility.

This entrypoint intentionally leaves the existing WHEN/OFT evaluators alone.
It imports the UPA-RFAS evaluator, replaces only its image-preparation hook in
this process, then writes results to a separate output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.when.config import LIBERO_SUITES, OFT_MODEL_PATHS, resolve_output_path
from VLAAttacker.when.patch_io import read_patch_manifest, require_patch_manifest
from VLAAttacker.when.trainer import assert_cuda_available


DEFAULT_UPA_RFAS_OFT_ROOT = Path("/home/caijiasheng25/UPA-RFAS/openvla-oft")
DEFAULT_OPENVLA_OFT_ROOT = Path("/home/caijiasheng25/Desktop/tex3D/openvla-oft")
DEFAULT_LIBERO_ROOT = Path("/home/caijiasheng25/Desktop/LIBERO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=LIBERO_SUITES, required=True)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visibility-id", required=True)
    parser.add_argument("--front-visibility", type=float, default=1.0)
    parser.add_argument("--wrist-visibility", type=float, default=1.0)
    parser.add_argument(
        "--visibility-mask",
        choices=["left", "right", "center", "random"],
        default="left",
    )
    parser.add_argument("--official-oft-root", default=str(DEFAULT_UPA_RFAS_OFT_ROOT))
    parser.add_argument("--openvla-oft-root", default=str(DEFAULT_OPENVLA_OFT_ROOT))
    parser.add_argument("--libero-root", default=str(DEFAULT_LIBERO_ROOT))
    parser.add_argument("--model-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _clamp_ratio(value: float) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"visibility ratio must be in [0, 1], got {value}")
    return float(value)


def _patch_slice(patch: Any | None, ratio: float, mode: str) -> tuple[Any | None, int]:
    if patch is None or ratio <= 0.0:
        return None, 0
    if ratio >= 1.0:
        return patch, 0
    if patch.ndim != 3:
        raise ValueError(f"expected patch [C,H,W], got shape {tuple(patch.shape)}")
    width = int(patch.shape[-1])
    keep = max(1, min(width, int(round(width * ratio))))
    if mode == "left":
        start = 0
    elif mode == "right":
        start = width - keep
    elif mode == "center":
        start = (width - keep) // 2
    elif mode == "random":
        start = random.randint(0, width - keep)
    else:
        raise ValueError(f"unsupported visibility mask: {mode}")
    return patch[:, :, start : start + keep].contiguous(), start


def _parse_official_log(path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {}
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "num_episodes": r"Total episodes:\s*(\d+)",
        "num_successes": r"Total successes:\s*(\d+)",
        "success_rate": r"Overall success rate:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            metrics[key] = float(value) if "." in value else int(value)
    return metrics


def _ensure_openvla_auto_map(checkpoint: str) -> None:
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_dir():
        return
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return

    wanted_auto_map = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }
    lock_path = checkpoint_path / ".when_config_json.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            if config.get("auto_map") == wanted_auto_map:
                return
            config["auto_map"] = wanted_auto_map
            tmp_path = checkpoint_path / f"config.json.when_tmp.{os.getpid()}"
            tmp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, config_path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _patch_official_prepare_observation(
    official: Any,
    *,
    front_visibility: float,
    wrist_visibility: float,
    visibility_mask: str,
) -> None:
    def prepare_observation_visibility(obs, resize_size, patch, geometry_data):  # noqa: ANN001
        img = official.get_libero_image(obs)
        wrist_img = official.get_libero_wrist_image(obs)

        img_resized = official.resize_image_for_policy(img, resize_size)
        wrist_img_resized = official.resize_image_for_policy(wrist_img, resize_size)

        if patch is not None:
            geometry = geometry_data[0]
            front_patch, front_x_offset = _patch_slice(patch, front_visibility, visibility_mask)
            wrist_patch, wrist_x_offset = _patch_slice(patch, wrist_visibility, visibility_mask)
            if front_patch is not None:
                img_resized = official.simulation_random_patch(
                    img_resized,
                    front_patch,
                    geometry=True,
                    colorjitter=False,
                    angle=geometry["angle"],
                    shx=geometry["shx"],
                    shy=geometry["shy"],
                    position=(geometry["x"] + front_x_offset, geometry["y"]),
                )
            if wrist_patch is not None:
                wrist_img_resized = official.simulation_random_patch(
                    wrist_img_resized,
                    wrist_patch,
                    geometry=True,
                    colorjitter=False,
                    angle=geometry["angle"],
                    shx=geometry["shx"],
                    shy=geometry["shy"],
                    position=(geometry["x"] + wrist_x_offset, geometry["y"]),
                )

        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], official.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
        return observation, img_resized

    official.prepare_observation = prepare_observation_visibility


def main() -> None:
    args = parse_args()
    front_visibility = _clamp_ratio(args.front_visibility)
    wrist_visibility = _clamp_ratio(args.wrist_visibility)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = require_patch_manifest(args.patch_manifest)
    manifest = read_patch_manifest(manifest_path)
    official_root = Path(args.official_oft_root).expanduser()
    openvla_oft_root = Path(args.openvla_oft_root).expanduser()
    official_script = official_root / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    model_checkpoint = args.model_checkpoint or str(OFT_MODEL_PATHS[args.suite])
    log_dir = output_dir / "official_logs"
    stdout_path = output_dir / "official_eval.out"

    payload = {
        "entrypoint": str(Path(__file__).resolve()),
        "official_oft_root": str(official_root),
        "openvla_oft_root": str(openvla_oft_root),
        "official_script": str(official_script),
        "suite": args.suite,
        "model_checkpoint": model_checkpoint,
        "patch_manifest": str(manifest_path),
        "patch_path": manifest["patch_path"],
        "patch_sha256": manifest.get("patch_sha256_verified", manifest.get("patch_sha256", "")),
        "visibility_id": args.visibility_id,
        "front_visibility": front_visibility,
        "wrist_visibility": wrist_visibility,
        "visibility_mask": args.visibility_mask,
        "output_dir": str(output_dir),
    }
    (output_dir / "visibility_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    assert_cuda_available()
    if not official_script.exists():
        raise FileNotFoundError(f"UPA-RFAS official evaluator not found: {official_script}")

    _ensure_openvla_auto_map(model_checkpoint)
    for module_name in ("experiments", "prismatic"):
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(official_root))
    sys.path.insert(1, str(openvla_oft_root))
    if args.libero_root:
        sys.path.insert(2, str(Path(args.libero_root).expanduser()))

    from experiments.robot.libero import run_libero_eval as official  # type: ignore
    from experiments.robot import openvla_utils as official_openvla_utils  # type: ignore

    official_openvla_utils.update_auto_map = lambda pretrained_checkpoint: None

    _patch_official_prepare_observation(
        official,
        front_visibility=front_visibility,
        wrist_visibility=wrist_visibility,
        visibility_mask=args.visibility_mask,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WHEN_ROLLOUT_DIR"] = str(output_dir / "videos")
    os.environ["WHEN_OFT_WORLD_PATCH"] = "0"

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = [
        str(official_script),
        "--pretrained_checkpoint",
        model_checkpoint,
        "--task_suite_name",
        args.suite,
        "--local_log_dir",
        str(log_dir),
        "--run_id_note",
        f"WHEN_VIS_{args.suite}_{args.visibility_id}",
        "--use_wandb",
        "False",
        "--patch_path",
        manifest["patch_path"],
    ]
    try:
        os.chdir(official_root)
        with stdout_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            official.eval_libero()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    log_files = sorted(log_dir.glob("*.txt"), key=lambda item: item.stat().st_mtime, reverse=True)
    official_log = log_files[0] if log_files else None
    parsed = _parse_official_log(official_log) if official_log else {}
    success_rate = parsed.get("success_rate")
    summary = {
        "mode": "oft_digital_visibility_proxy",
        "task_suite": args.suite,
        "visibility_id": args.visibility_id,
        "front_visibility": front_visibility,
        "wrist_visibility": wrist_visibility,
        "visibility_mask": args.visibility_mask,
        "patch_path": manifest["patch_path"],
        "patch_sha256": payload["patch_sha256"],
        "official_entrypoint": str(official_script),
        "official_log": str(official_log) if official_log else "",
        "stdout_log": str(stdout_path),
        **parsed,
    }
    if success_rate is not None:
        summary["attack_success_rate"] = 1.0 - float(success_rate)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
