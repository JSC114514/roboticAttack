#!/usr/bin/env python3
"""Evaluate WHEN patch transfer to OpenVLA-OFT digital protocols."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.openvla_oft_experiments import run_libero_rollout
from VLAAttacker.when.config import DIGITAL_PROTOCOLS, LIBERO_SUITES, OFT_MODEL_PATHS, load_when_config
from VLAAttacker.when.config import merge_when_defaults, resolve_output_path, suite_to_unnorm_key
from VLAAttacker.when.patch_io import read_patch_manifest, require_patch_manifest
from VLAAttacker.when.trainer import assert_cuda_available
from VLAAttacker.when.transforms import views_for_protocol


DEFAULT_OFT_ROOT = "/home/caijiasheng25/Desktop/tex3D/openvla-oft"
DEFAULT_UPA_RFAS_OFT_ROOT = "/home/caijiasheng25/UPA-RFAS/openvla-oft"
DEFAULT_LIBERO_ROOT = "/home/caijiasheng25/Desktop/LIBERO"
OFT_ATTACK_PROTOCOLS = tuple(DIGITAL_PROTOCOLS) + ("physical_both",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=LIBERO_SUITES)
    parser.add_argument("--attack-protocol", choices=OFT_ATTACK_PROTOCOLS)
    parser.add_argument("--mode", choices=["clean", "attack"], default="attack")
    parser.add_argument("--patch-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        if not match:
            continue
        value = match.group(1)
        metrics[key] = float(value) if "." in value else int(value)
    return metrics


def _link_official_videos(log_path: Path, *, official_root: Path, output_dir: Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    linked: list[str] = []
    video_dir = output_dir / "videos"
    for raw_path in re.findall(r"Saved rollout MP4 at path\s+(.+?\.mp4)", text):
        source = Path(raw_path.strip())
        if not source.is_absolute():
            source = official_root / source
        if not source.exists():
            continue
        video_dir.mkdir(parents=True, exist_ok=True)
        target = video_dir / source.name
        if source.resolve() == target.resolve():
            linked.append(str(target))
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
        linked.append(str(target))
    return linked


def _run_official_oft_eval(
    *,
    run_config: dict[str, str],
    output_dir: Path,
    dry_run: bool,
) -> dict[str, object]:
    """Run the UPA-RFAS/WHEN official OpenVLA-OFT LIBERO evaluator.

    This path is used for official clean and digital_both evaluations. The
    official evaluator always patches both front and wrist images when
    patch_path is provided, so front-only and wrist-only ablations remain on
    the local multiview evaluator.
    """
    official_root = Path(run_config["official_oft_root"]).expanduser()
    official_script = official_root / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "official_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(official_script),
        "--pretrained_checkpoint",
        run_config["model_checkpoint"],
        "--task_suite_name",
        run_config["task_suite_name"],
        "--local_log_dir",
        str(log_dir),
        "--run_id_note",
        run_config["exp_id"],
        "--use_wandb",
        "False",
    ]
    if run_config.get("patch_path"):
        cmd.extend(["--patch_path", run_config["patch_path"]])

    payload = {
        "official_entrypoint": str(official_script),
        "cwd": str(official_root),
        "command": cmd,
        "run_config": run_config,
    }
    (output_dir / "official_command.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return {"dry_run": True, **payload}

    if not official_script.exists():
        raise FileNotFoundError(f"Official UPA-RFAS OFT evaluator not found: {official_script}")

    env = os.environ.copy()
    libero_root = run_config.get("libero_root", "")
    pythonpath_parts = [str(official_root)]
    if libero_root:
        pythonpath_parts.append(libero_root)
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["WHEN_ROLLOUT_DIR"] = str(output_dir / "videos")
    if run_config["attack_protocol"] == "physical_both":
        env["WHEN_OFT_WORLD_PATCH"] = "1"
        if run_config.get("world_patch_pos", ""):
            env.setdefault("WHEN_WORLD_PATCH_POS", run_config["world_patch_pos"])
        env.setdefault("WHEN_WORLD_PATCH_ANCHOR_MODE", run_config.get("world_patch_anchor_mode", "object_cluster"))
        env.setdefault("WHEN_WORLD_PATCH_ANCHOR_SITE", run_config.get("world_patch_anchor_site", "table_top"))
        env.setdefault("WHEN_WORLD_PATCH_OFFSET", run_config.get("world_patch_offset", "0.00 0.00 0.001"))
        env.setdefault("WHEN_WORLD_PATCH_SIZE", run_config.get("world_patch_size", "0.055 0.055 0.001"))
        env.setdefault("WHEN_WORLD_PATCH_QUAT", run_config.get("world_patch_quat", "1 0 0 0"))
        env.setdefault("WHEN_WORLD_PATCH_SNAP_TO_SURFACE", run_config.get("world_patch_snap_to_surface", "1"))
        env.setdefault("WHEN_WORLD_PATCH_REQUIRE_SURFACE", run_config.get("world_patch_require_surface", "1"))
        env.setdefault("WHEN_WORLD_PATCH_Z_OFFSET", run_config.get("world_patch_z_offset", "0"))
    else:
        env["WHEN_OFT_WORLD_PATCH"] = "0"

    stdout_path = output_dir / "official_eval.out"
    with stdout_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=official_root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    log_files = sorted(log_dir.glob("*.txt"), key=lambda item: item.stat().st_mtime, reverse=True)
    official_log = log_files[0] if log_files else None
    parsed = _parse_official_log(official_log) if official_log else {}
    linked_videos = _link_official_videos(official_log, official_root=official_root, output_dir=output_dir) if official_log else []
    summary = {
        "mode": run_config["mode"],
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": run_config["model_checkpoint"],
        "task_suite": run_config["task_suite_name"],
        "attack_protocol": run_config["attack_protocol"],
        "official_entrypoint": str(official_script),
        "official_log": str(official_log) if official_log else "",
        "stdout_log": str(stdout_path),
        "video_dir": str(output_dir / "videos") if linked_videos else "",
        "videos": linked_videos,
        "patch_path": run_config.get("patch_path", ""),
        "patch_sha256": run_config.get("patch_sha256", ""),
        **parsed,
    }
    if "success_rate" in summary and run_config["mode"] in {"oft_official_attack", "oft_official_physical_attack"}:
        summary["attack_success_rate"] = 1.0 - float(summary["success_rate"])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    config = merge_when_defaults(load_when_config(args.config))
    suite = args.suite or config.get("task_suite", "libero_spatial")
    protocol = args.attack_protocol or config.get("attack_protocol", "digital_both")
    manifest_path = None
    manifest = None
    if args.mode == "attack":
        manifest_path = require_patch_manifest(args.patch_manifest or None)
        manifest = read_patch_manifest(manifest_path)
    output_dir = resolve_output_path(args.output_dir)
    attack_views = [] if protocol == "physical_both" else (
        list(views_for_protocol(protocol)) if args.mode == "attack" else []
    )
    use_official_oft = args.mode == "clean" or protocol in {"digital_both", "physical_both"}
    run_config = {
        "exp_id": f"WHEN_OFT_{suite}_{args.mode}_{protocol}",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": str(config.get("model_checkpoint") or OFT_MODEL_PATHS[suite]),
        "task_suite": suite,
        "task_suite_name": suite,
        "unnorm_key": suite_to_unnorm_key(suite),
        "input_views": "front,wrist",
        "attack_views": ",".join(attack_views),
        "eval_patch_space": str(config.get("eval_patch_space", "raw_image")),
        "eval_patch_position": f"upa_rfas_{suite}",
        "eval_raw_rgb_resize_before_patch": str(config.get("eval_raw_rgb_resize_before_patch", "true")),
        "eval_raw_rgb_resize_size": str(config.get("eval_raw_rgb_resize_size", "224")),
        "num_images_in_input": "2",
        "use_proprio": "true",
        "use_l1_regression": "true",
        "num_open_loop_steps": config.get("num_open_loop_steps", "8"),
        "num_steps_wait": config.get("num_steps_wait", "10"),
        "eval_tasks": config.get("eval_tasks", "10"),
        "eval_episodes_per_task": config.get("eval_episodes_per_task", "10"),
        "num_episodes": config.get("num_episodes", "100"),
        "seed": config.get("seed", "7"),
        "save_video": config.get("save_video", "true"),
        "video_views": "front,wrist",
        "openvla_oft_root": config.get("openvla_oft_root", DEFAULT_OFT_ROOT),
        "official_oft_root": config.get("official_oft_root", DEFAULT_UPA_RFAS_OFT_ROOT),
        "libero_root": config.get("libero_root", DEFAULT_LIBERO_ROOT),
        "attack_protocol": protocol if args.mode == "attack" else "clean",
        "patch_manifest": str(manifest_path) if manifest_path else "",
        "patch_path": manifest["patch_path"] if manifest else "",
        "patch_sha256": manifest["patch_sha256_verified"] if manifest else "",
        "world_patch_pos": str(config.get("world_patch_pos", "")),
        "world_patch_anchor_mode": str(config.get("world_patch_anchor_mode", "object_cluster")),
        "world_patch_anchor_site": str(config.get("world_patch_anchor_site", "table_top")),
        "world_patch_offset": str(config.get("world_patch_offset", "0.00 0.00 0.001")),
        "world_patch_size": str(config.get("world_patch_size", "0.055 0.055 0.001")),
        "world_patch_quat": str(config.get("world_patch_quat", "1 0 0 0")),
        "world_patch_snap_to_surface": str(config.get("world_patch_snap_to_surface", "1")),
        "world_patch_require_surface": str(config.get("world_patch_require_surface", "1")),
        "world_patch_z_offset": str(config.get("world_patch_z_offset", "0")),
        "official_evaluator": "upa_rfas_openvla_oft" if use_official_oft else "",
        "mode": "oft_official_physical_attack" if args.mode == "attack" and protocol == "physical_both" else (
            "oft_official_attack" if args.mode == "attack" and use_official_oft else
            "oft_official_clean" if use_official_oft else f"when_oft_{args.mode}_eval"
        ),
    }
    print(json.dumps({"run_config": run_config, "manifest": manifest, "output_dir": str(output_dir)}, indent=2))
    if use_official_oft:
        _run_official_oft_eval(
            run_config=run_config,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        return
    if args.dry_run:
        return
    assert_cuda_available()
    import torch

    patch = None
    if manifest:
        patch = torch.load(manifest["patch_path"], map_location="cuda:0")
    metrics = run_libero_rollout(
        run_config,
        output_dir,
        patch=patch,
        attack_views=attack_views,
        mode=f"when_oft_{args.mode}_eval",
    )
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
