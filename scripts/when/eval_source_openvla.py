#!/usr/bin/env python3
"""Evaluate clean/attacked source OpenVLA on LIBERO suites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.openvla_oft_experiments import TASK_MAX_STEPS
from VLAAttacker.multiview.openvla_oft_experiments import save_video, write_csv, write_json
from VLAAttacker.when.config import LIBERO_ROOT, LIBERO_SUITES, SOURCE_MODEL_CANDIDATES
from VLAAttacker.when.config import load_when_config, merge_when_defaults, resolve_output_path
from VLAAttacker.when.config import source_suite_to_unnorm_key
from VLAAttacker.when.patch_io import read_patch_manifest, require_patch_manifest
from VLAAttacker.when.trainer import assert_cuda_available
from VLAAttacker.when.transforms import apply_patch_to_rgb


ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=LIBERO_SUITES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["clean", "attack"], default="attack")
    parser.add_argument("--patch-manifest", default="")
    parser.add_argument("--source-model-checkpoint", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _int_config(config: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(str(config.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _bool_config(config: dict[str, Any], key: str, default: bool) -> bool:
    raw = str(config.get(key, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _mean_or_empty(values: list[float]) -> float | str:
    return float(sum(values) / len(values)) if values else ""


def _safe_task_name(task_description: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in task_description.lower())
    return "_".join(part for part in cleaned.split("_") if part)[:60] or "task"


def _ensure_runtime_paths(config: dict[str, Any]) -> None:
    for value in [config.get("project_root", ""), config.get("libero_root", str(LIBERO_ROOT))]:
        path = Path(str(value)).expanduser()
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _configure_cuda_runtime(config: dict[str, Any]) -> None:
    import torch

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = _bool_config(config, "allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = _bool_config(config, "allow_tf32", True)
    torch.backends.cudnn.deterministic = _bool_config(config, "cudnn_deterministic", True)
    torch.backends.cudnn.benchmark = _bool_config(config, "cudnn_benchmark", False)
    if _bool_config(config, "disable_cudnn", True):
        torch.backends.cudnn.enabled = False
    try:
        torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
    except Exception:
        pass


def _print_cuda_diagnostics(stage: str) -> None:
    import torch

    payload = {
        "stage": stage,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        payload.update({
            "current_device": device,
            "device_name": torch.cuda.get_device_name(device),
            "memory_free_gb": round(free_bytes / 1024 ** 3, 3),
            "memory_total_gb": round(total_bytes / 1024 ** 3, 3),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(device) / 1024 ** 3, 3),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(device) / 1024 ** 3, 3),
        })
    print("[source-openvla][cuda] " + json.dumps(payload, ensure_ascii=False))


def _resolve_source_checkpoint(args: argparse.Namespace, config: dict[str, Any], suite: str) -> Path:
    explicit = str(args.source_model_checkpoint or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    per_suite_key = f"source_model_checkpoint_{suite}"
    configured = str(config.get(per_suite_key, "")).strip()
    if configured:
        return Path(configured).expanduser()
    candidate = SOURCE_MODEL_CANDIDATES.get(suite)
    if candidate is not None:
        return candidate
    fallback = str(config.get("source_model_checkpoint", "")).strip()
    if fallback:
        return Path(fallback).expanduser()
    raise ValueError(f"No source OpenVLA checkpoint configured for {suite}")


def _resolve_unnorm_key(model: Any, requested: str) -> str:
    norm_stats = getattr(model, "norm_stats", {}) or {}
    if requested in norm_stats:
        return requested
    no_noops_key = f"{requested}_no_noops"
    if no_noops_key in norm_stats:
        return no_noops_key
    if len(norm_stats) == 1:
        only_key = next(iter(norm_stats.keys()))
        print(f"[source-openvla] requested unnorm_key={requested!r}; using only model key={only_key!r}")
        return only_key
    raise KeyError(f"Action unnorm key {requested!r} not found. Available keys: {sorted(norm_stats)}")


def _action_delta_metrics(clean_action: Any | None, attacked_action: Any | None) -> dict[str, float | str]:
    import numpy as np

    if clean_action is None or attacked_action is None:
        return {
            "action_l2": "",
            "position_error": "",
            "rotation_error": "",
            "gripper_error": "",
        }
    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    attacked = np.asarray(attacked_action, dtype=np.float32).reshape(-1)
    delta = attacked[:ACTION_DIM] - clean[:ACTION_DIM]
    return {
        "action_l2": float(np.linalg.norm(delta)),
        "position_error": float(np.linalg.norm(delta[:3])),
        "rotation_error": float(np.linalg.norm(delta[3:6])),
        "gripper_error": float(abs(delta[6])),
    }


def _build_official_openvla_cfg(
    *,
    checkpoint: Path,
    suite: str,
    center_crop: bool,
    seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(checkpoint),
        unnorm_key=source_suite_to_unnorm_key(suite),
        center_crop=center_crop,
        load_in_8bit=False,
        load_in_4bit=False,
        task_suite_name=suite,
        num_steps_wait=10,
        num_trials_per_task=50,
        seed=seed,
    )


def _build_official_observation(image: Any, obs: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from experiments.robot.libero.libero_utils import quat2axisangle

    return {
        "full_image": image,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }


def run_source_eval(
    *,
    config: dict[str, Any],
    suite: str,
    output_dir: Path,
    checkpoint: Path,
    mode: str,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    _ensure_runtime_paths(config)

    import numpy as np
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    from experiments.robot.openvla_utils import get_processor
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )

    seed = _int_config(config, "seed", 7)
    center_crop = _bool_config(config, "center_crop", True)
    official_cfg = _build_official_openvla_cfg(
        checkpoint=checkpoint,
        suite=suite,
        center_crop=center_crop,
        seed=seed,
    )
    set_seed_everywhere(seed)
    _configure_cuda_runtime(config)
    _print_cuda_diagnostics("before_model_load")

    device = "cuda:0"
    model = get_model(official_cfg)
    processor = get_processor(official_cfg)
    _print_cuda_diagnostics("after_model_load")
    unnorm_key = _resolve_unnorm_key(model, official_cfg.unnorm_key)
    official_cfg.unnorm_key = unnorm_key
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    patch = None
    if manifest is not None:
        import torch

        patch = torch.load(manifest["patch_path"], map_location=device)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    save_videos = _bool_config(config, "save_video", True)
    resize_size = get_image_resize_size(official_cfg)
    num_steps_wait = _int_config(config, "num_steps_wait", 10)
    eval_tasks = _int_config(config, "eval_tasks", 10)
    eval_episodes_per_task = _int_config(config, "eval_episodes_per_task", 10)
    max_steps = TASK_MAX_STEPS.get(suite, 220)

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task_limit = min(task_suite.n_tasks, eval_tasks) if eval_tasks > 0 else task_suite.n_tasks

    episode_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    task_successes: dict[int, int] = {}
    task_episodes: dict[int, int] = {}
    all_action_l2: list[float] = []
    all_position_error: list[float] = []
    all_rotation_error: list[float] = []
    all_gripper_error: list[float] = []
    all_first_step_position_error: list[float] = []
    total_successes = 0
    total_episodes = 0

    for task_id in range(task_limit):
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "openvla", resolution=256)
        initial_states = task_suite.get_task_init_states(task_id)
        episode_limit = min(len(initial_states), eval_episodes_per_task)
        task_successes.setdefault(task_id, 0)
        task_episodes.setdefault(task_id, 0)

        try:
            for local_episode_idx in range(episode_limit):
                env.reset()
                obs = env.set_init_state(initial_states[local_episode_idx])
                replay_images: list[np.ndarray] = []
                episode_action_l2: list[float] = []
                episode_position_error: list[float] = []
                episode_rotation_error: list[float] = []
                episode_gripper_error: list[float] = []
                first_step_position_error: float | str = ""
                success = False
                error = ""
                step_count = 0

                try:
                    for t in range(max_steps + num_steps_wait):
                        if t < num_steps_wait:
                            obs, _reward, done, _info = env.step(get_libero_dummy_action("openvla"))
                            continue

                        clean_image = get_libero_image(obs, resize_size)
                        attacked_image = (
                            apply_patch_to_rgb(clean_image, patch, position="upa_rfas_default", suite=suite)
                            if patch is not None
                            else clean_image
                        )
                        replay_images.append(np.asarray(attacked_image, dtype=np.uint8))

                        clean_action = None
                        if patch is not None:
                            clean_action = get_action(
                                official_cfg,
                                model,
                                _build_official_observation(clean_image, obs),
                                task_description,
                                processor=processor,
                            )
                        raw_action = get_action(
                            official_cfg,
                            model,
                            _build_official_observation(attacked_image, obs),
                            task_description,
                            processor=processor,
                        )
                        action_delta = _action_delta_metrics(clean_action, raw_action)
                        for metric_key, episode_list, global_list in [
                            ("action_l2", episode_action_l2, all_action_l2),
                            ("position_error", episode_position_error, all_position_error),
                            ("rotation_error", episode_rotation_error, all_rotation_error),
                            ("gripper_error", episode_gripper_error, all_gripper_error),
                        ]:
                            value = action_delta[metric_key]
                            if value != "":
                                episode_list.append(float(value))
                                global_list.append(float(value))
                        if first_step_position_error == "" and action_delta["position_error"] != "":
                            first_step_position_error = float(action_delta["position_error"])
                            all_first_step_position_error.append(float(action_delta["position_error"]))

                        recorded_clean_action = clean_action if patch is not None else raw_action
                        recorded_attacked_action = raw_action if patch is not None else None
                        action_rows.append({
                            "episode_id": total_episodes,
                            "step": step_count,
                            "task_name": task_description,
                            "view": "front" if patch is not None else "clean",
                            "clean_action": (
                                recorded_clean_action.tolist()
                                if recorded_clean_action is not None
                                else ""
                            ),
                            "attacked_action": (
                                recorded_attacked_action.tolist()
                                if recorded_attacked_action is not None
                                else ""
                            ),
                            "action_l2": action_delta["action_l2"],
                            "position_error": action_delta["position_error"],
                            "rotation_error": action_delta["rotation_error"],
                            "gripper_error": action_delta["gripper_error"],
                            "notes": f"source_openvla_{mode}; center_crop={center_crop}",
                        })

                        action = normalize_gripper_action(
                            np.asarray(raw_action, dtype=np.float32).copy(),
                            binarize=True,
                        )
                        action = invert_gripper_action(action)
                        obs, _reward, done, _info = env.step(action.tolist())
                        step_count += 1
                        if done:
                            success = True
                            break
                except Exception as exc:
                    error = repr(exc)

                if success:
                    task_successes[task_id] += 1
                    total_successes += 1
                task_episodes[task_id] += 1

                video_path = ""
                if save_videos:
                    outcome_tag = "task_succeeded" if success else "task_failed"
                    path = video_dir / (
                        f"episode_{total_episodes:04d}_{outcome_tag}_"
                        f"{_safe_task_name(task_description)}.mp4"
                    )
                    video_path = save_video(replay_images, path)

                episode_rows.append({
                    "episode_id": total_episodes,
                    "task_id": task_id,
                    "local_episode_id": local_episode_idx,
                    "task_name": task_description,
                    "success": int(success),
                    "attack_success": int(not success) if patch is not None else "",
                    "action_l2_mean": _mean_or_empty(episode_action_l2),
                    "position_error_mean": _mean_or_empty(episode_position_error),
                    "rotation_error_mean": _mean_or_empty(episode_rotation_error),
                    "gripper_error_mean": _mean_or_empty(episode_gripper_error),
                    "clean_adv_action_l2_mean": _mean_or_empty(episode_action_l2),
                    "clean_adv_position_l2_mean": _mean_or_empty(episode_position_error),
                    "clean_adv_rotation_l2_mean": _mean_or_empty(episode_rotation_error),
                    "clean_adv_gripper_abs_mean": _mean_or_empty(episode_gripper_error),
                    "clean_adv_first_step_position_l2": first_step_position_error,
                    "trajectory_deviation": "",
                    "view_disagreement_mean": "",
                    "latency_ms_mean": "",
                    "notes": (
                        f"mode=source_openvla_{mode}; task_id={task_id}; "
                        f"local_episode={local_episode_idx}; steps={step_count}; "
                        f"video={video_path}; error={error}"
                    ),
                })
                total_episodes += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

    per_task_success_rate = {
        str(task_id): (
            float(task_successes.get(task_id, 0)) / float(task_episodes[task_id])
            if task_episodes.get(task_id, 0)
            else 0.0
        )
        for task_id in sorted(task_episodes)
    }
    success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    metrics = {
        "mode": f"source_openvla_{mode}",
        "model_name": "OpenVLA",
        "model_checkpoint": str(checkpoint),
        "task_suite": suite,
        "unnorm_key": unnorm_key,
        "num_episodes": total_episodes,
        "eval_tasks": task_limit,
        "eval_episodes_per_task": eval_episodes_per_task,
        "num_successes": total_successes,
        "task_success_rate": success_rate,
        "success_rate": success_rate,
        "attack_success_rate": (1.0 - success_rate) if patch is not None and total_episodes else None,
        "per_task_success_rate": per_task_success_rate,
        "patch_used": patch is not None,
        "patch_sha256": manifest.get("patch_sha256_verified", "") if manifest else "",
        "attack_protocol": "source_front_digital" if patch is not None else "clean",
        "patch_position": "upa_rfas_default" if patch is not None else "",
        "center_crop": center_crop,
        "max_steps": max_steps,
        "num_steps_wait": num_steps_wait,
        "action_l2_mean": _mean_or_empty(all_action_l2),
        "position_error_mean": _mean_or_empty(all_position_error),
        "rotation_error_mean": _mean_or_empty(all_rotation_error),
        "gripper_error_mean": _mean_or_empty(all_gripper_error),
        "clean_adv_action_l2_mean": _mean_or_empty(all_action_l2),
        "clean_adv_position_l2_mean": _mean_or_empty(all_position_error),
        "clean_adv_rotation_l2_mean": _mean_or_empty(all_rotation_error),
        "clean_adv_gripper_abs_mean": _mean_or_empty(all_gripper_error),
        "clean_adv_first_step_position_l2": _mean_or_empty(all_first_step_position_error),
        "trajectory_deviation_mean": "",
    }
    write_json(output_dir / "rollout_metrics.json", metrics)
    write_csv(
        output_dir / "metrics_per_episode.csv",
        [
            "episode_id",
            "task_id",
            "local_episode_id",
            "task_name",
            "success",
            "attack_success",
            "action_l2_mean",
            "position_error_mean",
            "rotation_error_mean",
            "gripper_error_mean",
            "clean_adv_action_l2_mean",
            "clean_adv_position_l2_mean",
            "clean_adv_rotation_l2_mean",
            "clean_adv_gripper_abs_mean",
            "clean_adv_first_step_position_l2",
            "trajectory_deviation",
            "view_disagreement_mean",
            "latency_ms_mean",
            "notes",
        ],
        episode_rows,
    )
    write_csv(
        output_dir / "actions.csv",
        [
            "episode_id",
            "step",
            "task_name",
            "view",
            "clean_action",
            "attacked_action",
            "action_l2",
            "position_error",
            "rotation_error",
            "gripper_error",
            "notes",
        ],
        action_rows,
    )
    return metrics


def main() -> None:
    args = parse_args()
    config = merge_when_defaults(load_when_config(args.config))
    suite = args.suite or config.get("task_suite", "libero_10")
    output_dir = resolve_output_path(args.output_dir)
    checkpoint = _resolve_source_checkpoint(args, config, suite)
    manifest = None
    manifest_path = None
    if args.mode == "attack":
        manifest_path = require_patch_manifest(args.patch_manifest or None)
        manifest = read_patch_manifest(manifest_path)

    run_config = {
        "suite": suite,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "source_model_checkpoint": str(checkpoint),
        "source_model_exists": checkpoint.exists(),
        "libero_root": str(config.get("libero_root", LIBERO_ROOT)),
        "manifest_path": str(manifest_path) if manifest_path else "",
        "patch_sha256": manifest.get("patch_sha256_verified", "") if manifest else "",
        "eval_tasks": config.get("eval_tasks", "10"),
        "eval_episodes_per_task": config.get("eval_episodes_per_task", "10"),
        "seed": config.get("seed", "7"),
        "center_crop": config.get("center_crop", "true"),
        "official_model_family": "openvla",
        "official_pretrained_checkpoint": str(checkpoint),
        "official_unnorm_key_requested": source_suite_to_unnorm_key(suite),
        "official_load_in_8bit": False,
        "official_load_in_4bit": False,
    }
    print(json.dumps({"run_config": run_config, "manifest": manifest}, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Source OpenVLA checkpoint does not exist: {checkpoint}")
    libero_root = Path(str(config.get("libero_root", LIBERO_ROOT))).expanduser()
    if not libero_root.exists():
        raise FileNotFoundError(f"LIBERO root does not exist: {libero_root}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert_cuda_available()
    metrics = run_source_eval(
        config=config,
        suite=suite,
        output_dir=output_dir,
        checkpoint=checkpoint,
        mode=args.mode,
        manifest=manifest,
    )
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
