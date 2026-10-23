#!/usr/bin/env python3
"""统一运行 R/M 多视角实验。

默认只做 dry-run；真实运行必须显式设置 `RUN_MODE=run`。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.patch_manifest import (
    OFFICIAL_M_ATTACK_IMPLEMENTATION,
    OFFICIAL_M_ATTACK_LOSS,
    OFFICIAL_M_PATCH_SPACE,
    build_manifest,
    find_latest_manifest,
    find_latest_patch,
    official_m_dependency_error,
    write_manifest,
)
from VLAAttacker.multiview.view_batch_utils import (
    is_todo,
    parse_bool,
    read_simple_yaml,
    resolve_project_path,
    validate_view_sets,
    write_simple_yaml,
)


SUMMARY_FIELDS = [
    "exp_id",
    "timestamp",
    "model_name",
    "input_views",
    "patch_train_views",
    "attack_views",
    "patch_type",
    "train_patch",
    "use_pretrained_patch",
    "patch_family",
    "patch_size",
    "task_success_rate",
    "attack_success_rate",
    "action_l2_mean",
    "position_error_mean",
    "rotation_error_mean",
    "gripper_error_mean",
    "clean_adv_action_l2_mean",
    "clean_adv_position_l2_mean",
    "clean_adv_rotation_l2_mean",
    "clean_adv_gripper_abs_mean",
    "clean_adv_first_step_position_l2",
    "trajectory_deviation_mean",
    "view_disagreement_mean",
    "avg_latency_ms",
    "gpu_id",
    "gpu_memory_peak_mb",
    "output_dir",
    "notes",
]

CSV_HEADERS = {
    "metrics_per_episode.csv": [
        "episode_id",
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
    "actions.csv": [
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
    "trajectories.csv": [
        "episode_id",
        "step",
        "task_name",
        "clean_x",
        "clean_y",
        "clean_z",
        "attacked_x",
        "attacked_y",
        "attacked_z",
        "trajectory_deviation",
        "notes",
    ],
    "gpu_usage.csv": ["timestamp", "gpu_id", "memory_used_mb", "memory_peak_mb", "utilization_percent", "notes"],
}

WRAPPER_BY_METHOD = {
    "UADA": "VLAAttacker/UADA_wrapper.py",
    "TMA": "VLAAttacker/TMA_wrapper.py",
    "UPA": "VLAAttacker/UPA_wrapper.py",
}

PIXEL_GRAD_BLOCK_MESSAGE = (
    "OpenVLA-OFT differentiable patch training probe is not ready. "
    "Run/fix probe_openvla_oft_forward.py before submitting M1/M2/M5."
)
PREDICT_ACTION_BLOCK_MESSAGE = (
    "OpenVLA-OFT predict_action eval is not ready. Please run/fix probe_openvla_oft_forward.py first."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one R/M experiment with dry-run first")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dry_run", action="store_true", help="只做 dry-run，不加载模型")
    parser.add_argument("--run_mode", default=os.environ.get("RUN_MODE", "dry_run"), choices=["dry_run", "run"])
    parser.add_argument("--gpu_id", default=os.environ.get("GPU_ID", "0"), type=str)
    parser.add_argument("--allow_protected", action="store_true", help="临时允许运行受保护的 R1/A3 映射实验")
    return parser.parse_args()


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv_header(path: Path, headers: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)


def write_status(path: Path, stage: str, message: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def unique_output_dir(output_root: Path, exp_id: str) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{timestamp}_{exp_id}"
    if not base.exists():
        return timestamp, base
    idx = 1
    while True:
        candidate = output_root / f"{timestamp}_{exp_id}_{idx}"
        if not candidate.exists():
            return timestamp, candidate
        idx += 1


def resolve_local_paths(config: dict[str, str], config_path: Path) -> dict[str, str]:
    """用 local_paths.yaml 覆盖 TODO/LOCAL_PATHS 字段；文件不存在时 dry-run 仍可继续。"""
    resolved = dict(config)
    local_paths_value = resolved.get("local_paths_file", "configs/experiments/local_paths.yaml")
    local_paths_path = resolve_project_path(PROJECT_ROOT, local_paths_value)
    if local_paths_path.exists():
        local_paths = read_simple_yaml(local_paths_path)
        for key, value in local_paths.items():
            if key not in resolved or is_todo(resolved.get(key)):
                resolved[key] = value

        model_name = resolved.get("model_name", resolved.get("model_family", ""))
        if is_todo(resolved.get("model_checkpoint")):
            if model_name == "OpenVLA" and "OPENVLA_MODEL_PATH" in local_paths:
                resolved["model_checkpoint"] = local_paths["OPENVLA_MODEL_PATH"]
            if model_name == "OpenVLA-OFT" and "OPENVLA_OFT_MODEL_PATH" in local_paths:
                resolved["model_checkpoint"] = local_paths["OPENVLA_OFT_MODEL_PATH"]
    resolved["source_config"] = str(config_path)
    resolved["local_paths_file_resolved"] = str(local_paths_path)
    return resolved


def validate_for_mode(config: dict[str, str], run_mode: str, allow_protected: bool) -> list[str]:
    issues = validate_view_sets(config)
    exp_id = str(config.get("exp_id", "")).strip()
    model_name = str(config.get("model_name", config.get("model_family", ""))).strip()
    is_openvla_oft_m = model_name == "OpenVLA-OFT" and exp_id in {"M1", "M2", "M3", "M4", "M5", "M6", "M7"}
    is_a_series_oft_w = model_name == "OpenVLA-OFT_w" and exp_id in {"A0", "A1"}
    if is_openvla_oft_m and (
        parse_bool(config.get("train_patch", "false"))
        or parse_bool(config.get("use_pretrained_patch", "false"))
        or parse_bool(config.get("use_synthetic_patch", "false"))
    ):
        implementation = str(config.get("attack_implementation") or "").strip()
        if implementation != OFFICIAL_M_ATTACK_IMPLEMENTATION:
            issues.append(
                f"{exp_id} 必须显式使用 attack_implementation={OFFICIAL_M_ATTACK_IMPLEMENTATION}"
            )
        attack_loss_type = str(config.get("attack_loss_type") or "").strip().lower()
        if attack_loss_type != OFFICIAL_M_ATTACK_LOSS:
            issues.append(f"{exp_id} 必须显式使用 attack_loss_type={OFFICIAL_M_ATTACK_LOSS}")
        eval_patch_space = str(config.get("eval_patch_space") or "").strip().lower()
        if eval_patch_space != OFFICIAL_M_PATCH_SPACE:
            issues.append(f"{exp_id} 必须显式使用 eval_patch_space={OFFICIAL_M_PATCH_SPACE}")
    if is_openvla_oft_m and exp_id in {"M1", "M2", "M5"} and parse_bool(config.get("train_patch", "false")):
        train_patch_space = str(config.get("train_patch_space") or "").strip().lower()
        if train_patch_space != OFFICIAL_M_PATCH_SPACE:
            issues.append(f"{exp_id} 必须显式使用 train_patch_space={OFFICIAL_M_PATCH_SPACE}")
        if str(config.get("train_patch_position", "")).strip().lower() != "anchored_eot":
            issues.append(f"{exp_id} 必须使用 train_patch_position=anchored_eot")
        if str(config.get("eval_patch_position", "")).strip().lower() != "top_left":
            issues.append(f"{exp_id} 必须使用 eval_patch_position=top_left")
        if not parse_bool(config.get("eot_enabled", "false")):
            issues.append(f"{exp_id} 必须使用 eot_enabled=true")
        if parse_bool(config.get("eval_eot_enabled", "false")):
            issues.append(f"{exp_id} 必须使用 eval_eot_enabled=false")
        probs = [
            float(config.get("eot_fixed_anchor_probability", "0")),
            float(config.get("eot_local_jitter_probability", "0")),
            float(config.get("eot_global_random_probability", "0")),
        ]
        if abs(sum(probs) - 1.0) > 1e-6:
            issues.append(f"{exp_id} EOT 概率之和必须为 1.0，当前为 {sum(probs)}")
        if str(config.get("multiview_eot_transform_coupling", "")).strip().lower() != "shared":
            issues.append(f"{exp_id} 必须使用 multiview_eot_transform_coupling=shared")
        if str(config.get("action_divergence_type", "")).strip().lower() != "mse":
            issues.append(f"{exp_id} 必须显式记录 action_divergence_type=mse")
        if str(config.get("loss_weight_mode", "")).strip().lower() not in {"fixed", "gradient_calibrated_fixed"}:
            issues.append(f"{exp_id} loss_weight_mode 只支持 fixed 或 gradient_calibrated_fixed")
    if is_a_series_oft_w:
        if str(config.get("input_views", "")).strip() != "front":
            issues.append(f"{exp_id} 必须使用 input_views=front")
        if str(config.get("num_images_in_input", "")).strip() != "1":
            issues.append(f"{exp_id} 必须使用 num_images_in_input=1")
        if parse_bool(config.get("suite_specific_model", "false")):
            expected_subfolder = str(config.get("task_suite", "libero_spatial")).strip()
        else:
            expected_subfolder = "libero_spatial"
        if str(config.get("model_subfolder", "")).strip() != expected_subfolder:
            issues.append(f"{exp_id} 必须显式使用 model_subfolder={expected_subfolder}")
        if str(config.get("device", "cuda")).strip().lower() != "cuda":
            issues.append(f"{exp_id} 必须显式使用 device=cuda")
        if parse_bool(config.get("train_patch", "false")):
            issues.append(f"{exp_id} 不允许训练 patch")
        if exp_id == "A0" and parse_bool(config.get("use_pretrained_patch", "false")):
            issues.append("A0 是 clean baseline，不应加载 pretrained patch")
        if exp_id == "A1":
            if not parse_bool(config.get("use_pretrained_patch", "false")):
                issues.append("A1 必须 use_pretrained_patch=true")
            if str(config.get("pretrained_patch_source_exp_id", "")).strip() != "M1":
                issues.append("A1 必须直接加载 M1 patch")
            if str(config.get("patch_train_views", "")).strip() != "none":
                issues.append("A1 必须设置 patch_train_views=none，避免误解为重新训练")
            if str(config.get("attack_views", "")).strip() != "front":
                issues.append("A1 必须设置 attack_views=front")
            if str(config.get("eval_patch_space", "")).strip().lower() != OFFICIAL_M_PATCH_SPACE:
                issues.append(f"A1 必须使用 eval_patch_space={OFFICIAL_M_PATCH_SPACE}")
            if str(config.get("attack_implementation", "")).strip() != OFFICIAL_M_ATTACK_IMPLEMENTATION:
                issues.append(f"A1 必须使用 attack_implementation={OFFICIAL_M_ATTACK_IMPLEMENTATION}")
            if str(config.get("patch_retrained", "")).strip().lower() != "false":
                issues.append("A1 manifest/config 必须记录 patch_retrained=false")
    if run_mode == "run":
        if os.environ.get("RUN_MODE") != "run":
            issues.append("真实运行必须显式设置环境变量 RUN_MODE=run")
        if parse_bool(config.get("run_protected", "false")) and not (
            allow_protected or os.environ.get("ALLOW_PROTECTED_R1") == "1"
        ):
            issues.append("该实验受保护：A3/R1 正在训练映射未解锁")
        if not parse_bool(config.get("run_enabled", "true")):
            issues.append("该实验当前只生成配置和 sbatch，不允许 run")
        for key in ["model_checkpoint", "data_root", "libero_root", "project_root", "conda_env"]:
            if is_todo(config.get(key)):
                issues.append(f"{key} 仍是 TODO/LOCAL_PATHS")
    return issues


def build_train_command(config: dict[str, str], output_dir: Path, gpu_id: str, dry_run: bool) -> list[str]:
    method = config.get("attack_method", "UADA").upper()
    if method not in WRAPPER_BY_METHOD:
        raise ValueError(f"暂不支持的 attack_method: {method}")

    server_root = config.get("data_root", ".")
    if is_todo(server_root):
        server_root = "."

    command = [
        config.get("python_bin", "python"),
        WRAPPER_BY_METHOD[method],
        "--exp_id",
        config["exp_id"],
        "--dataset",
        config.get("task_suite", "libero_spatial"),
        "--server",
        server_root,
        "--device",
        "0",
        "--maskidx",
        config.get("maskidx", "0"),
        "--patch_size",
        config.get("wrapper_patch_size", "3,50,50"),
        "--lr",
        config.get("learning_rate", "0.001"),
        "--iter",
        config.get("max_train_iters", "2000"),
        "--bs",
        config.get("batch_size", "8"),
        "--accumulate",
        config.get("accumulate", "1"),
        "--warmup",
        config.get("warmup", "20"),
        "--innerLoop",
        config.get("inner_loop", "50"),
        "--geometry",
        config.get("geometry", "true"),
        "--filterGripTrainTo1",
        config.get("filter_grip_train_to_1", "false"),
        "--resize_patch",
        config.get("resize_patch", "false"),
        "--wandb_project",
        "false",
        "--input_views",
        config.get("input_views", "front"),
        "--patch_train_views",
        config.get("patch_train_views", "front"),
        "--attack_views",
        config.get("attack_views", "front"),
        "--patch_type",
        config.get("patch_type", "single_view"),
        "--multiview_fusion",
        config.get("multiview_fusion", "stitch"),
        "--output_dir",
        str(output_dir),
    ]
    model_path = config.get("model_checkpoint", config.get("MODEL_PATH", ""))
    if model_path and not is_todo(model_path):
        command.extend(["--vla_path", model_path])
    view_key_map = config.get("view_key_map", "")
    if view_key_map and not is_todo(view_key_map):
        command.extend(["--view_key_map", view_key_map])
    if parse_bool(config.get("multiview_debug", "false")):
        command.append("--multiview_debug")
    if dry_run:
        command.insert(2, "--dry_run")
    return command


def build_reserved_command(
    config: dict[str, str],
    output_dir: Path,
    dry_run: bool,
    dependency_manifest: str = "",
) -> list[str]:
    if not parse_bool(config.get("run_enabled", "true")):
        mode = "config_only"
    elif parse_bool(config.get("train_patch", "false")):
        mode = "train_patch"
    else:
        mode = (
            "patch_eval"
            if (
                parse_bool(config.get("use_pretrained_patch", "false"))
                or parse_bool(config.get("use_synthetic_patch", "false"))
            )
            else "clean_eval"
        )
    command = [
        config.get("python_bin", "python"),
        "VLAAttacker/multiview/multiview_attack_runner.py",
        "--config",
        config["source_config"],
        "--output_dir",
        str(output_dir),
        "--mode",
        mode,
    ]
    if dependency_manifest:
        command.extend(["--pretrained_patch_manifest", dependency_manifest])
    if dry_run:
        command.append("--dry_run")
    return command


def build_commands(config: dict[str, str], output_dir: Path, gpu_id: str, dependency_manifest: str) -> tuple[list[str], list[str]]:
    if not parse_bool(config.get("run_enabled", "true")):
        return (
            build_reserved_command(config, output_dir, dry_run=True, dependency_manifest=dependency_manifest),
            build_reserved_command(config, output_dir, dry_run=False, dependency_manifest=dependency_manifest),
        )
    if parse_bool(config.get("train_patch", "false")) and config.get("model_name", config.get("model_family", "")) == "OpenVLA":
        return (
            build_train_command(config, output_dir, gpu_id, dry_run=True),
            build_train_command(config, output_dir, gpu_id, dry_run=False),
        )
    return (
        build_reserved_command(config, output_dir, dry_run=True, dependency_manifest=dependency_manifest),
        build_reserved_command(config, output_dir, dry_run=False, dependency_manifest=dependency_manifest),
    )


def build_child_env(gpu_id: str) -> dict[str, str]:
    """Build subprocess env while respecting Slurm's allocated GPU mask."""
    env = os.environ.copy()
    gpu_id = str(gpu_id)
    parent_cvd = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    parent_gpu_id = str(os.environ.get("GPU_ID", "")).strip()

    # Under Slurm, CUDA_VISIBLE_DEVICES may already be a scheduler-provided mask.
    # Preserve it when the requested GPU is logical 0/the inherited GPU_ID; inside
    # the child process cuda:0 then points to the allocated card, not host GPU 0.
    if parent_cvd and (gpu_id == "0" or parent_gpu_id == gpu_id):
        env["CUDA_VISIBLE_DEVICES"] = parent_cvd
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["GPU_ID"] = gpu_id
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_gpu_preflight(command: list[str], env: dict[str, str], handle: Any) -> int:
    """Fail before the real run if the child Python cannot see CUDA."""
    python_bin = command[0] if command else sys.executable
    check = (
        "import os, torch\n"
        "print('[gpu-preflight] CUDA_VISIBLE_DEVICES=' + str(os.environ.get('CUDA_VISIBLE_DEVICES', '')))\n"
        "print('[gpu-preflight] torch.cuda.is_available()=' + str(torch.cuda.is_available()))\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('CUDA is not visible to the run process; refusing to run on CPU')\n"
        "print('[gpu-preflight] device_count=' + str(torch.cuda.device_count()))\n"
        "print('[gpu-preflight] current_device=' + str(torch.cuda.current_device()))\n"
        "print('[gpu-preflight] device_name=' + torch.cuda.get_device_name(torch.cuda.current_device()))\n"
    )
    handle.write(f"[gpu-preflight] command={shell_join([python_bin, '-c', check])}\n")
    handle.flush()
    result = subprocess.run(
        [python_bin, "-c", check],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        handle.write(result.stdout)
    handle.flush()
    return result.returncode


def run_and_log(command: list[str], log_path: Path, gpu_id: str, status_path: Path, phase: str) -> int:
    env = build_child_env(gpu_id)
    start_time = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[command] {shell_join(command)}\n")
        handle.write(
            "[gpu] "
            f"GPU_ID={env.get('GPU_ID', '')} "
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} "
            f"CUDA_DEVICE_ORDER={env.get('CUDA_DEVICE_ORDER', '')}\n"
        )
        handle.flush()
        if phase == "run":
            preflight_code = run_gpu_preflight(command, env, handle)
            if preflight_code != 0:
                elapsed = int(time.time() - start_time)
                handle.write(f"[exit] phase={phase}_gpu_preflight returncode={preflight_code} elapsed_sec={elapsed}\n")
                write_status(
                    status_path,
                    "run_gpu_preflight_failed",
                    "真实运行前 CUDA 检查失败，未执行实验",
                    {"returncode": preflight_code, "elapsed_sec": elapsed},
                )
                return preflight_code
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                handle.write(line)
                handle.flush()
            if line == "" and process.poll() is not None:
                break
        return_code = process.wait()
        elapsed = int(time.time() - start_time)
        handle.write(f"[exit] phase={phase} returncode={return_code} elapsed_sec={elapsed}\n")
        write_status(
            status_path,
            f"{phase}_finished" if return_code == 0 else f"{phase}_failed",
            "子进程结束",
            {"returncode": return_code, "elapsed_sec": elapsed},
        )
        return return_code


def write_command_script(path: Path, dry_command: list[str], run_command: list[str], gpu_id: str) -> None:
    content = f"""#!/usr/bin/env bash
set -euo pipefail

export GPU_ID=${{GPU_ID:-{shlex.quote(gpu_id)}}}
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-$GPU_ID}}
export CUDA_DEVICE_ORDER=${{CUDA_DEVICE_ORDER:-PCI_BUS_ID}}

DRY_RUN_COMMAND={shlex.quote(shell_join(dry_command))}
RUN_COMMAND={shlex.quote(shell_join(run_command))}

echo "[dry-run] $DRY_RUN_COMMAND"
$DRY_RUN_COMMAND
echo "[info] dry-run 通过后，真实运行必须使用 RUN_MODE=run。"
"""
    write_text(path, content)
    path.chmod(0o755)


def append_summary(summary_path: Path, metrics: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: metrics.get(field, "") for field in SUMMARY_FIELDS})


def metrics_template(
    config: dict[str, str],
    timestamp: str,
    output_dir: Path,
    gpu_id: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "exp_id": config.get("exp_id", ""),
        "timestamp": timestamp,
        "model_name": config.get("model_name", config.get("model_family", "")),
        "input_views": config.get("input_views", ""),
        "patch_train_views": config.get("patch_train_views", ""),
        "attack_views": config.get("attack_views", ""),
        "patch_type": config.get("patch_type", ""),
        "train_patch": parse_bool(config.get("train_patch", "false")),
        "use_pretrained_patch": parse_bool(config.get("use_pretrained_patch", "false")),
        "use_synthetic_patch": parse_bool(config.get("use_synthetic_patch", "false")),
        "patch_family": config.get("patch_family", ""),
        "patch_size": config.get("patch_size", ""),
        "task_success_rate": None,
        "attack_success_rate": None,
        "action_l2_mean": None,
        "position_error_mean": None,
        "rotation_error_mean": None,
        "gripper_error_mean": None,
        "clean_adv_action_l2_mean": None,
        "clean_adv_position_l2_mean": None,
        "clean_adv_rotation_l2_mean": None,
        "clean_adv_gripper_abs_mean": None,
        "clean_adv_first_step_position_l2": None,
        "trajectory_deviation_mean": None,
        "view_disagreement_mean": None,
        "avg_latency_ms": None,
        "gpu_id": gpu_id,
        "gpu_memory_peak_mb": None,
        "output_dir": str(output_dir),
        "notes": notes,
    }


def merge_runner_metrics(metrics: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Merge metrics produced by the concrete runner before writing summary rows."""
    rollout_path = output_dir / "rollout_metrics.json"
    train_path = output_dir / "train_metrics.json"
    if rollout_path.exists():
        try:
            rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
            if rollout.get("task_success_rate") is not None:
                metrics["task_success_rate"] = rollout.get("task_success_rate")
            if rollout.get("attack_success_rate") is not None:
                metrics["attack_success_rate"] = rollout.get("attack_success_rate")
            for field in [
                "action_l2_mean",
                "position_error_mean",
                "rotation_error_mean",
                "gripper_error_mean",
                "clean_adv_action_l2_mean",
                "clean_adv_position_l2_mean",
                "clean_adv_rotation_l2_mean",
                "clean_adv_gripper_abs_mean",
                "clean_adv_first_step_position_l2",
                "trajectory_deviation_mean",
            ]:
                if rollout.get(field) not in {None, ""}:
                    metrics[field] = rollout.get(field)
            if rollout.get("num_episodes") is not None:
                metrics["notes"] = f"{metrics.get('notes', '')}; rollout_episodes={rollout.get('num_episodes')}"
        except json.JSONDecodeError:
            metrics["notes"] = f"{metrics.get('notes', '')}; rollout_metrics.json parse failed"
    if train_path.exists():
        try:
            train_metrics = json.loads(train_path.read_text(encoding="utf-8"))
            if train_metrics.get("final_model_loss") is not None:
                metrics["notes"] = f"{metrics.get('notes', '')}; final_model_loss={train_metrics.get('final_model_loss')}"
            if train_metrics.get("final_total_attack_loss") is not None:
                metrics["notes"] = (
                    f"{metrics.get('notes', '')}; "
                    f"final_total_attack_loss={train_metrics.get('final_total_attack_loss')}"
                )
            if train_metrics.get("final_patch_grad_abs_mean") is not None:
                metrics["notes"] = (
                    f"{metrics.get('notes', '')}; "
                    f"final_patch_grad_abs_mean={train_metrics.get('final_patch_grad_abs_mean')}"
                )
        except json.JSONDecodeError:
            metrics["notes"] = f"{metrics.get('notes', '')}; train_metrics.json parse failed"
    return metrics


def write_report(
    path: Path,
    config: dict[str, str],
    output_dir: Path,
    dry_command: list[str],
    run_command: list[str],
    notes: str,
    dependency_manifest: str,
) -> None:
    lines = [
        f"# 实验报告: {config.get('exp_id', '')}",
        "",
        f"- 模型: {config.get('model_name', config.get('model_family', ''))}",
        f"- model_checkpoint: {config.get('model_checkpoint', '')}",
        f"- model_subfolder: {config.get('model_subfolder', '')}",
        f"- 目的: {config.get('purpose', '待确认')}",
        f"- input_views: {config.get('input_views', '')}",
        f"- patch_train_views: {config.get('patch_train_views', '')}",
        f"- attack_views: {config.get('attack_views', '')}",
        f"- train_patch: {config.get('train_patch', '')}",
        f"- use_pretrained_patch: {config.get('use_pretrained_patch', '')}",
        f"- use_synthetic_patch: {config.get('use_synthetic_patch', '')}",
        f"- synthetic_patch_type: {config.get('synthetic_patch_type', '')}",
        f"- synthetic_patch_seed: {config.get('synthetic_patch_seed', '')}",
        f"- patch_family: {config.get('patch_family', '')}",
        f"- patch size: {config.get('patch_size', '')}",
        f"- train_patch_position: {config.get('train_patch_position', '')}",
        f"- eval_patch_position: {config.get('eval_patch_position', '')}",
        f"- eot_enabled: {config.get('eot_enabled', '')}",
        f"- eot_samples: {config.get('eot_samples', '')}",
        f"- loss_weight_mode: {config.get('loss_weight_mode', '')}",
        f"- weight_source: {config.get('weight_source', '')}",
        f"- source_patch_experiment: {config.get('source_patch_experiment', config.get('patch_source_experiment', ''))}",
        f"- transfer_type: {config.get('transfer_type', '')}",
        f"- source_model: {config.get('source_model', '')}",
        f"- target_model: {config.get('target_model', '')}",
        f"- source_view: {config.get('source_view', '')}",
        f"- target_view: {config.get('target_view', '')}",
        f"- patch_retrained: {config.get('patch_retrained', '')}",
        f"- 输出目录: {output_dir}",
        f"- 依赖 manifest: {dependency_manifest if dependency_manifest else '无/未找到'}",
        f"- 待确认: {notes}",
        "",
        "## Dry-run 命令",
        "",
        "```bash",
        shell_join(dry_command),
        "```",
        "",
        "## 正式运行命令",
        "",
        "```bash",
        f"RUN_MODE=run {shell_join(run_command)}",
        "```",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_blocked_report(
    path: Path,
    config: dict[str, str],
    output_dir: Path,
    reason: str,
    issues: list[str],
    dependency_manifest: str = "",
) -> None:
    lines = [
        f"# 实验报告: {config.get('exp_id', '')}",
        "",
        "## 状态",
        "",
        f"- blocked: {reason}",
        f"- 输出目录: {output_dir}",
        f"- 依赖 manifest: {dependency_manifest if dependency_manifest else '无/未找到'}",
        "",
        "## 原因",
        "",
    ]
    lines.extend(f"- {issue}" for issue in issues)
    write_text(path, "\n".join(lines) + "\n")


def openvla_oft_probe_status_path(config: dict[str, str], output_root: Path) -> Path:
    probe_root = str(config.get("probe_output_root") or "").strip()
    if probe_root:
        return resolve_project_path(PROJECT_ROOT, probe_root) / "probe_openvla_oft_forward" / "status.json"
    return output_root / "probe_openvla_oft_forward" / "status.json"


def load_openvla_oft_probe_status(config: dict[str, str], output_root: Path) -> tuple[Path, dict[str, Any] | None, str]:
    path = openvla_oft_probe_status_path(config, output_root)
    if not path.exists():
        return path, None, ""
    try:
        return path, json.loads(path.read_text(encoding="utf-8")), ""
    except json.JSONDecodeError as exc:
        return path, None, f"OpenVLA-OFT probe status.json 无法解析: {path}: {exc}"


def validate_openvla_oft_probe(config: dict[str, str], output_root: Path, run_mode: str) -> tuple[list[str], str]:
    if config.get("model_name", config.get("model_family", "")) not in {"OpenVLA-OFT", "OpenVLA-OFT_w"}:
        return [], ""

    path, status, parse_error = load_openvla_oft_probe_status(config, output_root)
    if run_mode != "run":
        return [], f"dry-run: OpenVLA-OFT probe status not required; run mode expects {path}"
    if parse_error:
        return [parse_error], ""
    if status is None:
        return ["Please run probe_openvla_oft_forward.py first."], ""

    if parse_bool(config.get("train_patch", "false")):
        if not status.get("supports_pixel_values_grad_probe", False):
            return [PIXEL_GRAD_BLOCK_MESSAGE], f"probe_status={path}"
        if not status.get("dataloader_probe_success", False):
            return [
                "OpenVLA-OFT patch training requires a successful dataloader probe. "
                "Run probe_openvla_oft_forward.py with --use_dataloader --require_dataloader first."
            ], f"probe_status={path}"
        return [], f"probe_status={path}; pixel_values grad probe passed"

    if not status.get("supports_predict_action", False):
        return [PREDICT_ACTION_BLOCK_MESSAGE], f"probe_status={path}"
    return [], f"probe_status={path}; predict_action probe passed"


def resolve_dependency(config: dict[str, str], output_root: Path, run_mode: str) -> tuple[str, str]:
    """返回 dependency_manifest 和说明。"""
    if not parse_bool(config.get("use_pretrained_patch", "false")):
        return "", ""
    explicit_manifest = str(
        config.get("pretrained_patch_manifest")
        or config.get("source_patch_manifest")
        or ""
    ).strip()
    if explicit_manifest:
        manifest_path = resolve_project_path(PROJECT_ROOT, explicit_manifest)
        if manifest_path.exists():
            return str(manifest_path), f"使用显式依赖 patch manifest: {manifest_path}"
        message = f"显式依赖 patch manifest 不存在: {manifest_path}"
        if run_mode == "run":
            raise FileNotFoundError(message)
        return "", f"dry-run: {message}"

    source_output_root = str(
        config.get("pretrained_patch_output_root")
        or config.get("patch_source_output_root")
        or ""
    ).strip()
    dependency_root = (
        resolve_project_path(PROJECT_ROOT, source_output_root)
        if source_output_root
        else output_root
    )
    source_exp_id = config.get("pretrained_patch_source_exp_id", "")
    if source_exp_id and source_exp_id != "none":
        found_by_exp = find_latest_patch(source_exp_id, dependency_root)
        if found_by_exp:
            return found_by_exp["manifest_path"], f"找到依赖 {source_exp_id} patch manifest: {found_by_exp['manifest_path']}"
        if source_exp_id in {"M1", "M2", "M5"}:
            message = official_m_dependency_error(config.get("exp_id", ""), source_exp_id)
            if run_mode == "run":
                raise FileNotFoundError(message)
            return "", f"dry-run: {message}"
        if run_mode == "run":
            raise FileNotFoundError(f"未找到依赖 source_exp_id={source_exp_id} 的 patch manifest")
        return "", f"dry-run: 未找到依赖 source_exp_id={source_exp_id} 的 patch manifest"
    family = config.get("pretrained_patch_family") or config.get("patch_family", "")
    found = find_latest_manifest(
        dependency_root,
        family,
        require_existing_patch=(run_mode == "run"),
    )
    if found:
        manifest_path, _manifest = found
        return str(manifest_path), f"找到依赖 patch manifest: {manifest_path}"
    message = f"未找到依赖 patch_family={family} 的可用 manifest"
    if run_mode == "run":
        raise FileNotFoundError(message)
    return "", message


def main() -> None:
    args = parse_args()
    run_mode = "dry_run" if args.dry_run else args.run_mode
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = resolve_local_paths(read_simple_yaml(config_path), config_path)

    output_root = resolve_project_path(PROJECT_ROOT, config.get("output_root", "outputs/multiview_attack"))
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp, output_dir = unique_output_dir(output_root, config["exp_id"])
    output_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ["images", "videos", "visualizations", "patch_checkpoint"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    status_path = output_dir / "status.json"
    write_status(status_path, "created", "实验目录已创建")

    validation_issues = validate_for_mode(config, run_mode, args.allow_protected)
    notes = config.get("notes", "")
    if validation_issues:
        notes = f"{notes}; validation: {'; '.join(validation_issues)}"
        if run_mode == "run":
            write_text(output_dir / "run.log", "未执行真实实验: " + "; ".join(validation_issues) + "\n")
            write_status(status_path, "blocked", "run 前检查未通过", {"issues": validation_issues})
            write_blocked_report(output_dir / "report.md", config, output_dir, "run 前检查未通过", validation_issues)
            raise SystemExit("run 前检查未通过: " + "; ".join(validation_issues))

    probe_issues, probe_note = validate_openvla_oft_probe(config, output_root, run_mode)
    if probe_note:
        notes = f"{notes}; {probe_note}"
    if probe_issues:
        notes = f"{notes}; probe: {'; '.join(probe_issues)}"
        if run_mode == "run":
            write_text(output_dir / "run.log", "未执行真实实验: " + "; ".join(probe_issues) + "\n")
            write_status(status_path, "blocked", "OpenVLA-OFT probe 检查未通过", {"issues": probe_issues})
            write_blocked_report(output_dir / "report.md", config, output_dir, "OpenVLA-OFT probe 检查未通过", probe_issues)
            raise SystemExit("OpenVLA-OFT probe 检查未通过: " + "; ".join(probe_issues))

    dependency_manifest = ""
    dependency_note = ""
    try:
        dependency_manifest, dependency_note = resolve_dependency(config, output_root, run_mode)
    except FileNotFoundError as exc:
        write_text(output_dir / "run.log", f"未执行真实实验: {exc}\n")
        write_status(status_path, "blocked", str(exc))
        write_blocked_report(output_dir / "report.md", config, output_dir, "依赖 patch manifest 检查未通过", [str(exc)])
        raise SystemExit(str(exc))
    if dependency_note:
        notes = f"{notes}; {dependency_note}"

    dry_command, run_command = build_commands(config, output_dir, args.gpu_id, dependency_manifest)
    output_config = dict(config)
    output_config.update({"timestamp": timestamp, "gpu_id": args.gpu_id, "output_dir": str(output_dir)})
    write_simple_yaml(output_dir / "config.yaml", output_config)
    write_command_script(output_dir / "command.sh", dry_command, run_command, args.gpu_id)

    dry_code = run_and_log(dry_command, output_dir / "dry_run.log", args.gpu_id, status_path, "dry_run")
    if dry_code != 0:
        write_text(output_dir / "run.log", "未执行真实实验: dry-run 失败。\n")
        write_status(status_path, "dry_run_failed", "dry-run 失败，未执行真实实验")
        raise SystemExit(f"dry-run 失败: {output_dir / 'dry_run.log'}")

    run_status = "dry_run_placeholder"
    if run_mode == "run":
        write_status(status_path, "run_started", "dry-run 通过，开始真实运行")
        run_code = run_and_log(run_command, output_dir / "run.log", args.gpu_id, status_path, "run")
        if run_code != 0:
            run_status = "run_failed"
            notes = f"{notes}; 真实运行失败，请查看 run.log"
        else:
            run_status = "success"
    else:
        write_text(output_dir / "run.log", "未执行真实实验: dry-run。\n")
        write_status(status_path, "dry_run_only", "RUN_MODE=dry_run，未执行真实实验")

    patch_path = ""
    if parse_bool(config.get("train_patch", "false")):
        last_patch = output_dir / "last" / "patch.pt"
        patch_path = "last/patch.pt" if last_patch.exists() else "patch_checkpoint/patch.pt"
    elif parse_bool(config.get("use_synthetic_patch", "false")) and (output_dir / "patch_checkpoint" / "patch.pt").exists():
        patch_path = "patch_checkpoint/patch.pt"
    manifest = build_manifest(
        config,
        output_dir,
        status=run_status,
        patch_path=patch_path,
        dependency_manifest=dependency_manifest,
    )
    write_manifest(output_dir / "patch_manifest.json", manifest)

    metrics = metrics_template(config, timestamp, output_dir, args.gpu_id, notes)
    metrics = merge_runner_metrics(metrics, output_dir)
    write_text(output_dir / "metrics.json", json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    for filename, headers in CSV_HEADERS.items():
        write_csv_header(output_dir / filename, headers)
    write_report(output_dir / "report.md", config, output_dir, dry_command, run_command, notes, dependency_manifest)
    append_summary(output_root / "summary_all_experiments.csv", metrics)

    print(f"output_dir={output_dir}")
    print(f"dry_run_log={output_dir / 'dry_run.log'}")
    print(f"report={output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
