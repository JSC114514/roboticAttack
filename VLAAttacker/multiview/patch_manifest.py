"""Patch manifest 读写和依赖查找。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .view_batch_utils import parse_bool, parse_csv


OFFICIAL_M_ATTACK_IMPLEMENTATION = "openvla_oft_pixel_values_action_head_v1"
OFFICIAL_M_ATTACK_LOSS = "oft_clean_reference_hybrid_attack"
OFFICIAL_M_PATCH_SPACE = "pixel_values"
OFFICIAL_M_EOT_CONFIG: dict[str, str] = {
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
    "loss_weight_mode": "fixed",
    "action_divergence_type": "mse",
}
OFFICIAL_M_WEIGHT_SOURCES = {"manual", "uncalibrated_default", "gradient_calibrated_fixed", "sweep_selected"}

OFFICIAL_M_PATCH_SOURCES: dict[str, dict[str, Any]] = {
    "M1": {
        "patch_family": "P_oft_front_mvctx",
        "input_views": ["front", "wrist"],
        "patch_train_views": ["front"],
        "attack_views": ["front"],
        "patch_size": "0.05",
        "patch_position": "top_left",
        **OFFICIAL_M_EOT_CONFIG,
    },
    "M2": {
        "patch_family": "P_oft_wrist_mvctx",
        "input_views": ["front", "wrist"],
        "patch_train_views": ["wrist"],
        "attack_views": ["wrist"],
        "patch_size": "0.05",
        "patch_position": "top_left",
        **OFFICIAL_M_EOT_CONFIG,
    },
    "M5": {
        "patch_family": "P_oft_multiview",
        "input_views": ["front", "wrist"],
        "patch_train_views": ["front", "wrist"],
        "attack_views": ["front"],
        "patch_size": "0.05",
        "patch_position": "top_left",
        **OFFICIAL_M_EOT_CONFIG,
    },
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _same_csv(actual: Any, expected: list[str]) -> bool:
    return parse_csv(actual) == expected


def _is_openvla_oft_patch_config(config: dict[str, str]) -> bool:
    model_name = _text(config.get("model_name", config.get("model_family", "")))
    return model_name == "OpenVLA-OFT" and (
        parse_bool(config.get("train_patch", "false"))
        or parse_bool(config.get("use_pretrained_patch", "false"))
    )


def _default_attack_implementation(config: dict[str, str]) -> str:
    configured = _text(config.get("attack_implementation"))
    if configured:
        return configured
    return OFFICIAL_M_ATTACK_IMPLEMENTATION if _is_openvla_oft_patch_config(config) else ""


def _default_attack_loss(config: dict[str, str]) -> str:
    configured = _text(config.get("attack_loss_type"))
    if configured:
        return configured
    return OFFICIAL_M_ATTACK_LOSS if _is_openvla_oft_patch_config(config) else ""


def _default_train_patch_space(config: dict[str, str]) -> str:
    configured = _text(config.get("train_patch_space"))
    if configured:
        return configured
    if parse_bool(config.get("train_patch", "false")) and _is_openvla_oft_patch_config(config):
        return OFFICIAL_M_PATCH_SPACE
    return ""


def _default_eval_patch_space(config: dict[str, str], train_patch_space: str) -> str:
    configured = _text(config.get("eval_patch_space"))
    if configured:
        return configured
    if _is_openvla_oft_patch_config(config):
        return train_patch_space or OFFICIAL_M_PATCH_SPACE
    return ""


def build_manifest(
    config: dict[str, str],
    output_dir: Path,
    *,
    status: str,
    patch_path: str = "",
    dependency_manifest: str = "",
) -> dict[str, Any]:
    """构造统一 patch_manifest.json。"""
    train_patch = parse_bool(config.get("train_patch", "false"))
    use_pretrained = parse_bool(config.get("use_pretrained_patch", "false"))
    patch_family = config.get("patch_family", "none")
    if use_pretrained and not patch_family:
        patch_family = config.get("pretrained_patch_family", "none")
    train_patch_space = _default_train_patch_space(config)
    eval_patch_space = _default_eval_patch_space(config, train_patch_space)
    train_patch_position = (
        _text(config.get("train_patch_position"))
        or _text(config.get("patch_position"))
        or _text(config.get("eval_patch_position"))
    )
    eval_patch_position = (
        _text(config.get("eval_patch_position"))
        or _text(config.get("patch_position"))
        or train_patch_position
    )
    patch_position = eval_patch_position or train_patch_position

    manifest = {
        "exp_id": config.get("exp_id", ""),
        "experiment_id": config.get("exp_id", ""),
        "patch_family": patch_family,
        "patch_type": config.get("patch_type", "none"),
        "attack_implementation": _default_attack_implementation(config),
        "attack_loss_name": _default_attack_loss(config),
        "patch_space": train_patch_space or eval_patch_space,
        "train_patch_space": train_patch_space,
        "eval_patch_space": eval_patch_space,
        "input_views": parse_csv(config.get("input_views", "")),
        "patch_train_views": parse_csv(config.get("patch_train_views", "")),
        "attack_views": parse_csv(config.get("attack_views", "")),
        "patch_size": config.get("patch_size", ""),
        "patch_position": patch_position,
        "train_patch_position": train_patch_position,
        "eval_patch_position": eval_patch_position,
        "model_checkpoint": config.get("model_checkpoint", ""),
        "manifest_schema_version": config.get(
            "manifest_schema_version",
            "2" if _default_attack_implementation(config) else "1",
        ),
        "config_source": config.get("source_config", ""),
        "patch_path": patch_path,
        "patch_checkpoint_dir": "patch_checkpoint" if train_patch else "",
        "dependency_manifest": dependency_manifest,
        "train_patch": train_patch,
        "use_pretrained_patch": use_pretrained,
        "status": status,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
    }
    for key in [
        "eot_enabled",
        "eot_samples",
        "eot_force_eval_anchor_sample",
        "eot_warmup_fraction",
        "eot_fixed_anchor_probability",
        "eot_local_jitter_probability",
        "eot_global_random_probability",
        "eot_local_jitter_x",
        "eot_local_jitter_y",
        "eot_rotation_degrees",
        "eot_scale_min",
        "eot_scale_max",
        "eot_shear_degrees",
        "eot_brightness_delta",
        "eot_contrast_delta",
        "eval_eot_enabled",
        "multiview_eot_transform_coupling",
        "loss_weight_mode",
        "action_divergence_type",
        "action_boundary_weight",
        "action_divergence_weight",
        "feature_deviation_weight",
        "weight_source",
        "patch_seed",
        "eot_seed",
        "calibration_target_boundary_ratio",
        "calibration_target_divergence_ratio",
        "calibration_target_feature_ratio",
        "calibration_weight_min",
        "calibration_weight_max",
        "calibration_eps",
    ]:
        if key in config and _text(config.get(key)):
            manifest[key] = config.get(key)
    for key in [
        "source_patch_experiment",
        "patch_source_experiment",
        "transfer_type",
        "source_model",
        "target_model",
        "source_view",
        "target_view",
        "patch_retrained",
        "use_synthetic_patch",
        "synthetic_patch_type",
        "synthetic_patch_seed",
        "synthetic_patch_shape",
    ]:
        if key in config and _text(config.get(key)):
            manifest[key] = config.get(key)
    if dependency_manifest:
        manifest["source_patch_manifest"] = dependency_manifest
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """写 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_patch_manifest(
    output_dir: str | Path,
    exp_id: str,
    patch_family: str,
    patch_path: str,
    *,
    patch_type: str = "single_view",
    patch_train_views: Any = None,
    attack_views: Any = None,
    patch_checkpoint_dir: str = "patch_checkpoint",
    status: str = "success",
) -> Path:
    """保存标准 patch manifest，返回 manifest 路径。"""
    output_dir = Path(output_dir)
    manifest = {
        "exp_id": exp_id,
        "patch_family": patch_family,
        "patch_type": patch_type,
        "patch_train_views": parse_csv(patch_train_views),
        "attack_views": parse_csv(attack_views),
        "patch_path": patch_path,
        "patch_checkpoint_dir": patch_checkpoint_dir,
        "status": status,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
    }
    path = output_dir / "patch_manifest.json"
    write_manifest(path, manifest)
    return path


def read_manifest(path: Path) -> dict[str, Any] | None:
    """读取 manifest，失败时返回 None。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_patch_exists(manifest_path: Path, manifest: dict[str, Any]) -> bool:
    """判断 manifest 指向的 patch 是否存在。"""
    patch_path = str(manifest.get("patch_path", "")).strip()
    if not patch_path:
        return False
    path = Path(patch_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.exists()


def compatibility_errors_for_official_m_patch(
    manifest_path: Path,
    manifest: dict[str, Any],
    source_exp_id: str,
) -> list[str]:
    """返回 official M-series patch manifest 的兼容性错误。

    M3/M4/M6/M7 依赖的是当前正式 M1/M2/M5 训练出的 pixel_values/action-head
    patch。旧 M1 raw-image manifest、历史诊断 manifest 或缺少新版字段的历史
    manifest 都不能被静默加载。
    """
    source_exp_id = _text(source_exp_id)
    expected = OFFICIAL_M_PATCH_SOURCES.get(source_exp_id)
    if expected is None:
        return []

    errors: list[str] = []
    if _text(manifest.get("exp_id") or manifest.get("experiment_id")) != source_exp_id:
        errors.append(
            f"exp_id must be {source_exp_id}, got {manifest.get('exp_id') or manifest.get('experiment_id')!r}"
        )
    if _text(manifest.get("attack_implementation")) != OFFICIAL_M_ATTACK_IMPLEMENTATION:
        errors.append(
            "attack_implementation must be "
            f"{OFFICIAL_M_ATTACK_IMPLEMENTATION}, got {manifest.get('attack_implementation')!r}"
        )
    if _text(manifest.get("attack_loss_name")) != OFFICIAL_M_ATTACK_LOSS:
        errors.append(
            f"attack_loss_name must be {OFFICIAL_M_ATTACK_LOSS}, got {manifest.get('attack_loss_name')!r}"
        )
    patch_space = _text(manifest.get("patch_space") or manifest.get("train_patch_space"))
    if patch_space != OFFICIAL_M_PATCH_SPACE:
        errors.append(f"patch_space must be {OFFICIAL_M_PATCH_SPACE}, got {patch_space!r}")
    if _text(manifest.get("train_patch_space")) != OFFICIAL_M_PATCH_SPACE:
        errors.append(
            f"train_patch_space must be {OFFICIAL_M_PATCH_SPACE}, got {manifest.get('train_patch_space')!r}"
        )
    if _text(manifest.get("eval_patch_space")) != OFFICIAL_M_PATCH_SPACE:
        errors.append(
            f"eval_patch_space must be {OFFICIAL_M_PATCH_SPACE}, got {manifest.get('eval_patch_space')!r}"
        )
    for key in ["input_views", "patch_train_views", "attack_views"]:
        if not _same_csv(manifest.get(key), expected[key]):
            errors.append(f"{key} must be {expected[key]}, got {manifest.get(key)!r}")
    for key in ["patch_family", "patch_size", "patch_position"]:
        if _text(manifest.get(key)) != _text(expected[key]):
            errors.append(f"{key} must be {expected[key]!r}, got {manifest.get(key)!r}")
    for key, value in OFFICIAL_M_EOT_CONFIG.items():
        if _text(manifest.get(key)) != _text(value):
            errors.append(f"{key} must be {value!r}, got {manifest.get(key)!r}")
    weight_source = _text(manifest.get("weight_source"))
    if weight_source not in OFFICIAL_M_WEIGHT_SOURCES:
        errors.append(
            f"weight_source must be one of {sorted(OFFICIAL_M_WEIGHT_SOURCES)}, got {manifest.get('weight_source')!r}"
        )
    for key in ["action_boundary_weight", "action_divergence_weight", "feature_deviation_weight"]:
        if not _text(manifest.get(key)):
            errors.append(f"{key} is missing")
    if not _text(manifest.get("model_checkpoint")):
        errors.append("model_checkpoint is missing")
    if not _text(manifest.get("patch_path")):
        errors.append("patch_path is missing")
    if errors:
        return [f"{manifest_path}: {error}" for error in errors]
    return []


def is_compatible_official_m_patch(
    manifest_path: Path,
    manifest: dict[str, Any],
    source_exp_id: str,
) -> bool:
    return not compatibility_errors_for_official_m_patch(manifest_path, manifest, source_exp_id)


def find_latest_manifest(
    output_root: Path,
    patch_family: str,
    *,
    require_existing_patch: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """按 patch_family 查找最新可用 manifest。"""
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for manifest_path in output_root.glob("*/patch_manifest.json"):
        manifest = read_manifest(manifest_path)
        if not manifest:
            continue
        if manifest.get("patch_family") != patch_family:
            continue
        if manifest.get("status") not in {"success", "run_finished", "dry_run_placeholder"}:
            continue
        if require_existing_patch and not manifest_patch_exists(manifest_path, manifest):
            continue
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, manifest_path, manifest))
    if not candidates:
        return None
    _, path, manifest = sorted(candidates, key=lambda item: item[0])[-1]
    return path, manifest


def find_latest_patch(
    source_exp_id: str,
    output_root: str | Path,
    *,
    require_official_m_compatibility: bool = True,
) -> dict[str, Any] | None:
    """按 source_exp_id 查找最新 patch manifest。

    用于 M3<-M1、M4<-M2、M6/M7<-M5 这类依赖。
    """
    output_root = Path(output_root)
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for manifest_path in output_root.glob(f"*_{source_exp_id}/patch_manifest.json"):
        manifest = read_manifest(manifest_path)
        if not manifest:
            continue
        if manifest.get("status") not in {"success", "run_finished"}:
            continue
        if not manifest_patch_exists(manifest_path, manifest):
            continue
        if require_official_m_compatibility and not is_compatible_official_m_patch(
            manifest_path, manifest, source_exp_id
        ):
            continue
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, manifest_path, manifest))
    if not candidates:
        return None
    _, manifest_path, manifest = sorted(candidates, key=lambda item: item[0])[-1]
    manifest = dict(manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def official_m_dependency_error(consumer_exp_id: str, source_exp_id: str) -> str:
    """给 M 系列依赖缺失/不兼容时的明确错误。"""
    if source_exp_id == "M1":
        return (
            f"{consumer_exp_id} requires a patch generated by the current official M1 run, "
            "but no compatible M1 manifest was found."
        )
    if source_exp_id == "M2":
        return (
            f"{consumer_exp_id} requires a patch generated by the current official M2 run, "
            "but no compatible M2 manifest was found."
        )
    if source_exp_id == "M5":
        return (
            f"{consumer_exp_id} requires a patch generated by the current official M5 run, "
            "but no compatible M5 manifest was found."
        )
    return f"未找到 source_exp_id={source_exp_id} 的兼容 patch manifest"


def load_patch_from_manifest(manifest_path: str | Path, *, dry_run: bool = False) -> Any:
    """从 manifest 加载 patch；dry-run 只返回路径信息。"""
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"无法读取 patch manifest: {manifest_path}")
    patch_path = Path(str(manifest.get("patch_path", "")))
    if not patch_path.is_absolute():
        patch_path = manifest_path.parent / patch_path
    if dry_run:
        return {"manifest_path": str(manifest_path), "patch_path": str(patch_path), "exists": patch_path.exists()}
    if not patch_path.exists():
        raise FileNotFoundError(f"manifest 指向的 patch 不存在: {patch_path}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - 环境依赖由运行节点保证
        raise RuntimeError(f"加载 patch 需要 torch: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for patch evaluation; refusing to load patch on CPU. "
            "Check Slurm --gres/CUDA_VISIBLE_DEVICES."
        )
    return torch.load(patch_path, map_location="cuda:0")
