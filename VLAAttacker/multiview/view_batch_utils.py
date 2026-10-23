"""多视角实验的轻量工具。

这里不导入 torch/transformers，保证 dry-run 可以在登录节点完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


TODO_PREFIXES = ("TODO", "LOCAL_PATHS", "待确认")


def parse_bool(value: Any) -> bool:
    """把配置中的 true/false 字符串转成 bool。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_todo(value: Any) -> bool:
    """判断字段是否仍是占位值。"""
    text = str(value or "").strip()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in TODO_PREFIXES)


def parse_views(value: Any) -> list[str]:
    """解析 front,wrist / none 这类逗号分隔字段。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = str(value).split(",")
    return [str(item).strip() for item in raw if str(item).strip() and str(item).strip() != "none"]


def parse_csv(value: Any) -> list[str]:
    """兼容旧调用名。"""
    return parse_views(value)


def parse_key_value_csv(value: Any) -> dict[str, str]:
    """解析 front=image_primary,wrist=image_wrist。"""
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
    mapping: dict[str, str] = {}
    for item in parse_views(value):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if key.strip() and val.strip():
            mapping[key.strip()] = val.strip()
    return mapping


def read_simple_yaml(path: Path) -> dict[str, str]:
    """读取本项目生成的简单 key: value YAML。

    只支持一层标量字段；这让脚本不依赖 PyYAML。
    """
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def write_simple_yaml(path: Path, data: dict[str, Any]) -> None:
    """写一层 key: value YAML。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in data.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif value is None:
            value = "none"
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_project_path(project_root: Path, value: str) -> Path:
    """把相对路径解析到项目根目录下。"""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


DEFAULT_VIEW_KEYS = {
    "front": ["front", "primary", "image_primary", "pixel_values", "image"],
    "wrist": ["wrist", "hand", "image_wrist", "pixel_values_wrist"],
}


def _candidate_keys(view_name: str, view_key_map: Any = None) -> list[str]:
    """按配置和默认映射生成候选 key，不假设唯一字段名。"""
    mapping = parse_key_value_csv(view_key_map)
    keys = []
    if view_name in mapping:
        keys.append(mapping[view_name])
    keys.extend(DEFAULT_VIEW_KEYS.get(view_name, [view_name]))
    # 保持顺序去重。
    deduped = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _available_keys(batch: Any) -> list[str]:
    if isinstance(batch, dict):
        keys = sorted(str(key) for key in batch.keys())
        observation = batch.get("observation")
        if isinstance(observation, dict):
            keys.extend(f"observation.{key}" for key in sorted(observation.keys()))
        if isinstance(batch.get("pixel_values_by_view"), dict):
            keys.extend(f"pixel_values_by_view.{key}" for key in sorted(batch["pixel_values_by_view"].keys()))
        return keys
    return []


def inspect_batch_views(batch: Any) -> dict[str, Any]:
    """只读检查 batch 里可能的视角字段。"""
    keys = _available_keys(batch)
    result = {
        "available_keys": keys,
        "front_candidates": _candidate_keys("front"),
        "wrist_candidates": _candidate_keys("wrist"),
        "warnings": [],
    }
    if not keys:
        result["warnings"].append("batch 不是 dict 或没有可检查 key")
    return result


def get_view_tensor(batch: Any, view_name: str, view_key_map: Any = None) -> Any:
    """从 batch 中取某个视角；找不到时返回 warning dict 而不是崩溃。"""
    if not isinstance(batch, dict):
        return {"warning": "batch 不是 dict，无法读取视角", "view": view_name}

    if isinstance(batch.get("pixel_values_by_view"), dict):
        by_view = batch["pixel_values_by_view"]
        for key in _candidate_keys(view_name, view_key_map):
            if key in by_view:
                return by_view[key]
        if view_name in by_view:
            return by_view[view_name]

    observation = batch.get("observation")
    if isinstance(observation, dict):
        for key in _candidate_keys(view_name, view_key_map):
            if key in observation:
                return observation[key]

    for key in _candidate_keys(view_name, view_key_map):
        if key in batch:
            return batch[key]

    # 单视角旧逻辑：front 可以回退到 pixel_values。
    if view_name == "front" and "pixel_values" in batch:
        return batch["pixel_values"]
    return {
        "warning": f"找不到视角 {view_name}，请配置 view_key_map",
        "available_keys": _available_keys(batch),
        "view": view_name,
    }


def set_view_tensor(batch: Any, view_name: str, tensor: Any, view_key_map: Any = None) -> Any:
    """把某视角写回 batch；找不到字段时返回 warning dict。"""
    if not isinstance(batch, dict):
        return {"warning": "batch 不是 dict，无法写入视角", "view": view_name}

    if isinstance(batch.get("pixel_values_by_view"), dict):
        by_view = batch["pixel_values_by_view"]
        for key in _candidate_keys(view_name, view_key_map):
            if key in by_view:
                by_view[key] = tensor
                return batch
        by_view[view_name] = tensor
        return batch

    observation = batch.get("observation")
    if isinstance(observation, dict):
        for key in _candidate_keys(view_name, view_key_map):
            if key in observation:
                observation[key] = tensor
                return batch

    for key in _candidate_keys(view_name, view_key_map):
        if key in batch:
            batch[key] = tensor
            return batch

    if view_name == "front" and "pixel_values" in batch:
        batch["pixel_values"] = tensor
        return batch
    return {
        "warning": f"找不到视角 {view_name}，请配置 view_key_map",
        "available_keys": _available_keys(batch),
        "view": view_name,
    }


def validate_view_sets(config: dict[str, str]) -> list[str]:
    """返回视角配置中的潜在问题；不直接退出。"""
    issues: list[str] = []
    input_views = set(parse_views(config.get("input_views", "")))
    patch_train_views = set(parse_views(config.get("patch_train_views", "")))
    attack_views = set(parse_views(config.get("attack_views", "")))

    if patch_train_views and not patch_train_views.issubset(input_views):
        issues.append("patch_train_views 通常应是 input_views 的子集")
    if attack_views and not attack_views.issubset(input_views):
        issues.append("attack_views 通常应是 input_views 的子集")
    if parse_bool(config.get("train_patch", "false")) and not patch_train_views:
        issues.append("train_patch=true 但 patch_train_views 为空")
    if parse_bool(config.get("use_pretrained_patch", "false")) and not attack_views:
        issues.append("use_pretrained_patch=true 但 attack_views 为空")
    return issues
