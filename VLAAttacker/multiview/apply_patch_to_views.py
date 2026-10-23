"""按视角贴 patch 的接口层。

真实训练当前由 `white_patch.appply_random_transform.RandomPatchTransform`
中的最小多视角实现承担；本文件把研究概念和阶段选择集中起来，供后续替换为
OpenVLA-OFT 原生多图 forward 时复用。
"""

from __future__ import annotations

from copy import copy
from typing import Any

from .view_batch_utils import get_view_tensor, inspect_batch_views, parse_views, set_view_tensor, validate_view_sets


def select_patch_views(config: dict[str, str], phase: str) -> list[str]:
    """训练阶段用 patch_train_views，评估阶段用 attack_views。"""
    if phase == "train":
        return parse_views(config.get("patch_train_views", ""))
    return parse_views(config.get("attack_views", ""))


def describe_patch_application(config: dict[str, str], phase: str) -> str:
    """生成人类可读的贴 patch 摘要。"""
    return (
        f"phase={phase}; input_views={parse_views(config.get('input_views', ''))}; "
        f"patch_views={select_patch_views(config, phase)}; "
        f"fusion={config.get('multiview_fusion', 'stitch')}"
    )


def validate_patch_application(config: dict[str, str]) -> list[str]:
    """校验 input/train/attack 三组视角的基本关系。"""
    return validate_view_sets(config)


def _default_patch_fn(tensor: Any, patch: Any) -> Any:
    """默认只返回 patch/tensor 二元组，避免在轻量接口里导入 torch。"""
    return {"patched_from": tensor, "patch": patch}


def apply_patch_to_selected_views(
    batch: Any,
    patch: Any,
    attack_views: Any,
    view_key_map: Any = None,
    dry_run: bool = False,
    patch_fn: Any = None,
) -> Any:
    """只对 attack_views 指定的视角贴 patch。

    `patch_fn` 是真正张量贴图函数；未传入时只做结构级替换，供 dry-run/单元检查使用。
    训练主路径仍由 `RandomPatchTransform.apply_patch_to_selected_views` 执行真实 torch 操作。
    """
    views = parse_views(attack_views)
    summary = inspect_batch_views(batch)
    if dry_run:
        print(f"[dry-run] attack_views={views}")
        print(f"[dry-run] available_keys={summary['available_keys']}")
        return batch

    if not views:
        return batch
    if not isinstance(batch, dict):
        return {"warning": "batch 不是 dict，无法贴 patch", "batch": batch}

    patched_batch = copy(batch)
    apply_one = patch_fn or _default_patch_fn
    warnings = []
    for view in views:
        tensor = get_view_tensor(patched_batch, view, view_key_map=view_key_map)
        if isinstance(tensor, dict) and "warning" in tensor:
            warnings.append(tensor["warning"])
            continue
        patched = apply_one(tensor, patch)
        result = set_view_tensor(patched_batch, view, patched, view_key_map=view_key_map)
        if isinstance(result, dict) and "warning" in result:
            warnings.append(result["warning"])
        else:
            patched_batch = result

    if warnings:
        patched_batch.setdefault("multiview_warnings", []).extend(warnings)
    return patched_batch
