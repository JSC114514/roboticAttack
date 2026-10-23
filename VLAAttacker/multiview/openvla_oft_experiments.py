"""
OpenVLA-OFT M 系列实验核心逻辑。

本模块包含两个主要入口函数：
1. run_libero_rollout() — LIBERO 仿真评估（M0/M3/M4/M6/M7 等 clean/patch eval）
2. train_oft_patch()   — 对抗补丁训练（M1/M2/M5 等 train_patch 实验）

设计原则：所有重型依赖（torch, LIBERO, OpenVLA-OFT wrapper）只在运行时函数内部
导入，确保 dry-run 和配置检查保持轻量。
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from VLAAttacker.multiview.patch_manifest import (
    OFFICIAL_M_ATTACK_IMPLEMENTATION,
    OFFICIAL_M_ATTACK_LOSS,
    OFFICIAL_M_PATCH_SPACE,
)
from VLAAttacker.multiview.view_batch_utils import parse_bool, parse_views


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 各 LIBERO 任务套件的最大步数（超过视为失败）
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# OpenVLA 图像预处理的双通道 mean/std
# 通道 0-2（DINOv2 归一化）+ 通道 3-5（SigLIP 归一化）
PATCH_MEAN_STD = (
    ([0.484375, 0.455078125, 0.40625], [0.228515625, 0.2236328125, 0.224609375]),
    ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
)


# ===========================================================================
# 工具函数
# ===========================================================================

def _int_config(config: dict[str, str], key: str, default: int) -> int:
    """从配置字典安全读取整数值。"""
    try:
        return int(str(config.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _float_config(config: dict[str, str], key: str, default: float) -> float:
    """从配置字典安全读取浮点值。"""
    try:
        return float(str(config.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def _parse_patch_size(config: dict[str, str]) -> tuple[int, int, int]:
    """解析补丁尺寸，支持 "C,H,W" 字符串或标量面积比例。"""
    value = config.get("wrapper_patch_size") or config.get("patch_size") or "3,50,50"
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 3:
        try:
            c, h, w = (int(part) for part in parts)
            return c, h, w
        except ValueError:
            pass

    # 回退：标量面积比例 → 正方形补丁（基于 224×224 图像）
    ratio = _float_config(config, "patch_size", 0.05)
    side = max(1, int(round((224 * 224 * max(ratio, 0.0001)) ** 0.5)))
    return 3, side, side


def _mean_or_empty(values: list[float]) -> float | str:
    """返回均值；列表为空时用空字符串保持 CSV 字段兼容。"""
    if not values:
        return ""
    return float(sum(values) / len(values))


def _paired_action_delta_metrics(clean_action: Any, attacked_action: Any) -> dict[str, float | str]:
    """计算同一观测下 clean 与 attacked 动作的偏移。

    这些指标记录同一状态下攻击前后的 action head 输出差异；rollout 控制仍执行
    attacked_action，因此不会把干净动作混进攻击评估。
    """
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
    n = min(clean.size, attacked.size)
    if n == 0:
        return {
            "action_l2": "",
            "position_error": "",
            "rotation_error": "",
            "gripper_error": "",
        }
    delta = attacked[:n] - clean[:n]

    def norm_slice(start: int, end: int) -> float | str:
        if n <= start:
            return ""
        return float(np.linalg.norm(delta[start:min(end, n)]))

    return {
        "action_l2": float(np.linalg.norm(delta)),
        "position_error": norm_slice(0, 3),
        "rotation_error": norm_slice(3, 6),
        "gripper_error": float(abs(delta[6])) if n > 6 else "",
    }


def _ensure_runtime_paths(config: dict[str, str]) -> None:
    """将 openvla-oft、LIBERO 等外部仓库路径加入 sys.path。"""
    for key in ["project_root", "libero_root", "openvla_oft_root"]:
        value = config.get(key, "")
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _official_m_exp_id(config: dict[str, str]) -> str:
    exp_id = str(config.get("exp_id", "")).strip()
    return exp_id if exp_id in {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "A1"} else ""


def _require_official_pixel_values_eval(config: dict[str, str], eval_patch_space: str, patch: Any) -> None:
    if patch is None:
        return
    exp_id = _official_m_exp_id(config)
    model_name = str(config.get("model_name", config.get("model_family", ""))).strip()
    if not exp_id or model_name not in {"OpenVLA-OFT", "OpenVLA-OFT_w"}:
        return
    implementation = str(config.get("attack_implementation") or OFFICIAL_M_ATTACK_IMPLEMENTATION).strip()
    if implementation != OFFICIAL_M_ATTACK_IMPLEMENTATION:
        raise RuntimeError(
            f"{exp_id} official M-series patch eval requires attack_implementation="
            f"{OFFICIAL_M_ATTACK_IMPLEMENTATION}, got {implementation!r}."
        )
    loss_type = str(config.get("attack_loss_type") or OFFICIAL_M_ATTACK_LOSS).strip().lower()
    if loss_type != OFFICIAL_M_ATTACK_LOSS:
        raise RuntimeError(
            f"{exp_id} official M-series patch eval requires attack_loss_type="
            f"{OFFICIAL_M_ATTACK_LOSS}."
        )
    if eval_patch_space != OFFICIAL_M_PATCH_SPACE:
        raise RuntimeError(
            f"{exp_id} official M-series patch eval requires eval_patch_space="
            f"{OFFICIAL_M_PATCH_SPACE}; refusing to fall back to raw_image."
        )


def _require_official_pixel_values_training(config: dict[str, str], train_patch_space: str) -> None:
    exp_id = str(config.get("exp_id", "")).strip()
    if exp_id not in {"M1", "M2", "M5"}:
        return
    model_name = str(config.get("model_name", config.get("model_family", ""))).strip()
    if model_name != "OpenVLA-OFT":
        return
    implementation = str(config.get("attack_implementation") or OFFICIAL_M_ATTACK_IMPLEMENTATION).strip()
    if implementation != OFFICIAL_M_ATTACK_IMPLEMENTATION:
        raise RuntimeError(
            f"{exp_id} official patch training requires attack_implementation="
            f"{OFFICIAL_M_ATTACK_IMPLEMENTATION}, got {implementation!r}."
        )
    loss_type = str(config.get("attack_loss_type") or OFFICIAL_M_ATTACK_LOSS).strip().lower()
    if loss_type != OFFICIAL_M_ATTACK_LOSS:
        raise RuntimeError(
            f"{exp_id} official patch training requires attack_loss_type={OFFICIAL_M_ATTACK_LOSS}."
        )
    if train_patch_space != OFFICIAL_M_PATCH_SPACE:
        raise RuntimeError(
            f"{exp_id} official patch training requires train_patch_space="
            f"{OFFICIAL_M_PATCH_SPACE}; refusing to use the old raw_image path."
        )


def _set_seed(seed: int) -> None:
    """固定随机种子（Python、NumPy、PyTorch）。"""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ===========================================================================
# LIBERO 仿真相关工具
# ===========================================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """四元数 → 轴角表示（3 维旋转向量）。"""
    import numpy as np

    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def get_libero_image(obs: dict[str, Any], key: str) -> np.ndarray:
    """从 LIBERO 观测中提取图像数组（上下翻转以匹配 OpenVLA 的坐标系）。"""
    import numpy as np
    image = np.asarray(obs[key])
    return np.ascontiguousarray(image[::-1, ::-1])


def build_oft_observation(
    obs: dict[str, Any],
    *,
    input_views: list[str] | None = None,
) -> dict[str, Any]:
    """将 LIBERO 原始观测转换为 OpenVLA-OFT 标准格式。

    输出包含:
      - full_image:   前视图（agentview）RGB 图像
      - wrist_image:  腕部相机 RGB 图像（仅 input_views 包含 wrist 时）
      - state:        本体感知向量 [pos(3), axisangle(3), gripper_qpos(1)] = 8 维
    """
    import numpy as np

    views = input_views or ["front", "wrist"]
    observation = {
        "state": np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        ),
    }
    if "front" in views:
        observation["full_image"] = get_libero_image(obs, "agentview_image")
    if "wrist" in views:
        observation["wrist_image"] = get_libero_image(obs, "robot0_eye_in_hand_image")
    return observation


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """夹爪动作归一化：从 [0,1] 映射到 [-1,1]，可选二值化。"""
    import numpy as np
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2 * action[..., -1] - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """翻转夹爪方向（LIBERO 的夹爪控制与 OpenVLA 约定相反）。"""
    import numpy as np
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] *= -1
    return action


def process_libero_action(action: Any) -> np.ndarray:
    """处理 VLA 输出的原始动作 → 可供 LIBERO 环境执行的动作。

    流程: 归一化夹爪 → 二值化 → 翻转方向
    """
    import numpy as np
    processed = normalize_gripper_action(
        np.asarray(action, dtype=np.float32), binarize=True
    )
    return invert_gripper_action(processed)


def get_libero_dummy_action() -> list[float]:
    """LIBERO 预热阶段的空动作（夹爪设为 -1 表示闭合等待）。"""
    return [0, 0, 0, 0, 0, 0, -1]


# ===========================================================================
# 补丁贴图工具（评估用，numpy/PIL 路径，不需要 torch）
# ===========================================================================

def _patch_to_numpy_rgb(patch: Any) -> np.ndarray:
    """将 torch tensor / dict 格式的补丁转换为 HWC uint8 numpy 数组。"""
    import numpy as np

    # 支持 {"patch": tensor, ...} 的字典格式
    if isinstance(patch, dict):
        for key in ["patch", "rgb_patch", "tensor"]:
            if key in patch:
                patch = patch[key]
                break
    # torch tensor → numpy
    try:
        import torch
        if torch.is_tensor(patch):
            patch = patch.detach().cpu().float().clamp(0, 1).numpy()
    except Exception:
        pass

    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"patch must be CHW or HWC, got shape {arr.shape}")
    if arr.shape[0] == 3:       # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.max(initial=0) <= 1.5:  # [0,1] → [0,255]
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _select_patch_position(
    image_h: int, image_w: int, patch_h: int, patch_w: int, *, mode: str,
) -> tuple[int, int]:
    """根据模式选择补丁在图像上的粘贴位置。"""
    max_y = max(0, image_h - patch_h)
    max_x = max(0, image_w - patch_w)
    upa_rfas_positions = {
        "upa_rfas_libero_10": (5, 160),
        "upa_rfas_libero_object": (0, 0),
        "upa_rfas_libero_goal": (15, 158),
        "upa_rfas_libero_spatial": (120, 160),
    }
    if mode in upa_rfas_positions:
        x, y = upa_rfas_positions[mode]
        return min(max(0, x), max_x), min(max(0, y), max_y)
    if mode == "center":
        return max_x // 2, max_y // 2
    if mode == "bottom_right":
        return max_x, max_y
    if mode == "random":
        return random.randint(0, max_x), random.randint(0, max_y)
    return 0, 0  # 默认左上角


def validate_oft_eot_config(config: dict[str, str]) -> None:
    """Validate anchored EOT settings used by M1/M2/M5 pixel_values training."""
    position = str(config.get("train_patch_position") or "").strip().lower()
    eot_enabled = parse_bool(config.get("eot_enabled", "false"))
    if position == "anchored_eot" and not eot_enabled:
        raise ValueError("train_patch_position=anchored_eot requires eot_enabled=true.")
    if eot_enabled and position != "anchored_eot":
        raise ValueError("eot_enabled=true requires train_patch_position=anchored_eot.")
    if not eot_enabled:
        return

    if parse_bool(config.get("eval_eot_enabled", "false")):
        raise ValueError("eval_eot_enabled=true is not allowed for formal M-series rollout.")
    eot_samples = _int_config(config, "eot_samples", 1)
    if eot_samples < 1:
        raise ValueError(f"eot_samples must be >= 1, got {eot_samples}")
    force_eval_anchor = parse_bool(config.get("eot_force_eval_anchor_sample", "false"))
    if force_eval_anchor and eot_samples < 2:
        raise ValueError("eot_force_eval_anchor_sample=true requires eot_samples >= 2.")
    warmup_fraction = _float_config(config, "eot_warmup_fraction", 0.35)
    if not (0.0 <= warmup_fraction <= 1.0):
        raise ValueError(f"eot_warmup_fraction must be in [0,1], got {warmup_fraction}")

    probs = [
        _float_config(config, "eot_fixed_anchor_probability", 0.60),
        _float_config(config, "eot_local_jitter_probability", 0.30),
        _float_config(config, "eot_global_random_probability", 0.10),
    ]
    if any(prob < 0.0 for prob in probs):
        raise ValueError(f"EOT probabilities must be non-negative, got {probs}")
    if abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"EOT probabilities must sum to 1.0 exactly within 1e-6, got sum={sum(probs)}")

    if _int_config(config, "eot_local_jitter_x", 16) < 0 or _int_config(config, "eot_local_jitter_y", 16) < 0:
        raise ValueError("eot_local_jitter_x/y must be non-negative.")
    scale_min = _float_config(config, "eot_scale_min", 0.90)
    scale_max = _float_config(config, "eot_scale_max", 1.10)
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError(f"Invalid EOT scale range: [{scale_min}, {scale_max}]")

    coupling = str(config.get("multiview_eot_transform_coupling") or "shared").strip().lower()
    if coupling != "shared":
        raise NotImplementedError("Only multiview_eot_transform_coupling=shared is implemented.")


def sample_oft_eot_transforms(
    config: dict[str, str],
    *,
    step: int,
    max_steps: int,
    batch_size: int,
    image_h: int,
    image_w: int,
    patch_h: int,
    patch_w: int,
    force_eval_anchor: bool = False,
) -> list[dict[str, float | int | str]]:
    """Sample per-example anchored EOT transforms.

    Warmup uses strict top-left and identity geometry. After warmup, position mode is
    sampled from fixed-anchor / local-jitter / global-random, while rotation/scale/shear
    are sampled from configured ranges.
    """
    validate_oft_eot_config(config)
    max_x = max(0, int(image_w) - int(patch_w))
    max_y = max(0, int(image_h) - int(patch_h))
    warmup_steps = int(math.floor(max(0, max_steps) * _float_config(config, "eot_warmup_fraction", 0.35)))
    in_warmup = int(step) < warmup_steps

    fixed_prob = _float_config(config, "eot_fixed_anchor_probability", 0.60)
    local_prob = _float_config(config, "eot_local_jitter_probability", 0.30)
    jitter_x = min(max_x, _int_config(config, "eot_local_jitter_x", 16))
    jitter_y = min(max_y, _int_config(config, "eot_local_jitter_y", 16))
    rotation_degrees = abs(_float_config(config, "eot_rotation_degrees", 10.0))
    scale_min = _float_config(config, "eot_scale_min", 0.90)
    scale_max = _float_config(config, "eot_scale_max", 1.10)
    shear_degrees = abs(_float_config(config, "eot_shear_degrees", 5.0))

    transforms: list[dict[str, float | int | str]] = []
    for _sample_idx in range(int(batch_size)):
        if force_eval_anchor:
            transforms.append({
                "stage": "eval_anchor",
                "position_mode": "fixed_eval_anchor",
                "x": 0,
                "y": 0,
                "rotation": 0.0,
                "scale": 1.0,
                "shear": 0.0,
            })
            continue

        if in_warmup:
            transforms.append({
                "stage": "warmup",
                "position_mode": "fixed_anchor",
                "x": 0,
                "y": 0,
                "rotation": 0.0,
                "scale": 1.0,
                "shear": 0.0,
            })
            continue

        draw = random.random()
        if draw < fixed_prob:
            mode = "fixed_anchor"
            x, y = 0, 0
        elif draw < fixed_prob + local_prob:
            mode = "local_jitter"
            x = random.randint(0, jitter_x)
            y = random.randint(0, jitter_y)
        else:
            mode = "global_random"
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

        transforms.append({
            "stage": "eot",
            "position_mode": mode,
            "x": int(x),
            "y": int(y),
            "rotation": random.uniform(-rotation_degrees, rotation_degrees),
            "scale": random.uniform(scale_min, scale_max),
            "shear": random.uniform(-shear_degrees, shear_degrees),
        })
    return transforms


def _transform_rgb_patch_with_mask(patch: Any, transform: dict[str, Any]) -> tuple[Any, Any]:
    """Apply differentiable RGB affine transform and return transformed RGB patch + mask."""
    import torch
    import torch.nn.functional as F

    patch_rgb = patch.clamp(0, 1)
    if patch_rgb.ndim != 3 or patch_rgb.shape[0] != 3:
        raise ValueError(f"RGB patch must be [3,H,W], got {tuple(patch_rgb.shape)}")
    device = patch_rgb.device
    dtype = patch_rgb.dtype
    rotation = float(transform.get("rotation", 0.0))
    scale = float(transform.get("scale", 1.0))
    shear = float(transform.get("shear", 0.0))
    if abs(rotation) < 1e-8 and abs(scale - 1.0) < 1e-8 and abs(shear) < 1e-8:
        return patch_rgb, torch.ones((1, patch_rgb.shape[-2], patch_rgb.shape[-1]), device=device, dtype=dtype)

    angle = math.radians(rotation)
    shear_rad = math.radians(shear)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    shear_t = math.tan(shear_rad)
    affine = torch.tensor(
        [
            [scale * cos_a, -scale * sin_a + shear_t, 0.0],
            [scale * sin_a, scale * cos_a, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    theta = torch.linalg.inv(affine)[:2, :].unsqueeze(0).to(dtype=torch.float32)
    image = patch_rgb.unsqueeze(0).float()
    mask = torch.ones((1, 1, patch_rgb.shape[-2], patch_rgb.shape[-1]), device=device, dtype=torch.float32)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    transformed = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    transformed_mask = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return transformed.squeeze(0).to(dtype=dtype), transformed_mask.squeeze(0).clamp(0, 1).to(dtype=dtype)


def apply_rgb_patch_to_image(
    image: np.ndarray, patch: Any, *, position: str = "top_left",
) -> np.ndarray:
    """在单张 RGB 图像上粘贴补丁（numpy 级别，用于 rollout 评估贴图）。"""
    import numpy as np

    patched = np.asarray(image).copy()
    patch_np = _patch_to_numpy_rgb(patch)
    image_h, image_w = patched.shape[:2]
    patch_h, patch_w = patch_np.shape[:2]

    # 补丁比图像大时等比缩放
    if patch_h > image_h or patch_w > image_w:
        from PIL import Image
        scale = min(image_h / patch_h, image_w / patch_w)
        patch_h = max(1, int(patch_h * scale))
        patch_w = max(1, int(patch_w * scale))
        patch_np = np.asarray(Image.fromarray(patch_np).resize((patch_w, patch_h)))

    x, y = _select_patch_position(image_h, image_w, patch_h, patch_w, mode=position)
    patched[y : y + patch_h, x : x + patch_w] = patch_np
    return patched


def apply_patch_to_observation(
    observation: dict[str, Any],
    patch: Any,
    attack_views: list[str],
    *,
    position: str = "top_left",
) -> dict[str, Any]:
    """在观测的指定视角上粘贴补丁（仅评估路径使用）。

    attack_views 包含 "front" 和/或 "wrist"，对应观测中的 full_image 和 wrist_image。
    """
    if patch is None or not attack_views:
        return observation
    out = dict(observation)
    if "front" in attack_views and "full_image" in out:
        out["full_image"] = apply_rgb_patch_to_image(out["full_image"], patch, position=position)
    if "wrist" in attack_views and "wrist_image" in out:
        out["wrist_image"] = apply_rgb_patch_to_image(out["wrist_image"], patch, position=position)
    return out


def resize_oft_observation_images(
    observation: dict[str, Any],
    *,
    input_views: list[str],
    resize_size: int,
) -> dict[str, Any]:
    """Resize OFT RGB views to the policy input size before raw-image patching.

    UPA-RFAS's OpenVLA-OFT eval first resizes front/wrist images to 224x224 and
    then pastes the 50x50 RGB patch.  This helper keeps the local raw_image eval
    on the same coordinate system.
    """
    import numpy as np

    from experiments.robot.openvla_utils import resize_image_for_policy

    out = dict(observation)
    if "front" in input_views and "full_image" in out:
        out["full_image"] = np.asarray(
            resize_image_for_policy(np.asarray(out["full_image"]), int(resize_size)),
            dtype=np.uint8,
        )
    if "wrist" in input_views and "wrist_image" in out:
        out["wrist_image"] = np.asarray(
            resize_image_for_policy(np.asarray(out["wrist_image"]), int(resize_size)),
            dtype=np.uint8,
        )
    return out


# ===========================================================================
# 日志 / 输出工具
# ===========================================================================

def save_video(images: list[np.ndarray], path: Path, fps: int = 30) -> str:
    """将图像序列保存为 MP4 视频。"""
    if not images:
        return ""
    import imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for image in images:
            writer.append_data(np.asarray(image, dtype=np.uint8))
    finally:
        writer.close()
    return str(path)


# 视角到观测键的映射（用于视频录制）
VIDEO_OBS_KEYS = {"front": "full_image", "wrist": "wrist_image"}


def rollout_video_views(config: dict[str, str]) -> list[str]:
    """从配置中解析需要录制视频的视角列表。"""
    raw_views = config.get("video_views") or config.get("input_views") or "front"
    views = [view for view in parse_views(raw_views) if view in VIDEO_OBS_KEYS]
    return views or ["front"]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """写 CSV 文件（自动创建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def format_progress_bar(done: int, total: int, width: int = 28) -> str:
    """格式化进度条字符串 "[#####............]". """
    total = max(1, total)
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ===========================================================================
# 入口 1: LIBERO 仿真 rollout
# ===========================================================================

def run_libero_rollout(
    config: dict[str, str],
    output_dir: Path,
    *,
    patch: Any = None,
    attack_views: list[str] | None = None,
    mode: str,
) -> dict[str, Any]:
    """在 LIBERO 仿真环境中运行 VLA rollout 评估。

    这是 M0（干净基线）和 M3/M4/M6/M7（补丁评估）的核心逻辑。

    执行流程（每个 episode）:
      1. 初始化 LIBERO 环境和任务初始状态
      2. 等待 10 步预热（空动作）
      3. 每 8 步调用一次 VLA 的 predict_action() 获取动作序列
      4. 从动作队列中逐帧取出动作执行
      5. 记录成功/失败、保存视频和动作轨迹

    参数:
      config:       实验配置字典
      output_dir:   输出目录
      patch:        对抗补丁（None 表示干净评估）
      attack_views: 需要贴补丁的视角列表（如 ["front"]）
      mode:         运行模式标识（clean_eval/patch_eval 等）

    返回:
      包含 task_success_rate 等指标的字典
    """
    import numpy as np

    _ensure_runtime_paths(config)
    _set_seed(_int_config(config, "seed", 0))

    # 延迟导入重型依赖
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper

    attack_views = attack_views or []
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"

    # ---- 读取配置 ----
    task_suite_name = config.get("task_suite") or config.get("task_suite_name") or "libero_spatial"
    num_episodes = _int_config(config, "num_episodes", 5)
    eval_tasks = _int_config(config, "eval_tasks", 0)
    eval_episodes_per_task = _int_config(config, "eval_episodes_per_task", 0)
    num_steps_wait = _int_config(config, "num_steps_wait", 10)       # 预热步数
    env_img_res = _int_config(config, "env_img_res", 256)
    num_open_loop_steps = _int_config(config, "num_open_loop_steps", 8)  # 开环执行步数
    max_steps = TASK_MAX_STEPS.get(task_suite_name, 220)
    save_videos = parse_bool(config.get("save_video", "true"))
    video_views = rollout_video_views(config)
    input_views = parse_views(config.get("input_views", "front,wrist")) or ["front", "wrist"]
    patch_position = config.get("eval_patch_position", "top_left")
    if parse_bool(config.get("eval_eot_enabled", "false")):
        raise RuntimeError("eval_eot_enabled=true is not allowed; rollout evaluation must stay deterministic top_left.")
    eval_patch_space = str(
        config.get("eval_patch_space")
        or config.get("train_patch_space")
        or (OFFICIAL_M_PATCH_SPACE if patch is not None else "raw_image")
    ).strip().lower()
    if eval_patch_space not in {"raw_image", "pixel_values"}:
        raise ValueError(
            f"eval_patch_space must be 'raw_image' or 'pixel_values', got {eval_patch_space!r}"
        )
    _require_official_pixel_values_eval(config, eval_patch_space, patch)
    resize_raw_rgb_before_patch = (
        patch is not None
        and eval_patch_space == "raw_image"
        and parse_bool(config.get("eval_raw_rgb_resize_before_patch", "false"))
    )
    raw_rgb_resize_size = _int_config(config, "eval_raw_rgb_resize_size", 224)

    # ---- 加载模型 ----
    wrapper = OpenVLAOFTWrapper(config)
    wrapper.load_model()

    # ---- 获取任务套件 ----
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_limit = min(num_tasks, eval_tasks) if eval_tasks > 0 else num_tasks
    use_per_task_budget = eval_episodes_per_task > 0
    total_episode_limit = (
        task_limit * eval_episodes_per_task if use_per_task_budget else num_episodes
    )

    # ---- 统计变量 ----
    episode_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    all_action_l2: list[float] = []
    all_position_error: list[float] = []
    all_rotation_error: list[float] = []
    all_gripper_error: list[float] = []
    all_first_step_position_error: list[float] = []
    total_successes = 0
    total_episodes = 0
    evaluated_task_ids: set[int] = set()

    # ---- 遍历所有任务和初始状态 ----
    for task_id in range(task_limit):
        if total_episodes >= total_episode_limit:
            break

        # 获取任务描述和 BDDL 文件
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )

        # 创建 LIBERO 离线渲染环境
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=env_img_res,
            camera_widths=env_img_res,
        )
        env.seed(_int_config(config, "seed", 0))
        initial_states = task_suite.get_task_init_states(task_id)
        if use_per_task_budget:
            task_episode_limit = min(len(initial_states), eval_episodes_per_task)
        else:
            task_episode_limit = min(
                len(initial_states), max(0, total_episode_limit - total_episodes)
            )

        try:
            local_episode_idx = 0
            while local_episode_idx < task_episode_limit and total_episodes < total_episode_limit:
                # ---- Episode 初始化 ----
                evaluated_task_ids.add(task_id)
                env.reset()
                obs = env.set_init_state(initial_states[local_episode_idx])

                # 动作队列：VLA 每次预测 num_open_loop_steps 个动作，缓存后逐帧执行
                action_queue: deque[tuple[Any | None, Any | None]] = deque()
                replay_images_by_view: dict[str, list[np.ndarray]] = {
                    view: [] for view in video_views
                }
                episode_action_l2: list[float] = []
                episode_position_error: list[float] = []
                episode_rotation_error: list[float] = []
                episode_gripper_error: list[float] = []
                episode_first_step_position_error: float | str = ""
                success = False
                error = ""
                step_count = 0

                try:
                    # ---- 执行循环 ----
                    for t in range(max_steps + num_steps_wait):
                        # 预热阶段：执行空动作让环境稳定
                        if t < num_steps_wait:
                            obs, _reward, done, _info = env.step(get_libero_dummy_action())
                            continue

                        # 构建 OpenVLA-OFT 格式的观测。patch_eval 时保留 clean 观测，
                        # 用于记录同状态 clean/attacked 动作差异。
                        clean_observation = build_oft_observation(obs, input_views=input_views)
                        if resize_raw_rgb_before_patch:
                            clean_observation = resize_oft_observation_images(
                                clean_observation,
                                input_views=input_views,
                                resize_size=raw_rgb_resize_size,
                            )
                        raw_patched_observation = apply_patch_to_observation(
                            clean_observation, patch, attack_views, position=patch_position,
                        )
                        # 正式 M 系列使用 pixel_values 模式：动作预测先走 clean RGB
                        # 的官方预处理，再在 processor 输出张量上贴同一个 patch。
                        # raw_image 仅保留给非正式诊断配置，M 系列会在上方拒绝它。
                        observation_for_action = (
                            raw_patched_observation
                            if eval_patch_space == "raw_image"
                            else clean_observation
                        )
                        video_observation = raw_patched_observation

                        # 录制视频帧
                        for view, obs_key in VIDEO_OBS_KEYS.items():
                            if view in replay_images_by_view and obs_key in video_observation:
                                replay_images_by_view[view].append(
                                    np.asarray(video_observation[obs_key], dtype=np.uint8)
                                )

                        # 动作队列空了 → 让 VLA 预测下一批动作
                        if not action_queue:
                            if patch is not None:
                                clean_actions = list(wrapper.predict_action(clean_observation, task_description))
                                if eval_patch_space == "pixel_values":
                                    attacked_actions = list(wrapper.predict_action_with_pixel_patch(
                                        clean_observation,
                                        task_description,
                                        patch,
                                        attack_views,
                                        position=patch_position,
                                    ))
                                else:
                                    attacked_actions = list(wrapper.predict_action(
                                        observation_for_action, task_description
                                    ))
                                max_len = max(len(clean_actions), len(attacked_actions))
                                for idx in range(max_len):
                                    clean_raw = clean_actions[idx] if idx < len(clean_actions) else None
                                    attacked_raw = attacked_actions[idx] if idx < len(attacked_actions) else None
                                    action_queue.append((clean_raw, attacked_raw))
                            else:
                                clean_actions = list(wrapper.predict_action(clean_observation, task_description))
                                action_queue.extend((clean_raw, None) for clean_raw in clean_actions)

                        # 从队列取出一个动作执行
                        clean_raw_action, attacked_raw_action = action_queue.popleft()
                        raw_action = attacked_raw_action if attacked_raw_action is not None else clean_raw_action
                        action = process_libero_action(raw_action)
                        obs, _reward, done, _info = env.step(action.tolist())
                        action_delta = _paired_action_delta_metrics(clean_raw_action, attacked_raw_action)
                        for metric_key, target_list, global_list in [
                            ("action_l2", episode_action_l2, all_action_l2),
                            ("position_error", episode_position_error, all_position_error),
                            ("rotation_error", episode_rotation_error, all_rotation_error),
                            ("gripper_error", episode_gripper_error, all_gripper_error),
                        ]:
                            metric_value = action_delta[metric_key]
                            if metric_value != "":
                                target_list.append(float(metric_value))
                                global_list.append(float(metric_value))
                        if episode_first_step_position_error == "" and action_delta["position_error"] != "":
                            episode_first_step_position_error = float(action_delta["position_error"])
                            all_first_step_position_error.append(float(action_delta["position_error"]))

                        # 记录每步动作
                        action_rows.append({
                            "episode_id": total_episodes,
                            "step": step_count,
                            "task_name": task_description,
                            "view": ",".join(attack_views) if attack_views else "clean",
                            "clean_action": np.asarray(clean_raw_action).tolist() if clean_raw_action is not None else "",
                            "attacked_action": np.asarray(attacked_raw_action).tolist() if attacked_raw_action is not None else "",
                            "action_l2": action_delta["action_l2"],
                            "position_error": action_delta["position_error"],
                            "rotation_error": action_delta["rotation_error"],
                            "gripper_error": action_delta["gripper_error"],
                            "notes": f"{mode}; eval_patch_space={eval_patch_space}",
                        })
                        step_count += 1

                        if done:
                            success = True
                            break
                except Exception as exc:
                    error = repr(exc)

                # ---- Episode 结束，记录结果 ----
                if success:
                    total_successes += 1

                # 保存视频
                video_path = ""
                video_paths: dict[str, str] = {}
                if save_videos:
                    safe_task = task_description.lower().replace(" ", "_").replace("/", "_")[:60]
                    if patch is not None:
                        outcome_tag = (
                            "task_succeeded_attack_failed"
                            if success else
                            "task_failed_attack_succeeded"
                        )
                    else:
                        outcome_tag = "task_succeeded_clean" if success else "task_failed_clean"
                    for view in video_views:
                        images = replay_images_by_view.get(view, [])
                        if not images:
                            continue
                        suffix = "" if view == "front" else f"_{view}"
                        path = video_dir / (
                            f"episode_{total_episodes:04d}_{outcome_tag}"
                            f"_{safe_task}{suffix}.mp4"
                        )
                        saved_path = save_video(images, path)
                        if saved_path:
                            video_paths[view] = saved_path
                    video_path = video_paths.get("front") or next(iter(video_paths.values()), "")
                video_note = "|".join(f"{view}:{path}" for view, path in video_paths.items())

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
                    "clean_adv_first_step_position_l2": episode_first_step_position_error,
                    "trajectory_deviation": "",
                    "view_disagreement_mean": "",
                    "latency_ms_mean": "",
                    "notes": (
                        f"mode={mode}; eval_patch_space={eval_patch_space}; "
                        f"task_id={task_id}; local_episode={local_episode_idx}; "
                        f"steps={step_count}; video={video_path}; videos={video_note}; error={error}"
                    ),
                })
                total_episodes += 1
                local_episode_idx += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

    # ---- 汇总指标 ----
    success_rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    metrics = {
        "mode": mode,
        "exp_id": config.get("exp_id", ""),
        "task_suite": task_suite_name,
        "num_episodes": total_episodes,
        "num_episodes_config": num_episodes,
        "eval_tasks": eval_tasks if eval_tasks > 0 else "",
        "eval_episodes_per_task": (
            eval_episodes_per_task if use_per_task_budget else ""
        ),
        "num_tasks_evaluated": len(evaluated_task_ids),
        "num_successes": total_successes,
        "task_success_rate": success_rate,
        "attack_success_rate": (
            (1.0 - success_rate) if patch is not None and total_episodes else None
        ),
        "attack_views": attack_views,
        "patch_used": patch is not None,
        "eval_patch_space": eval_patch_space,
        "patch_position": patch_position,
        "video_views": video_views if save_videos else [],
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

    # ---- 写输出文件 ----
    write_json(output_dir / "rollout_metrics.json", metrics)
    write_csv(
        output_dir / "metrics_per_episode.csv",
        ["episode_id", "task_id", "local_episode_id", "task_name",
         "success", "attack_success", "action_l2_mean",
         "position_error_mean", "rotation_error_mean", "gripper_error_mean",
         "clean_adv_action_l2_mean", "clean_adv_position_l2_mean",
         "clean_adv_rotation_l2_mean", "clean_adv_gripper_abs_mean",
         "clean_adv_first_step_position_l2",
         "trajectory_deviation", "view_disagreement_mean", "latency_ms_mean", "notes"],
        episode_rows,
    )
    write_csv(
        output_dir / "actions.csv",
        ["episode_id", "step", "task_name", "view", "clean_action", "attacked_action",
         "action_l2", "position_error", "rotation_error", "gripper_error", "notes"],
        action_rows,
    )
    return metrics


# ===========================================================================
# 补丁贴图工具（训练用，torch tensor 路径）
# ===========================================================================

def _rgb_patch_to_processor_channels(patch: Any, device: Any, dtype: Any) -> Any:
    """将 RGB 补丁转换为 OpenVLA processor 的 6 通道格式。

    通道 0-2: DINOv2 归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    通道 3-5: SigLIP 归一化 (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    """
    import torch

    if isinstance(patch, dict):
        patch = patch.get("patch", patch.get("rgb_patch", patch))
    if not torch.is_tensor(patch):
        patch = torch.as_tensor(patch)
    patch = patch.to(device=device, dtype=torch.float32)
    # HWC → CHW
    if patch.ndim == 3 and patch.shape[-1] == 3:
        patch = patch.permute(2, 0, 1)
    patch = patch.clamp(0, 1)

    channels = []
    for mean, std in PATCH_MEAN_STD:
        mean_t = torch.tensor(mean, device=device, dtype=torch.float32)[:, None, None]
        std_t = torch.tensor(std, device=device, dtype=torch.float32)[:, None, None]
        channels.append((patch - mean_t) / std_t)
    return torch.cat(channels, dim=0).to(dtype=dtype)


def _rgb_patch_and_mask_to_processor_channels(patch_rgb: Any, mask: Any, device: Any, dtype: Any) -> tuple[Any, Any]:
    """Normalize a transformed RGB patch to processor channels and expand the RGB mask to 6 channels."""
    import torch

    patch_rgb = patch_rgb.to(device=device, dtype=torch.float32).clamp(0, 1)
    mask = mask.to(device=device, dtype=torch.float32).clamp(0, 1)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != 1:
        mask = mask[:1]

    channels = []
    masks = []
    for mean, std in PATCH_MEAN_STD:
        mean_t = torch.tensor(mean, device=device, dtype=torch.float32)[:, None, None]
        std_t = torch.tensor(std, device=device, dtype=torch.float32)[:, None, None]
        channels.append((patch_rgb - mean_t) / std_t)
        masks.append(mask.expand(3, -1, -1))
    return torch.cat(channels, dim=0).to(dtype=dtype), torch.cat(masks, dim=0).to(dtype=dtype)


def apply_patch_to_pixel_values(
    pixel_values: Any,
    patch: Any,
    *,
    input_views: list[str],
    patch_views: list[str],
    position: str = "random",
    eot_transforms: list[dict[str, Any]] | None = None,
) -> Any:
    """在 pixel_values 的指定视角通道上粘贴补丁（训练路径）。

    OpenVLA-OFT 的多视角 pixel_values 布局：
      每个视角占 6 个通道，按 input_views 顺序拼接。
      如 input_views=["front","wrist"] → pixel_values 形状为 [B, 12, H, W]
      通道 [0:6] 是 front，通道 [6:12] 是 wrist。

    参数:
      pixel_values: 模型的像素输入 [B, 6*N_views, H, W]
      patch:        RGB 补丁 tensor [3, patch_h, patch_w]
      input_views:  输入视角列表
      patch_views:  需要贴补丁的视角列表
      position:     粘贴位置模式（"random"/"center"/"top_left"/"anchored_eot"）
      eot_transforms: optional per-sample anchored EOT transform records. When provided,
                      the same transform is reused across all patched views for that sample.

    返回:
      修改后的 pixel_values（含 patch 通道的 clone）
    """
    import torch

    if not patch_views:
        return pixel_values

    patched = pixel_values.clone()
    device = patched.device
    dtype = patched.dtype

    if isinstance(patch, dict):
        patch = patch.get("patch", patch.get("rgb_patch", patch))
    if not torch.is_tensor(patch):
        patch = torch.as_tensor(patch)
    patch = patch.to(device=device, dtype=torch.float32).clamp(0, 1)

    # Non-EOT path: convert the RGB patch directly to processor 6-channel values.
    patch_channels = _rgb_patch_to_processor_channels(patch, device, dtype)
    _, image_h, image_w = patched.shape[-3:]
    patch_h, patch_w = patch_channels.shape[-2:]
    eot_cache: dict[int, tuple[Any, Any, int, int]] = {}

    if eot_transforms is not None and len(eot_transforms) != patched.shape[0]:
        raise ValueError(
            f"eot_transforms length must match batch size {patched.shape[0]}, got {len(eot_transforms)}"
        )

    # 补丁比图像大时缩放
    if patch_h > image_h or patch_w > image_w:
        patch_channels = torch.nn.functional.interpolate(
            patch_channels.unsqueeze(0).float(),
            size=(min(patch_h, image_h), min(patch_w, image_w)),
            mode="bilinear", align_corners=False,
        ).squeeze(0).to(dtype=dtype)
        patch_h, patch_w = patch_channels.shape[-2:]

    def eot_patch_for_sample(sample_idx: int) -> tuple[Any, Any, int, int]:
        if sample_idx in eot_cache:
            return eot_cache[sample_idx]
        assert eot_transforms is not None
        transform = eot_transforms[sample_idx]
        transformed_rgb, transformed_mask = _transform_rgb_patch_with_mask(patch, transform)
        local_h, local_w = transformed_rgb.shape[-2:]
        if local_h > image_h or local_w > image_w:
            import torch.nn.functional as F

            target_h = min(local_h, image_h)
            target_w = min(local_w, image_w)
            transformed_rgb = F.interpolate(
                transformed_rgb.unsqueeze(0).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(dtype=transformed_rgb.dtype)
            transformed_mask = F.interpolate(
                transformed_mask.unsqueeze(0).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).to(dtype=transformed_mask.dtype)
            local_h, local_w = target_h, target_w
        max_x = max(0, image_w - local_w)
        max_y = max(0, image_h - local_h)
        x = min(max(0, int(transform.get("x", 0))), max_x)
        y = min(max(0, int(transform.get("y", 0))), max_y)
        local_channels, local_mask = _rgb_patch_and_mask_to_processor_channels(
            transformed_rgb, transformed_mask, device, dtype
        )
        eot_cache[sample_idx] = (local_channels, local_mask, x, y)
        return eot_cache[sample_idx]

    # 安全检查：确保 channel 数与视角数匹配
    # 如果 dataloader 给出了拼接后的图而非 per-view channels，应直接报错
    expected_channels = 6 * len(input_views)
    if patched.ndim != 4 or patched.shape[1] < expected_channels:
        raise ValueError(
            "OpenVLA-OFT per-view patching expects pixel_values with 6 channels per input view. "
            f"Got shape={tuple(patched.shape)}, input_views={input_views}, "
            f"expected at least {expected_channels} channels. "
            "This usually means the dataloader produced stitched/single-view pixel_values; "
            "do not silently train a front/wrist patch under that condition."
        )

    # 逐个视角粘贴补丁
    for view in patch_views:
        if view not in input_views:
            raise ValueError(f"patch view {view!r} is not in input_views={input_views}")
        # 计算该视角对应的通道区间 [start, end)
        view_idx = input_views.index(view)
        start = view_idx * 6
        end = start + 6
        if end > patched.shape[1]:
            raise ValueError(
                f"Cannot patch view={view}: channel slice {start}:{end} "
                f"exceeds pixel_values shape={tuple(patched.shape)}"
            )
        # 逐样本粘贴；EOT 模式下同一 sample 的变换在所有 patch_views 间共享。
        for sample_idx in range(patched.shape[0]):
            if eot_transforms is not None:
                local_channels, local_mask, x, y = eot_patch_for_sample(sample_idx)
                local_h, local_w = local_channels.shape[-2:]
                target_slice = patched[sample_idx, start:end, y : y + local_h, x : x + local_w]
                patched[sample_idx, start:end, y : y + local_h, x : x + local_w] = (
                    target_slice * (1.0 - local_mask) + local_channels * local_mask
                )
            else:
                x, y = _select_patch_position(
                    image_h, image_w, patch_h, patch_w, mode=position
                )
                patched[sample_idx, start:end, y : y + patch_h, x : x + patch_w] = patch_channels

    return patched


def _normalization_params_from_processor(image_processor: Any) -> list[tuple[list[float], list[float]]]:
    """读取 OpenVLA processor 的多 backbone mean/std；缺失时回退到当前 checkpoint 的 6 通道配置。"""
    params = getattr(image_processor, "tvf_normalize_params", None)
    if params:
        return [(list(item["mean"]), list(item["std"])) for item in params]
    means = getattr(image_processor, "means", None)
    stds = getattr(image_processor, "stds", None)
    if means and stds:
        return [(list(mean), list(std)) for mean, std in zip(means, stds)]
    return [(list(mean), list(std)) for mean, std in PATCH_MEAN_STD]


def _processor_target_size(image_processor: Any, default: int = 224) -> tuple[int, int]:
    """取 processor 输出尺寸；OpenVLA-OFT 当前为 224×224。"""
    resize_params = getattr(image_processor, "tvf_resize_params", None)
    if resize_params:
        size = resize_params[0].get("size", default)
        if isinstance(size, int):
            return int(size), int(size)
        if len(size) == 2:
            return int(size[0]), int(size[1])
    input_sizes = getattr(image_processor, "input_sizes", None)
    if input_sizes:
        return int(input_sizes[0][-2]), int(input_sizes[0][-1])
    return default, default


def _center_crop_resize_tensor(images: Any, *, output_size: tuple[int, int], crop_scale: float = 0.9) -> Any:
    """torch 版官方 center_crop_image：中心裁掉 sqrt(0.9) 面积后 resize 回 224。"""
    import torch
    import torch.nn.functional as F

    if crop_scale >= 0.999:
        return F.interpolate(images, size=output_size, mode="bicubic", align_corners=False, antialias=True)

    bsz = images.shape[0]
    side_scale = float(math.sqrt(crop_scale))
    ys = torch.linspace(-side_scale, side_scale, output_size[0], device=images.device, dtype=images.dtype)
    xs = torch.linspace(-side_scale, side_scale, output_size[1], device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _resize_patch_for_raw_image(
    patch: Any,
    *,
    image_h: int,
    image_w: int,
    eval_image_size: int,
    match_eval_resize: bool,
) -> Any:
    """把训练时贴到 224 raw 图上的 patch 尺寸缩放到等价于 eval 256→224 后的大小。"""
    import torch
    import torch.nn.functional as F

    patch_h, patch_w = patch.shape[-2:]
    if match_eval_resize and eval_image_size > 0:
        target_h = max(1, int(round(patch_h * image_h / float(eval_image_size))))
        target_w = max(1, int(round(patch_w * image_w / float(eval_image_size))))
    else:
        target_h, target_w = patch_h, patch_w
    target_h, target_w = min(target_h, image_h), min(target_w, image_w)
    if (target_h, target_w) == (patch_h, patch_w):
        return patch
    return F.interpolate(
        patch.unsqueeze(0).float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).to(dtype=patch.dtype)


def _apply_patch_to_raw_view_tensor(
    images: Any,
    patch: Any,
    *,
    position: str,
    eval_image_size: int,
    match_eval_resize: bool,
) -> Any:
    """在 raw RGB tensor [B,3,H,W] 上贴 patch；所有像素操作都保持 torch 可微。"""
    if patch is None:
        return images
    patched = images.clone()
    image_h, image_w = int(patched.shape[-2]), int(patched.shape[-1])
    patch_rgb = patch.to(device=patched.device, dtype=patched.dtype).clamp(0, 1)
    patch_rgb = _resize_patch_for_raw_image(
        patch_rgb,
        image_h=image_h,
        image_w=image_w,
        eval_image_size=eval_image_size,
        match_eval_resize=match_eval_resize,
    )
    patch_h, patch_w = int(patch_rgb.shape[-2]), int(patch_rgb.shape[-1])

    for sample_idx in range(patched.shape[0]):
        x, y = _select_patch_position(image_h, image_w, patch_h, patch_w, mode=position)
        patched[sample_idx, :, y : y + patch_h, x : x + patch_w] = patch_rgb
    return patched


def _processor_normalize_tensor(images: Any, image_processor: Any, *, center_crop: bool) -> Any:
    """用 torch 复刻 OpenVLA-OFT 的 prepare_images_for_vla + PrismaticImageProcessor。"""
    import torch
    import torch.nn.functional as F

    target_size = _processor_target_size(image_processor)
    if images.shape[-2:] != target_size:
        images = F.interpolate(images, size=target_size, mode="bicubic", align_corners=False, antialias=True)
    if center_crop:
        # 官方 rollout 会在 processor 前做 0.9 center crop；训练必须走同一条链路。
        images = _center_crop_resize_tensor(images, output_size=target_size, crop_scale=0.9)

    channels = []
    for mean, std in _normalization_params_from_processor(image_processor):
        mean_t = torch.tensor(mean, device=images.device, dtype=images.dtype)[None, :, None, None]
        std_t = torch.tensor(std, device=images.device, dtype=images.dtype)[None, :, None, None]
        channels.append((images - mean_t) / std_t)
    return torch.cat(channels, dim=1)


def build_raw_patched_pixel_values(
    raw_pixel_values_by_view: dict[str, Any],
    patch: Any,
    *,
    input_views: list[str],
    patch_views: list[str],
    image_processor: Any,
    device: Any,
    dtype: Any,
    position: str,
    center_crop: bool,
    eval_image_size: int,
    match_eval_resize: bool,
) -> Any:
    """非正式诊断路径：从 raw RGB 视角图像生成模型 pixel_values。"""
    import torch

    if not raw_pixel_values_by_view:
        raise RuntimeError("train_patch_space=raw_image requires raw_pixel_values_by_view from the dataloader.")

    per_view = []
    for view in input_views:
        if view not in raw_pixel_values_by_view:
            raise KeyError(
                f"raw_pixel_values_by_view missing view={view!r}; available={sorted(raw_pixel_values_by_view)}"
            )
        raw_images = raw_pixel_values_by_view[view].to(device=device, dtype=torch.float32)
        if raw_images.ndim != 4 or raw_images.shape[1] != 3:
            raise ValueError(
                f"raw image tensors must be [B,3,H,W], got view={view} shape={tuple(raw_images.shape)}"
            )
        # 该路径保留给非正式 raw-image 诊断；正式 M1/M2/M5 已在入口处要求
        # train_patch_space=pixel_values，避免静默回到旧攻击路径。
        if view in patch_views:
            raw_images = _apply_patch_to_raw_view_tensor(
                raw_images,
                patch,
                position=position,
                eval_image_size=eval_image_size,
                match_eval_resize=match_eval_resize,
            )
        per_view.append(_processor_normalize_tensor(raw_images, image_processor, center_crop=center_crop))

    return torch.cat(per_view, dim=1).to(dtype=dtype)


def _move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """将 batch 中的 tensor 移到指定 device。"""
    import torch
    moved = {}
    for key in ["input_ids", "attention_mask", "labels", "pixel_values", "actions", "proprio"]:
        value = batch.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device)
    return moved


def _uses_oft_action_head_loss(config: dict[str, str]) -> bool:
    loss_type = str(config.get("attack_loss_type") or OFFICIAL_M_ATTACK_LOSS).strip().lower()
    return loss_type in {
        "oft_clean_reference_hybrid_attack",
        "oft_action_mse_far_boundary",
        "oft_action_far_boundary",
        "oft_uada_far_boundary",
    }


def _get_oft_num_visual_tokens(wrapper: Any) -> int:
    vla = getattr(wrapper.vla, "module", wrapper.vla)
    return int(vla.vision_backbone.get_num_patches()) * int(vla.vision_backbone.get_num_images_in_input())


def _get_oft_num_patches(wrapper: Any, *, use_proprio: bool, use_diffusion: bool) -> int:
    num_patches = _get_oft_num_visual_tokens(wrapper)
    if use_proprio:
        num_patches += 1
    if use_diffusion:
        num_patches += 1
    return num_patches


def _mean_float(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std_float(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean_float(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def summarize_eot_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-EOT-sample records for one optimizer step."""
    if not records:
        return {
            "eot_enabled": 0,
            "eot_sample_count": 0,
            "eot_stage": "",
            "eot_position_mode": "",
            "eot_position_x": "",
            "eot_position_y": "",
            "eot_rotation": "",
            "eot_scale": "",
            "eot_shear": "",
            "eot_total_loss_mean": "",
            "eot_total_loss_std": "",
            "eot_boundary_mean": "",
            "eot_action_divergence_mean": "",
            "eot_feature_deviation_mean": "",
            "eot_sample_records_json": "",
        }
    total_losses = [float(record["total_loss"]) for record in records]
    boundary = [float(record["boundary_loss"]) for record in records]
    divergence = [float(record["action_divergence"]) for record in records]
    feature = [float(record["feature_deviation"]) for record in records]
    xs = [float(record["position_x"]) for record in records]
    ys = [float(record["position_y"]) for record in records]
    rotations = [float(record["rotation"]) for record in records]
    scales = [float(record["scale"]) for record in records]
    shears = [float(record["shear"]) for record in records]
    return {
        "eot_enabled": 1,
        "eot_sample_count": len(records),
        "eot_stage": ",".join(sorted({str(record["stage"]) for record in records})),
        "eot_position_mode": ",".join(sorted({str(record["position_mode"]) for record in records})),
        "eot_position_x": _mean_float(xs),
        "eot_position_y": _mean_float(ys),
        "eot_rotation": _mean_float(rotations),
        "eot_scale": _mean_float(scales),
        "eot_shear": _mean_float(shears),
        "eot_total_loss_mean": _mean_float(total_losses),
        "eot_total_loss_std": _std_float(total_losses),
        "eot_boundary_mean": _mean_float(boundary),
        "eot_action_divergence_mean": _mean_float(divergence),
        "eot_feature_deviation_mean": _mean_float(feature),
        "eot_sample_records_json": json.dumps(records, ensure_ascii=False),
    }


def hybrid_component_grad_diagnostics(patch: Any, components: dict[str, Any]) -> dict[str, float]:
    """Compute non-accumulating component-gradient norms/cosines for diagnostics."""
    import torch

    component_names = {
        "boundary": "boundary_objective",
        "divergence": "divergence_objective",
        "feature": "feature_objective",
    }
    grads: dict[str, Any] = {}
    for short_name, component_key in component_names.items():
        tensor = components.get(component_key)
        if tensor is None or not torch.is_tensor(tensor) or not tensor.requires_grad:
            grads[short_name] = torch.zeros_like(patch).reshape(-1).float()
            continue
        grad = torch.autograd.grad(
            tensor,
            patch,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(patch)
        grads[short_name] = grad.detach().reshape(-1).float()

    def norm(name: str) -> float:
        return float(torch.linalg.vector_norm(grads[name]).item())

    def cosine(a: str, b: str) -> float:
        denom = torch.linalg.vector_norm(grads[a]) * torch.linalg.vector_norm(grads[b])
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(torch.dot(grads[a], grads[b]).div(denom).item())

    return {
        "grad_boundary_l2": norm("boundary"),
        "grad_divergence_l2": norm("divergence"),
        "grad_feature_l2": norm("feature"),
        "cos_boundary_divergence": cosine("boundary", "divergence"),
        "cos_boundary_feature": cosine("boundary", "feature"),
        "cos_divergence_feature": cosine("divergence", "feature"),
    }


# ===========================================================================
# 入口 2: 对抗补丁训练
# ===========================================================================

def train_oft_patch(
    config: dict[str, str], output_dir: Path,
) -> dict[str, Any]:
    """训练对抗补丁（M1/M2/M5 的核心逻辑）。

    训练流程（每步）:
      1. 从 dataloader 取一个 batch
      2. 正式 M 系列在 pixel_values 指定视角通道上粘贴当前补丁
      3. （可选）做一次干净前向传播获取 clean 参考（clean_adv_kl / hybrid 需要）
      4. 攻击前向传播 → compute_attack_loss() → loss.backward()
      5. 优化器更新补丁，clamp 到 [0, 1]
      6. 定期保存 checkpoint 和训练指标

    关键配置字段:
      - attack_loss_type: 攻击损失类型（oft_clean_reference_hybrid_attack / oft_action_mse_far_boundary / uada_far_boundary / clean_adv_kl / model_ce_max）
      - maskidx: 攻击的自由度索引（如 "0,1,2" 只攻击位置）
      - patch_train_views: 在哪些视角上训练补丁
      - input_views: VLA 输入视角列表

    返回:
      包含 final_model_loss、final_attack_loss 等指标的字典
    """
    _ensure_runtime_paths(config)
    _set_seed(_int_config(config, "seed", 0))

    import torch
    import transformers
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper
    from VLAAttacker.multiview.openvla_oft_attack_losses import (
        compute_attack_loss,
        oft_clean_reference_hybrid_attack_components,
        parse_maskidx,
    )
    from VLAAttacker.white_patch.openvla_dataloader import get_dataloader

    # ---- 准备工作目录 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "last").mkdir(parents=True, exist_ok=True)
    (output_dir / "patch_checkpoint").mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    wrapper = OpenVLAOFTWrapper(config)
    wrapper.load_model()
    device = wrapper.device
    dtype = wrapper.dtype
    train_patch_space = str(config.get("train_patch_space") or "pixel_values").strip().lower()
    if train_patch_space not in {"pixel_values", "raw_image"}:
        raise ValueError(
            f"train_patch_space must be 'pixel_values' or 'raw_image', got {train_patch_space!r}"
        )
    _require_official_pixel_values_training(config, train_patch_space)
    use_raw_patch_training = train_patch_space == "raw_image"
    if use_raw_patch_training and getattr(wrapper.processor, "image_processor", None) is None:
        raise RuntimeError("train_patch_space=raw_image requires wrapper.processor.image_processor.")

    # ---- 构建 dataloader ----
    train_loader, _val_loader = get_dataloader(
        batch_size=_int_config(config, "batch_size", 8),
        dataset=config.get("task_suite", "libero_spatial"),
        server=config.get("data_root") or config.get("project_root", "."),
        vla_path=config.get("model_checkpoint", ""),
        input_views=config.get("input_views", "front,wrist"),
        view_key_map=config.get("view_key_map", ""),
        multiview_fusion=config.get("multiview_fusion", "stitch"),
        use_proprio=parse_bool(config.get("use_proprio", "false")),
        num_images_in_input=_int_config(
            config, "num_images_in_input",
            len(parse_views(config.get("input_views", "front,wrist"))),
        ),
        action_chunk_len=_int_config(config, "num_open_loop_steps", 8),
        # The stock multiview transform keeps PIL images in pixel_values. The
        # PatchAware path applies the same processor transform per requested view
        # and collates channel-stacked tensors, which official M pixel-space
        # training and anchored EOT require.
        return_raw_images=True,
        multiview_debug=parse_bool(config.get("multiview_debug", "false")),
    )

    # ---- 初始化补丁和优化器 ----
    c, h, w = _parse_patch_size(config)
    if c != 3:
        raise ValueError(f"Only RGB patches are supported for OpenVLA-OFT, got channels={c}")
    # 随机初始化补丁 ∈ [0, 1]，设 requires_grad=True
    patch_seed = _int_config(config, "patch_seed", _int_config(config, "seed", 0))
    try:
        patch_generator = torch.Generator(device=device)
    except TypeError:
        patch_generator = torch.Generator()
    patch_generator.manual_seed(patch_seed)
    patch = torch.rand(
        (3, h, w),
        device=device,
        dtype=torch.float32,
        generator=patch_generator,
        requires_grad=True,
    )
    optimizer = transformers.AdamW([patch], lr=_float_config(config, "learning_rate", 0.001))

    # ---- 训练超参 ----
    max_iters = _int_config(config, "max_train_iters", 2000)
    save_every = max(1, _int_config(config, "save_every", 100))
    progress_every = max(1, _int_config(config, "progress_every", 10))
    input_views = parse_views(config.get("input_views", "front,wrist"))
    patch_views = parse_views(config.get("patch_train_views", "front"))
    # 训练位置和 rollout 评估位置分开配置；当前 front 强化实验用 random 训练，
    # 评估仍固定 top_left，以检验位置鲁棒训练后的单点攻击效果。
    patch_position = (
        config.get("train_patch_position")
        or config.get("patch_position")
        or config.get("eval_patch_position")
        or "top_left"
    )
    eval_patch_position = (
        config.get("eval_patch_position")
        or config.get("patch_position")
        or patch_position
    )
    attack_loss_type = str(config.get("attack_loss_type") or OFFICIAL_M_ATTACK_LOSS).strip().lower()
    use_clean_reference_hybrid = attack_loss_type == "oft_clean_reference_hybrid_attack"
    if use_clean_reference_hybrid and not parse_bool(config.get("clean_forward_enabled", "true")):
        raise RuntimeError("oft_clean_reference_hybrid_attack requires clean_forward_enabled=true.")
    eot_enabled = parse_bool(config.get("eot_enabled", "false"))
    if eot_enabled and not use_clean_reference_hybrid:
        raise RuntimeError("anchored EOT is implemented for oft_clean_reference_hybrid_attack only.")
    if eot_enabled or patch_position == "anchored_eot":
        validate_oft_eot_config(config)
        random.seed(_int_config(config, "eot_seed", _int_config(config, "seed", 0)))
    if parse_bool(config.get("eval_eot_enabled", "false")):
        raise RuntimeError("eval_eot_enabled=true is not allowed during train_patch_eval.")
    gradient_conflict_mode = str(config.get("gradient_conflict_mode") or "none").strip().lower()
    if gradient_conflict_mode == "pcgrad":
        raise NotImplementedError("gradient_conflict_mode=pcgrad is reserved but not enabled.")
    if gradient_conflict_mode != "none":
        raise ValueError(f"Unsupported gradient_conflict_mode={gradient_conflict_mode!r}")
    use_oft_action_loss = _uses_oft_action_head_loss(config)
    requested_proprio = parse_bool(config.get("use_proprio", "false"))
    use_proprio = requested_proprio and wrapper.proprio_projector is not None
    if requested_proprio and wrapper.proprio_projector is None:
        raise RuntimeError("use_proprio=true but OpenVLA-OFT wrapper did not load proprio_projector.")
    use_diffusion = parse_bool(config.get("use_diffusion", "false"))
    use_film = parse_bool(config.get("use_film", "false"))
    patch_grad_clip_norm = _float_config(config, "patch_grad_clip_norm", 0.0)
    raw_train_center_crop = parse_bool(config.get("center_crop", "true"))
    raw_train_match_eval_resize = parse_bool(config.get("raw_train_match_eval_resize", "true"))
    raw_train_eval_image_size = _int_config(
        config,
        "raw_train_eval_image_size",
        _int_config(config, "env_img_res", 256),
    )
    num_patches = _get_oft_num_patches(
        wrapper,
        use_proprio=use_proprio,
        use_diffusion=use_diffusion,
    )
    num_visual_tokens = _get_oft_num_visual_tokens(wrapper)
    train_rows: list[dict[str, Any]] = []
    data_iter = iter(train_loader)

    def patch_grad_stats() -> dict[str, float]:
        if patch.grad is None:
            return {
                "patch_grad_abs_mean": 0.0,
                "patch_grad_l2": 0.0,
                "patch_grad_abs_max": 0.0,
                "patch_grad_nonzero_frac": 0.0,
            }
        grad = patch.grad.detach().float()
        return {
            "patch_grad_abs_mean": float(grad.abs().mean().item()),
            "patch_grad_l2": float(torch.linalg.vector_norm(grad.reshape(-1)).item()),
            "patch_grad_abs_max": float(grad.abs().max().item()),
            "patch_grad_nonzero_frac": float((grad.abs() > 0).float().mean().item()),
        }

    def patch_pixel_stats() -> dict[str, float]:
        values = patch.detach().float()
        eps = 1e-6
        return {
            "patch_min": float(values.min().item()),
            "patch_max": float(values.max().item()),
            "patch_mean": float(values.mean().item()),
            "patch_std": float(values.std(unbiased=False).item()),
            "patch_clamp_low_frac": float((values <= eps).float().mean().item()),
            "patch_clamp_high_frac": float((values >= 1.0 - eps).float().mean().item()),
        }

    def vla_forward(
        inputs: dict[str, Any],
        *,
        output_hidden_states: bool = False,
        output_projector_features: bool = False,
    ) -> Any:
        forward_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"].to(dtype=dtype),
            "labels": inputs["labels"],
            "return_dict": True,
        }
        if output_hidden_states:
            forward_kwargs["output_hidden_states"] = True
        if output_projector_features:
            forward_kwargs["output_projector_features"] = True
        if use_proprio:
            if "proprio" not in inputs:
                raise RuntimeError(
                    "use_proprio=true but batch has no proprio tensor. "
                    "This breaks OFT num_patches alignment and weakens/misplaces the attack gradient."
                )
            proprio = inputs["proprio"]
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim > 2:
                proprio = proprio.reshape(proprio.shape[0], -1)
            forward_kwargs["proprio"] = proprio.to(dtype=dtype)
            forward_kwargs["proprio_projector"] = wrapper.proprio_projector
        if use_film:
            forward_kwargs["use_film"] = True
        if use_diffusion:
            raise NotImplementedError("Patch training for OpenVLA-OFT diffusion action head is not implemented.")
        try:
            return wrapper.vla(**forward_kwargs)
        except TypeError as exc:
            if output_projector_features and "output_projector_features" in str(exc):
                forward_kwargs.pop("output_projector_features", None)
                return wrapper.vla(**forward_kwargs)
            raise

    # ---- 训练主循环 ----
    for step in range(max_iters):
        # 取 batch（无限循环 dataloader）
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # 将 batch 移到 GPU；clean forward 保持未贴补丁，adv forward 负责贴图。
        moved = _move_batch_to_device(batch, device)
        if use_oft_action_loss and "actions" not in moved:
            raise RuntimeError(
                "OpenVLA-OFT action-head attack loss requires batch['actions']. "
                "Check RLDSBatchTransform/collator: the continuous action chunk must be preserved."
            )
        if use_proprio and "proprio" not in moved:
            raise RuntimeError(
                "use_proprio=true but dataloader did not return proprio. "
                "This would make num_patches differ from the OFT rollout path."
            )
        if eot_enabled and use_raw_patch_training:
            raise NotImplementedError("anchored EOT is implemented only for train_patch_space=pixel_values.")

        # 梯度清零
        optimizer.zero_grad(set_to_none=True)
        wrapper.vla.zero_grad(set_to_none=True)

        # ---- 可选：干净前向传播（clean_adv_kl / hybrid 需要） ----
        clean_outputs = None
        if attack_loss_type == "clean_adv_kl" or use_clean_reference_hybrid:
            # 用未贴补丁的同一 batch 做干净前向；clean 分支只作为 detached reference。
            clean_moved = _move_batch_to_device(batch, device)
            if use_raw_patch_training:
                clean_moved["pixel_values"] = build_raw_patched_pixel_values(
                    batch.get("raw_pixel_values_by_view", {}),
                    None,
                    input_views=input_views,
                    patch_views=[],
                    image_processor=wrapper.processor.image_processor,
                    device=device,
                    dtype=dtype,
                    position=patch_position,
                    center_crop=raw_train_center_crop,
                    eval_image_size=raw_train_eval_image_size,
                    match_eval_resize=raw_train_match_eval_resize,
                )
            else:
                clean_moved["pixel_values"] = clean_moved["pixel_values"].to(dtype=dtype)
            with torch.no_grad():
                clean_outputs = wrapper._with_cudnn_fallback(
                    "train_oft_patch clean forward",
                    lambda: vla_forward(
                        clean_moved,
                        output_hidden_states=use_clean_reference_hybrid,
                        output_projector_features=use_clean_reference_hybrid,
                    ),
                )

        def aggregate_loss_info(info_rows: list[dict[str, Any]]) -> dict[str, Any]:
            if not info_rows:
                return {}
            keys = sorted({key for row_info in info_rows for key in row_info})
            out: dict[str, Any] = {}
            for key in keys:
                values = [row_info[key] for row_info in info_rows if key in row_info]
                numeric = [float(value) for value in values if isinstance(value, (int, float))]
                if numeric and len(numeric) == len(values):
                    out[key] = _mean_float(numeric)
                else:
                    out[key] = values[0]
            return out

        def compute_hybrid_components(outputs: Any, adv_inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            maskidx = parse_maskidx(config.get("maskidx"), default=(0, 1, 2, 3, 4, 5, 6))
            return oft_clean_reference_hybrid_attack_components(
                outputs=outputs,
                clean_outputs=clean_outputs,
                labels=adv_inputs["labels"],
                actions=adv_inputs["actions"],
                action_head=wrapper.action_head,
                num_patches=num_patches,
                maskidx=maskidx,
                config=config,
                num_visual_tokens=num_visual_tokens,
            )

        model_loss_values: list[Any] = []
        eot_records: list[dict[str, Any]] = []
        component_for_grad: dict[str, Any] | None = None
        logged_pixel_values = moved.get("pixel_values")

        if eot_enabled:
            eot_samples = _int_config(config, "eot_samples", 1)
            losses = []
            info_rows: list[dict[str, Any]] = []
            component_rows: list[dict[str, Any]] = []
            clean_pixel_values = moved["pixel_values"].to(dtype=dtype)
            _, image_h, image_w = clean_pixel_values.shape[-3:]
            patch_h, patch_w = int(patch.shape[-2]), int(patch.shape[-1])
            for eot_idx in range(eot_samples):
                transforms = sample_oft_eot_transforms(
                    config,
                    step=step,
                    max_steps=max_iters,
                    batch_size=int(clean_pixel_values.shape[0]),
                    image_h=int(image_h),
                    image_w=int(image_w),
                    patch_h=patch_h,
                    patch_w=patch_w,
                    force_eval_anchor=(
                        parse_bool(config.get("eot_force_eval_anchor_sample", "false")) and eot_idx == 0
                    ),
                )
                adv_moved = dict(moved)
                adv_moved["pixel_values"] = apply_patch_to_pixel_values(
                    clean_pixel_values,
                    patch,
                    input_views=input_views,
                    patch_views=patch_views,
                    position=patch_position,
                    eot_transforms=transforms,
                )
                logged_pixel_values = adv_moved["pixel_values"]
                outputs = wrapper._with_cudnn_fallback(
                    f"train_oft_patch forward eot_sample={eot_idx}",
                    lambda adv_inputs=adv_moved: vla_forward(
                        adv_inputs,
                        output_hidden_states=use_oft_action_loss,
                        output_projector_features=use_clean_reference_hybrid,
                    ),
                )
                model_loss = getattr(outputs, "loss", None)
                if model_loss is not None:
                    model_loss_values.append(model_loss.detach().float())
                components, sample_info = compute_hybrid_components(outputs, adv_moved)
                losses.append(components["total"])
                component_rows.append(components)
                info_rows.append(sample_info)

                numeric_transforms = transforms
                eot_records.append({
                    "eot_sample_index": eot_idx,
                    "stage": ",".join(sorted({str(item["stage"]) for item in numeric_transforms})),
                    "position_mode": ",".join(sorted({str(item["position_mode"]) for item in numeric_transforms})),
                    "position_x": _mean_float([float(item["x"]) for item in numeric_transforms]),
                    "position_y": _mean_float([float(item["y"]) for item in numeric_transforms]),
                    "rotation": _mean_float([float(item["rotation"]) for item in numeric_transforms]),
                    "scale": _mean_float([float(item["scale"]) for item in numeric_transforms]),
                    "shear": _mean_float([float(item["shear"]) for item in numeric_transforms]),
                    "total_loss": float(components["total"].detach().item()),
                    "boundary_loss": float(components["action_boundary_mse"].detach().item()),
                    "action_divergence": float(components["action_divergence_score"].detach().item()),
                    "feature_deviation": float(components["feature_deviation_score"].detach().item()),
                })

            attack_loss = torch.stack(losses).mean()
            loss_info = aggregate_loss_info(info_rows)
            loss_info.update(summarize_eot_records(eot_records))
            component_for_grad = {
                key: torch.stack([component[key] for component in component_rows]).mean()
                for key in component_rows[0]
            }
            loss_info["eot_attack_forward_count"] = float(eot_samples)
            loss_info["eot_force_eval_anchor_sample"] = int(
                parse_bool(config.get("eot_force_eval_anchor_sample", "false"))
            )
        else:
            if use_raw_patch_training:
                moved["pixel_values"] = build_raw_patched_pixel_values(
                    batch.get("raw_pixel_values_by_view", {}),
                    patch,
                    input_views=input_views,
                    patch_views=patch_views,
                    image_processor=wrapper.processor.image_processor,
                    device=device,
                    dtype=dtype,
                    position=patch_position,
                    center_crop=raw_train_center_crop,
                    eval_image_size=raw_train_eval_image_size,
                    match_eval_resize=raw_train_match_eval_resize,
                )
            else:
                moved["pixel_values"] = apply_patch_to_pixel_values(
                    moved["pixel_values"].to(dtype=dtype),
                    patch,
                    input_views=input_views,
                    patch_views=patch_views,
                    position=patch_position,
                )
            logged_pixel_values = moved["pixel_values"]
            outputs = wrapper._with_cudnn_fallback(
                "train_oft_patch forward",
                lambda: vla_forward(
                    moved,
                    output_hidden_states=use_oft_action_loss,
                    output_projector_features=use_clean_reference_hybrid,
                ),
            )
            model_loss = getattr(outputs, "loss", None)
            if model_loss is not None:
                model_loss_values.append(model_loss.detach().float())
            if use_clean_reference_hybrid:
                component_for_grad, loss_info = compute_hybrid_components(outputs, moved)
                attack_loss = component_for_grad["total"]
                loss_info.update(summarize_eot_records([]))
            else:
                attack_loss, loss_info = compute_attack_loss(
                    outputs=outputs,
                    labels=moved["labels"],
                    config=config,
                    clean_outputs=clean_outputs,
                    actions=moved.get("actions"),
                    action_head=wrapper.action_head,
                    num_patches=num_patches,
                    num_visual_tokens=num_visual_tokens,
                )
                loss_info.update(summarize_eot_records([]))

        model_loss = torch.stack(model_loss_values).mean() if model_loss_values else None
        component_grad_info = (
            hybrid_component_grad_diagnostics(patch, component_for_grad)
            if component_for_grad is not None else
            {
                "grad_boundary_l2": 0.0,
                "grad_divergence_l2": 0.0,
                "grad_feature_l2": 0.0,
                "cos_boundary_divergence": 0.0,
                "cos_boundary_feature": 0.0,
                "cos_divergence_feature": 0.0,
            }
        )

        # ---- 反向传播更新补丁 ----
        attack_loss.backward()
        grad_info = patch_grad_stats()
        grad_info["patch_grad_preclip_l2"] = grad_info["patch_grad_l2"]
        if patch_grad_clip_norm > 0 and patch.grad is not None:
            # 默认不裁剪；保留这个开关用于后续验证强梯度/稳定性方案。
            torch.nn.utils.clip_grad_norm_([patch], max_norm=patch_grad_clip_norm)
            grad_info.update(patch_grad_stats())
        patch_before_step = patch.detach().clone()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)  # 保持补丁像素值在合法范围
            patch_update_l2 = float(torch.linalg.vector_norm((patch - patch_before_step).reshape(-1)).item())

        # ---- 记录训练日志 ----
        row = {
            "step": step,
            "model_loss": float(model_loss.detach().float().item()) if model_loss is not None else "",
            "attack_loss": float(attack_loss.detach().float().item()),
            "total_attack_loss": float(attack_loss.detach().float().item()),
            "patch_views": ",".join(patch_views),
            "patch_position": patch_position,
            "train_patch_position": patch_position,
            "eval_patch_position": eval_patch_position,
            "patch_update_l2": patch_update_l2,
            "patch_grad_clip_norm": patch_grad_clip_norm,
            "proprio_in_batch": int("proprio" in moved),
            "actions_shape": "x".join(str(dim) for dim in tuple(moved["actions"].shape)) if "actions" in moved else "",
            "proprio_shape": "x".join(str(dim) for dim in tuple(moved["proprio"].shape)) if "proprio" in moved else "",
            "train_patch_space": train_patch_space,
            "raw_train_match_eval_resize": int(raw_train_match_eval_resize) if use_raw_patch_training else "",
            "raw_train_eval_image_size": raw_train_eval_image_size if use_raw_patch_training else "",
            "pixel_values_shape": "x".join(str(dim) for dim in tuple(logged_pixel_values.shape)),
            "patch_seed": patch_seed,
            "eot_seed": _int_config(config, "eot_seed", _int_config(config, "seed", 0)),
            "eot_force_eval_anchor_sample": int(parse_bool(config.get("eot_force_eval_anchor_sample", "false"))),
            "weight_source": config.get("weight_source", "manual"),
        }
        row.update(grad_info)
        row.update(component_grad_info)
        row.update(patch_pixel_stats())
        row.update(loss_info)  # 合并损失函数返回的额外信息
        train_rows.append(row)

        completed_steps = step + 1

        # ---- 进度打印 ----
        if step % progress_every == 0 or step == max_iters - 1:
            percent = 100.0 * completed_steps / float(max_iters)
            progress_payload = dict(row)
            progress_payload.update({
                "exp_id": config.get("exp_id", ""),
                "completed_steps": completed_steps,
                "total_steps": max_iters,
                "percent": percent,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            write_json(output_dir / "train_progress.json", progress_payload)
            bar = format_progress_bar(completed_steps, max_iters)
            model_loss_text = (
                f"{row['model_loss']:.4f}" if isinstance(row["model_loss"], float) else str(row["model_loss"])
            )
            print(
                "[openvla-oft-progress] "
                f"{config.get('exp_id', '')} {bar} {completed_steps}/{max_iters} "
                f"({percent:.1f}%) loss={model_loss_text} "
                f"grad_mean={row['patch_grad_abs_mean']:.6f} "
                f"grad_l2={row['patch_grad_l2']:.6f} "
                f"space={row['train_patch_space']} views={row['patch_views']}",
                flush=True,
            )

        # ---- 定期保存 checkpoint ----
        if step % save_every == 0 or step == max_iters - 1:
            torch.save(patch.detach().cpu(), output_dir / "last" / "patch.pt")
            torch.save(patch.detach().cpu(), output_dir / "patch_checkpoint" / "patch.pt")
            write_json(output_dir / "openvla_oft_train_status.json", row)
            print(f"[openvla-oft-train] {row}")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- 写最终训练指标 ----
    train_fieldnames = [
        "step", "model_loss", "attack_loss", "total_attack_loss", "patch_grad_abs_mean", "patch_grad_l2",
        "patch_grad_abs_max", "patch_grad_nonzero_frac", "patch_grad_preclip_l2",
        "patch_update_l2", "patch_min", "patch_max", "patch_mean", "patch_std",
        "patch_clamp_low_frac", "patch_clamp_high_frac", "patch_grad_clip_norm",
        "patch_views", "patch_position", "train_patch_position", "eval_patch_position",
        "proprio_in_batch", "actions_shape", "proprio_shape",
        "train_patch_space", "raw_train_match_eval_resize", "raw_train_eval_image_size", "pixel_values_shape",
        "patch_seed", "eot_seed", "eot_force_eval_anchor_sample", "attack_loss_type", "maskidx",
        "eot_enabled", "eot_sample_count", "eot_attack_forward_count", "eot_stage", "eot_position_mode",
        "eot_position_x", "eot_position_y", "eot_rotation", "eot_scale", "eot_shear",
        "eot_total_loss_mean", "eot_total_loss_std", "eot_boundary_mean",
        "eot_action_divergence_mean", "eot_feature_deviation_mean", "eot_sample_records_json",
        "loss_uada", "selected_action_tokens", "token_distance_mean",
        "loss_oft_action_mse_far_boundary", "selected_action_values", "pred_action_abs_mean",
        "gt_action_abs_mean", "selected_action_abs_error_mean", "action_l2_mean", "target_l2_mean",
        "loss_oft_clean_reference_hybrid_attack", "action_boundary_mse", "action_divergence_score",
        "feature_deviation_score", "feature_raw_mse", "feature_cosine_distance_mean",
        "feature_clean_norm_mean", "feature_adv_norm_mean", "feature_token_l2_mean",
        "clean_action_abs_mean", "adv_action_abs_mean", "clean_adv_action_l2_mean",
        "adv_gt_action_l2_mean", "adv_target_action_l2_mean", "clean_target_action_l2_mean",
        "clean_action_deadzone_fallback_frac", "action_boundary_weight", "action_divergence_weight",
        "feature_deviation_weight", "clean_action_deadzone", "action_boundary_value", "time_decay",
        "loss_weight_mode", "action_divergence_type", "weight_source",
        "grad_boundary_l2", "grad_divergence_l2", "grad_feature_l2",
        "cos_boundary_divergence", "cos_boundary_feature", "cos_divergence_feature",
        "feature_source_projector_features", "action_head_total_values", "action_chunk_len",
        "action_dim", "selected_hidden_count",
        "loss_clean_adv_kl", "kl",
    ]
    write_csv(output_dir / "train_metrics.csv", train_fieldnames, train_rows)

    metrics = {
        "mode": "train_patch",
        "train_steps": max_iters,
        "patch_path": "last/patch.pt",
        "patch_views": patch_views,
        "attack_loss_type": config.get("attack_loss_type", OFFICIAL_M_ATTACK_LOSS),
        "maskidx": config.get("maskidx", "0,1,2,3,4,5,6"),
        "final_model_loss": train_rows[-1]["model_loss"] if train_rows else None,
        "final_attack_loss": train_rows[-1]["attack_loss"] if train_rows else None,
        "final_total_attack_loss": train_rows[-1]["total_attack_loss"] if train_rows else None,
        "final_patch_grad_abs_mean": train_rows[-1]["patch_grad_abs_mean"] if train_rows else None,
        "final_patch_grad_l2": train_rows[-1]["patch_grad_l2"] if train_rows else None,
        "final_patch_update_l2": train_rows[-1]["patch_update_l2"] if train_rows else None,
        "final_patch_mean": train_rows[-1]["patch_mean"] if train_rows else None,
        "final_patch_std": train_rows[-1]["patch_std"] if train_rows else None,
        "final_patch_min": train_rows[-1]["patch_min"] if train_rows else None,
        "final_patch_max": train_rows[-1]["patch_max"] if train_rows else None,
        "final_patch_clamp_low_frac": train_rows[-1]["patch_clamp_low_frac"] if train_rows else None,
        "final_patch_clamp_high_frac": train_rows[-1]["patch_clamp_high_frac"] if train_rows else None,
        "patch_position": patch_position,
        "train_patch_position": patch_position,
        "eval_patch_position": eval_patch_position,
        "train_patch_space": train_patch_space,
        "patch_seed": patch_seed,
        "eot_seed": _int_config(config, "eot_seed", _int_config(config, "seed", 0)),
        "eot_enabled": eot_enabled,
        "eot_samples": _int_config(config, "eot_samples", 1),
        "eot_force_eval_anchor_sample": parse_bool(config.get("eot_force_eval_anchor_sample", "false")),
        "loss_weight_mode": config.get("loss_weight_mode", "fixed"),
        "action_boundary_weight": _float_config(config, "action_boundary_weight", 1.0),
        "action_divergence_weight": _float_config(config, "action_divergence_weight", 0.25),
        "feature_deviation_weight": _float_config(config, "feature_deviation_weight", 0.1),
        "action_divergence_type": config.get("action_divergence_type", "mse"),
        "weight_source": config.get("weight_source", "manual"),
        "raw_train_match_eval_resize": raw_train_match_eval_resize if use_raw_patch_training else None,
        "raw_train_eval_image_size": raw_train_eval_image_size if use_raw_patch_training else None,
    }
    write_json(output_dir / "train_metrics.json", metrics)
    return metrics
