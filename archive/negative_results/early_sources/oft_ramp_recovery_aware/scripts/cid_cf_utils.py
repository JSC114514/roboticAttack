from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import ROOT, require


CID_CONFIG_PATH = ROOT / "configs" / "CID_CF.json"
METHODS = ("action_eot", "dino_cf", "dino_cf_tail", "cid_cf")
OBJECTIVES = ("cf", "tail", "consistency", "full_action")
METHOD_OBJECTIVES = {
    "action_eot": ("full_action",),
    "dino_cf": ("cf",),
    "dino_cf_tail": ("cf", "tail", "full_action"),
    "cid_cf": OBJECTIVES,
}


@dataclass(frozen=True)
class EOTParameters:
    translation_x: float
    translation_y: float
    scale: float
    rotation_degrees: float
    brightness: float
    contrast: float


def patch_path(method: str, update: int) -> Path:
    require(method in METHODS, f"Unknown CID-CF method: {method}")
    return ROOT / "artifacts" / "cid_cf" / "patches" / method / f"update_{update:04d}.pt"


def record_key(record: dict[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_id"]), int(record["state_index"]),
            int(record["decision_index"]))


def task_pair_indices(update: int, trajectories: list[list[dict[str, Any]]], seed: int) -> tuple[int, int]:
    """Balanced deterministic schedule: two distinct tasks and one rank per task."""
    require(update >= 1, "Update index must be one-based")
    by_task: dict[int, dict[int, int]] = {}
    for index, trajectory in enumerate(trajectories):
        first = trajectory[0]
        by_task.setdefault(int(first["task_id"]), {})[int(first["trajectory_rank"])] = index
    require(set(by_task) == set(range(10)), "CID-CF requires Object tasks 0..9")
    require(all(set(values) == set(range(5)) for values in by_task.values()),
            "CID-CF requires five ranked trajectories per Object task")

    round_index, pair_slot = divmod(update - 1, 5)
    tasks = list(range(10))
    random.Random(seed + round_index).shuffle(tasks)
    selected_tasks = tasks[2 * pair_slot:2 * pair_slot + 2]
    result = []
    for task_id in selected_tasks:
        cycle, offset = divmod(round_index, 5)
        ranks = list(range(5))
        random.Random(seed + 100_000 + 101 * cycle + task_id).shuffle(ranks)
        result.append(by_task[task_id][ranks[offset]])
    require(len(result) == 2 and selected_tasks[0] != selected_tasks[1],
            "Task-pair schedule produced duplicate tasks")
    return int(result[0]), int(result[1])


def sample_eot(cfg: dict[str, Any], update: int, context_index: int) -> EOTParameters:
    require(0 <= context_index < int(cfg["contexts_per_update"]), "Invalid context index")
    ranges = cfg["eot"]
    rng = random.Random(int(cfg["seed"]) + 1_000_003 * update + 7_919 * context_index)
    draw = lambda key: rng.uniform(float(ranges[key][0]), float(ranges[key][1]))
    return EOTParameters(
        translation_x=draw("translation_x_pixels"),
        translation_y=draw("translation_y_pixels"),
        scale=draw("scale"),
        rotation_degrees=draw("rotation_degrees"),
        brightness=draw("brightness"),
        contrast=draw("contrast"),
    )


def eot_dict(value: EOTParameters) -> dict[str, float]:
    return {key: float(item) for key, item in asdict(value).items()}


def apply_eot_patch(image: torch.Tensor, patch: torch.Tensor, xy: tuple[int, int],
                    parameters: EOTParameters) -> torch.Tensor:
    """Differentiably resize/rotate/color-transform and alpha-blend a Wrist patch."""
    require(image.ndim == 3 and patch.ndim == 3 and image.shape[0] == patch.shape[0] == 3,
            "Expected RGB CHW image and patch")
    require(bool(torch.isfinite(patch).all()), "Non-finite CID-CF patch")
    transformed = patch.float()
    mean = transformed.mean(dim=(-2, -1), keepdim=True)
    transformed = ((transformed - mean) * float(parameters.contrast) + mean)
    transformed = (transformed * float(parameters.brightness)).clamp(0, 1)

    height = max(1, int(round(patch.shape[-2] * float(parameters.scale))))
    width = max(1, int(round(patch.shape[-1] * float(parameters.scale))))
    transformed = F.interpolate(transformed.unsqueeze(0), size=(height, width),
                                mode="bilinear", align_corners=False)
    alpha = torch.ones((1, 1, height, width), device=patch.device, dtype=transformed.dtype)
    angle = math.radians(float(parameters.rotation_degrees))
    theta = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0],
                          [math.sin(angle), math.cos(angle), 0.0]],
                         device=patch.device, dtype=transformed.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, transformed.shape, align_corners=False)
    transformed = F.grid_sample(transformed, grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)
    alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False).clamp(0, 1)

    base_x, base_y = map(int, xy)
    center_x = base_x + patch.shape[-1] / 2.0 + float(parameters.translation_x)
    center_y = base_y + patch.shape[-2] / 2.0 + float(parameters.translation_y)
    left, top = int(round(center_x - width / 2.0)), int(round(center_y - height / 2.0))
    require(0 <= left and left + width <= image.shape[-1]
            and 0 <= top and top + height <= image.shape[-2],
            "EOT-transformed patch left the Wrist image")
    canvas = torch.zeros_like(image, dtype=transformed.dtype)
    mask = torch.zeros((1, image.shape[-2], image.shape[-1]),
                       device=image.device, dtype=transformed.dtype)
    canvas[:, top:top + height, left:left + width] = transformed.squeeze(0) * alpha.squeeze(0)
    mask[:, top:top + height, left:left + width] = alpha.squeeze(0)
    return (image.float() * (1.0 - mask) + canvas).clamp(0, 1)


def standardized_action_l2(action: torch.Tensor, clean: torch.Tensor,
                           scale: torch.Tensor) -> torch.Tensor:
    delta = (action.float() - clean.float()) / scale.float()
    require(delta.numel() == 56, "CID-CF expects one 8x7 action chunk")
    return torch.linalg.vector_norm(delta.reshape(-1))


def rms_displacement(value: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    delta = value.float() - clean.float()
    return torch.sqrt(torch.mean(delta.square()))


def unit_displacement(value: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    delta = (value.float() - clean.float()).reshape(-1)
    return delta / torch.linalg.vector_norm(delta).clamp_min(1e-12)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 0:
        return 0.0
    return float(torch.sum(left * right) / denominator)


def scale_auxiliary(reference: torch.Tensor, auxiliary: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
    ref_norm = torch.linalg.vector_norm(reference)
    aux_norm = torch.linalg.vector_norm(auxiliary)
    require(float(ref_norm) > 0 and float(aux_norm) > 0, "Degenerate CID-CF objective gradient")
    factor = float(ratio) * ref_norm / aux_norm
    return factor * auxiliary, float(factor)
