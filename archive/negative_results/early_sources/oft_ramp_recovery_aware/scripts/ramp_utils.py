from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import torch

from common import require


METHODS = ("dino_continue", "ramp_t", "ramp_tg")


def record_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["task_id"]), int(row["state_index"]), int(row.get("decision_index", -1))


def group_trajectories(records: list[dict[str, Any]], expected: int = 50) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        rank = int(row.get("trajectory_rank", row.get("state_index", -1)))
        groups[(int(row["task_id"]), rank)].append(row)
    trajectories = []
    phase_order = {"pre_1": 0, "pre_2": 1, "contact_1": 2, "contact_2": 3,
                   "post_1": 4, "post_2": 5}
    for key in sorted(groups):
        rows = groups[key]
        rows.sort(key=lambda row: (int(row.get("decision_index", 10**9)),
                                   phase_order.get(str(row.get("phase_proxy", "")), 99)))
        require(len(rows) == 6, f"Trajectory {key} has {len(rows)} states, expected 6")
        trajectories.append(rows)
    require(len(trajectories) == expected, f"Found {len(trajectories)} trajectories, expected {expected}")
    return trajectories


def training_index(update: int, size: int, seed: int) -> int:
    epoch, offset = divmod(update - 1, size)
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def normalized_phase_weights(values: list[float], device: torch.device) -> torch.Tensor:
    weights = torch.as_tensor(values, device=device, dtype=torch.float32)
    require(weights.shape == (6,) and torch.all(weights > 0), "Invalid phase weights")
    return weights / weights.mean()


def project_and_scale(base: torch.Tensor, auxiliary: torch.Tensor, ratio: float):
    dot = torch.sum(base * auxiliary)
    conflict = bool(dot < 0)
    adjusted = auxiliary
    if conflict:
        adjusted = auxiliary - dot / torch.sum(base * base).clamp_min(1e-12) * base
    scale = float(ratio) * torch.linalg.vector_norm(base) / torch.linalg.vector_norm(adjusted).clamp_min(1e-12)
    scaled = scale * adjusted
    effective = float(torch.linalg.vector_norm(scaled) / torch.linalg.vector_norm(base).clamp_min(1e-12))
    return scaled, float(scale), conflict, effective


def action_quantities(action: torch.Tensor, clean: torch.Tensor, scale: torch.Tensor, beta: float):
    delta = (action.float() - clean.float()) / scale
    action_l2 = torch.linalg.vector_norm(delta, dim=-1).mean()
    mean_chunk = delta[..., :6].mean(dim=1)
    persistent = torch.linalg.vector_norm(mean_chunk, dim=-1).mean()
    state_score = action_l2 + float(beta) * persistent
    six_dim_mean = mean_chunk.reshape(-1)
    return delta, action_l2, persistent, state_score, six_dim_mean
