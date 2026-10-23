from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import ROOT, require
from ramp_utils import record_key

BRANCH_CONFIG_PATH = ROOT / "configs" / "BRANCH_B.json"
CRITICAL_ARTIFACT_PATH = ROOT / "artifacts" / "branch_b" / "CRITICAL_DIRECTIONS.pt"
METHODS = ("dino_continue_b", "action_critical_b", "dino_critical_b")


def critical_patch_path(method: str, update: int) -> Path:
    return ROOT / "artifacts" / "branch_b" / "patches" / method / f"update_{update:04d}.pt"


def failure_action_direction(clean_action: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unit action direction opposing the clean task/phase action chunk."""
    clean_scaled = clean_action.float() / scale.float()
    direction = -clean_scaled.reshape(-1)
    require(float(torch.linalg.vector_norm(direction)) > 0, "Degenerate clean action direction")
    return direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)


def critical_coefficients(response: torch.Tensor, failure_direction: torch.Tensor) -> torch.Tensor:
    """Steepest failure-aligned coefficient in the two-dimensional geometry basis."""
    require(response.ndim == 2 and response.shape[1] == 2, "Expected action-response matrix [D,2]")
    require(response.shape[0] == failure_direction.numel(), "Action-response dimension mismatch")
    coefficient = response.float().T @ failure_direction.float().reshape(-1)
    require(float(torch.linalg.vector_norm(coefficient)) > 0, "Degenerate critical coefficient")
    return coefficient / torch.linalg.vector_norm(coefficient).clamp_min(1e-12)


def artifact_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return record_key(row)


def dino_failure_direction(record: dict[str, Any], direction: dict[str, Any],
                           device: torch.device) -> torch.Tensor:
    coefficient = direction["critical_coefficients"].to(device).float()
    range_direction = record["range_direction"].to(device).float().reshape(-1)
    lateral_direction = record["lateral_direction"].to(device).float().reshape(-1)
    combined = (coefficient[0] * float(direction["range_scale"]) * range_direction
                + coefficient[1] * float(direction["lateral_scale"]) * lateral_direction)
    require(float(torch.linalg.vector_norm(combined)) > 0, "Degenerate DINO failure direction")
    return combined / torch.linalg.vector_norm(combined).clamp_min(1e-12)


def critical_scores(delta_action: torch.Tensor, delta_dino: torch.Tensor,
                    record: dict[str, Any], direction: dict[str, Any], artifact: dict[str, Any]):
    action_direction = direction["failure_action_direction"].to(delta_action.device).float()
    dino_direction = dino_failure_direction(record, direction, delta_dino.device)
    signed_action = (torch.dot(delta_action.float().reshape(-1), action_direction)
                     / float(artifact["sigma_action"]))
    signed_dino = (torch.dot(delta_dino.float().reshape(-1), dino_direction)
                   / float(artifact["sigma_dino"]))
    return signed_action, signed_dino


def margin_loss(score: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    return F.softplus((float(margin) - score) / float(temperature))


def load_critical_artifact(path: Path = CRITICAL_ARTIFACT_PATH) -> dict[str, Any]:
    require(path.is_file(), f"Missing critical-direction artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    require(len(artifact["records"]) == 300, "Critical-direction coverage drift")
    require(float(artifact["sigma_action"]) > 0 and float(artifact["sigma_dino"]) > 0,
            "Invalid critical-direction normalization")
    return artifact


def derive_local_direction(wrapper, observation: dict[str, Any], task_name: str,
                           clean_action: torch.Tensor, clean_dino: torch.Tensor,
                           range_dino: torch.Tensor, lateral_dino: torch.Tensor,
                           scale: torch.Tensor, fraction: float,
                           xy: tuple[int, int]):
    """Build the frozen critical definition for one held-out task/phase state."""
    from runtime import action_with_dino_value

    units, natural_scales, columns = [], [], []
    for perturbed_dino in (range_dino, lateral_dino):
        difference = perturbed_dino.float() - clean_dino.float()
        natural_scale = torch.linalg.vector_norm(difference).clamp_min(1e-8)
        unit = difference / natural_scale
        epsilon = float(fraction) * natural_scale
        with torch.no_grad():
            plus = action_with_dino_value(
                wrapper, observation, task_name, clean_dino + epsilon * unit, None, xy)
            minus = action_with_dino_value(
                wrapper, observation, task_name, clean_dino - epsilon * unit, None, xy)
        columns.append(((plus.float() - minus.float()) / scale / (2.0 * float(fraction))).reshape(-1))
        units.append(unit.reshape(-1))
        natural_scales.append(float(natural_scale))
    response = torch.stack(columns, dim=1)
    action_direction = failure_action_direction(clean_action, scale)
    coefficient = critical_coefficients(response, action_direction)
    record = {"range_direction": units[0], "lateral_direction": units[1]}
    direction = {"range_scale": natural_scales[0], "lateral_scale": natural_scales[1],
                 "critical_coefficients": coefficient,
                 "failure_action_direction": action_direction}
    return record, direction


def direction_map(artifact: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {artifact_key(row): row for row in artifact["records"]}
    require(len(result) == len(artifact["records"]), "Duplicate critical-direction key")
    return result
