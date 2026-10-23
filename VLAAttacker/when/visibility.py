"""Visibility metric names and helpers for world-coordinate patch experiments."""

from __future__ import annotations


VISIBILITY_FIELDS = (
    "front_visible",
    "wrist_visible",
    "front_area_ratio",
    "wrist_area_ratio",
    "front_occlusion_ratio",
    "wrist_occlusion_ratio",
    "front_visibility_rate",
    "wrist_visibility_rate",
    "both_visible_rate",
    "neither_visible_rate",
    "first_visible_step",
    "last_visible_step",
)


def empty_visibility_record() -> dict[str, float | int | bool | str]:
    """Return an empty visibility record with stable keys."""
    return {field: "" for field in VISIBILITY_FIELDS}
