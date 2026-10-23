#!/usr/bin/env python3
"""Render one OFT physical-patch LIBERO observation for visibility checking."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPA_RFAS_OFT_ROOT = Path("/home/caijiasheng25/UPA-RFAS/openvla-oft")
DEFAULT_LIBERO_ROOT = Path("/home/caijiasheng25/Desktop/LIBERO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--official-oft-root", default=str(DEFAULT_UPA_RFAS_OFT_ROOT))
    parser.add_argument("--libero-root", default=str(DEFAULT_LIBERO_ROOT))
    parser.add_argument(
        "--offsets",
        default="-0.08 -0.08 0.001;-0.08 0.00 0.001;-0.08 0.08 0.001;"
        "0.00 -0.08 0.001;0.00 0.00 0.001;0.00 0.08 0.001;"
        "0.08 -0.08 0.001;0.08 0.00 0.001;0.08 0.08 0.001",
        help="Semicolon-separated WHEN_WORLD_PATCH_OFFSET candidates.",
    )
    parser.add_argument("--size", default="0.055 0.055 0.001")
    parser.add_argument(
        "--positions",
        default="",
        help="Semicolon-separated absolute WHEN_WORLD_PATCH_POS candidates. Overrides --offsets.",
    )
    parser.add_argument(
        "--object-slots",
        action="store_true",
        help="Compute absolute patch candidate positions from tabletop object coordinates.",
    )
    parser.add_argument(
        "--camera-slots",
        action="store_true",
        help="Compute absolute patch candidate positions under the wrist camera and its center ray.",
    )
    parser.add_argument("--camera-name", default="robot0_eye_in_hand")
    parser.add_argument(
        "--z-values",
        default="",
        help="Semicolon-separated absolute z values for --object-slots, for example 0.68;0.70;0.72.",
    )
    parser.add_argument(
        "--camera-offsets",
        default="0 0",
        help="Semicolon-separated 2D offsets in the wrist camera image plane for --camera-slots.",
    )
    parser.add_argument(
        "--anchor-mode",
        default="object_cluster",
        choices=("object_cluster", "site"),
        help="Anchor offsets at the detected tabletop object cluster or at WHEN_WORLD_PATCH_ANCHOR_SITE.",
    )
    parser.add_argument(
        "--marker-patch",
        action="store_true",
        help="Use a high-contrast marker texture for placement debugging.",
    )
    return parser.parse_args()


def _labelled(img: Image.Image, label: str) -> Image.Image:
    canvas = ImageOps.expand(img.convert("RGB"), border=(0, 24, 0, 0), fill="white")
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def _write_contact_sheet(images: list[Image.Image], output_path: Path) -> None:
    contact = Image.new("RGB", (sum(img.width for img in images), max(img.height for img in images)), "white")
    x = 0
    for img in images:
        contact.paste(img, (x, 0))
        x += img.width
    contact.save(output_path)


def _render_camera(env, camera_name: str, resolution: int = 224) -> Image.Image:
    frame = env.sim.render(camera_name=camera_name, width=resolution, height=resolution, depth=False)
    return Image.fromarray(frame[::-1].copy())


def _parse_offsets(raw: str) -> list[str]:
    offsets = []
    for item in raw.split(";"):
        item = " ".join(item.replace(",", " ").split())
        if not item:
            continue
        parts = item.split()
        if len(parts) != 3:
            raise ValueError(f"Offset must have 3 values, got {item!r}")
        offsets.append(item)
    if not offsets:
        raise ValueError("At least one offset is required")
    return offsets


def _parse_positions(raw: str) -> list[list[float]]:
    positions = []
    for item in raw.split(";"):
        item = " ".join(item.replace(",", " ").split())
        if not item:
            continue
        parts = [float(part) for part in item.split()]
        if len(parts) != 3:
            raise ValueError(f"Position must have 3 values, got {item!r}")
        positions.append(parts)
    return positions


def _parse_z_values(raw: str, default_z: float) -> list[float]:
    values = [float(item.strip()) for item in raw.replace(",", ";").split(";") if item.strip()]
    return values or [default_z]


def _parse_camera_offsets(raw: str) -> list[list[float]]:
    offsets = []
    for item in raw.split(";"):
        item = " ".join(item.replace(",", " ").split())
        if not item:
            continue
        values = [float(part) for part in item.split()]
        if len(values) != 2:
            raise ValueError(f"Camera offset must have 2 values, got {item!r}")
        offsets.append(values)
    return offsets or [[0.0, 0.0]]


def _candidate_name(offset: str) -> str:
    return "offset_" + "_".join(part.replace("-", "m").replace(".", "p") for part in offset.split())


def _values_name(prefix: str, values: list[float]) -> str:
    parts = [f"{value:.4f}".rstrip("0").rstrip(".") for value in values]
    return prefix + "_" + "_".join(part.replace("-", "m").replace(".", "p") for part in parts)


def _short_object_name(name: str) -> str:
    return (
        name.replace("_main", "")
        .replace("_visual", "")
        .replace("_collision", "")
        .replace("/", "_")
    )


def _collect_object_points(task, initial_state, num_steps_wait, get_libero_env, get_libero_dummy_action, table_z_fn, object_points_fn):
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    table_z = float(table_z_fn(env))
    points = object_points_fn(env, table_z)
    return table_z, points


def _object_slot_candidates(object_points: list[dict[str, object]], z_values: list[float]) -> list[dict[str, object]]:
    if not object_points:
        raise RuntimeError("No tabletop object coordinates were detected; cannot compute object-slot patch positions.")

    xy = np.array([point["pos"][:2] for point in object_points], dtype=float)
    raw_slots: list[tuple[str, np.ndarray, dict[str, object]]] = []
    raw_slots.append(("centroid_all", xy.mean(axis=0), {"objects": object_points}))
    raw_slots.append(("bbox_center_all", (xy.min(axis=0) + xy.max(axis=0)) / 2.0, {"objects": object_points}))

    for left, right in combinations(object_points, 2):
        left_xy = np.array(left["pos"][:2], dtype=float)
        right_xy = np.array(right["pos"][:2], dtype=float)
        distance = float(np.linalg.norm(left_xy - right_xy))
        name = f"pair_{_short_object_name(left['name'])}__{_short_object_name(right['name'])}"
        raw_slots.append(
            (
                name,
                (left_xy + right_xy) / 2.0,
                {
                    "objects": [left, right],
                    "pair_distance_xy": distance,
                },
            )
        )

    candidates = []
    seen = set()
    for slot_name, slot_xy, meta in raw_slots:
        for z in z_values:
            pos = [float(slot_xy[0]), float(slot_xy[1]), float(z)]
            key = (slot_name, round(pos[0], 5), round(pos[1], 5), round(pos[2], 5))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "kind": "position",
                    "name": _values_name(slot_name, pos),
                    "position": pos,
                    "slot_name": slot_name,
                    "slot_meta": meta,
                }
            )
    return candidates


def _camera_slot_candidates(
    task,
    initial_state,
    num_steps_wait,
    get_libero_env,
    get_libero_dummy_action,
    table_z_fn,
    object_points_fn,
    find_named_id_fn,
    camera_name: str,
    z_values: list[float],
    camera_offsets: list[list[float]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    camera_id = find_named_id_fn(env.sim.model, "camera", camera_name)
    if camera_id is None:
        raise RuntimeError(f"Camera {camera_name!r} was not found in the MuJoCo model.")

    table_z = float(table_z_fn(env))
    object_points = object_points_fn(env, table_z)
    cam_pos = np.array(env.sim.data.cam_xpos[camera_id], dtype=float)
    cam_mat = np.array(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    camera_info = {
        "camera_name": camera_name,
        "camera_id": int(camera_id),
        "camera_pos": [float(value) for value in cam_pos],
        "camera_mat": [[float(value) for value in row] for row in cam_mat],
        "table_z": table_z,
        "object_points": object_points,
    }

    raw_slots: list[tuple[str, list[float], dict[str, object]]] = []
    optical_direction = -cam_mat[:, 2]
    optical_norm = float(np.linalg.norm(optical_direction))
    if optical_norm > 1e-8:
        optical_direction = optical_direction / optical_norm
        right_axis = cam_mat[:, 0]
        up_axis = cam_mat[:, 1]
        for z in z_values:
            if abs(float(optical_direction[2])) < 1e-8:
                continue
            t = (float(z) - float(cam_pos[2])) / float(optical_direction[2])
            if not (0.02 < t < 5.0):
                continue
            base_point = cam_pos + t * optical_direction
            for dx, dy in camera_offsets:
                point = base_point + float(dx) * right_axis + float(dy) * up_axis
                point[2] = float(z)
                if abs(float(point[0])) > 1.5 or abs(float(point[1])) > 1.5:
                    continue
                distances = [
                    float(np.linalg.norm(np.array(point[:2], dtype=float) - np.array(obj["pos"][:2], dtype=float)))
                    for obj in object_points
                ]
                raw_slots.append(
                    (
                        f"camera_plane_dx{dx:.3f}_dy{dy:.3f}_z{z:.4f}",
                        [float(point[0]), float(point[1]), float(point[2])],
                        {
                            "camera": camera_info,
                            "projection": "camera_image_plane_offset",
                            "base_projection": "col2_neg_center_ray",
                            "camera_offset": [float(dx), float(dy)],
                            "ray_t": float(t),
                            "min_object_xy_distance": min(distances) if distances else None,
                        },
                    )
                )

    for z in z_values:
        raw_slots.append(
            (
                f"camera_vertical_z{z:.4f}",
                [float(cam_pos[0]), float(cam_pos[1]), float(z)],
                {"camera": camera_info, "projection": "vertical_under_camera"},
            )
        )

    directions: list[tuple[str, np.ndarray]] = []
    for axis in range(3):
        directions.append((f"col{axis}_pos", cam_mat[:, axis]))
        directions.append((f"col{axis}_neg", -cam_mat[:, axis]))
        directions.append((f"row{axis}_pos", cam_mat[axis, :]))
        directions.append((f"row{axis}_neg", -cam_mat[axis, :]))

    seen_dirs = set()
    for name, direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        unit = direction / norm
        dir_key = tuple(round(float(value), 6) for value in unit)
        if dir_key in seen_dirs:
            continue
        seen_dirs.add(dir_key)
        if abs(float(unit[2])) < 1e-8:
            continue
        for z in z_values:
            t = (float(z) - float(cam_pos[2])) / float(unit[2])
            if not (0.02 < t < 5.0):
                continue
            point = cam_pos + t * unit
            point[2] = float(z)
            if abs(float(point[0])) > 1.5 or abs(float(point[1])) > 1.5:
                continue
            raw_slots.append(
                (
                    f"camera_ray_{name}_z{z:.4f}",
                    [float(point[0]), float(point[1]), float(point[2])],
                    {
                        "camera": camera_info,
                        "projection": "center_ray_candidate",
                        "direction_name": name,
                        "direction": [float(value) for value in unit],
                        "ray_t": float(t),
                    },
                )
            )

    candidates = []
    seen_positions = set()
    for slot_name, pos, meta in raw_slots:
        key = (round(pos[0], 5), round(pos[1], 5), round(pos[2], 5), slot_name)
        if key in seen_positions:
            continue
        seen_positions.add(key)
        candidates.append(
            {
                "kind": "position",
                "name": _values_name(slot_name, pos),
                "position": pos,
                "slot_name": slot_name,
                "slot_meta": meta,
            }
        )
    return candidates, camera_info


def _marker_patch(size: int = 64) -> torch.Tensor:
    yy = torch.arange(size).view(size, 1)
    xx = torch.arange(size).view(1, size)
    checker = ((xx // 8 + yy // 8) % 2).bool()
    patch = torch.zeros(3, size, size)
    patch[0, checker] = 1.0
    patch[2, checker] = 1.0
    patch[1, ~checker] = 1.0
    patch[2, ~checker] = 1.0
    return patch


def _baseline_wrist_image(task, initial_state, num_steps_wait, get_libero_env, get_libero_dummy_action, get_libero_wrist_image, resize_image_for_policy) -> np.ndarray:
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    wrist = resize_image_for_policy(get_libero_wrist_image(obs), 224)
    return np.asarray(wrist, dtype=np.int16)


def _wrist_marker_score(wrist: Image.Image, baseline_wrist: np.ndarray | None) -> dict[str, object]:
    if baseline_wrist is None:
        return {
            "visible_pixels": 0,
            "centroid_xy": None,
            "center_distance_px": None,
        }
    arr = np.asarray(wrist.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - baseline_wrist).sum(axis=2)
    magenta = (arr[:, :, 0] > 130) & (arr[:, :, 2] > 130) & (arr[:, :, 1] < 110)
    cyan = (arr[:, :, 1] > 130) & (arr[:, :, 2] > 130) & (arr[:, :, 0] < 110)
    mask = (diff > 40) & (magenta | cyan)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "visible_pixels": 0,
            "centroid_xy": None,
            "center_distance_px": None,
        }
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    center = (np.array(arr.shape[:2][::-1], dtype=float) - 1.0) / 2.0
    distance = float(np.linalg.norm(np.array([centroid_x, centroid_y]) - center))
    return {
        "visible_pixels": int(len(xs)),
        "centroid_xy": [centroid_x, centroid_y],
        "center_distance_px": distance,
    }


def _annotate_wrist(wrist: Image.Image, wrist_score: dict[str, object]) -> Image.Image:
    annotated = wrist.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    cx = (annotated.width - 1) / 2.0
    cy = (annotated.height - 1) / 2.0
    draw.line((cx - 10, cy, cx + 10, cy), fill=(255, 255, 0), width=2)
    draw.line((cx, cy - 10, cx, cy + 10), fill=(255, 255, 0), width=2)
    centroid = wrist_score.get("centroid_xy")
    if centroid is not None:
        mx, my = centroid
        draw.ellipse((mx - 5, my - 5, mx + 5, my + 5), outline=(255, 0, 0), width=2)
    return annotated


def _model_names(model, attr: str) -> list[str]:
    values = getattr(model, attr, [])
    if callable(values):
        values = values()
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="ignore"))
        else:
            names.append(str(value))
    return names


def _name_id(model, kind: str, name: str) -> int | None:
    name2id = getattr(model, f"{kind}_name2id", None)
    if callable(name2id):
        try:
            return int(name2id(name))
        except Exception:
            pass
    names = _model_names(model, f"{kind}_names")
    try:
        return names.index(name)
    except ValueError:
        pass
    for idx, value in enumerate(names):
        if name in value:
            return idx
    return None


def _project_points(env, camera_name: str, points: list[list[float]], resolution: int = 224) -> dict[str, object]:
    camera_id = _name_id(env.sim.model, "camera", camera_name)
    if camera_id is None:
        return {"camera_name": camera_name, "camera_id": None, "points": []}

    cam_pos = np.array(env.sim.data.cam_xpos[camera_id], dtype=float)
    cam_mat = np.array(env.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    fovy = float(env.sim.model.cam_fovy[camera_id])
    focal = 0.5 * float(resolution) / np.tan(np.deg2rad(fovy) / 2.0)
    projected = []
    for point in points:
        world = np.array(point, dtype=float)
        camera = cam_mat.T @ (world - cam_pos)
        depth = -float(camera[2])
        if depth <= 1e-8:
            projected.append(
                {
                    "world": [float(value) for value in world],
                    "pixel": None,
                    "depth": depth,
                    "in_frame": False,
                }
            )
            continue
        px = float(resolution) / 2.0 + focal * float(camera[0]) / depth
        py = float(resolution) / 2.0 - focal * float(camera[1]) / depth
        projected.append(
            {
                "world": [float(value) for value in world],
                "pixel": [px, py],
                "depth": depth,
                "in_frame": bool(0.0 <= px < resolution and 0.0 <= py < resolution),
            }
        )

    return {
        "camera_name": camera_name,
        "camera_id": int(camera_id),
        "camera_pos": [float(value) for value in cam_pos],
        "camera_fovy": fovy,
        "points": projected,
    }


def _patch_geometry(env) -> dict[str, object]:
    geom_id = _name_id(env.sim.model, "geom", "when_physical_patch_geom")
    body_id = _name_id(env.sim.model, "body", "when_physical_patch_body")
    if geom_id is None:
        return {"found": False}

    center = np.array(env.sim.data.geom_xpos[geom_id], dtype=float)
    xmat = np.array(env.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    size = np.array(env.sim.model.geom_size[geom_id], dtype=float)

    def face_corners(z_sign: float) -> list[list[float]]:
        corners = []
        for sx, sy in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]:
            local = np.array([sx * size[0], sy * size[1], z_sign * size[2]], dtype=float)
            world = center + xmat @ local
            corners.append([float(value) for value in world])
        return corners

    top_corners = face_corners(1.0)
    bottom_corners = face_corners(-1.0)
    all_points = [center.tolist()] + top_corners + bottom_corners
    z_values = [point[2] for point in all_points]

    payload = {
        "found": True,
        "geom_id": int(geom_id),
        "body_id": int(body_id) if body_id is not None else None,
        "geom_center": [float(value) for value in center],
        "geom_size": [float(value) for value in size],
        "geom_xmat": [[float(value) for value in row] for row in xmat],
        "top_corners": top_corners,
        "bottom_corners": bottom_corners,
        "z_min": float(min(z_values)),
        "z_max": float(max(z_values)),
        "z_axis": [float(value) for value in xmat[:, 2]],
    }
    if body_id is not None:
        payload["body_center"] = [float(value) for value in np.array(env.sim.data.body_xpos[body_id], dtype=float)]
    return payload


def _nearby_geoms(env, xy: list[float] | None, radius: float = 0.45) -> list[dict[str, object]]:
    if xy is None:
        xy = [0.0, 0.0]
    xy_arr = np.array(xy, dtype=float)
    names = _model_names(env.sim.model, "geom_names")
    rows = []
    for geom_id, name in enumerate(names):
        try:
            pos = np.array(env.sim.data.geom_xpos[geom_id], dtype=float)
            size = np.array(env.sim.model.geom_size[geom_id], dtype=float)
            geom_type = int(env.sim.model.geom_type[geom_id])
        except Exception:
            continue
        dist = float(np.linalg.norm(pos[:2] - xy_arr))
        is_large_horizontal = len(size) >= 2 and float(size[0]) >= 0.12 and float(size[1]) >= 0.12
        if dist > radius and not is_large_horizontal and "table" not in name.lower():
            continue
        rows.append(
            {
                "geom_id": int(geom_id),
                "name": name,
                "type": geom_type,
                "pos": [float(value) for value in pos],
                "size": [float(value) for value in size],
                "xy_distance_to_patch": dist,
                "approx_z_min": float(pos[2] - size[2]) if len(size) >= 3 else None,
                "approx_z_max": float(pos[2] + size[2]) if len(size) >= 3 else None,
            }
        )
    rows.sort(key=lambda item: (float(item["xy_distance_to_patch"]), item["name"]))
    return rows[:120]


def _image_diff_score(image: Image.Image, baseline: np.ndarray | None, threshold: int = 45) -> dict[str, object]:
    if baseline is None:
        return {"visible_pixels": 0, "bbox_xyxy": None, "centroid_xy": None, "area_ratio": 0.0}
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr - baseline.astype(np.int16)).sum(axis=2)
    mask = diff > threshold
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"visible_pixels": 0, "bbox_xyxy": None, "centroid_xy": None, "area_ratio": 0.0}
    return {
        "visible_pixels": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "area_ratio": float(len(xs) / mask.size),
    }


def _draw_diff_box(image: Image.Image, score: dict[str, object], color: tuple[int, int, int]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    bbox = score.get("bbox_xyxy")
    if bbox is None:
        return annotated
    draw = ImageDraw.Draw(annotated)
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
    centroid = score.get("centroid_xy")
    if centroid is not None:
        cx, cy = centroid
        draw.line((cx - 5, cy, cx + 5, cy), fill=color, width=2)
        draw.line((cx, cy - 5, cx, cy + 5), fill=color, width=2)
    return annotated


def _draw_projection(image: Image.Image, projection: dict[str, object], color: tuple[int, int, int]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    points = projection.get("points", [])
    if len(points) < 5:
        return annotated
    draw = ImageDraw.Draw(annotated)

    def pixel(idx: int):
        value = points[idx].get("pixel")
        if value is None:
            return None
        return (float(value[0]), float(value[1]))

    center = pixel(0)
    if center is not None:
        cx, cy = center
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=color, width=2)

    top = [pixel(idx) for idx in range(1, 5)]
    if all(item is not None for item in top):
        draw.line(top + [top[0]], fill=color, width=2)
    return annotated


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = _parse_offsets(args.offsets)
    os.environ.setdefault("WHEN_WORLD_PATCH_SIZE", args.size)
    os.environ["WHEN_WORLD_PATCH_ANCHOR_MODE"] = args.anchor_mode

    official_root = Path(args.official_oft_root).expanduser()
    libero_root = Path(args.libero_root).expanduser()
    sys.path.insert(0, str(libero_root))
    sys.path.insert(0, str(official_root))

    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
    )
    from experiments.robot.libero.run_libero_eval import inject_world_patch
    from experiments.robot.libero.run_libero_eval import _find_named_id, _scene_object_points, _table_center_z
    from experiments.robot.openvla_utils import resize_image_for_policy

    manifest = json.loads(Path(args.patch_manifest).read_text(encoding="utf-8"))
    patch_path = manifest["patch_path"]
    patch = torch.load(patch_path, map_location="cpu")
    if args.marker_patch:
        patch = _marker_patch()

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    initial_state = task_suite.get_task_init_states(args.task_id)[args.episode_id]
    task_description = task.language

    explicit_positions = _parse_positions(args.positions)
    camera_slot_info = None
    if args.object_slots:
        table_z, object_points = _collect_object_points(
            task,
            initial_state,
            args.num_steps_wait,
            get_libero_env,
            get_libero_dummy_action,
            _table_center_z,
            _scene_object_points,
        )
        candidate_specs = _object_slot_candidates(object_points, _parse_z_values(args.z_values, table_z - 0.10))
    elif args.camera_slots:
        table_z = None
        object_points = []
        candidate_specs, camera_slot_info = _camera_slot_candidates(
            task,
            initial_state,
            args.num_steps_wait,
            get_libero_env,
            get_libero_dummy_action,
            _table_center_z,
            _scene_object_points,
            _find_named_id,
            args.camera_name,
            _parse_z_values(args.z_values, 0.68),
            _parse_camera_offsets(args.camera_offsets),
        )
    elif explicit_positions:
        table_z = None
        object_points = []
        candidate_specs = [
            {
                "kind": "position",
                "name": _values_name("pos", position),
                "position": position,
                "slot_name": "explicit",
                "slot_meta": {},
            }
            for position in explicit_positions
        ]
    else:
        table_z = None
        object_points = []
        candidate_specs = [
            {
                "kind": "offset",
                "name": _candidate_name(offset),
                "offset": offset,
                "slot_name": "",
                "slot_meta": {},
            }
            for offset in offsets
        ]

    baseline_wrist = None
    if args.marker_patch:
        baseline_wrist = _baseline_wrist_image(
            task,
            initial_state,
            args.num_steps_wait,
            get_libero_env,
            get_libero_dummy_action,
            get_libero_wrist_image,
            resize_image_for_policy,
        )

    rows = []
    candidates = []
    for spec in candidate_specs:
        candidate_dir = output_dir / str(spec["name"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "position":
            position = spec["position"]
            os.environ["WHEN_WORLD_PATCH_POS"] = " ".join(f"{value:.8g}" for value in position)
            os.environ.pop("WHEN_WORLD_PATCH_OFFSET", None)
        else:
            offset = str(spec["offset"])
            os.environ.pop("WHEN_WORLD_PATCH_POS", None)
            os.environ["WHEN_WORLD_PATCH_OFFSET"] = offset
        os.environ["WHEN_ROLLOUT_DIR"] = str(candidate_dir / "videos")

        env, _ = get_libero_env(task, "openvla", resolution=256)
        env.reset()
        obs = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        baseline_front = np.asarray(resize_image_for_policy(get_libero_image(obs), 224), dtype=np.int16)
        baseline_wrist_candidate = np.asarray(resize_image_for_policy(get_libero_wrist_image(obs), 224), dtype=np.int16)
        inject_world_patch(env, patch)
        world_patch_info = getattr(env, "_when_world_patch_info", {})
        obs = env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

        front = Image.fromarray(resize_image_for_policy(get_libero_image(obs), 224))
        wrist = Image.fromarray(resize_image_for_policy(get_libero_wrist_image(obs), 224))
        front_diff_score = _image_diff_score(front, baseline_front)
        wrist_diff_score = _image_diff_score(wrist, baseline_wrist_candidate)
        wrist_score = _wrist_marker_score(wrist, baseline_wrist)
        bird = _render_camera(env, "birdview", 224)
        geometry = _patch_geometry(env)
        patch_points = []
        if geometry.get("found"):
            patch_points = [geometry["geom_center"]] + geometry["top_corners"] + geometry["bottom_corners"]
            geometry["front_projection"] = _project_points(env, "agentview", patch_points, 224)
            geometry["wrist_projection"] = _project_points(env, "robot0_eye_in_hand", patch_points, 224)
            geometry["birdview_projection"] = _project_points(env, "birdview", patch_points, 224)
            geometry["nearby_geoms"] = _nearby_geoms(env, geometry["geom_center"][:2])
        else:
            geometry["front_projection"] = {"points": []}
            geometry["wrist_projection"] = {"points": []}
            geometry["birdview_projection"] = {"points": []}
            geometry["nearby_geoms"] = []

        front_annotated = _draw_projection(
            _draw_diff_box(front, front_diff_score, (255, 255, 0)),
            geometry["front_projection"],
            (0, 255, 0),
        )
        wrist_annotated = _draw_projection(
            _draw_diff_box(_annotate_wrist(wrist, wrist_score), wrist_diff_score, (255, 255, 0)),
            geometry["wrist_projection"],
            (0, 255, 0),
        )
        front_path = candidate_dir / "front.png"
        front_annotated_path = candidate_dir / "front_annotated.png"
        wrist_path = candidate_dir / "wrist.png"
        wrist_annotated_path = candidate_dir / "wrist_annotated.png"
        bird_path = candidate_dir / "birdview.png"
        contact_path = candidate_dir / "contact.png"
        geometry_path = candidate_dir / "geometry.json"
        front.save(front_path)
        front_annotated.save(front_annotated_path)
        wrist.save(wrist_path)
        wrist_annotated.save(wrist_annotated_path)
        bird.save(bird_path)
        geometry_path.write_text(json.dumps(geometry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        patch_texture_path = candidate_dir / "physical_patch_assets" / "P_WHEN_physical.png"
        patch_texture = Image.open(patch_texture_path).convert("RGB").resize((224, 224))
        wrist_label = "wrist"
        if wrist_score["center_distance_px"] is not None:
            wrist_label = f"wrist d={wrist_score['center_distance_px']:.1f} pix={wrist_score['visible_pixels']}"
        pos = world_patch_info.get("pos")
        front_label = f"front {spec['name']}"
        if pos is not None:
            front_label += " pos=" + ",".join(f"{float(value):.2f}" for value in pos)
        row = [
            _labelled(front_annotated, front_label),
            _labelled(wrist_annotated, wrist_label),
            _labelled(bird, "birdview"),
            _labelled(patch_texture, "patch texture"),
        ]
        _write_contact_sheet(row, contact_path)
        rows.append(row)
        candidates.append(
            {
                "kind": spec["kind"],
                "offset": spec.get("offset", ""),
                "position": spec.get("position", None),
                "slot_name": spec.get("slot_name", ""),
                "slot_meta": spec.get("slot_meta", {}),
                "directory": str(candidate_dir),
                "front": str(front_path),
                "front_annotated": str(front_annotated_path),
                "wrist": str(wrist_path),
                "wrist_annotated": str(wrist_annotated_path),
                "birdview": str(bird_path),
                "contact": str(contact_path),
                "geometry": str(geometry_path),
                "patch_texture": str(patch_texture_path),
                "front_diff_score": front_diff_score,
                "wrist_diff_score": wrist_diff_score,
                "wrist_marker_score": wrist_score,
                "patch_geometry": geometry,
                "world_patch_info": world_patch_info,
            }
        )

    row_width = max(sum(img.width for img in row) for row in rows)
    row_height = max(max(img.height for img in row) for row in rows)
    sweep = Image.new("RGB", (row_width, row_height * len(rows)), "white")
    y = 0
    for row in rows:
        x = 0
        for img in row:
            sweep.paste(img, (x, y))
            x += img.width
        y += row_height
    sweep_path = output_dir / "sweep_contact.png"
    sweep.save(sweep_path)

    first = candidates[0]
    scored = [
        candidate
        for candidate in candidates
        if candidate["wrist_marker_score"]["center_distance_px"] is not None
    ]
    best_candidate = min(
        scored,
        key=lambda candidate: candidate["wrist_marker_score"]["center_distance_px"],
        default=None,
    )
    payload = {
        "suite": args.suite,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task_description": task_description,
        "patch_path": patch_path,
        "patch_texture": first["patch_texture"],
        "front": first["front"],
        "wrist": first["wrist"],
        "birdview": first["birdview"],
        "contact": first["contact"],
        "sweep_contact": str(sweep_path),
        "candidates": candidates,
        "object_slot_table_z": table_z,
        "object_slot_points": object_points,
        "world_patch_anchor_site": os.environ.get("WHEN_WORLD_PATCH_ANCHOR_SITE", "table_top"),
        "world_patch_anchor_mode": os.environ.get("WHEN_WORLD_PATCH_ANCHOR_MODE", args.anchor_mode),
        "world_patch_offsets": offsets,
        "world_patch_positions": [candidate.get("position") for candidate in candidates if candidate.get("position")],
        "object_slots": bool(args.object_slots),
        "camera_slots": bool(args.camera_slots),
        "camera_slot_info": camera_slot_info,
        "z_values": (
            _parse_z_values(args.z_values, table_z - 0.10 if table_z is not None else 0.68)
            if args.object_slots or args.camera_slots
            else []
        ),
        "world_patch_size": os.environ.get("WHEN_WORLD_PATCH_SIZE", args.size),
        "world_patch_quat": os.environ.get("WHEN_WORLD_PATCH_QUAT", "1 0 0 0"),
        "world_patch_table_z": os.environ.get("WHEN_WORLD_PATCH_TABLE_Z", ""),
        "world_patch_z": os.environ.get("WHEN_WORLD_PATCH_Z", ""),
        "marker_patch": bool(args.marker_patch),
        "best_wrist_center_candidate": best_candidate,
    }
    (output_dir / "visibility_check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
