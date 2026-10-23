#!/usr/bin/env python3
"""Show compact progress bars for M-series OpenVLA-OFT patch training jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPS = ["M1", "M2", "M5"]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return data
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def latest_output_dir(output_root: Path, exp_id: str) -> Path | None:
    candidates = [path for path in output_root.glob(f"*_{exp_id}") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def bar(done: int, total: int, width: int) -> str:
    total = max(1, total)
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def progress_from_dir(path: Path) -> dict[str, Any]:
    config = read_simple_yaml(path / "config.yaml")
    total = int(config.get("max_train_iters", "2000") or 2000)
    status = read_json(path / "status.json")
    progress = read_json(path / "train_progress.json")
    train_status = read_json(path / "openvla_oft_train_status.json")
    train_metrics = read_json(path / "train_metrics.json")

    source = "none"
    row: dict[str, Any] = {}
    if progress:
        source = "train_progress.json"
        row = progress
        done = int(row.get("completed_steps", 0) or 0)
    elif train_metrics:
        source = "train_metrics.json"
        row = train_metrics
        done = int(row.get("train_steps", total) or total)
    elif train_status:
        source = "openvla_oft_train_status.json"
        row = train_status
        done = int(row.get("step", -1) or 0) + 1
    else:
        done = 0

    return {
        "exp_id": config.get("exp_id", path.name.rsplit("_", 1)[-1]),
        "stage": status.get("stage", "unknown"),
        "done": done,
        "total": total,
        "percent": 100.0 * done / max(1, total),
        "loss": row.get("model_loss", row.get("final_model_loss", "")),
        "grad": row.get("patch_grad_abs_mean", row.get("final_patch_grad_abs_mean", "")),
        "views": row.get("patch_views", ",".join(row.get("patch_views", [])) if isinstance(row.get("patch_views"), list) else ""),
        "source": source,
        "path": path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize M-series OpenVLA-OFT training progress")
    parser.add_argument("--output_root", default="outputs/multiview_attack")
    parser.add_argument("--exp", nargs="*", default=DEFAULT_EXPS, help="Experiment ids, e.g. M1 M2 M5")
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    rows = []
    for exp_id in args.exp:
        path = latest_output_dir(output_root, exp_id)
        if path is None:
            rows.append({"exp_id": exp_id, "missing": True})
            continue
        rows.append(progress_from_dir(path))

    for row in rows:
        if row.get("missing"):
            print(f"{row['exp_id']}  [missing latest output dir]")
            continue
        loss = row["loss"]
        grad = row["grad"]
        loss_text = f" loss={float(loss):.4f}" if isinstance(loss, (int, float)) else ""
        grad_text = f" grad={float(grad):.6f}" if isinstance(grad, (int, float)) else ""
        print(
            f"{row['exp_id']} {bar(row['done'], row['total'], args.width)} "
            f"{row['done']}/{row['total']} ({row['percent']:.1f}%) "
            f"stage={row['stage']} views={row['views']}{loss_text}{grad_text}"
        )
        print(f"    source={row['source']} dir={row['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
