#!/usr/bin/env python3
"""汇总 R/M 多视角实验结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "exp_id",
    "timestamp",
    "model_name",
    "input_views",
    "patch_train_views",
    "attack_views",
    "patch_type",
    "train_patch",
    "use_pretrained_patch",
    "patch_family",
    "patch_size",
    "task_success_rate",
    "attack_success_rate",
    "action_l2_mean",
    "position_error_mean",
    "rotation_error_mean",
    "gripper_error_mean",
    "trajectory_deviation_mean",
    "view_disagreement_mean",
    "avg_latency_ms",
    "gpu_id",
    "gpu_memory_peak_mb",
    "output_dir",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment metrics")
    parser.add_argument("--output_root", default="outputs/multiview_attack", type=str)
    return parser.parse_args()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        metrics = read_json(metrics_path)
        if not metrics:
            continue
        row = {field: metrics.get(field, "") for field in FIELDS}
        if not row["output_dir"]:
            row["output_dir"] = str(metrics_path.parent)
        manifest = read_json(metrics_path.parent / "patch_manifest.json") or {}
        if manifest.get("patch_family") and not row["patch_family"]:
            row["patch_family"] = manifest["patch_family"]
        rows.append(row)

    summary_path = output_root / "summary_all_experiments.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    final_dir = output_root / "final_analysis"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_summary = final_dir / "final_summary.md"
    completed = [row for row in rows if str(row.get("task_success_rate", "")).strip()]
    lines = [
        "# R/M 实验汇总",
        "",
        f"- summary CSV: `{summary_path}`",
        f"- metrics rows: {len(rows)}",
        f"- rows with task_success_rate: {len(completed)}",
        "- 空指标表示 dry-run 或真实评估尚未回填，不代表实验结论。",
    ]
    final_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_csv={summary_path}")
    print(f"final_summary={final_summary}")
    print(f"num_rows={len(rows)}")


if __name__ == "__main__":
    main()
