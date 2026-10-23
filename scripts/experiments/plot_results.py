#!/usr/bin/env python3
"""为汇总结果生成基础图表。

没有真实数值或 matplotlib 不可用时，只写说明文件，不伪造结果。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS = [
    ("task_success_rate", "success_rate_comparison.png"),
    ("action_l2_mean", "action_error_comparison.png"),
    ("position_error_mean", "position_error_comparison.png"),
    ("trajectory_deviation_mean", "trajectory_deviation_comparison.png"),
    ("view_disagreement_mean", "view_disagreement_comparison.png"),
    ("avg_latency_ms", "latency_comparison.png"),
    ("gpu_memory_peak_mb", "gpu_usage_comparison.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot collected experiment metrics")
    parser.add_argument("--summary_csv", default="outputs/multiview_attack/summary_all_experiments.csv", type=str)
    parser.add_argument("--output_dir", default="outputs/multiview_attack/final_analysis", type=str)
    return parser.parse_args()


def to_float(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    if not summary_csv.is_absolute():
        summary_csv = PROJECT_ROOT / summary_csv
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_csv.exists():
        warning = f"warning: 找不到 summary_csv: {summary_csv}\n"
        (output_dir / "plot_notes.txt").write_text(warning, encoding="utf-8")
        print(warning.strip())
        return

    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "plot_notes.txt").write_text(f"matplotlib 不可用: {exc}\n", encoding="utf-8")
        print(f"plot_notes={output_dir / 'plot_notes.txt'}")
        return

    notes = []
    for metric, filename in PLOTS:
        values = [(row.get("exp_id", ""), to_float(row.get(metric, ""))) for row in rows]
        values = [(exp_id, value) for exp_id, value in values if value is not None]
        if not values:
            notes.append(f"{metric}: 暂无真实数值，未生成图。")
            continue
        labels = [item[0] for item in values]
        numbers = [item[1] for item in values]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        ax.bar(range(len(numbers)), numbers)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)
    (output_dir / "plot_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"plot_dir={output_dir}")


if __name__ == "__main__":
    main()
