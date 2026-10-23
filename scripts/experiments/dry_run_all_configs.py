#!/usr/bin/env python3
"""顺序 dry-run 所有 R/M 配置。

不提交 sbatch，不加载大模型，不训练。
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ORDER = [
    "R0_openvla_front_clean.yaml",
    "R1_openvla_front_attack.yaml",
    "M0_oft_front_wrist_clean.yaml",
    "M1_oft_train_front_attack_front.yaml",
    "M2_oft_train_wrist_attack_wrist.yaml",
    "M3_oft_front_patch_attack_wrist.yaml",
    "M4_oft_wrist_patch_attack_front.yaml",
    "M5_oft_train_multiview_attack_front.yaml",
    "M6_oft_multiview_patch_attack_wrist.yaml",
    "M7_oft_multiview_patch_attack_both.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run all experiment configs")
    parser.add_argument("--config_dir", default="configs/experiments", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    output_root = PROJECT_ROOT / "outputs" / "multiview_attack"
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / f"dry_run_all_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    lines = ["# R/M dry-run 汇总", ""]
    failures = []
    for config_name in CONFIG_ORDER:
        config_path = config_dir / config_name
        command = [
            "python3",
            "scripts/experiments/run_experiment.py",
            "--config",
            str(config_path),
            "--dry_run",
            "--gpu_id",
            args.gpu_id,
        ]
        lines.append(f"## {config_name}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(command))
        lines.append("```")
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
        lines.append(f"- returncode: {result.returncode}")
        if result.stdout:
            lines.append("- stdout:")
            lines.append("```text")
            lines.append(result.stdout.strip())
            lines.append("```")
        if result.stderr:
            lines.append("- stderr:")
            lines.append("```text")
            lines.append(result.stderr.strip())
            lines.append("```")
        lines.append("")
        if result.returncode != 0:
            failures.append(config_name)
            if not args.continue_on_error:
                break

    if failures:
        lines.append("失败配置: " + ", ".join(failures))
    else:
        lines.append("全部 dry-run 通过。")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dry_run_all_log={log_path}")
    if failures:
        raise SystemExit("dry-run 失败: " + ", ".join(failures))


if __name__ == "__main__":
    main()
