#!/usr/bin/env python3
"""检查单个 R/M 实验配置。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.view_batch_utils import is_todo, parse_bool, read_simple_yaml, validate_view_sets


REQUIRED_FIELDS = [
    "exp_id",
    "series",
    "model_name",
    "model_checkpoint",
    "input_adapter",
    "input_views",
    "patch_train_views",
    "attack_views",
    "attack_method",
    "train_patch",
    "use_pretrained_patch",
    "pretrained_patch_source_exp_id",
    "pretrained_patch_family",
    "patch_family",
    "baseline_exp_id",
    "patch_size",
    "task_suite",
    "num_episodes",
    "seed",
    "output_root",
    "project_root",
    "data_root",
    "libero_root",
    "conda_env",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check one experiment config")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--strict", action="store_true", help="TODO 路径也视为失败")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = read_simple_yaml(path)

    issues = []
    for field in REQUIRED_FIELDS:
        if field not in config:
            issues.append(f"缺少字段: {field}")
    issues.extend(validate_view_sets(config))

    if parse_bool(config.get("use_pretrained_patch", "false")) and is_todo(config.get("pretrained_patch_family")):
        issues.append("use_pretrained_patch=true 但 pretrained_patch_family 未设置")
    if parse_bool(config.get("use_pretrained_patch", "false")) and is_todo(config.get("pretrained_patch_source_exp_id")):
        issues.append("use_pretrained_patch=true 但 pretrained_patch_source_exp_id 未设置")
    if args.strict:
        for field in ["model_checkpoint", "data_root", "project_root", "libero_root", "conda_env", "view_key_map"]:
            if is_todo(config.get(field)):
                issues.append(f"{field} 仍是 TODO/LOCAL_PATHS")

    print(f"config={path}")
    print(f"exp_id={config.get('exp_id', '')}")
    print(f"model_name={config.get('model_name', '')}")
    print(f"input_views={config.get('input_views', '')}")
    print(f"patch_train_views={config.get('patch_train_views', '')}")
    print(f"attack_views={config.get('attack_views', '')}")
    print(f"train_patch={config.get('train_patch', '')}")
    print(f"use_pretrained_patch={config.get('use_pretrained_patch', '')}")
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("status=ok")


if __name__ == "__main__":
    main()
