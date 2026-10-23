"""RLDS batch 结构检查工具。

该模块只提供轻量函数；真正加载 RLDS 数据集仍由现有 dataloader 完成。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def summarize_observation_keys(batch: dict[str, Any]) -> dict[str, Any]:
    """提取 batch/observation 中可能的图像字段。"""
    observation = batch.get("observation", batch)
    keys = sorted(str(key) for key in observation.keys()) if isinstance(observation, dict) else []
    image_keys = [key for key in keys if key.startswith("image")]
    return {
        "keys": keys,
        "image_keys": image_keys,
        "has_image_primary": "image_primary" in image_keys,
        "has_image_wrist": "image_wrist" in image_keys,
        "has_image_secondary": "image_secondary" in image_keys,
    }


def format_batch_summary(batch: dict[str, Any]) -> str:
    """格式化 batch 字段摘要。"""
    summary = summarize_observation_keys(batch)
    return (
        f"keys={summary['keys']}; image_keys={summary['image_keys']}; "
        f"primary={summary['has_image_primary']}; wrist={summary['has_image_wrist']}; "
        f"secondary={summary['has_image_secondary']}"
    )


def check_paths(data_root: str, model_checkpoint: str, libero_root: str) -> dict[str, Any]:
    """检查数据管线关键路径是否存在，不导入 torch/tensorflow。"""
    dataset_dir = Path(data_root) / "openvla-main" / "dataset" / "libero_spatial_no_noops" / "1.0.0"
    paths = {
        "data_root": Path(data_root),
        "libero_spatial_dataset": dataset_dir,
        "model_checkpoint": Path(model_checkpoint),
        "libero_root": Path(libero_root),
    }
    return {name: {"path": str(path), "exists": path.exists()} for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight multi-view dataloader checks")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--libero_root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, item in check_paths(args.data_root, args.model_checkpoint, args.libero_root).items():
        print(f"{name}: exists={item['exists']} path={item['path']}")


if __name__ == "__main__":
    main()
