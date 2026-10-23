"""多视角实验参数的轻量级工具。

当前实现已接入最小多视角数据流: 视角加载、拼接输入和按视角贴 patch。
真实字段名和融合方式仍需要通过实验环境继续确认。
"""

import argparse


def resolve_vla_path(dataset: str) -> str:
    """根据现有 wrapper 规则解析 OpenVLA checkpoint 名称。"""
    if "bridge_orig" in dataset:
        return "openvla/openvla-7b"
    if "libero_spatial" in dataset:
        return "openvla/openvla-7b-finetuned-libero-spatial"
    if "libero_object" in dataset:
        return "openvla/openvla-7b-finetuned-libero-object"
    if "libero_goal" in dataset:
        return "openvla/openvla-7b-finetuned-libero-goal"
    if "libero_10" in dataset:
        return "openvla/openvla-7b-finetuned-libero-10"
    raise ValueError(f"Invalid dataset: {dataset}")


def parse_views(value: str):
    """解析逗号分隔视角名，保留用户输入的命名。"""
    return [view.strip() for view in value.split(",") if view.strip()]


def add_multiview_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向现有 wrapper 增加多视角预留参数。"""
    parser.add_argument("--input_views", default="front", type=str, help="预留: VLA 输入视角, 如 front 或 front,wrist")
    parser.add_argument("--patch_train_views", default="front", type=str, help="预留: patch 训练视角")
    parser.add_argument("--attack_views", default="front", type=str, help="预留: 实际贴 patch 的视角")
    parser.add_argument("--patch_type", default="single_view", type=str, help="预留: single_view/cross_view/multi_view")
    parser.add_argument("--multiview_fusion", default="stitch", type=str, help="多视角输入融合方式; 当前最小实现为 stitch")
    parser.add_argument("--exp_id", default="", type=str, help="预留: 外部指定实验编号")
    parser.add_argument("--dry_run", action="store_true", help="只打印配置, 不加载模型, 不构造 dataloader, 不训练")
    parser.add_argument("--multiview_debug", action="store_true", help="打印 batch 中 image/camera/view 字段结构")
    parser.add_argument("--view_key_map", default="", type=str, help="预留: 视角到 batch key 的映射, 如 front=image_primary,wrist=image_wrist")
    parser.add_argument("--output_dir", default="", type=str, help="可选: 统一实验输出目录, 为空时使用原有 run 路径")
    return parser


def print_dry_run_config(args, method_name: str, vla_path: str, run_path: str) -> None:
    """打印 dry-run 配置摘要。"""
    print("[dry-run] 不会加载 OpenVLA 模型, 不会构造 dataloader, 不会启动训练。")
    print(f"[dry-run] method: {method_name}")
    print(f"[dry-run] exp_id: {args.exp_id if args.exp_id else 'auto-uuid'}")
    print(f"[dry-run] dataset: {args.dataset}")
    print(f"[dry-run] vla_path: {vla_path}")
    print(f"[dry-run] run_path: {run_path}")
    print(f"[dry-run] output_dir: {args.output_dir if args.output_dir else '使用默认 run 路径'}")
    print(f"[dry-run] maskidx: {args.maskidx}")
    print(f"[dry-run] lr: {args.lr}")
    print(f"[dry-run] iter: {args.iter}")
    print(f"[dry-run] batch_size: {args.bs}")
    print(f"[dry-run] patch_size: {args.patch_size}")
    print(f"[dry-run] geometry: {args.geometry}")
    print(f"[dry-run] input_views: {parse_views(args.input_views)}")
    print(f"[dry-run] patch_train_views: {parse_views(args.patch_train_views)}")
    print(f"[dry-run] attack_views: {parse_views(args.attack_views)}")
    print(f"[dry-run] patch_type: {args.patch_type}")
    print(f"[dry-run] multiview_fusion: {args.multiview_fusion}")
    print(f"[dry-run] view_key_map: {args.view_key_map if args.view_key_map else '待确认'}")
    print(f"[dry-run] multiview_debug: {args.multiview_debug}")
    print("[dry-run] 多视角最小数据流已接入; 字段名和融合方式仍需通过 multiview_debug/正式小样本确认。")
