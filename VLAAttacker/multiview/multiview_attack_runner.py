#!/usr/bin/env python3
"""OpenVLA-OFT multi-view experiment runner for M-series experiments."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.openvla_oft_experiments import run_libero_rollout, train_oft_patch, write_json
from VLAAttacker.multiview.patch_manifest import (
    find_latest_patch,
    load_patch_from_manifest,
    official_m_dependency_error,
)
from VLAAttacker.multiview.view_batch_utils import parse_bool, parse_views, read_simple_yaml


PIXEL_GRAD_BLOCK_MESSAGE = "OpenVLA-OFT differentiable patch training probe is not ready."
PREDICT_ACTION_BLOCK_MESSAGE = (
    "OpenVLA-OFT predict_action eval is not ready. Please run/fix probe_openvla_oft_forward.py first."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reserved runner for multi-view clean/eval experiments")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--mode", choices=["auto", "clean_eval", "patch_eval", "train_patch", "config_only"], default="auto")
    parser.add_argument("--pretrained_patch_manifest", default="", type=str)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_mode(config: dict[str, str], mode: str) -> str:
    if mode != "auto":
        return mode
    if parse_bool(config.get("train_patch", "false")):
        return "train_patch"
    if parse_bool(config.get("use_pretrained_patch", "false")) or parse_bool(config.get("use_synthetic_patch", "false")):
        return "patch_eval"
    return "clean_eval"


def probe_status_path(config: dict[str, str]) -> Path:
    output_root = Path(config.get("probe_output_root") or config.get("output_root", "outputs/multiview_attack"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root / "probe_openvla_oft_forward" / "status.json"


def load_probe_status(config: dict[str, str]) -> tuple[Path, dict[str, Any] | None]:
    path = probe_status_path(config)
    if not path.exists():
        return path, None
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"OpenVLA-OFT probe status.json 无法解析: {path}: {exc}") from exc


def require_openvla_oft_probe(config: dict[str, str], mode: str, dry_run: bool) -> None:
    if config.get("model_name", config.get("model_family", "")) not in {"OpenVLA-OFT", "OpenVLA-OFT_w"}:
        return
    path, status = load_probe_status(config)
    if dry_run:
        print(f"[multiview-runner] dry-run: OpenVLA-OFT probe status not required; expected={path}")
        return
    if status is None:
        raise SystemExit("Please run probe_openvla_oft_forward.py first.")

    print(f"[multiview-runner] probe_status={path}")
    if mode in {"clean_eval", "patch_eval"} and not status.get("supports_predict_action", False):
        raise SystemExit(PREDICT_ACTION_BLOCK_MESSAGE)
    if mode == "train_patch" and not status.get("supports_pixel_values_grad_probe", False):
        raise SystemExit(PIXEL_GRAD_BLOCK_MESSAGE)
    if mode == "train_patch" and not status.get("dataloader_probe_success", False):
        raise SystemExit(
            "OpenVLA-OFT patch training requires a successful dataloader probe. "
            "Run probe_openvla_oft_forward.py with --use_dataloader --require_dataloader first."
        )


def run_clean_eval(config: dict[str, str], output_dir: Path) -> dict[str, Any]:
    return run_libero_rollout(config, output_dir, patch=None, attack_views=[], mode="clean_eval")


def build_synthetic_patch(config: dict[str, str], output_dir: Path) -> Any:
    """Build a non-adversarial RGB patch for black/noise control rollouts."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Synthetic patch controls require torch: {exc}") from exc

    shape_text = str(config.get("synthetic_patch_shape") or config.get("wrapper_patch_size") or "3,50,50")
    shape = [int(part.strip()) for part in shape_text.split(",") if part.strip()]
    if len(shape) != 3 or shape[0] != 3 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(f"synthetic_patch_shape/wrapper_patch_size must be 3,H,W, got {shape_text!r}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synthetic patch controls; refusing to run on CPU.")
    patch_kind = str(config.get("synthetic_patch_type") or "black").strip().lower()
    device = "cuda:0"
    if patch_kind in {"black", "black_block"}:
        patch = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    elif patch_kind in {"random", "random_noise", "noise"}:
        seed = int(str(config.get("synthetic_patch_seed") or config.get("seed") or "0"))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        patch = torch.rand(tuple(shape), dtype=torch.float32, generator=generator).to(device=device)
    else:
        raise ValueError(f"Unsupported synthetic_patch_type={patch_kind!r}; expected black or random_noise.")

    patch_dir = output_dir / "patch_checkpoint"
    patch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(patch.detach().cpu(), patch_dir / "patch.pt")
    return patch


def run_patch_eval(config: dict[str, str], output_dir: Path, manifest_path: str) -> dict[str, Any]:
    if parse_bool(config.get("use_synthetic_patch", "false")):
        patch = build_synthetic_patch(config, output_dir)
    else:
        patch = load_patch_from_manifest(manifest_path, dry_run=False)
    attack_views = parse_views(config.get("attack_views", ""))
    return run_libero_rollout(config, output_dir, patch=patch, attack_views=attack_views, mode="patch_eval")


def run_train_then_eval(config: dict[str, str], output_dir: Path) -> dict[str, Any]:
    train_metrics = train_oft_patch(config, output_dir)
    patch_path = output_dir / "last" / "patch.pt"
    if not patch_path.exists():
        raise SystemExit(f"OpenVLA-OFT patch training finished but patch is missing: {patch_path}")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Loading trained patch requires torch: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for train+eval patch rollout; refusing to load patch on CPU. "
            "Check Slurm --gres/CUDA_VISIBLE_DEVICES."
        )
    patch = torch.load(patch_path, map_location="cuda:0")
    rollout_metrics = run_libero_rollout(
        config,
        output_dir,
        patch=patch,
        attack_views=parse_views(config.get("attack_views", "")),
        mode="train_patch_eval",
    )
    summary = {"train": train_metrics, "rollout": rollout_metrics}
    write_json(output_dir / "openvla_oft_run_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_simple_yaml(Path(args.config))
    mode = resolve_mode(config, args.mode)
    print(f"[multiview-runner] mode={mode}")
    print(f"[multiview-runner] config={args.config}")
    print(f"[multiview-runner] output_dir={output_dir}")

    manifest_path = (
        args.pretrained_patch_manifest
        or config.get("pretrained_patch_manifest", "")
        or config.get("source_patch_manifest", "")
    )
    use_synthetic_patch = parse_bool(config.get("use_synthetic_patch", "false"))
    if mode == "patch_eval" and not manifest_path and not use_synthetic_patch:
        source_exp_id = config.get("pretrained_patch_source_exp_id", "")
        source_root = (
            config.get("pretrained_patch_output_root", "")
            or config.get("patch_source_output_root", "")
            or config.get("output_root", "outputs/multiview_attack")
        )
        found = find_latest_patch(source_exp_id, source_root)
        if found:
            manifest_path = found["manifest_path"]
        elif source_exp_id in {"M1", "M2", "M5"}:
            message = official_m_dependency_error(config.get("exp_id", ""), source_exp_id)
            if args.dry_run:
                print(f"[multiview-runner] dry-run: {message}")
            else:
                raise SystemExit(message)
    if manifest_path:
        print(f"[multiview-runner] pretrained_patch_manifest={manifest_path}")
        if args.dry_run:
            print("[multiview-runner] dry-run: skip reading patch manifest.")
        else:
            print(f"[multiview-runner] patch_info={load_patch_from_manifest(manifest_path, dry_run=True)}")

    require_openvla_oft_probe(config, mode, args.dry_run)

    if args.dry_run:
        print("[multiview-runner] dry-run only; 不加载模型、不跑仿真/训练。")
        return
    if mode == "patch_eval" and not manifest_path and not use_synthetic_patch:
        raise SystemExit("run 模式找不到依赖 patch manifest，停止。")
    if mode == "clean_eval":
        metrics = run_clean_eval(config, output_dir)
        print(f"[multiview-runner] clean rollout finished: {metrics}")
        return
    if mode == "patch_eval":
        metrics = run_patch_eval(config, output_dir, manifest_path)
        print(f"[multiview-runner] patch rollout finished: {metrics}")
        return
    if mode == "train_patch":
        metrics = run_train_then_eval(config, output_dir)
        print(f"[multiview-runner] train+eval finished: {metrics}")
        return
    print("[multiview-runner] config_only; nothing to run.")


if __name__ == "__main__":
    main()
