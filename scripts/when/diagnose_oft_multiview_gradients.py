#!/usr/bin/env python3
"""Diagnose front/wrist patch-gradient conflict for OpenVLA-OFT-libero10."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = "configs/experiments/oft_libero10_gradconf.yaml"
DEFAULT_RUNS = {
    "static": "G0_random_static",
    "dynamic": "G1_dynamic_when_naive",
}
LIBERO10_ALIASES = {
    "libero10",
    "libero_10",
    "libero-10",
    "LIBERO-10",
    "libero_long",
    "LIBERO-Long",
    "libero-long",
    "LIBERO_LONG",
}


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(char in text for char in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _minimal_yaml_load(path: Path) -> dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    processed: list[tuple[int, str, int]] = []
    for line_no, raw in enumerate(raw_lines):
        line = _strip_inline_comment(raw).rstrip()
        if not line.strip():
            continue
        processed.append((len(line) - len(line.lstrip(" ")), line.strip(), line_no))

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, text, _line_no) in enumerate(processed):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if text.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"YAML list item found under non-list container: {text}")
            container.append(_parse_scalar(text[2:]))
            continue
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            if not isinstance(container, dict):
                raise ValueError(f"YAML key found under non-dict container: {text}")
            container[key] = _parse_scalar(value)
            continue
        next_text = processed[index + 1][1] if index + 1 < len(processed) else ""
        child: Any = [] if next_text.startswith("- ") else {}
        if not isinstance(container, dict):
            raise ValueError(f"YAML nested key found under non-dict container: {text}")
        container[key] = child
        stack.append((indent, child))
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")
        return data
    except ModuleNotFoundError:
        return _minimal_yaml_load(config_path)


def cfg_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", " ").split() if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _compact_alias(value: str) -> str:
    text = Path(str(value)).name
    if text.endswith("_no_noops"):
        text = text[: -len("_no_noops")]
    return "".join(char for char in text.lower() if char.isalnum())


def _canonical_libero10_name(value: str, default: str = "libero_10") -> str:
    text = str(value or default).strip()
    aliases = {"".join(char for char in item.lower() if char.isalnum()) for item in LIBERO10_ALIASES}
    return "libero_10" if _compact_alias(text) in aliases else text


def _load_data_checker() -> Any:
    path = PROJECT_ROOT / "scripts" / "data" / "check_libero10_oft.py"
    spec = importlib.util.spec_from_file_location("check_libero10_oft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import data checker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_data(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_root = args.data_root or str(cfg_get(config, "data_root", "/home/caijiasheng25/data"))
    search_roots = args.search_roots if args.search_roots else _as_list(cfg_get(config, "search_roots", []))
    dataset_name = _canonical_libero10_name(str(cfg_get(config, "dataset_name", "libero_10")))
    checker = _load_data_checker()
    result = checker.discover_libero10(
        data_root=data_root,
        dataset_name=dataset_name,
        search_roots=search_roots,
        dry_run=True,
    )
    if not result.get("selected"):
        for error in result.get("errors", []):
            print(f"[gradconf][data][error] {error}", file=sys.stderr)
        raise SystemExit(2)
    field_check = result.get("field_check") or {}
    if field_check.get("status") == "missing_required_fields":
        print("[gradconf][data][error] libero_10 / LIBERO-Long candidate is missing required fields.", file=sys.stderr)
        print(json.dumps(field_check, indent=2, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
    if field_check.get("status") == "unknown":
        print("[gradconf][data][warn] Data fields could not be confirmed from metadata; runtime dataloader will verify them.")
    return result


def infer_dataloader_server_and_dataset(selected_path: str, requested_dataset: str) -> tuple[str, str]:
    path = Path(selected_path)
    if path.name == "1.0.0":
        path = path.parent
    dataset_dir = path
    dataset_name = dataset_dir.name
    if dataset_name == "dataset" or dataset_name == "openvla-main":
        dataset_name = requested_dataset

    if dataset_dir.parent.name == "dataset" and dataset_dir.parent.parent.name == "openvla-main":
        server = dataset_dir.parent.parent.parent
    elif dataset_dir.parent.name == "dataset":
        server = dataset_dir.parent.parent
    else:
        server = dataset_dir

    if _canonical_libero10_name(dataset_name) == "libero_10":
        dataset_arg = "libero_10"
    else:
        dataset_arg = _canonical_libero10_name(requested_dataset)
    return str(server), dataset_arg


def output_dir_for(args: argparse.Namespace, config: dict[str, Any], run_name: str) -> Path:
    output_root = args.output_root or str(cfg_get(config, "output_root", "outputs/when_oft_gradconf"))
    output_stage = args.output_stage or str(cfg_get(config, "output_stage", "gradconf_oft_multiview"))
    suite_name = _canonical_libero10_name(args.suite_name or str(cfg_get(config, "suite_name", "libero_10")))
    root = resolve_project_path(output_root)
    if args.validation:
        return root / "_validation" / run_name
    return root / output_stage / suite_name / run_name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def config_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator.strip()) / float(denominator.strip())
        except (TypeError, ValueError, ZeroDivisionError):
            return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def configure_when_runtime(config: dict[str, Any], torch_module: Any) -> None:
    """Match the CUDA/seed controls used by scripts/when/train_sim_patch.py."""
    seed = int(cfg_get(config, "train.seed", cfg_get(config, "patch.seed", 42)) or 42)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)

    allow_tf32 = config_bool(cfg_get(config, "train.allow_tf32", True), True)
    if hasattr(torch_module.backends, "cuda"):
        torch_module.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch_module.backends.cudnn.allow_tf32 = allow_tf32
    torch_module.backends.cudnn.deterministic = config_bool(
        cfg_get(config, "train.cudnn_deterministic", True),
        True,
    )
    torch_module.backends.cudnn.benchmark = config_bool(
        cfg_get(config, "train.cudnn_benchmark", False),
        False,
    )
    if config_bool(cfg_get(config, "train.disable_cudnn", True), True):
        torch_module.backends.cudnn.enabled = False
    try:
        torch_module.set_float32_matmul_precision(str(cfg_get(config, "train.float32_matmul_precision", "high")))
    except Exception:
        pass


def build_when_scheduler(optimizer: Any, *, warmup_steps: int, total_steps: int) -> Any:
    try:
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=max(0, int(warmup_steps)),
            num_training_steps=max(1, int(total_steps)),
            num_cycles=0.5,
            last_epoch=-1,
        )
    except Exception:
        import torch

        warmup = max(0, int(warmup_steps))
        total = max(1, int(total_steps))

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--search_roots", nargs="*", default=[])
    parser.add_argument("--output_root", default="")
    parser.add_argument("--output_stage", default="")
    parser.add_argument("--suite_name", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True)
    parser.add_argument("--num_batches", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--cleanup_validation_on_success", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    suite_name = _canonical_libero10_name(args.suite_name or str(cfg_get(config, "suite_name", "libero_10")))
    dataset_name = _canonical_libero10_name(str(cfg_get(config, "dataset_name", "libero_10")))

    run_name = args.run_name or DEFAULT_RUNS[args.mode]
    run_dir = output_dir_for(args, config, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "official_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_log = run_dir / "official_eval.out"
    text_log = log_dir / f"{run_name}.txt"

    data_result = discover_data(args, config)
    selected_data_path = data_result["selected"]["path"]
    dataloader_server, dataloader_dataset = infer_dataloader_server_and_dataset(selected_data_path, dataset_name)
    model_path = args.model_path or str(cfg_get(config, "model_path", ""))
    if not model_path:
        raise SystemExit("--model_path or config model_path is required.")

    batch_size = int(args.batch_size or cfg_get(config, "train.batch_size", 1))
    if args.num_batches > 0:
        num_batches = int(args.num_batches)
    elif args.mode == "static":
        num_batches = int(cfg_get(config, "train.num_batches_static_full", 100))
    else:
        num_batches = int(cfg_get(config, "train.num_steps_dynamic_full", 500))
    if args.dry_run:
        num_batches = min(num_batches, int(cfg_get(config, "train.num_batches_debug", 5)))

    resolved_config = {
        "experiment_name": cfg_get(config, "experiment_name", "oft_libero10_gradconf"),
        "dataset_name": dataset_name,
        "dataset_semantic_name": cfg_get(config, "dataset_semantic_name", "LIBERO-Long"),
        "suite_name": suite_name,
        "mode": args.mode,
        "run_name": run_name,
        "model_family": cfg_get(config, "model_family", "openvla_oft"),
        "model_variant": cfg_get(config, "model_variant", "oft_libero10"),
        "model_path": model_path,
        "patch_source": cfg_get(config, "patch.source", "random"),
        "patch_size": cfg_get(config, "patch.size", [3, 50, 50]),
        "patch_position": cfg_get(config, "patch.position", "upa_rfas_libero_10"),
        "patch_train_space": cfg_get(config, "patch.train_space", cfg_get(config, "patch.space", "pixel_values")),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "when_baseline_alignment": {
            "baseline_config": "configs/when/train/openvla_libero10_when.yaml",
            "optimizer": cfg_get(config, "train.optimizer", "adamw"),
            "learning_rate": cfg_get(config, "train.learning_rate", 1.0e-3),
            "batch_size": batch_size,
            "iterations_or_outer_steps": num_batches,
            "warmup_steps": cfg_get(config, "train.warmup_steps", 20),
            "inner_loop": cfg_get(config, "train.inner_loop", 50),
            "inner_steps": cfg_get(config, "train.inner_steps", 8),
            "inner_eps": cfg_get(config, "train.inner_eps", "2/255"),
            "gradient_merge": cfg_get(config, "train.gradient_merge", "naive_average"),
            "patch_train_space": cfg_get(config, "patch.train_space", cfg_get(config, "patch.space", "pixel_values")),
            "geometry": cfg_get(config, "train.geometry", True),
            "dual_rupa": cfg_get(config, "loss.use_rupa_inner_loop", True),
            "use_pad": cfg_get(config, "loss.use_pad", True),
            "use_psm": cfg_get(config, "loss.use_psm", True),
            "lambda_con": cfg_get(config, "loss.lambda_con", 10.0),
            "lambda_attn": cfg_get(config, "loss.lambda_attn", 1.0),
            "lambda_text": cfg_get(config, "loss.lambda_text", 0.5),
            "lambda_1": cfg_get(config, "loss.lambda_1", 0.1),
            "tau": cfg_get(config, "loss.tau", 0.3),
            "beta_neg": cfg_get(config, "loss.beta_neg", 0.5),
            "probe_phrases": cfg_get(config, "loss.probe_phrases", []),
            "seed": cfg_get(config, "train.seed", cfg_get(config, "patch.seed", 42)),
        },
        "dry_run": bool(args.dry_run),
        "validation": bool(args.validation),
        "output_dir": str(run_dir),
        "data_path_resolved": selected_data_path,
        "dataloader_server": dataloader_server,
        "dataloader_dataset": dataloader_dataset,
        "data_discovery": data_result,
    }
    write_json(run_dir / "gradconf_config.json", resolved_config)
    write_json(
        run_dir / "official_command.json",
        {
            "entrypoint": str(Path(__file__).resolve()),
            "cwd": str(PROJECT_ROOT),
            "argv": sys.argv,
            "note": "Gradient-conflict diagnostic; no official WHEN patch is loaded.",
            "resolved_config": resolved_config,
        },
    )
    eval_log.write_text(
        "OpenVLA-OFT multiview gradient-conflict diagnostic.\n"
        "This file is kept for WHEN-compatible output layout; it is not a rollout evaluation log.\n",
        encoding="utf-8",
    )
    text_log.write_text(
        f"[gradconf] started {datetime.now().isoformat(timespec='seconds')}\n"
        "[gradconf] WHEN baseline objective enabled: dual/RUPA outer perturbation, "
        "contrastive, attention, text, and feature terms.\n",
        encoding="utf-8",
    )

    summary = run_diagnostic(args, config, resolved_config, run_dir, text_log)
    write_json(run_dir / "summary.json", summary)
    with eval_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if args.validation and args.cleanup_validation_on_success:
        shutil.rmtree(run_dir)
        print(f"[gradconf][validation] success; removed {run_dir}")
    else:
        print(f"[gradconf] wrote {run_dir}")
    return 0


def run_diagnostic(
    args: argparse.Namespace,
    config: dict[str, Any],
    resolved: dict[str, Any],
    run_dir: Path,
    text_log: Path,
) -> dict[str, Any]:
    import torch

    configure_when_runtime(config, torch)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this OFT gradient-conflict job; refusing to run on CPU. "
            "Check Slurm --gres/CUDA_VISIBLE_DEVICES."
        )

    from VLAAttacker.multiview.openvla_oft_experiments import (
        _get_oft_num_visual_tokens,
        _select_patch_position,
        apply_patch_to_pixel_values,
        build_raw_patched_pixel_values,
    )
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper
    from VLAAttacker.white_patch.openvla_dataloader import get_dataloader

    if str(resolved["patch_source"]).lower() != "random":
        raise ValueError("Only random patch source is supported in this diagnostic stage.")

    patch_size = resolved["patch_size"]
    if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
        raise ValueError(f"patch.size must be [3,H,W], got {patch_size!r}")
    patch_c, patch_h, patch_w = [int(value) for value in patch_size]
    if patch_c != 3:
        raise ValueError(f"Only RGB patches are supported, got patch.size={patch_size!r}")

    use_proprio_requested = True
    wrapper_config = {
        "model_checkpoint": resolved["model_path"],
        "model_name": "OpenVLA-OFT",
        "model_family": "OpenVLA-OFT",
        "task_suite": resolved["suite_name"],
        "task_suite_name": resolved["suite_name"],
        "openvla_oft_root": str(cfg_get(config, "openvla_oft_root", "")),
        "libero_root": str(cfg_get(config, "libero_root", "")),
        "unnorm_key": f"{resolved['suite_name']}_no_noops",
        "input_views": "front,wrist",
        "view_key_map": "front=image_primary,wrist=image_wrist",
        "num_images_in_input": 2,
        "num_open_loop_steps": 8,
        "use_proprio": use_proprio_requested,
        "use_l1_regression": True,
        "center_crop": True,
        "disable_cudnn": bool(cfg_get(config, "train.disable_cudnn", True)),
        "torch_dtype": str(cfg_get(config, "train.precision", "bf16")),
    }
    wrapper = OpenVLAOFTWrapper(wrapper_config)
    wrapper.load_model()
    device = wrapper.device
    if device.type != "cuda":
        raise RuntimeError(f"OpenVLA-OFT wrapper resolved device={device}; refusing CPU execution.")
    torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    model_device = next(wrapper.vla.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"OpenVLA-OFT model parameters are on {model_device}; refusing CPU execution.")
    dtype = wrapper.dtype
    if wrapper.proprio_projector is None:
        raise RuntimeError("OpenVLA-OFT wrapper did not load proprio_projector; refusing mismatched OFT diagnostics.")

    train_loader, _val_loader = get_dataloader(
        batch_size=int(resolved["batch_size"]),
        server=resolved["dataloader_server"],
        dataset=resolved["dataloader_dataset"],
        vla_path=resolved["model_path"],
        input_views="front,wrist",
        view_key_map="front=image_primary,wrist=image_wrist",
        multiview_fusion="stitch",
        use_proprio=True,
        num_images_in_input=2,
        action_chunk_len=8,
        return_raw_images=True,
        multiview_debug=False,
    )

    patch_seed = int(cfg_get(config, "patch.seed", 0) or 0)
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(patch_seed)
    patch = torch.rand(
        (patch_c, patch_h, patch_w),
        device=device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    try:
        from transformers import AdamW

        optimizer = AdamW([patch], lr=float(cfg_get(config, "train.learning_rate", 1.0e-3)))
    except Exception:
        optimizer = torch.optim.AdamW([patch], lr=float(cfg_get(config, "train.learning_rate", 1.0e-3)))
    accumulate_steps = max(1, int(cfg_get(config, "train.gradient_accumulation_steps", 1) or 1))
    scheduler = build_when_scheduler(
        optimizer,
        warmup_steps=int(cfg_get(config, "train.warmup_steps", 20) or 20),
        total_steps=max(1, int(resolved["num_batches"]) // accumulate_steps),
    )
    clamp_min = float(cfg_get(config, "patch.clamp_min", 0.0))
    clamp_max = float(cfg_get(config, "patch.clamp_max", 1.0))
    patch_position = str(resolved.get("patch_position") or cfg_get(config, "patch.position", "upa_rfas_libero_10"))
    patch_train_space = str(
        cfg_get(config, "patch.train_space", cfg_get(config, "patch.space", "pixel_values")) or "pixel_values"
    ).strip().lower()
    if patch_train_space in {"raw", "raw_image", "raw_rgb", "rgb"}:
        patch_train_space = "raw_rgb"
    elif patch_train_space in {"pixel", "pixel_value", "pixel_values", "processed"}:
        patch_train_space = "pixel_values"
    else:
        raise ValueError(
            f"Unsupported patch.train_space={patch_train_space!r}; expected pixel_values or raw_rgb."
        )
    raw_eval_image_size = int(cfg_get(config, "patch.raw_eval_image_size", 256) or 256)
    raw_match_eval_resize = config_bool(cfg_get(config, "patch.raw_match_eval_resize", True), True)
    raw_center_crop = config_bool(
        cfg_get(config, "patch.raw_center_crop", getattr(wrapper.cfg, "center_crop", True)),
        bool(getattr(wrapper.cfg, "center_crop", True)),
    )
    conflict_threshold = float(cfg_get(config, "diagnostic.conflict_threshold", 0.0))
    severe_threshold = float(cfg_get(config, "diagnostic.severe_conflict_threshold", -0.2))
    dominance_threshold = float(cfg_get(config, "diagnostic.dominance_ratio_threshold", 2.0))
    enable_interference = bool(cfg_get(config, "diagnostic.enable_interference_test", True))
    lambda_con = float(cfg_get(config, "loss.lambda_con", 1.0))
    lambda_attn = float(cfg_get(config, "loss.lambda_attn", 1.0))
    lambda_text = float(cfg_get(config, "loss.lambda_text", 0.5))
    lambda_1 = float(cfg_get(config, "loss.lambda_1", 0.1))
    lambda_nonpatch = float(cfg_get(config, "loss.lambda_nonpatch", 0.8))
    tau = float(cfg_get(config, "loss.tau", 0.3))
    beta_neg = float(cfg_get(config, "loss.beta_neg", 0.5))
    last_n = max(1, int(cfg_get(config, "loss.last_n", 2) or 2))
    topk_text = cfg_get(config, "loss.topk_text", 0.3)
    topk_text = None if topk_text is None else float(topk_text)
    margin = cfg_get(config, "loss.margin", 0.1)
    margin = None if margin is None else float(margin)
    probe_phrases = _as_list(
        cfg_get(
            config,
            "loss.probe_phrases",
            "put,pick up,place,turn on,push,open,close,left,right,bottom,back,middle,top,front",
        )
    )
    use_feature_loss = bool(cfg_get(config, "loss.use_feature_loss", True))
    use_contrastive_loss = bool(cfg_get(config, "loss.use_contrastive_loss", True))
    use_attention_loss = bool(cfg_get(config, "loss.use_pad", True))
    use_text_loss = bool(cfg_get(config, "loss.use_psm", True))
    use_rupa_inner_loop = bool(cfg_get(config, "loss.use_rupa_inner_loop", True))
    use_l2_feature = bool(cfg_get(config, "loss.use_l2_feature", False))
    ignore_index = int(cfg_get(config, "loss.ignore_index", -100) or -100)
    record_every = max(1, int(cfg_get(config, "train.record_every_steps", 10)))
    inner_loop_steps = max(1, int(cfg_get(config, "train.inner_loop", 1) or 1))
    inner_steps = max(1, int(cfg_get(config, "train.inner_steps", 8) or 8))
    inner_eps = config_float(cfg_get(config, "train.inner_eps", "2/255"), 2.0 / 255.0)
    gradient_merge_mode = str(cfg_get(config, "train.gradient_merge", "naive_average") or "naive_average")
    num_visual_tokens = _get_oft_num_visual_tokens(wrapper)
    input_views = ["front", "wrist"]
    view_token_count = max(1, int(num_visual_tokens) // len(input_views))
    view_token_side = int(math.sqrt(view_token_count))
    if view_token_side * view_token_side != view_token_count:
        raise ValueError(
            f"Expected square per-view visual token grid, got {view_token_count} tokens per view."
        )
    phrase_prototypes: torch.Tensor | None = None

    def append_log(message: str) -> None:
        with text_log.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    append_log(
        "[gradconf] "
        f"patch_train_space={patch_train_space} raw_eval_image_size={raw_eval_image_size} "
        f"raw_match_eval_resize={int(raw_match_eval_resize)} raw_center_crop={int(raw_center_crop)}"
    )

    def move_batch(batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key in ["input_ids", "attention_mask", "labels", "pixel_values", "actions", "proprio"]:
            value = batch.get(key)
            if torch.is_tensor(value):
                moved[key] = value.to(device)
        raw_by_view = batch.get("raw_pixel_values_by_view")
        if isinstance(raw_by_view, dict):
            moved["raw_pixel_values_by_view"] = {
                str(view): value.to(device) if torch.is_tensor(value) else value
                for view, value in raw_by_view.items()
            }
        if "proprio" not in moved:
            raise RuntimeError("use_proprio=true but dataloader did not return proprio.")
        return moved

    def vla_forward(inputs: dict[str, Any], *, output_features: bool = True) -> Any:
        forward_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"].to(dtype=dtype),
            "labels": inputs["labels"],
            "return_dict": True,
            "output_hidden_states": True,
            "output_attentions": True,
        }
        if output_features:
            forward_kwargs["output_projector_features"] = True
        proprio = inputs["proprio"]
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim > 2:
            proprio = proprio.reshape(proprio.shape[0], -1)
        forward_kwargs["proprio"] = proprio.to(dtype=dtype)
        forward_kwargs["proprio_projector"] = wrapper.proprio_projector
        try:
            return wrapper._with_cudnn_fallback("gradconf vla forward", lambda: wrapper.vla(**forward_kwargs))
        except TypeError as exc:
            if output_features and "output_projector_features" in str(exc):
                forward_kwargs.pop("output_projector_features", None)
                return wrapper._with_cudnn_fallback("gradconf vla forward", lambda: wrapper.vla(**forward_kwargs))
            raise

    def extract_features(outputs: Any) -> torch.Tensor:
        features = getattr(outputs, "projector_features", None)
        if torch.is_tensor(features):
            return features.float()
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("WHEN-style objective requires projector_features or hidden_states.")
        layer = 0
        seq = hidden_states[layer]
        start = 1 if seq.shape[1] >= int(num_visual_tokens) + 1 else 0
        end = start + int(num_visual_tokens)
        if end > seq.shape[1]:
            raise RuntimeError(
                f"Cannot slice visual tokens [{start}:{end}] from hidden state shape={tuple(seq.shape)}"
            )
        return seq[:, start:end, :].float()

    def align_text_mask_to_sequence(mask: torch.Tensor, seq_len: int, feature_tokens: int) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(device=device, dtype=torch.bool)
        batch_size = int(mask.shape[0])
        aligned = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
        text_len = int(mask.shape[1])
        prefix_len = max(0, seq_len - text_len)
        copy_len = min(text_len, max(0, seq_len - prefix_len))
        if copy_len > 0:
            aligned[:, prefix_len : prefix_len + copy_len] = mask[:, :copy_len]
        visual_start = 1 if seq_len >= int(feature_tokens) + 1 else 0
        visual_end = min(seq_len, visual_start + int(feature_tokens))
        aligned[:, visual_start:visual_end] = False
        return aligned

    def action_query_mask(inputs: dict[str, Any], seq_len: int, feature_tokens: int) -> torch.Tensor:
        labels = inputs.get("labels")
        if not torch.is_tensor(labels):
            return torch.zeros((1, seq_len), device=device, dtype=torch.bool)
        labels = labels.to(device=device)
        return align_text_mask_to_sequence(labels.ne(ignore_index), seq_len, feature_tokens)

    def instruction_query_mask(inputs: dict[str, Any], seq_len: int, feature_tokens: int) -> torch.Tensor:
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        if torch.is_tensor(attention_mask):
            instruction_mask = attention_mask.to(device=device).bool()
        elif torch.is_tensor(labels):
            instruction_mask = torch.ones_like(labels.to(device=device), dtype=torch.bool)
        else:
            return torch.zeros((1, seq_len), device=device, dtype=torch.bool)
        if torch.is_tensor(labels) and tuple(labels.shape) == tuple(instruction_mask.shape):
            instruction_mask = instruction_mask & labels.to(device=device).eq(ignore_index)
        return align_text_mask_to_sequence(instruction_mask, seq_len, feature_tokens)

    def feature_distance(adv_features: torch.Tensor, clean_features: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if use_l2_feature:
            return F.mse_loss(adv_features, clean_features.detach(), reduction="mean")
        return F.l1_loss(adv_features, clean_features.detach(), reduction="mean")

    def bicontrastive_ce_objective(adv_features: torch.Tensor, clean_features: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        bsz = adv_features.shape[0]
        adv = F.normalize(adv_features.reshape(bsz, -1, adv_features.shape[-1]), dim=-1)
        clean = F.normalize(clean_features.detach().reshape(bsz, -1, clean_features.shape[-1]), dim=-1)
        logits = torch.bmm(adv, clean.transpose(1, 2)) / 0.07
        labels = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).expand(logits.shape[0], -1)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")

    def image_patch_mask(batch_size: int, image_h: int, image_w: int) -> torch.Tensor:
        mask = torch.zeros((batch_size, 1, image_h, image_w), device=device, dtype=torch.float32)
        if patch_position == "random":
            raise ValueError(
                "patch.position=random cannot be paired with WHEN PAD/PSM masks; use fixed upa_rfas_libero_10."
            )
        local_h = min(patch_h, image_h)
        local_w = min(patch_w, image_w)
        for sample_idx in range(batch_size):
            x, y = _select_patch_position(image_h, image_w, local_h, local_w, mode=patch_position)
            mask[sample_idx, :, y : y + local_h, x : x + local_w] = 1.0
        return mask

    def patch_channel_mask(pixel_values: torch.Tensor, views: list[str]) -> torch.Tensor:
        batch_size, channels, image_h, image_w = pixel_values.shape
        mask = torch.zeros((batch_size, channels, image_h, image_w), device=device, dtype=torch.float32)
        local_image_mask = image_patch_mask(batch_size, image_h, image_w)
        for view in views:
            if view not in input_views:
                continue
            view_idx = input_views.index(view)
            start = view_idx * 6
            end = min(start + 6, channels)
            mask[:, start:end] = local_image_mask
        return mask

    def patch_token_mask(pixel_values: torch.Tensor, views: list[str], feature_tokens: int) -> torch.Tensor:
        import torch.nn.functional as F

        batch_size, _channels, image_h, image_w = pixel_values.shape
        local_image_mask = image_patch_mask(batch_size, image_h, image_w)
        local_token_mask = F.interpolate(
            local_image_mask,
            size=(view_token_side, view_token_side),
            mode="area",
        ).flatten(1).clamp(0, 1)
        per_view_masks = []
        for view in input_views:
            if view in views:
                per_view_masks.append(local_token_mask)
            else:
                per_view_masks.append(torch.zeros_like(local_token_mask))
        mask = torch.cat(per_view_masks, dim=1)
        if mask.shape[1] < feature_tokens:
            pad = torch.zeros((batch_size, feature_tokens - mask.shape[1]), device=device, dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=1)
        elif mask.shape[1] > feature_tokens:
            mask = mask[:, :feature_tokens]
        return mask

    def attention_hijack_loss(
        clean_outputs: Any,
        adv_outputs: Any,
        clean_inputs: dict[str, Any],
        token_mask: torch.Tensor,
        feature_tokens: int,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        clean_attn = getattr(clean_outputs, "attentions", None)
        adv_attn = getattr(adv_outputs, "attentions", None)
        if not clean_attn or not adv_attn:
            return token_mask.new_tensor(0.0)

        def text_to_visual_share(attention_blocks: Any) -> tuple[torch.Tensor, torch.Tensor]:
            shares = []
            valids = []
            visual_start = 1
            visual_end = visual_start + feature_tokens
            for attention in attention_blocks[-last_n:]:
                attn = attention.float().mean(dim=1)
                if attn.shape[-1] <= visual_end:
                    continue
                action_mask = action_query_mask(clean_inputs, int(attn.shape[-2]), feature_tokens)
                if bool(action_mask.any().item()):
                    text_to_visual = attn[:, :, visual_start:visual_end]
                    valid_queries = action_mask[:, : text_to_visual.shape[1]].to(dtype=text_to_visual.dtype)
                else:
                    text_to_visual = attn[:, visual_end:, visual_start:visual_end]
                    valid_queries = torch.ones(
                        text_to_visual.shape[:2],
                        device=text_to_visual.device,
                        dtype=text_to_visual.dtype,
                    )
                share = text_to_visual / (text_to_visual.sum(dim=-1, keepdim=True) + 1e-6)
                shares.append(share)
                valids.append(valid_queries)
            if not shares:
                fallback_share = token_mask.new_zeros((token_mask.shape[0], 1, feature_tokens))
                fallback_valid = token_mask.new_ones((token_mask.shape[0], 1))
                return fallback_share, fallback_valid
            return torch.stack(shares, dim=0).mean(dim=0), torch.stack(valids, dim=0).amax(dim=0)

        clean_share, valid_queries = text_to_visual_share(clean_attn)
        adv_share, _ = text_to_visual_share(adv_attn)
        delta = adv_share - clean_share.detach()
        mask = token_mask[:, None, :].expand_as(delta)
        valid_queries = valid_queries.to(device=delta.device, dtype=delta.dtype)
        if topk_text is not None and 0 < topk_text < 1:
            text_weights = clean_share.sum(dim=-1)
            text_weights = text_weights.masked_fill(valid_queries <= 0, float("-inf"))
            keep_count = max(1, int(text_weights.shape[1] * topk_text))
            keep_idx = text_weights.topk(keep_count, dim=1).indices
            text_mask = torch.zeros_like(text_weights, dtype=torch.bool).scatter(1, keep_idx, True)
            delta = torch.where(text_mask[:, :, None], delta, torch.zeros_like(delta))
            mask = torch.where(text_mask[:, :, None], mask, torch.zeros_like(mask))
            valid_queries = valid_queries * text_mask.to(dtype=valid_queries.dtype)

        patch_sum = mask.sum(dim=-1).clamp_min(1.0)
        non_mask = 1.0 - mask
        non_sum = non_mask.sum(dim=-1).clamp_min(1.0)
        patch_delta = (delta * mask).sum(dim=-1) / patch_sum
        nonpatch_delta = (delta * non_mask).sum(dim=-1) / non_sum
        valid_sum = valid_queries.sum().clamp_min(1.0)
        loss_core = -((patch_delta * valid_queries).sum() / valid_sum)
        loss_core = loss_core + float(lambda_nonpatch) * ((F.relu(nonpatch_delta) * valid_queries).sum() / valid_sum)
        if margin is not None and margin > 0:
            non_top = (delta * non_mask).amax(dim=-1)
            rank_term = F.relu(float(margin) - (patch_delta - non_top))
            loss_core = loss_core + (rank_term * valid_queries).sum() / valid_sum
        return loss_core

    def get_phrase_prototypes() -> torch.Tensor:
        nonlocal phrase_prototypes
        import torch.nn.functional as F

        if phrase_prototypes is not None:
            return phrase_prototypes
        tokenizer = getattr(getattr(wrapper, "processor", None), "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("WHEN text similarity loss requires wrapper.processor.tokenizer.")
        embeddings = wrapper.vla.get_input_embeddings().weight.detach().to(device).float()
        prototypes = []
        for phrase in probe_phrases:
            token_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            ids = torch.tensor(token_ids, device=device, dtype=torch.long)
            prototypes.append(embeddings.index_select(0, ids).mean(dim=0))
        if not prototypes:
            phrase_prototypes = torch.empty((0, embeddings.shape[-1]), device=device, dtype=torch.float32)
        else:
            phrase_prototypes = F.normalize(torch.stack(prototypes, dim=0), dim=-1)
        return phrase_prototypes

    def text_similarity_loss(
        adv_outputs: Any,
        adv_features: torch.Tensor,
        clean_inputs: dict[str, Any],
        token_mask: torch.Tensor,
        feature_tokens: int,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        prototypes = get_phrase_prototypes()
        if prototypes.numel() == 0:
            return adv_features.new_tensor(0.0)
        mask = token_mask[:, : adv_features.shape[1]].to(dtype=adv_features.dtype)
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        patch_repr = (adv_features.float() * weights.unsqueeze(-1)).sum(dim=1)
        patch_repr = F.normalize(patch_repr, dim=-1)

        pos_sims = torch.matmul(patch_repr, prototypes.t()) / float(tau)
        pos_loss = -torch.logsumexp(pos_sims, dim=-1).mean()

        hidden_states = getattr(adv_outputs, "hidden_states", None)
        if not hidden_states:
            return pos_loss
        last_hidden = hidden_states[-1].float()
        instruction_mask = instruction_query_mask(clean_inputs, int(last_hidden.shape[1]), feature_tokens)
        if bool(instruction_mask.any().item()):
            weights = instruction_mask.to(device=last_hidden.device, dtype=last_hidden.dtype)
            text_repr = (last_hidden * weights.unsqueeze(-1)).sum(dim=1)
            text_repr = text_repr / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            text_repr = F.normalize(text_repr, dim=-1)
        else:
            text_start = 1 + feature_tokens
            if last_hidden.shape[1] <= text_start:
                return pos_loss
            text_repr = F.normalize(last_hidden[:, text_start:, :].mean(dim=1), dim=-1)
        neg_loss = (patch_repr * text_repr).sum(dim=-1).mean()
        return pos_loss + float(beta_neg) * neg_loss

    def when_attack_loss_and_objective(
        clean_inputs: dict[str, Any],
        clean_outputs: Any,
        adv_outputs: Any,
        clean_features: torch.Tensor,
        adv_features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        feature_tokens = adv_features.shape[1]
        loss_con = adv_features.new_tensor(0.0)
        loss_1 = adv_features.new_tensor(0.0)
        loss_attn = adv_features.new_tensor(0.0)
        loss_text = adv_features.new_tensor(0.0)

        if use_contrastive_loss:
            loss_con = -bicontrastive_ce_objective(adv_features, clean_features)
        if use_feature_loss:
            loss_1 = -feature_distance(adv_features, clean_features)
        if use_attention_loss:
            loss_attn = attention_hijack_loss(clean_outputs, adv_outputs, clean_inputs, token_mask, feature_tokens)
        if use_text_loss:
            loss_text = text_similarity_loss(adv_outputs, adv_features, clean_inputs, token_mask, feature_tokens)

        attack_loss = (
            float(lambda_con) * loss_con
            + float(lambda_attn) * loss_attn
            + float(lambda_text) * loss_text
            + float(lambda_1) * loss_1
        )
        objective = -attack_loss
        terms = {
            "loss_con": float(loss_con.detach().float().item()),
            "loss_attn": float(loss_attn.detach().float().item()),
            "loss_text": float(loss_text.detach().float().item()),
            "loss_1": float(loss_1.detach().float().item()),
            "attack_loss": float(attack_loss.detach().float().item()),
            "objective": float(objective.detach().float().item()),
        }
        return attack_loss, objective, terms

    def build_adv_pixel_values(
        clean_inputs: dict[str, Any],
        patch_tensor: torch.Tensor | None,
        views: list[str],
    ) -> torch.Tensor:
        if patch_train_space == "raw_rgb":
            raw_by_view = clean_inputs.get("raw_pixel_values_by_view")
            if not isinstance(raw_by_view, dict):
                raise RuntimeError(
                    "patch.train_space=raw_rgb requires raw_pixel_values_by_view; "
                    "check get_dataloader(return_raw_images=True)."
                )
            return build_raw_patched_pixel_values(
                raw_by_view,
                patch_tensor,
                input_views=input_views,
                patch_views=views,
                image_processor=wrapper.processor.image_processor,
                device=device,
                dtype=dtype,
                position=patch_position,
                center_crop=raw_center_crop,
                eval_image_size=raw_eval_image_size,
                match_eval_resize=raw_match_eval_resize,
            )
        return apply_patch_to_pixel_values(
            clean_inputs["pixel_values"].to(dtype=dtype),
            patch_tensor,
            input_views=input_views,
            patch_views=views,
            position=patch_position,
        )

    def objective_for_views(
        clean_inputs: dict[str, Any],
        clean_outputs: Any,
        clean_features: torch.Tensor,
        patch_tensor: torch.Tensor,
        views: list[str],
        *,
        grad_enabled: bool,
        perturbation: torch.Tensor | None = None,
        channel_mask: torch.Tensor | None = None,
        return_terms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            adv_inputs = dict(clean_inputs)
            adv_pixels = build_adv_pixel_values(clean_inputs, patch_tensor, views)
            if perturbation is not None:
                if channel_mask is None:
                    channel_mask = patch_channel_mask(clean_inputs["pixel_values"].float(), views)
                adv_pixels = adv_pixels.float() + perturbation.to(device=device, dtype=torch.float32) * (
                    1.0 - channel_mask
                )
            adv_inputs["pixel_values"] = adv_pixels.to(dtype=dtype)
            adv_outputs = vla_forward(adv_inputs)
            adv_features = extract_features(adv_outputs)
            token_mask = patch_token_mask(
                clean_inputs["pixel_values"].float(),
                views,
                adv_features.shape[1],
            )
            attack_loss, objective, terms = when_attack_loss_and_objective(
                clean_inputs,
                clean_outputs,
                adv_outputs,
                clean_features,
                adv_features,
                token_mask,
            )
            if return_terms:
                return attack_loss, objective, terms
            return objective

    def grad_for_views(
        clean_inputs: dict[str, Any],
        clean_outputs: Any,
        clean_features: torch.Tensor,
        views: list[str],
    ) -> tuple[torch.Tensor, float]:
        objective = objective_for_views(
            clean_inputs,
            clean_outputs,
            clean_features,
            patch,
            views,
            grad_enabled=True,
        )
        assert torch.is_tensor(objective)
        grad = torch.autograd.grad(objective, patch, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(patch)
        return grad.detach().reshape(-1).float(), float(objective.detach().float().item())

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(torch.dot(a, b).div(denom).item())

    def norm(a: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(a).item())

    def patch_stats() -> dict[str, float]:
        values = patch.detach().float()
        return {
            "patch_min": float(values.min().item()),
            "patch_max": float(values.max().item()),
            "patch_mean": float(values.mean().item()),
            "patch_std": float(values.std(unbiased=False).item()),
        }

    def build_rupa_perturbation(
        clean_inputs: dict[str, Any],
        clean_features: torch.Tensor,
        views: list[str],
    ) -> tuple[torch.Tensor | None, float]:
        if not use_rupa_inner_loop or inner_steps <= 0 or inner_eps <= 0:
            return None, 0.0

        clean_pixels = clean_inputs["pixel_values"].float()
        channel_mask = patch_channel_mask(clean_pixels, views)
        perturbation = torch.zeros_like(clean_pixels, device=device, dtype=torch.float32, requires_grad=True)
        outer_lr = 2.0 * float(inner_eps) / float(max(1, inner_steps))
        last_outer_loss = 0.0

        base_pixels = build_adv_pixel_values(clean_inputs, patch.detach(), views).float()
        token_mask = patch_token_mask(clean_pixels, views, clean_features.shape[1])

        for _outer_index in range(inner_steps):
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            adv_inputs = dict(clean_inputs)
            adv_inputs["pixel_values"] = (base_pixels + perturbation * (1.0 - channel_mask)).to(dtype=dtype)
            adv_outputs = vla_forward(adv_inputs)
            adv_features = extract_features(adv_outputs)
            outer_loss = (
                float(lambda_con) * bicontrastive_ce_objective(adv_features, clean_features)
                + float(lambda_1) * feature_distance(adv_features, clean_features)
            )
            outer_loss.backward()
            last_outer_loss = float(outer_loss.detach().float().item())
            with torch.no_grad():
                if perturbation.grad is not None:
                    perturbation.add_(-outer_lr * perturbation.grad.sign())
                perturbation.clamp_(-float(inner_eps), float(inner_eps))
                perturbation.mul_(1.0 - channel_mask)
            perturbation.requires_grad_(True)

        return perturbation.detach(), last_outer_loss

    metric_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    interference_rows: list[dict[str, Any]] = []
    data_iter = iter(train_loader)

    for batch_index in range(int(resolved["num_batches"])):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        clean_inputs = move_batch(batch)
        clean_inputs["pixel_values"] = clean_inputs["pixel_values"].to(dtype=dtype)

        with torch.no_grad():
            clean_outputs = vla_forward(clean_inputs)
            clean_features = extract_features(clean_outputs).detach()

        g_front, obj_front = grad_for_views(clean_inputs, clean_outputs, clean_features, ["front"])
        g_wrist, obj_wrist = grad_for_views(clean_inputs, clean_outputs, clean_features, ["wrist"])
        g_both, obj_both = grad_for_views(clean_inputs, clean_outputs, clean_features, ["front", "wrist"])

        cos_front_wrist = cosine(g_front, g_wrist)
        cos_both_front = cosine(g_both, g_front)
        cos_both_wrist = cosine(g_both, g_wrist)
        norm_front = norm(g_front)
        norm_wrist = norm(g_wrist)
        norm_both = norm(g_both)
        norm_ratio = norm_wrist / max(norm_front, 1e-12)
        row = {
            "batch_index": batch_index,
            "mode": args.mode,
            "objective_front": obj_front,
            "objective_wrist": obj_wrist,
            "objective_both": obj_both,
            "cos_front_wrist": cos_front_wrist,
            "cos_both_front": cos_both_front,
            "cos_both_wrist": cos_both_wrist,
            "norm_front": norm_front,
            "norm_wrist": norm_wrist,
            "norm_both": norm_both,
            "norm_ratio_wrist_front": norm_ratio,
            "conflict_flag": int(cos_front_wrist < conflict_threshold),
            "severe_conflict_flag": int(cos_front_wrist < severe_threshold),
            "wrist_dominance_flag": int(norm_ratio >= dominance_threshold),
        }
        row.update(patch_stats())
        metric_rows.append(row)

        if enable_interference:
            step_lr = float(cfg_get(config, "train.learning_rate", 1.0e-3))
            directions = {
                "g_front": g_front.reshape_as(patch),
                "g_wrist": g_wrist.reshape_as(patch),
                "g_average": ((g_front + g_wrist) * 0.5).reshape_as(patch),
            }
            for source, direction in directions.items():
                temp_patch = (patch.detach() + step_lr * direction.to(device=device, dtype=torch.float32)).clamp(
                    clamp_min,
                    clamp_max,
                )
                updated_front = objective_for_views(
                    clean_inputs,
                    clean_outputs,
                    clean_features,
                    temp_patch,
                    ["front"],
                    grad_enabled=False,
                )
                updated_wrist = objective_for_views(
                    clean_inputs,
                    clean_outputs,
                    clean_features,
                    temp_patch,
                    ["wrist"],
                    grad_enabled=False,
                )
                interference_rows.append(
                    {
                        "batch_index": batch_index,
                        "update_source": source,
                        "base_front_objective": obj_front,
                        "base_wrist_objective": obj_wrist,
                        "updated_front_objective": float(updated_front.detach().float().item()),
                        "updated_wrist_objective": float(updated_wrist.detach().float().item()),
                        "delta_front_objective": float(updated_front.detach().float().item()) - obj_front,
                        "delta_wrist_objective": float(updated_wrist.detach().float().item()) - obj_wrist,
                    }
                )

        if args.mode == "dynamic":
            inner_objectives: list[float] = []
            inner_losses: list[float] = []
            inner_loss_con: list[float] = []
            inner_loss_attn: list[float] = []
            inner_loss_text: list[float] = []
            inner_loss_1: list[float] = []
            grad_l2 = 0.0
            training_branches = [
                ("front", ["front"]),
                ("wrist", ["wrist"]),
            ]
            branch_states: list[dict[str, Any]] = []
            for branch_name, branch_views in training_branches:
                branch_rupa_perturbation, branch_rupa_outer_loss = build_rupa_perturbation(
                    clean_inputs,
                    clean_features,
                    branch_views,
                )
                branch_states.append(
                    {
                        "name": branch_name,
                        "views": branch_views,
                        "perturbation": branch_rupa_perturbation,
                        "outer_loss": branch_rupa_outer_loss,
                        "channel_mask": patch_channel_mask(clean_inputs["pixel_values"].float(), branch_views),
                    }
                )
            rupa_outer_losses = [float(state["outer_loss"]) for state in branch_states]
            rupa_enabled = any(state["perturbation"] is not None for state in branch_states)
            merge_divisor = float(len(branch_states)) if gradient_merge_mode == "naive_average" else 1.0
            for _inner_index in range(inner_loop_steps):
                merged_grad = torch.zeros_like(patch, dtype=torch.float32)
                branch_count = 0
                for state in branch_states:
                    optimizer.zero_grad(set_to_none=True)
                    attack_loss, objective, terms = objective_for_views(
                        clean_inputs,
                        clean_outputs,
                        clean_features,
                        patch,
                        state["views"],
                        grad_enabled=True,
                        perturbation=state["perturbation"],
                        channel_mask=state["channel_mask"],
                        return_terms=True,
                    )
                    assert torch.is_tensor(attack_loss)
                    assert torch.is_tensor(objective)
                    attack_loss.backward()
                    if patch.grad is not None:
                        merged_grad.add_(patch.grad.detach().float())
                    branch_count += 1
                    inner_objectives.append(float(objective.detach().float().item()))
                    inner_losses.append(float(attack_loss.detach().float().item()))
                    inner_loss_con.append(terms["loss_con"])
                    inner_loss_attn.append(terms["loss_attn"])
                    inner_loss_text.append(terms["loss_text"])
                    inner_loss_1.append(terms["loss_1"])
                if branch_count <= 0:
                    raise RuntimeError("G1 dynamic training has no front/wrist branches to merge.")
                merged_grad.div_(merge_divisor)
                patch.grad = merged_grad.to(device=device, dtype=patch.dtype).detach().clone()
                grad_l2 = float(torch.linalg.vector_norm(merged_grad.reshape(-1)).item())
                optimizer.step()
                with torch.no_grad():
                    patch.clamp_(clamp_min, clamp_max)
                optimizer.zero_grad(set_to_none=True)
            if (batch_index + 1) % accumulate_steps == 0 or batch_index + 1 == int(resolved["num_batches"]):
                scheduler.step()
            train_row = {
                "step": batch_index,
                "inner_loop": inner_loop_steps,
                "inner_steps": inner_steps,
                "inner_eps": inner_eps,
                "gradient_merge": gradient_merge_mode,
                "patch_train_space": patch_train_space,
                "training_branch_count": len(branch_states),
                "rupa_enabled": int(rupa_enabled),
                "rupa_outer_loss": mean(rupa_outer_losses),
                "rupa_outer_loss_front": rupa_outer_losses[0] if len(rupa_outer_losses) > 0 else 0.0,
                "rupa_outer_loss_wrist": rupa_outer_losses[1] if len(rupa_outer_losses) > 1 else 0.0,
                "training_objective_naive": mean(inner_objectives[-len(branch_states) :]) if inner_objectives else 0.0,
                "training_objective_naive_mean": mean(inner_objectives),
                "training_objective_both": mean(inner_objectives[-len(branch_states) :]) if inner_objectives else 0.0,
                "training_objective_both_mean": mean(inner_objectives),
                "training_loss_minimized": mean(inner_losses[-len(branch_states) :]) if inner_losses else 0.0,
                "training_loss_minimized_mean": mean(inner_losses),
                "training_loss_con_mean": mean(inner_loss_con),
                "training_loss_attn_mean": mean(inner_loss_attn),
                "training_loss_text_mean": mean(inner_loss_text),
                "training_loss_1_mean": mean(inner_loss_1),
                "patch_grad_l2": grad_l2,
                "learning_rate": float(optimizer.param_groups[0].get("lr", 0.0)),
                "recorded": int(batch_index % record_every == 0 or batch_index + 1 == int(resolved["num_batches"])),
            }
            train_row.update(patch_stats())
            training_rows.append(train_row)

        append_log(
            "[gradconf] "
            f"batch={batch_index} cos_front_wrist={cos_front_wrist:.6f} "
            f"norm_ratio_wrist_front={norm_ratio:.6f}"
        )

    metric_fields = [
        "batch_index",
        "mode",
        "objective_front",
        "objective_wrist",
        "objective_both",
        "cos_front_wrist",
        "cos_both_front",
        "cos_both_wrist",
        "norm_front",
        "norm_wrist",
        "norm_both",
        "norm_ratio_wrist_front",
        "conflict_flag",
        "severe_conflict_flag",
        "wrist_dominance_flag",
        "patch_min",
        "patch_max",
        "patch_mean",
        "patch_std",
    ]
    write_csv(run_dir / "metrics_per_batch.csv", metric_rows, metric_fields)
    write_csv(
        run_dir / "interference_matrix.csv",
        interference_rows,
        [
            "batch_index",
            "update_source",
            "base_front_objective",
            "base_wrist_objective",
            "updated_front_objective",
            "updated_wrist_objective",
            "delta_front_objective",
            "delta_wrist_objective",
        ],
    )
    if args.mode == "dynamic":
        write_csv(
            run_dir / "training_grad_metrics.csv",
            training_rows,
            [
                "step",
                "inner_loop",
                "inner_steps",
                "inner_eps",
                "gradient_merge",
                "patch_train_space",
                "training_branch_count",
                "rupa_enabled",
                "rupa_outer_loss",
                "rupa_outer_loss_front",
                "rupa_outer_loss_wrist",
                "training_objective_naive",
                "training_objective_naive_mean",
                "training_objective_both",
                "training_objective_both_mean",
                "training_loss_minimized",
                "training_loss_minimized_mean",
                "training_loss_con_mean",
                "training_loss_attn_mean",
                "training_loss_text_mean",
                "training_loss_1_mean",
                "patch_grad_l2",
                "learning_rate",
                "recorded",
                "patch_min",
                "patch_max",
                "patch_mean",
                "patch_std",
            ],
        )
        torch.save(patch.detach().cpu(), run_dir / "patch_checkpoint_last.pt")

    cos_values = [safe_float(row["cos_front_wrist"]) for row in metric_rows]
    summary = {
        "dataset": resolved["dataset_name"],
        "dataset_semantic_name": resolved["dataset_semantic_name"],
        "suite": resolved["suite_name"],
        "model_family": resolved["model_family"],
        "model_variant": resolved["model_variant"],
        "model_path": resolved["model_path"],
        "patch_source": resolved["patch_source"],
        "patch_position": resolved.get("patch_position", ""),
        "patch_train_space": patch_train_space,
        "when_baseline_alignment": resolved.get("when_baseline_alignment", {}),
        "mode": args.mode,
        "mean_cos_front_wrist": mean(cos_values),
        "median_cos_front_wrist": float(median(cos_values)) if cos_values else 0.0,
        "conflict_rate": mean([safe_float(row["conflict_flag"]) for row in metric_rows]),
        "severe_conflict_rate": mean([safe_float(row["severe_conflict_flag"]) for row in metric_rows]),
        "wrist_dominance_rate": mean([safe_float(row["wrist_dominance_flag"]) for row in metric_rows]),
        "mean_norm_front": mean([safe_float(row["norm_front"]) for row in metric_rows]),
        "mean_norm_wrist": mean([safe_float(row["norm_wrist"]) for row in metric_rows]),
        "mean_norm_both": mean([safe_float(row["norm_both"]) for row in metric_rows]),
        "mean_norm_ratio_wrist_front": mean([safe_float(row["norm_ratio_wrist_front"]) for row in metric_rows]),
        "mean_cos_both_front": mean([safe_float(row["cos_both_front"]) for row in metric_rows]),
        "mean_cos_both_wrist": mean([safe_float(row["cos_both_wrist"]) for row in metric_rows]),
        "num_batches": len(metric_rows),
        "data_path_resolved": resolved["data_path_resolved"],
        "interference_matrix": str(run_dir / "interference_matrix.csv"),
        "notes": [
            "Only random patch is used.",
            "libero_10 is treated as the repository's LIBERO-Long dataset alias.",
            "Optimizer, scheduler, seed, batch size, patch size, dual/RUPA loop, loss weights, and dynamic inner-loop defaults are aligned to configs/when/train/openvla_libero10_when.yaml.",
            "Dynamic G1 training uses complete front and wrist WHEN branches with per-branch RAUP and naive averaged patch gradients; g_both is kept for conflict diagnostics.",
            "The deliberate difference from the source WHEN baseline is the OFT front/wrist input branch and the extra front/wrist/both conflict diagnostic.",
            f"Patch train space is {patch_train_space}. raw_rgb means the patch is applied on raw RGB views before differentiable resize/crop/normalization into OFT pixel_values.",
            "Self-trained patch, G0-when, GRAC, physical occlusion, visibility proxy, and physical masking are not enabled.",
        ],
    }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
