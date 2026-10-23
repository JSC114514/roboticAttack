#!/usr/bin/env python3
"""Calibrate OpenVLA-OFT hybrid loss weights from patch-gradient norms.

This script reads M1/M2/M5 configs, uses the same patch initialization and EOT
sampling code as training, and computes component gradients without updating the
patch. It intentionally does not submit Slurm jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _early_set_cuda_visible_devices(argv: list[str]) -> None:
    if "--gpu_id" not in argv:
        return
    idx = argv.index("--gpu_id")
    if idx + 1 >= len(argv):
        return
    gpu_id = argv[idx + 1].strip()
    if gpu_id:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
        os.environ.setdefault("GPU_ID", gpu_id)


_early_set_cuda_visible_devices(sys.argv)


DEFAULT_CONFIGS = [
    "configs/experiments/M1_oft_train_front_attack_front.yaml",
    "configs/experiments/M2_oft_train_wrist_attack_wrist.yaml",
    "configs/experiments/M5_oft_train_multiview_attack_front.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate OFT hybrid loss weights.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--batches_per_config", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/loss_weight_calibration")
    parser.add_argument("--gpu_id", default="")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def require_cuda_or_die() -> None:
    import torch

    print(f"[cuda-preflight] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[cuda-preflight] torch.cuda.is_available()={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Weight calibration requires CUDA. Submit with sbatch on a GPU partition; "
            "CPU fallback is intentionally disabled."
        )
    print(f"[cuda-preflight] device_count={torch.cuda.device_count()}")
    print(f"[cuda-preflight] current_device={torch.cuda.current_device()}")
    print(f"[cuda-preflight] device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")


def validate_cuda_config(config: dict[str, str], config_path: Path) -> None:
    device = str(config.get("device") or "cuda").strip().lower()
    if device != "cuda":
        raise ValueError(f"{config_path}: calibration must use device=cuda, got {device!r}")
    allow_cpu = str(config.get("allow_cpu") or "false").strip().lower() in {"1", "true", "yes", "y", "on"}
    if allow_cpu:
        raise ValueError(f"{config_path}: allow_cpu=true is forbidden for calibration")


def init_patch(config: dict[str, str], device: Any) -> Any:
    import torch

    value = config.get("wrapper_patch_size") or config.get("patch_size") or "3,50,50"
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 3:
        c, h, w = (int(part) for part in parts)
    else:
        ratio = float(config.get("patch_size", "0.05"))
        side = max(1, int(round((224 * 224 * max(ratio, 0.0001)) ** 0.5)))
        c, h, w = 3, side, side
    if c != 3:
        raise ValueError(f"Only RGB patches are supported, got {c} channels")
    patch_seed = int(config.get("patch_seed") or config.get("seed") or 0)
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(patch_seed)
    return torch.rand((3, h, w), device=device, dtype=torch.float32, generator=generator, requires_grad=True)


def flatten_grad(grad: Any, patch: Any) -> Any:
    import torch

    if grad is None:
        grad = torch.zeros_like(patch)
    return grad.detach().reshape(-1).float()


def grad_norm_and_cosines(grads: dict[str, Any]) -> dict[str, float]:
    import torch

    def norm(name: str) -> float:
        return float(torch.linalg.vector_norm(grads[name]).item())

    def cosine(a: str, b: str) -> float:
        denom = torch.linalg.vector_norm(grads[a]) * torch.linalg.vector_norm(grads[b])
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(torch.dot(grads[a], grads[b]).div(denom).item())

    return {
        "grad_boundary_l2": norm("boundary"),
        "grad_divergence_l2": norm("divergence"),
        "grad_feature_l2": norm("feature"),
        "cos_boundary_divergence": cosine("boundary", "divergence"),
        "cos_boundary_feature": cosine("boundary", "feature"),
        "cos_divergence_feature": cosine("divergence", "feature"),
    }


def aggregate_components(component_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        key: torch.stack([components[key] for components in component_rows]).mean()
        for key in component_rows[0]
    }


def run_one_config(config_path: Path, batches: int) -> list[dict[str, Any]]:
    import torch
    from VLAAttacker.multiview.openvla_oft_attack_losses import (
        oft_clean_reference_hybrid_attack_components,
        parse_maskidx,
    )
    from VLAAttacker.multiview.openvla_oft_experiments import (
        _get_oft_num_patches,
        _get_oft_num_visual_tokens,
        _int_config,
        _move_batch_to_device,
        apply_patch_to_pixel_values,
        sample_oft_eot_transforms,
        validate_oft_eot_config,
    )
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper
    from VLAAttacker.multiview.view_batch_utils import parse_bool, parse_views, read_simple_yaml
    from VLAAttacker.white_patch.openvla_dataloader import get_dataloader

    config = read_simple_yaml(config_path)
    validate_cuda_config(config, config_path)
    validate_oft_eot_config(config)
    random.seed(int(config.get("eot_seed") or config.get("seed") or 0))
    torch.manual_seed(int(config.get("seed") or 0))

    wrapper = OpenVLAOFTWrapper(config)
    wrapper.load_model()
    device = wrapper.device
    dtype = wrapper.dtype
    use_proprio = parse_bool(config.get("use_proprio", "false")) and wrapper.proprio_projector is not None
    use_diffusion = parse_bool(config.get("use_diffusion", "false"))
    use_film = parse_bool(config.get("use_film", "false"))
    if use_diffusion:
        raise NotImplementedError("Calibration for diffusion action head is not implemented.")
    num_patches = _get_oft_num_patches(wrapper, use_proprio=use_proprio, use_diffusion=use_diffusion)
    num_visual_tokens = _get_oft_num_visual_tokens(wrapper)
    input_views = parse_views(config.get("input_views", "front,wrist"))
    patch_views = parse_views(config.get("patch_train_views", "front"))
    patch = init_patch(config, device)

    train_loader, _val_loader = get_dataloader(
        batch_size=_int_config(config, "batch_size", 8),
        dataset=config.get("task_suite", "libero_spatial"),
        server=config.get("data_root") or config.get("project_root", "."),
        vla_path=config.get("model_checkpoint", ""),
        input_views=config.get("input_views", "front,wrist"),
        view_key_map=config.get("view_key_map", ""),
        multiview_fusion=config.get("multiview_fusion", "stitch"),
        use_proprio=parse_bool(config.get("use_proprio", "false")),
        num_images_in_input=_int_config(config, "num_images_in_input", len(input_views)),
        action_chunk_len=_int_config(config, "num_open_loop_steps", 8),
        # Match official M pixel-space training: use the PatchAware transform so
        # the dataloader returns channel-stacked pixel_values tensors rather than
        # PIL images from the stock multiview transform.
        return_raw_images=True,
        multiview_debug=False,
    )

    def vla_forward(inputs: dict[str, Any], *, output_projector_features: bool = True) -> Any:
        forward_kwargs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"].to(dtype=dtype),
            "labels": inputs["labels"],
            "return_dict": True,
            "output_hidden_states": True,
        }
        if output_projector_features:
            forward_kwargs["output_projector_features"] = True
        if use_proprio:
            proprio = inputs["proprio"]
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim > 2:
                proprio = proprio.reshape(proprio.shape[0], -1)
            forward_kwargs["proprio"] = proprio.to(dtype=dtype)
            forward_kwargs["proprio_projector"] = wrapper.proprio_projector
        if use_film:
            forward_kwargs["use_film"] = True
        try:
            return wrapper.vla(**forward_kwargs)
        except TypeError as exc:
            if output_projector_features and "output_projector_features" in str(exc):
                forward_kwargs.pop("output_projector_features", None)
                return wrapper.vla(**forward_kwargs)
            raise

    rows: list[dict[str, Any]] = []
    data_iter = iter(train_loader)
    for batch_idx in range(batches):
        batch = next(data_iter)
        moved = _move_batch_to_device(batch, device)
        if "actions" not in moved:
            raise RuntimeError("Calibration requires batch['actions'].")
        if use_proprio and "proprio" not in moved:
            raise RuntimeError("Calibration requires proprio when use_proprio=true.")
        clean_moved = _move_batch_to_device(batch, device)
        clean_moved["pixel_values"] = clean_moved["pixel_values"].to(dtype=dtype)
        with torch.no_grad():
            clean_outputs = wrapper._with_cudnn_fallback(
                "calibration clean forward",
                lambda: vla_forward(clean_moved),
            )

        clean_pixel_values = moved["pixel_values"].to(dtype=dtype)
        _, image_h, image_w = clean_pixel_values.shape[-3:]
        component_rows: list[dict[str, Any]] = []
        eot_samples = _int_config(config, "eot_samples", 1)
        for eot_idx in range(eot_samples):
            transforms = sample_oft_eot_transforms(
                config,
                step=batch_idx,
                max_steps=max(1, batches),
                batch_size=int(clean_pixel_values.shape[0]),
                image_h=int(image_h),
                image_w=int(image_w),
                patch_h=int(patch.shape[-2]),
                patch_w=int(patch.shape[-1]),
            )
            adv_moved = dict(moved)
            adv_moved["pixel_values"] = apply_patch_to_pixel_values(
                clean_pixel_values,
                patch,
                input_views=input_views,
                patch_views=patch_views,
                position=config.get("train_patch_position", "anchored_eot"),
                eot_transforms=transforms,
            )
            outputs = wrapper._with_cudnn_fallback(
                "calibration adv forward",
                lambda adv_inputs=adv_moved: vla_forward(adv_inputs),
            )
            maskidx = parse_maskidx(config.get("maskidx"), default=(0, 1, 2, 3, 4, 5, 6))
            components, _info = oft_clean_reference_hybrid_attack_components(
                outputs=outputs,
                clean_outputs=clean_outputs,
                labels=adv_moved["labels"],
                actions=adv_moved["actions"],
                action_head=wrapper.action_head,
                num_patches=num_patches,
                maskidx=maskidx,
                config=config,
                num_visual_tokens=num_visual_tokens,
            )
            component_rows.append(components)

        components = aggregate_components(component_rows)
        grads = {}
        for name, key in [
            ("boundary", "boundary_objective"),
            ("divergence", "divergence_objective"),
            ("feature", "feature_objective"),
        ]:
            patch.grad = None
            wrapper.vla.zero_grad(set_to_none=True)
            grad = torch.autograd.grad(
                components[key],
                patch,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grads[name] = flatten_grad(grad, patch)
        stats = grad_norm_and_cosines(grads)
        rows.append({
            "exp_id": config.get("exp_id", config_path.stem),
            "config": str(config_path),
            "batch_index": batch_idx,
            "eot_samples": eot_samples,
            "boundary_value": float(components["action_boundary_mse"].detach().item()),
            "divergence_value": float(components["action_divergence_score"].detach().item()),
            "feature_value": float(components["feature_deviation_score"].detach().item()),
            **stats,
        })
        patch.grad = None
        wrapper.vla.zero_grad(set_to_none=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return rows


def median_of(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(statistics.median(values)) if values else 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], configs: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "batch_gradients.csv"
    fieldnames = [
        "exp_id", "config", "batch_index", "eot_samples",
        "boundary_value", "divergence_value", "feature_value",
        "grad_boundary_l2", "grad_divergence_l2", "grad_feature_l2",
        "cos_boundary_divergence", "cos_boundary_feature", "cos_divergence_feature",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_exp: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_exp.setdefault(str(row["exp_id"]), []).append(row)
    per_exp = {
        exp_id: {
            "batches": len(exp_rows),
            "G_boundary": median_of(exp_rows, "grad_boundary_l2"),
            "G_divergence": median_of(exp_rows, "grad_divergence_l2"),
            "G_feature": median_of(exp_rows, "grad_feature_l2"),
            "cos_boundary_divergence_median": median_of(exp_rows, "cos_boundary_divergence"),
            "cos_boundary_feature_median": median_of(exp_rows, "cos_boundary_feature"),
            "cos_divergence_feature_median": median_of(exp_rows, "cos_divergence_feature"),
        }
        for exp_id, exp_rows in sorted(by_exp.items())
    }
    G_boundary = median_of(rows, "grad_boundary_l2")
    G_divergence = median_of(rows, "grad_divergence_l2")
    G_feature = median_of(rows, "grad_feature_l2")
    # Defaults mirror the experiment config; command-line customization can be added later.
    target_boundary = 1.0
    target_divergence = 0.4
    target_feature = 0.2
    eps = 1.0e-12
    weight_min = 0.001
    weight_max = 2.0
    lambda_boundary = 1.0
    lambda_divergence = clamp(target_divergence * G_boundary / (G_divergence + eps), weight_min, weight_max)
    lambda_feature = clamp(target_feature * G_boundary / (G_feature + eps), weight_min, weight_max)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "configs": configs,
        "total_batches": len(rows),
        "per_experiment": per_exp,
        "overall": {
            "G_boundary": G_boundary,
            "G_divergence": G_divergence,
            "G_feature": G_feature,
            "cos_boundary_divergence_median": median_of(rows, "cos_boundary_divergence"),
            "cos_boundary_feature_median": median_of(rows, "cos_boundary_feature"),
            "cos_divergence_feature_median": median_of(rows, "cos_divergence_feature"),
        },
        "targets": {
            "boundary": target_boundary,
            "divergence": target_divergence,
            "feature": target_feature,
        },
        "limits": {"min": weight_min, "max": weight_max, "eps": eps},
        "recommended_weights": {
            "action_boundary_weight": lambda_boundary,
            "action_divergence_weight": lambda_divergence,
            "feature_deviation_weight": lambda_feature,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    yaml_lines = [
        "weight_source: gradient_calibrated_fixed",
        "loss_weight_mode: gradient_calibrated_fixed",
        f"action_boundary_weight: {lambda_boundary:.10g}",
        f"action_divergence_weight: {lambda_divergence:.10g}",
        f"feature_deviation_weight: {lambda_feature:.10g}",
        f"calibration_batches_total: {len(rows)}",
        f"calibration_G_boundary: {G_boundary:.10g}",
        f"calibration_G_divergence: {G_divergence:.10g}",
        f"calibration_G_feature: {G_feature:.10g}",
        f"calibration_target_boundary_ratio: {target_boundary}",
        f"calibration_target_divergence_ratio: {target_divergence}",
        f"calibration_target_feature_ratio: {target_feature}",
        f"calibration_weight_min: {weight_min}",
        f"calibration_weight_max: {weight_max}",
        f"calibration_eps: {eps}",
    ]
    (output_dir / "recommended_weights.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    conflict_lines = []
    for exp_id, stats in per_exp.items():
        for key in ["cos_boundary_divergence_median", "cos_boundary_feature_median", "cos_divergence_feature_median"]:
            if float(stats[key]) < 0:
                conflict_lines.append(f"- {exp_id}: {key}={stats[key]:.6f}")
    report = [
        "# OpenVLA-OFT Hybrid Loss Weight Calibration",
        "",
        f"- Total batches: {len(rows)}",
        f"- Recommended action_divergence_weight: {lambda_divergence:.6g}",
        f"- Recommended feature_deviation_weight: {lambda_feature:.6g}",
        "",
        "## Gradient Conflicts",
        "",
        "\n".join(conflict_lines) if conflict_lines else "No median cosine conflicts below 0.",
        "",
        "## Outputs",
        "",
        f"- {csv_path}",
        f"- {output_dir / 'summary.json'}",
        f"- {output_dir / 'recommended_weights.yaml'}",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.gpu_id:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_id)
        os.environ.setdefault("GPU_ID", args.gpu_id)
    require_cuda_or_die()
    all_rows: list[dict[str, Any]] = []
    config_paths = [resolve(path) for path in args.configs]
    for config_path in config_paths:
        all_rows.extend(run_one_config(config_path, args.batches_per_config))
    write_outputs(resolve(args.output_dir), all_rows, [str(path) for path in config_paths])
    print(f"output_dir={resolve(args.output_dir)}")
    print(f"rows={len(all_rows)}")


if __name__ == "__main__":
    main()
