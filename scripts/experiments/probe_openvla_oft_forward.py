#!/usr/bin/env python3
"""
Probe OpenVLA-OFT forward/eval behavior before running M-series experiments.

Copy to:
  roboticAttack-main/scripts/experiments/probe_openvla_oft_forward.py

Example:
  python scripts/experiments/probe_openvla_oft_forward.py \
    --config configs/experiments/M0_oft_front_wrist_clean.yaml \
    --dry_run

  python scripts/experiments/probe_openvla_oft_forward.py \
    --config configs/experiments/M0_oft_front_wrist_clean.yaml \
    --max_batches 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def add_repo_to_path() -> Path:
    here = Path(__file__).resolve()
    # scripts/experiments/probe_openvla_oft_forward.py -> repo root
    root = here.parents[2]
    sys.path.insert(0, str(root))
    return root


ROOT = add_repo_to_path()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_yaml(path: str) -> Dict[str, Any]:
    """Load config without making PyYAML a dry-run dependency."""
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}

    data: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def make_dummy_obs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback observation for API probing when dataloader is unavailable."""
    import numpy as np

    h = int(cfg.get("probe_image_h", 256))
    w = int(cfg.get("probe_image_w", 256))
    front = np.zeros((h, w, 3), dtype=np.uint8)
    wrist = np.zeros((h, w, 3), dtype=np.uint8)
    front[..., 0] = 127
    wrist[..., 1] = 127
    return {
        "image_primary": front,
        "image_wrist": wrist,
        "state": np.zeros(8, dtype=np.float32),
        "task_description": "pick up the object",
    }


def try_get_real_batch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try local dataloader, but do not rely on it. Many environments fail due to tfds/CPU issues.
    Codex can replace this with your existing get_dataloader(...) once verified.
    """
    info: Dict[str, Any] = {
        "requested": _parse_bool(cfg.get("probe_use_dataloader", False)),
        "success": False,
        "fallback_dummy": False,
        "error": None,
    }
    cfg["_dataloader_probe_info"] = info
    if not info["requested"]:
        info["fallback_dummy"] = True
        return make_dummy_obs(cfg)

    try:
        # Keep this aligned with VLAAttacker/white_patch/openvla_dataloader.py.
        from VLAAttacker.white_patch.openvla_dataloader import get_dataloader  # type: ignore

        batch_size = int(cfg.get("probe_batch_size", 1))
        dataset = cfg.get("task_suite", "libero_spatial")
        data_root = cfg.get("data_root") or cfg.get("project_root")
        model_checkpoint = cfg.get("model_checkpoint")
        dl = get_dataloader(
            batch_size=batch_size,
            dataset=dataset,
            server=data_root,
            vla_path=model_checkpoint,
            input_views=cfg.get("input_views", "front,wrist"),
            view_key_map=cfg.get("view_key_map", ""),
            multiview_fusion=cfg.get("multiview_fusion", "stitch"),
            use_proprio=_parse_bool(cfg.get("use_proprio", False)),
            num_images_in_input=int(cfg.get("num_images_in_input", 2)),
            action_chunk_len=int(cfg.get("num_open_loop_steps", 8)),
            multiview_debug=_parse_bool(cfg.get("multiview_debug", False)),
        )
        batch = next(iter(dl[0] if isinstance(dl, (tuple, list)) else dl))
        info["success"] = True
        return batch
    except Exception as e:
        if _parse_bool(cfg.get("_require_dataloader", False)):
            info["error"] = repr(e)
            raise
        info["fallback_dummy"] = True
        info["error"] = repr(e)
        print(f"[WARN] dataloader probe failed, falling back to dummy obs: {repr(e)}")
        return make_dummy_obs(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--use_dataloader", action="store_true", help="强制尝试真实 dataloader batch")
    parser.add_argument("--require_dataloader", action="store_true", help="dataloader 失败时直接失败，不回退 dummy obs")
    parser.add_argument("--dataloader_only", action="store_true", help="只验证 dataloader，不加载 OpenVLA-OFT 模型")
    parser.add_argument("--output_dir", default="outputs/multiview_attack/probe_openvla_oft_forward")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.use_dataloader or args.require_dataloader or args.dataloader_only:
        cfg["probe_use_dataloader"] = True
    if args.require_dataloader or args.dataloader_only:
        cfg["_require_dataloader"] = True
    out_dir = ensure_dir(ROOT / args.output_dir)
    status_path = out_dir / "status.json"
    log_path = out_dir / ("dry_run.log" if args.dry_run else "probe.log")

    status: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "dry_run": args.dry_run,
        "dataloader_only": args.dataloader_only,
        "repo_root": str(ROOT),
        "model_checkpoint": cfg.get("model_checkpoint"),
        "openvla_oft_root": cfg.get("openvla_oft_root") or cfg.get("oft_repo_root"),
        "libero_root": cfg.get("libero_root"),
        "input_views": cfg.get("input_views"),
        "view_key_map": cfg.get("view_key_map") or cfg.get("view_map"),
        "disable_cudnn": cfg.get("disable_cudnn", cfg.get("probe_disable_cudnn", False)),
        "torch_cudnn_enabled": None,
        "torch_cudnn_enabled_initial": None,
        "torch_cudnn_enabled_final": None,
        "cudnn_fallback_used": False,
        "model_load_success": False,
        "processor_load_success": False,
        "supports_predict_action": False,
        "supports_forward_logits": False,
        "supports_pixel_values_grad_probe": False,
        "supports_native_multiview": False,
        "dataloader_probe_requested": False,
        "dataloader_probe_success": False,
        "dataloader_probe_fallback_dummy": False,
        "dataloader_probe_error": None,
        "accepted_image_format": None,
        "pixel_values_shape": None,
        "logits_shape": None,
        "notes": [],
    }

    with open(log_path, "w", encoding="utf-8") as log:
        def logprint(msg: str) -> None:
            print(msg)
            log.write(msg + "\n")
            log.flush()

        logprint(f"[INFO] repo root: {ROOT}")
        logprint(f"[INFO] config: {args.config}")
        logprint(f"[INFO] dry_run: {args.dry_run}")

        required = ["model_checkpoint", "input_views"]
        missing = [k for k in required if not cfg.get(k)]
        if missing:
            status["notes"].append(f"missing config fields: {missing}")
            logprint(f"[WARN] missing config fields: {missing}")

        if args.dry_run:
            status["notes"].append("dry-run: skipped model/dataloader loading; run mode requires a real probe first.")
            logprint("[DRY-RUN] skip model/dataloader loading.")
            dump_json(status_path, status)
            return 0

        try:
            obs = try_get_real_batch(cfg)
            dataloader_info = cfg.get("_dataloader_probe_info", {})
            status["dataloader_probe_requested"] = bool(dataloader_info.get("requested", False))
            status["dataloader_probe_success"] = bool(dataloader_info.get("success", False))
            status["dataloader_probe_fallback_dummy"] = bool(dataloader_info.get("fallback_dummy", False))
            status["dataloader_probe_error"] = dataloader_info.get("error")
            if dataloader_info.get("error"):
                status["notes"].append(f"dataloader probe failed: {dataloader_info['error']}")
            logprint(f"[INFO] obs/batch type: {type(obs)}")
            if isinstance(obs, dict):
                logprint(f"[INFO] obs/batch keys: {list(obs.keys())}")
        except Exception as e:
            dataloader_info = cfg.get("_dataloader_probe_info", {})
            status["dataloader_probe_requested"] = bool(dataloader_info.get("requested", False))
            status["dataloader_probe_success"] = bool(dataloader_info.get("success", False))
            status["dataloader_probe_fallback_dummy"] = bool(dataloader_info.get("fallback_dummy", False))
            status["dataloader_probe_error"] = dataloader_info.get("error") or repr(e)
            status["notes"].append(f"batch loading failed: {repr(e)}")
            dump_json(status_path, status)
            raise

        if args.dataloader_only:
            status["notes"].append("dataloader-only probe finished before model loading.")
            dump_json(status_path, status)
            logprint(f"[DONE] dataloader-only probe wrote {status_path}")
            return 0

        from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper  # noqa: E402

        instruction = (
            obs.get("task_description")
            if isinstance(obs, dict) and "task_description" in obs
            else cfg.get("probe_instruction", "pick up the object")
        )
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0]
        instruction = str(instruction)

        try:
            wrapper = OpenVLAOFTWrapper(cfg)
            wrapper.load_model()
            status["model_load_success"] = wrapper.vla is not None
            status["processor_load_success"] = wrapper.processor is not None
            status["disable_cudnn"] = bool(wrapper.cfg.disable_cudnn)
            try:
                import torch  # noqa: E402

                status["torch_cudnn_enabled_initial"] = bool(torch.backends.cudnn.enabled)
                status["torch_cudnn_enabled"] = status["torch_cudnn_enabled_initial"]
            except Exception:
                status["torch_cudnn_enabled"] = None
            logprint("[INFO] OpenVLA-OFT model and processor loaded.")
            logprint(
                "[INFO] OpenVLA-OFT disable_cudnn="
                f"{status['disable_cudnn']}; torch.backends.cudnn.enabled.initial="
                f"{status['torch_cudnn_enabled_initial']}"
            )
        except Exception as e:
            status["notes"].append(f"model/processor load failed: {repr(e)}")
            dump_json(status_path, status)
            raise

        try:
            inputs = wrapper.prepare_hf_inputs(obs, instruction)
            status["accepted_image_format"] = "official_processor_primary_plus_wrist_concat"
            if "pixel_values" in inputs:
                status["pixel_values_shape"] = tuple(inputs["pixel_values"].shape)
                # Official convention: image dimension >1 means primary+wrist.
                status["supports_native_multiview"] = inputs["pixel_values"].shape[1] >= 2
            logprint(f"[INFO] processor input keys: {list(inputs.keys())}")
            logprint(f"[INFO] pixel_values shape: {status['pixel_values_shape']}")
        except Exception as e:
            status["notes"].append(f"prepare_hf_inputs failed: {repr(e)}")
            dump_json(status_path, status)
            raise

        try:
            actions = wrapper.predict_action(obs, instruction)
            status["supports_predict_action"] = True
            logprint(f"[INFO] predict_action success; len(actions)={len(actions)}")
        except Exception as e:
            status["notes"].append(f"predict_action failed: {repr(e)}")
            logprint(f"[WARN] predict_action failed: {repr(e)}")

        try:
            result = wrapper.forward_for_attack(obs, instruction)
            if "logits" in result:
                status["supports_forward_logits"] = True
                status["logits_shape"] = tuple(result["logits"].shape)
            if "forward_error" in result:
                status["notes"].append(f"forward_error: {result['forward_error']}")
            if "predict_action_warning" in result:
                status["notes"].append(result["predict_action_warning"])
            logprint(f"[INFO] forward_for_attack keys: {list(result.keys())}")
        except Exception as e:
            status["notes"].append(f"forward_for_attack failed: {repr(e)}")
            logprint(f"[WARN] forward_for_attack failed: {repr(e)}")

        try:
            grad_status = wrapper.probe_differentiable_forward(obs, instruction)
            status["supports_pixel_values_grad_probe"] = bool(grad_status.get("pixel_delta_grad", False))
            status["supports_forward_logits"] = status["supports_forward_logits"] or bool(grad_status.get("supports_forward_logits", False))
            if grad_status.get("logits_shape"):
                status["logits_shape"] = grad_status["logits_shape"]
            status["notes"].extend(grad_status.get("notes", []))
            logprint(f"[INFO] gradient probe: {grad_status}")
        except Exception as e:
            status["notes"].append(f"gradient probe failed: {repr(e)}")
            logprint(f"[WARN] gradient probe failed: {repr(e)}")

        try:
            import torch  # noqa: E402

            status["torch_cudnn_enabled_final"] = bool(torch.backends.cudnn.enabled)
            status["torch_cudnn_enabled"] = status["torch_cudnn_enabled_final"]
        except Exception:
            status["torch_cudnn_enabled_final"] = None
        status["cudnn_fallback_used"] = bool(getattr(wrapper, "_cudnn_disabled_by_wrapper", False))
        logprint(
            "[INFO] OpenVLA-OFT cudnn final: "
            f"enabled={status['torch_cudnn_enabled_final']}; fallback_used={status['cudnn_fallback_used']}"
        )

    dump_json(status_path, status)
    print(f"[DONE] wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
