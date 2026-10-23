#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "VLAAttacker", PROJECT_ROOT / "VLAAttacker" / "white_patch"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_wrapper(cfg: dict):
    for key in ("openvla_oft_root", "libero_root", "attribution_root"):
        value = str(cfg[key])
        sys.path[:] = [entry for entry in sys.path if entry != value]
        sys.path.insert(0, value)
    torch.backends.cudnn.enabled = False
    from VLAAttacker.multiview.openvla_oft_wrapper import OpenVLAOFTWrapper

    wrapper = OpenVLAOFTWrapper({
        "model_checkpoint": cfg["checkpoint"],
        "task_suite": cfg["suite"],
        "unnorm_key": cfg["unnorm_key"],
        "input_views": "front,wrist",
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "num_open_loop_steps": cfg["num_open_loop_steps"],
        "device": "cuda",
        "dtype": cfg["dtype"],
        "allow_cpu": False,
        "hf_local_files_only": True,
        "openvla_oft_root": cfg["openvla_oft_root"],
        "libero_root": cfg["libero_root"],
    })
    wrapper.load_model()
    require(wrapper.action_head is not None, "OFT L1 action head is required")
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "exploring_oft_dino" / "object_timeboxed.json"))
    parser.add_argument("--method", choices=(
        "exploring_oft_uada", "exploring_oft_dino_cf",
        "exploring_oft_dino_cf_worst_chunk", "exploring_oft_full_worst_chunk",
        "exploring_oft_dino_restore",
        "exploring_oft_siglip_cf"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    require(args.method in cfg["methods"], f"Method {args.method} is not frozen in {cfg_path}")
    bank_path = Path(cfg["training_bank"])
    require(bank_path.is_file(), f"Missing training bank: {bank_path}")
    require(Path(cfg["checkpoint"], "config.json").is_file(), "Missing OFT checkpoint")
    model_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text(encoding="utf-8"))
    timm_ids = model_cfg.get("timm_model_ids", [])
    require(len(timm_ids) == 2 and "dinov2" in timm_ids[0].lower()
            and "siglip" in timm_ids[1].lower(),
            "Checkpoint visual branch order drift: expected featurizer=DINO, fused_featurizer=SigLIP")
    require(cfg["batch_size"] == 1, "Frozen single-GPU contract requires batch_size=1")
    if cfg["experiment_id"] == "EXPLORING-OFT-DINO-TARGET5-OBJECT2000-01":
        require(cfg["maskidx"] == [5] and cfg["methods"] == ["exploring_oft_uada"],
                "DINO-guided target experiment must use unchanged UADA core and action index 5")
    else:
        require(cfg["maskidx"] == [0], "Exploring reproduction maskidx drift")
    require(cfg["outer_iterations"] in (200, 500, 2000) and cfg["inner_loop"] == 50,
            "Frozen Exploring-OFT training budget drift")
    require(cfg["checkpoint_outer_iterations"][-1] == cfg["outer_iterations"],
            "Final checkpoint must equal outer_iterations")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "method": args.method, "config": str(cfg_path),
                          "old_patch_warm_start": False, "source": "white_patch/UADA.py",
                          "optimizer_updates": cfg["outer_iterations"] * cfg["inner_loop"]}, ensure_ascii=False, indent=2))
        return
    require(torch.cuda.is_available(), "CUDA GPU required")
    require(torch.cuda.device_count() == 1, "Exactly one Slurm-mapped GPU is required")
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}), flush=True)
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == 300, "Expected 300 frozen source-suite training states")
    from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker

    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTOpenVLAAttacker(wrapper, cfg["output_root"], resize_patch=bool(cfg["resize_patch"]))
    attacker.patchattack_unconstrained_oft(records, cfg, args.method)


if __name__ == "__main__":
    main()
