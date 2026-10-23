#!/usr/bin/env python3
"""Action-screen one-token Wrist-DINO spatial transports before pixel training.

Four directions are compared on ranks 0-3 of each Object task.  The selected
direction is then tested once on held-out rank 4 with identical action-sampling
RNG and clean Front/SigLIP.  This is a feature-level *training target screen*,
not a deployable attack or closed-loop result.
"""

from __future__ import annotations

import argparse
import json
import random
import stat
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "VLAAttacker", ROOT / "VLAAttacker/white_patch", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from UADA_OFT_DINO_wrapper import load_wrapper  # noqa: E402
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file  # noqa: E402
from white_patch.UADA_OFT_DINO_SPATIAL import (  # noqa: E402
    SHIFTS, ExploringOFTDinoSpatialAttacker, site_token_mask,
    spatial_teacher, target_support,
)
from white_patch.UADA_OFT_JOINT500 import opposite_boundary_target  # noqa: E402
from white_patch.UADA_OFT_OBJECT_DECOY import load_object_masks, record_key  # noqa: E402

SOURCE_SHA256 = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
EXPERIMENT_ID = "EXPLORING-OFT-WRIST-DINO-SPATIAL500-V1"


def validate(cfg: dict) -> None:
    require(cfg["experiment_id"] == EXPERIMENT_ID, "Experiment ID drift")
    require(cfg["site_xy"] == [91, 140] and cfg["patch_shape"] == [3, 50, 50],
            "Frozen matched Wrist site/patch shape drift")
    require(cfg["suite"] == "libero_object" and cfg["maskidx"] == [0],
            "Frozen Object UADA training contract drift")
    require(cfg["pilot_discovery_ranks"] == [0, 1, 2, 3]
            and cfg["pilot_holdout_rank"] == 4
            and cfg["pilot_directions"] == list(SHIFTS), "Pilot split/candidate drift")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_SHA256
            and not (source.stat().st_mode & stat.S_IWUSR), "Read-only UADA source drift")
    require(Path(cfg["training_bank"]).is_file(), "Missing frozen training bank")
    require(Path(cfg["object_mask_root"]).is_dir(), "Missing Object mask receipts")
    checkpoint_cfg = json.loads(Path(cfg["checkpoint"], "config.json").read_text())
    ids = checkpoint_cfg.get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower()
            and "siglip" in ids[1].lower(), "Expected DINO then SigLIP checkpoint branches")


def seed_forward(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_record(attacker: ExploringOFTDinoSpatialAttacker, record: dict,
                    target_tokens: torch.Tensor, roi: torch.Tensor, cfg: dict,
                    directions: list[str]) -> dict:
    key = record_key(record)
    row = {"task_id": key[0], "trajectory_rank": key[1], "decision_index": key[2],
           "state_index": int(record["state_index"]), "candidates": {}}
    if all(float(target_support(roi, target_tokens, SHIFTS[name]).sum()) < 0.5
           for name in directions):
        row["eligible"] = False
        return row
    row["eligible"] = True
    inputs, proprio = attacker._base_inputs(record)
    observation = record["observation"]
    seed = int(cfg["pilot_seed"]) + key[0] * 100000 + key[1] * 1000 + key[2]
    with torch.no_grad():
        seed_forward(seed)
        clean_action, clean_dino, clean_siglip = attacker._forward(
            inputs, proprio, observation, None, False, capture_siglip=True)
        require(clean_siglip is not None, "Missing clean Wrist-SigLIP capture")
        low = torch.as_tensor(cfg["action_head_q01"], device=clean_action.device).reshape(1, 1, 7)
        high = torch.as_tensor(cfg["action_head_q99"], device=clean_action.device).reshape(1, 1, 7)
        boundary = opposite_boundary_target(clean_action, low, high)
        clean_mse = float(F.mse_loss(clean_action[..., 0], boundary[..., 0]))
        action_range = float((high[..., 0] - low[..., 0]).mean())
        for name in directions:
            shift = SHIFTS[name]
            support = target_support(roi, target_tokens, shift)
            if float(support.sum()) < 0.5:
                continue
            teacher = spatial_teacher(clean_dino, roi, shift)
            seed_forward(seed)
            intervened_action, _, _ = attacker._forward(
                inputs, proprio, observation, None, False,
                clean_dino=teacher, clean_siglip=clean_siglip)
            mse = float(F.mse_loss(intervened_action[..., 0], boundary[..., 0]))
            row["candidates"][name] = {
                "boundary_mse_improvement": clean_mse - mse,
                "action_delta_over_range": float(
                    (intervened_action[..., 0] - clean_action[..., 0]).abs().mean()) / action_range,
                "support_mass": float(support.sum()),
            }
    return row


def run_task(cfg: dict, config_path: Path, task_id: int) -> None:
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one Slurm-mapped CUDA GPU required")
    bank = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)
    records = bank["records"]
    require(len(records) == 300, "Expected frozen 300-record Object bank")
    task_records = [record for record in records if int(record["task_id"]) == task_id]
    require(len(task_records) == 30, "Expected 30 records per task")
    masks = load_object_masks(cfg, records)
    wrapper = load_wrapper(cfg)
    attacker = ExploringOFTDinoSpatialAttacker(wrapper, cfg["output_root"], tuple(cfg["site_xy"]))
    roi = site_token_mask(tuple(cfg["site_xy"])).to(wrapper.device)
    discovery = []
    for record in task_records:
        if int(record["trajectory_rank"]) not in cfg["pilot_discovery_ranks"]:
            continue
        row = evaluate_record(attacker, record, masks[record_key(record)]["target_object"].to(wrapper.device),
                              roi, cfg, list(SHIFTS))
        discovery.append(row)
        print(f"DINO_SPATIAL_PILOT task={task_id} discovery={len(discovery)}/24", flush=True)
    require(len(discovery) == 24, "Discovery split size drift")
    means = {name: float(np.mean([
        row["candidates"][name]["boundary_mse_improvement"]
        for row in discovery if name in row["candidates"]]))
        if any(name in row["candidates"] for row in discovery) else float("-inf")
        for name in SHIFTS}
    selected = max(SHIFTS, key=lambda name: means[name])
    holdout = []
    for record in task_records:
        if int(record["trajectory_rank"]) != cfg["pilot_holdout_rank"]:
            continue
        row = evaluate_record(attacker, record, masks[record_key(record)]["target_object"].to(wrapper.device),
                              roi, cfg, [selected])
        holdout.append(row)
        print(f"DINO_SPATIAL_PILOT task={task_id} holdout={len(holdout)}/6", flush=True)
    require(len(holdout) == 6, "Held-out split size drift")
    result = {"status": "TASK_COMPLETE", "experiment_id": EXPERIMENT_ID,
              "task_id": task_id, "site_xy": cfg["site_xy"],
              "source_config_sha256": sha256_file(config_path),
              "training_bank_sha256": sha256_file(cfg["training_bank"]),
              "source_uada_sha256": SOURCE_SHA256,
              "selected_shift": selected, "discovery_means": means,
              "discovery_eligible": sum(bool(row["eligible"]) for row in discovery),
              "holdout_eligible": sum(bool(row["eligible"]) for row in holdout),
              "discovery": discovery, "holdout": holdout}
    output = Path(cfg["pilot_output_root"]) / f"task_{task_id:02d}.json"
    require(not output.exists(), f"Pilot shard already exists: {output}")
    atomic_json(result, output)
    print(f"DINO_SPATIAL_PILOT_TASK_COMPLETE task={task_id} shift={selected}", flush=True)


def summarize(cfg: dict, config_path: Path) -> dict:
    shards = []
    for task_id in range(10):
        path = Path(cfg["pilot_output_root"]) / f"task_{task_id:02d}.json"
        require(path.is_file(), f"Missing pilot task shard: {task_id}")
        shard = json.loads(path.read_text())
        require(shard["status"] == "TASK_COMPLETE" and shard["task_id"] == task_id
                and shard["experiment_id"] == EXPERIMENT_ID
                and shard["source_config_sha256"] == sha256_file(config_path)
                and shard["training_bank_sha256"] == sha256_file(cfg["training_bank"])
                and shard["site_xy"] == cfg["site_xy"], f"Pilot task lineage drift: {task_id}")
        shards.append(shard)
    selected = {str(shard["task_id"]): shard["selected_shift"] for shard in shards}
    task_results = {}
    enabled = []
    for shard in shards:
        name = shard["selected_shift"]
        values = [row["candidates"][name] for row in shard["holdout"]
                  if name in row["candidates"]]
        mean_improvement = float(np.mean([v["boundary_mse_improvement"] for v in values])) if values else 0.0
        mean_delta = float(np.mean([v["action_delta_over_range"] for v in values])) if values else 0.0
        task_results[str(shard["task_id"])] = {
            "shift": name, "holdout_eligible": len(values),
            "heldout_boundary_mse_improvement": mean_improvement,
            "heldout_action_delta_over_range": mean_delta,
        }
        if len(values) >= 2 and mean_improvement > 0 and mean_delta >= 0.01:
            enabled.append(shard["task_id"])
    status = "PASS" if len(enabled) >= 6 else "FAIL_NO_ACTION_SCREENED_DINO_TARGET"
    result = {"status": status, "experiment_id": EXPERIMENT_ID,
              "source_config_sha256": sha256_file(config_path),
              "training_bank_sha256": sha256_file(cfg["training_bank"]),
              "site_xy": cfg["site_xy"], "selected_shifts": selected,
              "enabled_tasks": enabled, "task_results": task_results,
              "gate": "at least six tasks with >=2 eligible held-out records, positive mean boundary improvement, and >=1% mean normalized action change",
              "interpretation": "Feature-level Wrist-DINO action screen only; pixel realizability and closed-loop strength remain unproven"}
    output = Path(cfg["pilot_receipt"])
    require(not output.exists(), f"Pilot receipt already exists: {output}")
    atomic_json(result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exploring_oft_dino/dino_spatial500_v1.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--task-id", type=int, choices=range(10))
    group.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text())
    validate(cfg)
    if args.preflight:
        print(json.dumps({"status": "PREFLIGHT_PASS", "experiment_id": EXPERIMENT_ID,
                          "source_uada": "read-only", "gpu_job_submitted": False,
                          "pilot_tasks": 10, "candidate_shifts": list(SHIFTS),
                          "site_xy": cfg["site_xy"]}, indent=2))
    elif args.summarize:
        print(json.dumps(summarize(cfg, config_path), indent=2))
    else:
        run_task(cfg, config_path, args.task_id)


if __name__ == "__main__":
    main()
