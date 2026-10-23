#!/usr/bin/env python3
"""Three-arm paired Wrist-patch evaluation, reusing the frozen Scale-500 runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "oft_ramp_recovery_aware" / "scripts"))

from common import load_json, require, require_gpu, sha256_file  # noqa: E402
from run_exploring_oft_scale500_cross_suite import run_shard  # noqa: E402


CONDITIONS = (
    "exploring_oft_uada",
    "exploring_oft_dino_jacobian_dense_r03",
    "exploring_oft_dino_jacobian_dense_r10",
)
MATCH_REFERENCE_KEYS = (
    "training_suite", "suites", "task_count", "state_indices", "frozen_object_clean",
    "frozen_cross_suite_clean", "order_seed", "bootstrap_seed", "num_steps_wait",
    "max_steps_by_suite", "num_open_loop_steps", "image_resolution", "patch_shape",
    "patch_xy", "dtype", "checkpoints", "unnorm_keys", "paths",
)


def validate_contract(cfg: dict) -> None:
    reference = load_json(cfg["reference_eval_config"])
    for key in MATCH_REFERENCE_KEYS:
        require(cfg[key] == reference[key], f"Scale-500 evaluator drift: {key}")
    require(tuple(cfg["conditions"]) == CONDITIONS, "Three-arm method grid drift")
    require(cfg["analysis_kind"] == "DINO_JACOBIAN500_VS_UADA500", "Analysis kind drift")
    require(cfg["manual_visual_review"] == "USER_REQUESTED_TEST_20260925",
            "User-requested visual review marker missing")


def load_jacobian_patches(cfg: dict, device: torch.device) -> dict[str, tuple[torch.Tensor, str]]:
    """Require completed training, matching checkpoint identity, and frozen hashes."""
    validate_contract(cfg)
    require(set(cfg["patch_sources"]) == set(CONDITIONS), "Patch source coverage drift")
    patches: dict[str, tuple[torch.Tensor, str]] = {}
    hashes: set[str] = set()
    for condition in CONDITIONS:
        source = cfg["patch_sources"][condition]
        method = source["source_method"]
        require(method == condition, f"Source method drift: {condition}")
        status = load_json(source["training_status"])
        if condition == "exploring_oft_uada":
            require(status["status"] == "READY_FOR_MANUAL_VISUAL_REVIEW"
                    and any(row.get("method") == method and row.get("status") == "COMPLETE"
                            and row.get("experiment_id") == source["experiment_id"]
                            for row in status["methods"]), "UADA-500 baseline is not complete")
        else:
            require(status["status"] == "COMPLETE"
                    and status["method"] == method
                    and status["experiment_id"] == source["experiment_id"]
                    and int(status["outer_iterations"]) == 500,
                    f"DINO-Jacobian training incomplete: {method}")
            gate = load_json(Path(source["patch_path"]).with_name("structure_gate_0500.json"))
            require(gate["outer_iteration"] == 500 and gate["passed"],
                    f"DINO-Jacobian structure gate failed: {method}")
        path = Path(source["patch_path"])
        require(path.is_file(), f"Missing patch: {path}")
        digest = sha256_file(path)
        require(digest == source["sha256"] and digest not in hashes,
                f"Patch hash mismatch or duplicate: {method}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload.get("experiment_id") == source["experiment_id"]
                and payload.get("method") == method
                and int(payload.get("outer_iterations", -1)) == 500,
                f"Patch training identity drift: {method}")
        patch = payload["patch"].float()
        require(tuple(patch.shape) == tuple(cfg["patch_shape"])
                and torch.isfinite(patch).all()
                and bool(((patch >= 0) & (patch <= 1)).all()),
                f"Invalid patch tensor: {method}")
        hashes.add(digest)
        patches[condition] = (patch.to(device), digest)
    return patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--array-index", type=int, choices=range(40), required=True)
    args = parser.parse_args()
    cfg = load_json(args.config.resolve())
    validate_contract(cfg)
    print(json.dumps(require_gpu()), flush=True)
    run_shard(args.array_index, args.config.resolve(), patch_loader=load_jacobian_patches)


if __name__ == "__main__":
    main()
