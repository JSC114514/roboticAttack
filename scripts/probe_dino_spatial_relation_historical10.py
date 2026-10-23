#!/usr/bin/env python3
"""Exploratory ten-task masked DINO screen across frozen, heterogeneous archives.

This is not a validation on the 300-state MGWP/Object training bank and includes
no patch training, action restoration, or Slurm submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from probe_dino_spatial_relation_stage0a import (
    patch_exclusion, prepare_array, require, run, sha256_file,
)


def load_historical_records(cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    base = Path(cfg["source_base"])
    require(set(map(int, cfg["sources"])) == set(map(int, cfg["task_ids"])),
            "Exactly one source is required per Object task")
    frames = []
    hashes = {}
    for task_id in map(int, cfg["task_ids"]):
        relative = cfg["sources"][str(task_id)]
        path = base / relative
        require(path.is_file(), f"Missing frozen source: {path}")
        hashes[str(task_id)] = sha256_file(path)
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".json":
            frame = pd.DataFrame(json.loads(path.read_text()))
        else:
            raise RuntimeError(f"Unsupported source format: {path}")
        require(set(frame.task_id.astype(int)) == {task_id}, f"Task identity drift: {path}")
        require(frame.clean_success.eq(1).all(), f"Source contains failed clean trajectory: {path}")
        require(not frame.groupby("trajectory_id").split.nunique().gt(1).any(),
                f"Split leakage within source trajectory: {path}")
        frame = frame.loc[frame.split.isin(cfg["included_splits"])].copy()
        require(set(frame.split) == set(cfg["included_splits"]), f"Missing split: {path}")
        counts = frame.groupby("split").trajectory_id.nunique()
        require(all(int(counts[split]) >= 3 for split in cfg["included_splits"]),
                f"Too few independent trajectories in {path}: {counts.to_dict()}")
        frame["source_origin"] = relative.split("/")[0]
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    require(records.state_id.is_unique, "Global source state collision")
    require(records.array_path.map(lambda value: Path(value).is_file()).all(), "Missing source NPZ")
    require(set(records.task_id.astype(int)) == set(range(10)), "Not all ten Object tasks present")
    return records.sort_values(["task_id", "split", "trajectory_id", "state_id"]).reset_index(drop=True), hashes


def check_only(cfg: dict, records: pd.DataFrame, hashes: dict[str, str]) -> dict:
    exclude = patch_exclusion(cfg)
    coverage = {}
    for row in records.itertuples(index=False):
        _, masks = prepare_array(row, cfg)
        oracle_valid = all(float(mask.sum()) > 0 for mask in masks.values())
        outside_valid = oracle_valid and all(float(masks[role].masked_fill(exclude, 0).sum()) > 0
                                             for role in ("target_object", "gripper"))
        key = f"{row.task_id}:{row.split}"
        entry = coverage.setdefault(key, {"rows": 0, "oracle_valid": 0, "outside_patch_valid": 0})
        entry["rows"] += 1
        entry["oracle_valid"] += int(oracle_valid)
        entry["outside_patch_valid"] += int(outside_valid)
    model = json.loads(Path(cfg["checkpoint_config"]).read_text())
    ids = model.get("timm_model_ids", [])
    require(len(ids) == 2 and "dinov2" in ids[0].lower() and "siglip" in ids[1].lower(),
            "Visual branch order drift")
    return {"status": "CHECK_ONLY_PASS", "design_status": cfg["design_status"],
            "rows": len(records), "source_tasks": sorted(map(int, cfg["task_ids"])),
            "coverage": coverage, "source_manifest_hashes": hashes,
            "gpu_job_submitted": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    records, hashes = load_historical_records(cfg)
    if args.check_only:
        print(json.dumps(check_only(cfg, records, hashes), ensure_ascii=False, indent=2))
        return
    cfg["source_manifest_hashes"] = hashes
    status = run(cfg, records)
    require(status["status"] == "COMPLETE_FEASIBILITY_ONLY", "Unexpected exploratory result status")


if __name__ == "__main__":
    main()
