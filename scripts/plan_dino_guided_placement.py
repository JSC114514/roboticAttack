#!/usr/bin/env python3
"""Choose one frozen Wrist patch site from the Object training bank, CPU only.

The DINO rule uses task-specific target/background prototypes fitted on four
trajectories per task. A fifth trajectory is used only to audit the selected
site against an oracle-mask position prior; it never selects the site.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from readout_object_bank_spatial_cpu import load_manifest
from replay_object_bank_mask_pilot import require, sha256_file
from probe_dino_spatial_relation_stage0a import transform_mask


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT.parent / "oft_mgwp_wrist_only/artifacts/GEOMETRY_TRAIN_BANK.pt"
OUTPUT = ROOT / "outputs/exploring_oft_dino/dino_guided_placement_v1"
PATCH_SIZE = 50
IMAGE_SIZE = 224
GRID_STEP = 7
TEMPERATURE = 0.05
OLD_XY = (26, 137)


def candidate_masks() -> tuple[list[tuple[int, int]], torch.Tensor]:
    axis = list(range(0, IMAGE_SIZE - PATCH_SIZE + 1, GRID_STEP))
    if axis[-1] != IMAGE_SIZE - PATCH_SIZE:
        axis.append(IMAGE_SIZE - PATCH_SIZE)
    xy = [(x, y) for y in axis for x in axis]
    masks = []
    for x, y in xy:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = 1
        masks.append(transform_mask(mask))
    return xy, torch.stack(masks).float()


def prepare(manifest: list[dict]) -> list[dict]:
    rows = []
    for item in manifest:
        with np.load(item["mask_path"], allow_pickle=False) as arrays:
            masks = {role: transform_mask(np.asarray(arrays[f"wrist_pixel_mask_{role}"]))
                     for role in ("target_object", "matched_background")}
        features = item["row"]["clean_dino"]
        require(isinstance(features, torch.Tensor) and tuple(features.shape) == (1, 256, 1024),
                "Unexpected clean Wrist-DINO feature shape")
        rows.append({"task_id": item["task_id"], "rank": item["rank"],
                     "split": item["split"], "features": features[0].float(), "masks": masks})
    return rows


def fit_prototypes(rows: list[dict]) -> dict[int, dict[str, torch.Tensor]]:
    sums = {task: {role: torch.zeros(1024) for role in ("target_object", "matched_background")}
            for task in range(10)}
    masses = {task: {role: 0.0 for role in ("target_object", "matched_background")}
              for task in range(10)}
    for item in rows:
        if item["split"] != "discovery":
            continue
        task = item["task_id"]
        for role, mask in item["masks"].items():
            sums[task][role] += (item["features"] * mask[:, None]).sum(0)
            masses[task][role] += float(mask.sum())
    require(all(masses[t][r] > 0 for t in range(10) for r in masses[t]),
            "Empty task prototype")
    return {task: {role: F.normalize(sums[task][role] / masses[task][role], dim=0)
                   for role in sums[task]} for task in range(10)}


def score_sites(rows: list[dict], candidates: torch.Tensor,
                prototypes: dict[int, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    dino = torch.zeros(len(candidates))
    prior = torch.zeros(len(candidates))
    used = 0
    for item in rows:
        if item["split"] != "discovery":
            continue
        task_proto = prototypes[item["task_id"]]
        features = F.normalize(item["features"], dim=-1)
        logits = features @ (task_proto["target_object"] - task_proto["matched_background"])
        probability = torch.softmax(logits / TEMPERATURE, dim=0)
        truth = item["masks"]["target_object"]
        dino += candidates @ probability
        prior += candidates @ (truth / truth.sum().clamp_min(1e-6))
        used += 1
    require(used == 240, f"Discovery coverage drift: {used}")
    return dino / used, prior / used


def heldout_audit(rows: list[dict], sites: dict[str, tuple[int, int]],
                  xy: list[tuple[int, int]], candidates: torch.Tensor,
                  prototypes: dict[int, dict[str, torch.Tensor]]) -> pd.DataFrame:
    indices = {name: xy.index(site) for name, site in sites.items() if site in xy}
    require(set(indices) == set(sites), "A site is outside the candidate grid")
    result = []
    for item in rows:
        if item["split"] != "test":
            continue
        target = item["masks"]["target_object"]
        task_proto = prototypes[item["task_id"]]
        features = F.normalize(item["features"], dim=-1)
        logits = features @ (task_proto["target_object"] - task_proto["matched_background"])
        probability = torch.softmax(logits / TEMPERATURE, dim=0)
        for name, index in indices.items():
            footprint = candidates[index]
            result.append({"task_id": item["task_id"], "trajectory_rank": item["rank"],
                           "site": name, "x": sites[name][0], "y": sites[name][1],
                           "dino_probability_mass": float(footprint @ probability),
                           "target_mask_coverage": float((footprint * target).sum() / target.sum()),
                           "patch_target_fraction": float((footprint * target).sum() / footprint.sum())})
    frame = pd.DataFrame(result)
    require(len(frame) == 60 * len(sites), "Heldout audit coverage drift")
    return frame


def plan(output: Path) -> dict:
    require(not output.exists(), f"Refusing to overwrite placement output: {output}")
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    manifest = load_manifest(bank)
    rows = prepare(manifest)
    xy, candidates = candidate_masks()
    prototypes = fit_prototypes(rows)
    dino, prior = score_sites(rows, candidates, prototypes)
    # Tie-breaking is deterministic because candidates are ordered by y, then x.
    dino_site = xy[int(torch.argmax(dino))]
    prior_site = xy[int(torch.argmax(prior))]
    sites = {"dino": dino_site, "position_prior": prior_site}
    audit = heldout_audit(rows, sites, xy, candidates, prototypes)
    summary = audit.groupby("site", as_index=False).agg(
        heldout_target_mask_coverage=("target_mask_coverage", "mean"),
        heldout_patch_target_fraction=("patch_target_fraction", "mean"),
        heldout_dino_probability_mass=("dino_probability_mass", "mean"),
    )
    require(audit.groupby(["site", "task_id"]).size().eq(6).all(),
            "Heldout trajectories are not paired")
    output.mkdir(parents=True)
    audit.to_csv(output / "heldout_audit.csv", index=False)
    summary.to_csv(output / "heldout_summary.csv", index=False)
    result = {"status": "PLACEMENT_CPU_AUDIT_COMPLETE", "training_bank_sha256": sha256_file(BANK),
              "selection_split": "discovery_only_240_states_40_trajectories",
              "audit_split": "heldout_60_states_10_trajectories",
              "patch_shape": [3, PATCH_SIZE, PATCH_SIZE], "image_size": IMAGE_SIZE,
              "candidate_step": GRID_STEP, "candidate_count": len(xy),
              "temperature": TEMPERATURE, "dino_xy": list(dino_site),
              "position_prior_xy": list(prior_site), "old_xy": list(OLD_XY),
              "discovery_dino_score_at_dino_xy": float(dino.max()),
              "discovery_prior_score_at_prior_xy": float(prior.max()),
              "heldout": summary.to_dict("records"),
              "interpretation_limit": "Site choice uses dense DINO features, but superior site or attack effect must not be attributed to DINO without position-matched controls."}
    (output / "placement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(plan(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
