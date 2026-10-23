"""Frozen identity, three-decision clips, and matched sampling for window replay."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from dino_replay_common import (ROOT, SOURCE_SHA, runtime_cfg, selected_decisions,
                                source_records)
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.oft_correspondence_objectives import total_variation

CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_window_refine200_v1.json"
METHODS = ("clean_continue", "window_uniform", "window_dino", "window_shuffled")
SEEDS = (42, 137)


def config(path: Path = CONFIG) -> dict:
    overlay = json.loads(Path(path).read_text())
    parent = ROOT / "configs/exploring_oft_dino/object_dino_replay_refine200_v1.json"
    require(Path(overlay["parent_config"]).resolve() == parent.resolve()
            and sha256_file(parent) == overlay["parent_config_sha256"], "Parent config drift")
    cfg = {**json.loads(parent.read_text()), **overlay}
    require(cfg["experiment_id"] == "OFT-DINO-WINDOW-REFINE200-V1"
            and tuple(cfg["methods"]) == METHODS and tuple(cfg["training_seeds"]) == SEEDS,
            "Window experiment identity drift")
    require(cfg["outer_iterations"] == 200 and cfg["inner_loop"] == 50
            and cfg["base_outer_iterations"] == 2200 and cfg["clip_length"] == 3
            and cfg["centers_per_trajectory"] == 6 and cfg["patch_xy"] == [26, 137]
            and cfg["patch_shape"] == [3, 50, 50] and cfg["maskidx"] == [0]
            and cfg["learning_rate"] == .002 and cfg["tv_weight"] == 1
            and cfg["geometry"] is True, "Threat or training budget drift")
    require(Path(cfg["output_root"]).resolve() == (ROOT / "research_runs/wrist_dino_window_v1").resolve()
            and Path(cfg["replay_bank"]).resolve() ==
            (ROOT / "research_runs/wrist_dino_window_v1/data/WINDOW_BANK.pt").resolve(),
            "Window data location drift")
    original = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(original) == SOURCE_SHA and original.stat().st_mode & 0o222 == 0,
            "Read-only Exploring source changed")
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"]
            and sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"],
            "Frozen source bank or common RGB start changed")
    status = json.loads((Path(cfg["baseline_patch"]).parent / "status.json").read_text())
    require(status["status"] == "COMPLETE" and status["method"] == "replay_uniform"
            and status["total_outer"] == 2200
            and status["final_patch_sha256"] == cfg["baseline_patch_sha256"],
            "Common start provenance drift")
    patch = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"].float()
    require(tuple(patch.shape) == (3, 50, 50) and torch.isfinite(patch).all()
            and bool(((patch >= 0) & (patch <= 1)).all())
            and abs(float(total_variation(patch)) - cfg["tv_threshold"]) < 1e-8,
            "Common start RGB or TV drift")
    require(sha256_file(cfg["eval_reference"]) == cfg["eval_reference_sha256"],
            "Frozen four-suite reference changed")
    return cfg


def clip_indices(center: int, length: int) -> tuple[int, int, int]:
    require(length > 0 and 0 <= center < length, "Invalid clip center")
    return max(center - 1, 0), center, min(center + 1, length - 1)


def score_and_weights(records: list[dict], seed: int) -> list[dict]:
    require(records, "Empty window bank")
    result = [dict(r) for r in records]
    for task in range(10):
        for phase in ("early", "middle", "late"):
            ids = [i for i, r in enumerate(result)
                   if int(r["task_id"]) == task and r["phase_proxy"] == phase]
            require(ids, f"Missing task/phase window: {task}/{phase}")
            order = sorted(ids, key=lambda i: (-float(result[i]["dino_score"]),
                                               int(result[i]["state_index"]),
                                               int(result[i]["decision_index"])))
            strong = set(i for i in order if float(result[i]["dino_score"]) > 0)
            strong = set(order[:(len(order) + 1) // 2]) & strong
            weights = [2.0 if i in strong else 1.0 for i in ids]
            shuffled = weights.copy()
            random.Random(seed + 100 * task + ("early", "middle", "late").index(phase)).shuffle(shuffled)
            for i, weight, control in zip(ids, weights, shuffled, strict=True):
                result[i]["dino_weight"] = weight
                result[i]["shuffled_weight"] = control
    return result


def draw_record(clean: list[dict], replay: list[dict], method: str, outer: int,
                base_index: int, seed: int) -> tuple[dict, str, int]:
    require(method in METHODS and 1 <= outer <= 200 and 0 <= base_index < len(clean),
            "Invalid matched window draw")
    if method == "clean_continue" or outer % 2 == 1:
        return clean[base_index], "clean", base_index
    task_id = int(clean[base_index]["task_id"])
    phase = ("early", "middle", "late")[(outer // 2 - 1) % 3]
    candidates = [i for i, r in enumerate(replay)
                  if int(r["task_id"]) == task_id and r["phase_proxy"] == phase]
    require(candidates, f"No replay clip for task/phase {task_id}/{phase}")
    field = {"window_uniform": None, "window_dino": "dino_weight",
             "window_shuffled": "shuffled_weight"}[method]
    weights = None if field is None else [float(replay[i][field]) for i in candidates]
    index = random.Random(seed * 1_000_000 + outer).choices(candidates, weights=weights, k=1)[0]
    return replay[index], "replay", index
