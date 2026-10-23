"""From-scratch contracts. No historical trained patch or replay bank is read."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from dino_replay_common import ROOT, SOURCE_SHA, BANK_SHA, source_records, runtime_cfg, selected_decisions
from dino_window_common import clip_indices, score_and_weights
from white_patch.UADA_OFT_DINO import require, sha256_file

CONFIG = Path(os.environ.get("OFT_DINO_SCRATCH_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_scratch1000_v1.json"))
METHODS = ("uada", "uniform", "dino")
CHECKPOINTS = (500, 750, 1000)


def config(path=CONFIG, *, require_prefix=False):
    cfg = json.loads(Path(path).read_text())
    require(cfg["experiment_id"] in ("OFT-DINO-SCRATCH1000-V1", "OFT-DINO-SCRATCH1000-V2")
            and tuple(cfg["methods"]) == METHODS and cfg["seed"] == 42,
            "Scratch experiment identity changed")
    require(cfg["base_outer_iterations"] == 0 and cfg["bootstrap_outer"] == 250
            and cfg["outer_iterations"] == 1000 and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500, 750, 1000],
            "Scratch budget/checkpoint drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0] and cfg["clip_length"] == 3
            and cfg["centers_per_trajectory"] == 6
            and cfg["learning_rate"] == .002 and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20 and cfg["geometry"] is True
            and cfg["tv_threshold"] == .4 and cfg["tv_weight"] == 1,
            "Scratch threat/optimizer/appearance changed")
    version = cfg["experiment_id"].rsplit("-", 1)[-1].lower()
    root = ROOT / f"research_runs/wrist_dino_scratch_{version}"
    require(Path(cfg["output_root"]).resolve() == root.resolve()
            and Path(cfg["replay_bank"]).resolve() == (root / "data/WINDOW_BANK.pt").resolve(),
            "Scratch output must remain isolated")
    require(cfg.get("auto_eval_after_training", False) is (version == "v2"),
            "Scratch evaluation policy drift")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(sha256_file(source) == SOURCE_SHA and source.stat().st_mode & 0o222 == 0,
            "Read-only Exploring source changed")
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"] == BANK_SHA
            and sha256_file(cfg["eval_reference"]) == cfg["eval_reference_sha256"],
            "Clean bank/evaluation identity changed")
    require(not any(k in cfg for k in ("baseline_patch", "baseline_patch_sha256", "baseline_elapsed_s")),
            "Historical warm start is prohibited")
    if require_prefix:
        prefix_state(cfg)
    return cfg


def prefix_state(cfg):
    folder = Path(cfg["output_root"]) / "training/warmup"
    status = json.loads((folder / "status.json").read_text())
    resume = folder / "resume.pt"
    patch_path = folder / f"outer_{cfg['bootstrap_outer']:04d}.pt"
    require(status["status"] == "COMPLETE" and status["method"] == "warmup"
            and status["outer"] == cfg["bootstrap_outer"]
            and status["optimizer_updates"] == cfg["bootstrap_outer"] * cfg["inner_loop"]
            and status["final_patch_sha256"] == sha256_file(patch_path)
            and status["resume_sha256"] == sha256_file(resume),
            "Incomplete/changed scratch prefix")
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    require(saved["method"] == "warmup" and saved["outer"] == cfg["bootstrap_outer"]
            and saved["signature"] == status["signature"] and saved["config"] == cfg
            and len(saved["rows"]) == cfg["bootstrap_outer"]
            and torch.equal(saved["patch"], torch.load(patch_path, map_location="cpu",
                                                       weights_only=False)["patch"]),
            "Scratch prefix state/patch/config mismatch")
    return saved, sha256_file(resume), patch_path


def collection_config(path=CONFIG, *, require_prefix=True):
    cfg = config(path)
    folder = Path(cfg["output_root"]) / "training/warmup"
    patch = folder / f"outer_{cfg['bootstrap_outer']:04d}.pt"
    if require_prefix:
        prefix_state(cfg)
    cfg["baseline_patch"] = str(patch)
    cfg["baseline_patch_sha256"] = sha256_file(patch) if patch.is_file() else None
    return cfg


def replay_records(cfg):
    path = Path(cfg["replay_bank"])
    status = json.loads(path.with_name("STATUS.json").read_text())
    _, _, prefix = prefix_state(cfg)
    require(status["status"] == "COMPLETE"
            and status["config_sha256"] == sha256_file(CONFIG)
            and status["bank_sha256"] == sha256_file(path)
            and status["baseline_patch_sha256"] == sha256_file(prefix)
            and status["source_bank_sha256"] == cfg["training_bank_sha256"],
            "Replay bank did not come from this scratch prefix")
    records = torch.load(path, map_location="cpu", weights_only=False)["records"]
    require(len(records) == status["records"] and
            {int(r["task_id"]) for r in records} == set(range(10)), "Scratch bank coverage")
    return records, status["bank_sha256"]


def draw_record(clean, replay, method, total_outer, cfg):
    from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker
    require(method in ("warmup", *METHODS) and 1 <= total_outer <= cfg["outer_iterations"],
            "Invalid scratch method/budget")
    seed = cfg["data_seed"] + cfg["seed"]
    base_index = ExploringOFTOpenVLAAttacker._training_index(total_outer, len(clean), seed)
    base = clean[base_index]
    local = total_outer - cfg["bootstrap_outer"]
    if method in ("warmup", "uada") or local <= 0 or local % 2:
        return base, "clean", base_index, base_index
    phase = ("early", "middle", "late")[(local // 2 - 1) % 3]
    candidates = [i for i, r in enumerate(replay)
                  if int(r["task_id"]) == int(base["task_id"]) and r["phase_proxy"] == phase]
    require(candidates, f"Missing task/phase: {base['task_id']}/{phase}")
    weights = [float(replay[i]["dino_weight"]) for i in candidates] if method == "dino" else None
    index = random.Random(seed * 1_000_000 + local).choices(candidates, weights=weights, k=1)[0]
    return replay[index], "replay", index, base_index
