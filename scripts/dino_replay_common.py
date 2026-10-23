"""Frozen contracts and deterministic sampling for the Wrist-DINO replay experiment."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
from white_patch.UADA_OFT_DINO import require, sha256_file
from white_patch.UADA_OFT_CORRESPONDENCE import record_key
from white_patch.oft_correspondence_objectives import total_variation

CONFIG = ROOT / "configs/exploring_oft_dino/object_dino_replay_refine200_v1.json"
METHODS = ("clean_continue", "replay_uniform", "replay_dino", "replay_shuffled")
SOURCE_SHA = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
BASE_SHA = "ab2529a1957a654613402e4173a04af3f3659a24a9ce97e5a2c3165dd9541c52"
BANK_SHA = "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8"


def config(path: Path = CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text())
    require(cfg["experiment_id"] == "OFT-DINO-REPLAY-REFINE200-V1"
            and tuple(cfg["methods"]) == METHODS, "Replay experiment identity drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0] and cfg["outer_iterations"] == 200
            and cfg["base_outer_iterations"] == 2000 and cfg["inner_loop"] == 50,
            "Frozen threat/budget drift")
    require(cfg["learning_rate"] == 0.002 and cfg["warmup_outer_iterations"] == 20
            and cfg["tv_weight"] == 1 and cfg["geometry"] is True
            and cfg["resume_every_outer_iterations"] == 5,
            "Exploring optimization drift")
    require(Path(cfg["output_root"]).resolve() == (ROOT / "research_runs/wrist_dino_replay_v1").resolve()
            and Path(cfg["replay_bank"]).resolve() ==
            (ROOT / "research_runs/wrist_dino_replay_v1/data/REPLAY_BANK.pt").resolve(),
            "Replay output path drift")
    require(sha256_file(ROOT / "VLAAttacker/white_patch/UADA.py") == SOURCE_SHA
            and (ROOT / "VLAAttacker/white_patch/UADA.py").stat().st_mode & 0o222 == 0,
            "Read-only Exploring source changed")
    require(sha256_file(cfg["training_bank"]) == cfg["training_bank_sha256"] == BANK_SHA,
            "Source Object bank changed")
    require(sha256_file(cfg["baseline_patch"]) == cfg["baseline_patch_sha256"] == BASE_SHA,
            "Frozen UADA2000 patch changed")
    baseline = torch.load(cfg["baseline_patch"], map_location="cpu", weights_only=False)["patch"].float()
    require(tuple(baseline.shape) == (3, 50, 50) and bool(torch.isfinite(baseline).all())
            and bool(((baseline >= 0) & (baseline <= 1)).all())
            and abs(float(total_variation(baseline)) - cfg["tv_threshold"]) < 1e-8,
            "Frozen UADA2000 RGB/TV drift")
    status = json.loads((Path(cfg["baseline_patch"]).parent / "status.json").read_text())
    require(status["status"] == "COMPLETE" and status["outer_iterations"] == 2000
            and status["inner_loop"] == 50 and status["elapsed_s"] == cfg["baseline_elapsed_s"],
            "Frozen UADA2000 status drift")
    require(sha256_file(cfg["eval_reference"]) == cfg["eval_reference_sha256"],
            "Four-suite evaluation reference drift")
    checkpoint = json.loads((Path(cfg["checkpoint"]) / "config.json").read_text())
    branches = checkpoint.get("timm_model_ids", [])
    require(len(branches) == 2 and "dinov2" in branches[0].lower() and "siglip" in branches[1].lower(),
            "Visual branch order drift")
    return cfg


def source_records(cfg: dict) -> list[dict]:
    records = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)["records"]
    keys = [record_key(row) for row in records]
    require(len(records) == len(set(keys)) == 300, "Expected 300 distinct clean source states")
    for task_id in range(10):
        subset = [r for r in records if int(r["task_id"]) == task_id]
        starts = {int(r["state_index"]) for r in subset}
        require(len(subset) == 30 and len(starts) == 5 and all(10 <= x < 40 for x in starts),
                f"Source/evaluation state split drift: task {task_id}")
    return records


def selected_decisions(count: int, maximum: int = 6) -> list[int]:
    require(count > 0 and maximum == 6, "Invalid source decision count")
    if count <= maximum:
        return list(range(count))
    return [round(i * (count - 1) / (maximum - 1)) for i in range(maximum)]


def runtime_cfg(cfg: dict) -> dict:
    """Use the frozen four-suite runner's Object model and simulator setup."""
    from run_exploring_oft_scale500_cross_suite import runtime_config
    base = json.loads(Path(cfg["eval_reference"]).read_text())
    result = runtime_config(base, "libero_object")
    result["max_steps"] = base["max_steps_by_suite"]["libero_object"]
    return result


def score_and_weights(records: list[dict], seed: int) -> list[dict]:
    """Keep successful attacked trajectories where available; rank DINO restoration within task."""
    chosen = []
    for task_id in range(10):
        task_rows = [r for r in records if int(r["task_id"]) == task_id]
        require(task_rows, f"Missing replay task {task_id}")
        survivors = [r for r in task_rows if r["attacked_episode_success"]]
        chosen.extend(survivors if survivors else task_rows)
    require(all(math.isfinite(float(r["dino_score"])) and float(r["dino_score"]) >= 0 for r in chosen),
            "Nonfinite/negative DINO score")
    result = [dict(row) for row in chosen]
    for task_id in range(10):
        ids = [i for i, r in enumerate(result) if int(r["task_id"]) == task_id]
        ranked = sorted(ids, key=lambda i: (-float(result[i]["dino_score"]),
                                            int(result[i]["state_index"]), int(result[i]["decision_index"])))
        positive = [i for i in ranked if float(result[i]["dino_score"]) > 0]
        strong = set(positive[:(len(ranked) + 1) // 2])
        for i in ids:
            result[i]["dino_weight"] = 2.0 if i in strong else 1.0
        phases = sorted({str(result[i]["phase_proxy"]) for i in ids})
        for phase in phases:
            phase_ids = [i for i in ids if str(result[i]["phase_proxy"]) == phase]
            weights = [result[i]["dino_weight"] for i in phase_ids]
            random.Random(seed + 100 * task_id + {"early": 0, "middle": 1, "late": 2}[phase]).shuffle(weights)
            for i, weight in zip(phase_ids, weights, strict=True):
                result[i]["shuffled_weight"] = weight
    return result


def draw_record(clean: list[dict], replay: list[dict], method: str, outer: int,
                base_index: int, seed: int) -> tuple[dict, str, int]:
    require(method in METHODS and 1 <= outer <= 200 and 0 <= base_index < len(clean),
            "Invalid matched replay draw")
    base = clean[base_index]
    if method == METHODS[0] or outer % 2 == 1:
        return base, "clean", base_index
    task_id = int(base["task_id"])
    candidates = [i for i, record in enumerate(replay) if int(record["task_id"]) == task_id]
    require(candidates, f"No replay states for task {task_id}")
    weight_field = {METHODS[1]: None, METHODS[2]: "dino_weight", METHODS[3]: "shuffled_weight"}[method]
    weights = None if weight_field is None else [float(replay[i][weight_field]) for i in candidates]
    selected = random.Random(seed * 1_000_000 + outer).choices(candidates, weights=weights, k=1)[0]
    return replay[selected], "replay", selected
