from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "VLAAttacker", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from VLAAttacker.UADA_OFT_DINO_SAMPLING_wrapper import validate
from white_patch.UADA_OFT_DINO_SAMPLING import METHODS, draw_record_index
from white_patch.UADA_OFT_DINO import sha256_file
from plan_dino_guided_placement import candidate_masks


CONFIG = ROOT / "configs/exploring_oft_dino/dino_sampling500_v1.json"


def test_plan_and_training_contract_are_frozen() -> None:
    cfg = json.loads(CONFIG.read_text())
    assert tuple(cfg["methods"]) == METHODS
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 25_000
    assert validate(cfg, CONFIG, METHODS[0]) == (91, 140)
    assert validate(cfg, CONFIG, METHODS[1]) == (91, 140)
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    assert sha256_file(source) == "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
    assert not (source.stat().st_mode & stat.S_IWUSR)


def test_candidate_footprints_are_valid_for_224_wrist_image() -> None:
    sites, masks = candidate_masks()
    assert len(sites) == len(set(sites)) == len(masks) == 676
    assert (91, 140) in sites and (84, 140) in sites
    assert masks.shape == (676, 256)
    assert (masks >= 0).all() and (masks <= 1).all()


def test_dino_sampling_changes_record_but_preserves_frozen_task_schedule() -> None:
    records = [{"task_id": task, "trajectory_rank": rank, "decision_index": decision}
               for task in range(10) for rank in range(5) for decision in range(6)]
    scores = {(r["task_id"], r["trajectory_rank"], r["decision_index"]):
              (100.0 if r["trajectory_rank"] == 0 and r["decision_index"] == 0 else 1e-6)
              for r in records}
    changed = 0
    for outer in range(1, 301):
        control = draw_record_index(outer, records, scores, 20260920, METHODS[0])
        guided = draw_record_index(outer, records, scores, 20260920, METHODS[1])
        assert records[control]["task_id"] == records[guided]["task_id"]
        assert guided == draw_record_index(outer, records, scores, 20260920, METHODS[1])
        changed += control != guided
    assert changed > 200


def test_manual_submission_style_and_no_source_edit() -> None:
    slurm = (ROOT / "slurm/exploring_oft_dino/44_train_dino_sampling500_3090.sbatch").read_text()
    submit = (ROOT / "submit_exploring_oft_dino_sampling500.sh").read_text()
    trainer = (ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO_SAMPLING.py").read_text()
    assert "#SBATCH --array=0-1%2" in slurm
    assert "#SBATCH --gres=gpu:3090:1" in slurm
    assert "sbatch " not in slurm
    assert "CHECK_ONLY" in submit
    assert "from white_patch.UADA_OFT_SOURCE_COPY import" in trainer
    assert "action_loss.backward()" in trainer
    assert "decoy_loss" not in trainer
