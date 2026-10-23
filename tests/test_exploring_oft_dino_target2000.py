from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/exploring_oft_dino/uada_object2000.json"
CANDIDATE = ROOT / "configs/exploring_oft_dino/dino_target5_object2000.json"
ARTIFACT = ROOT.parents[0] / "oft_mgwp_wrist_only/artifacts/mgwp_v2/ACTION_SENSITIVE_DIRECTION.pt"


def test_candidate_changes_only_predeclared_target_and_output() -> None:
    base = json.loads(BASE.read_text())
    candidate = json.loads(CANDIDATE.read_text())
    assert candidate["maskidx"] == [5]
    assert candidate["methods"] == ["exploring_oft_uada"]
    assert candidate["output_root"] != base["output_root"]
    assert "initial_patch" not in candidate
    changed = {key for key in base if base[key] != candidate[key]}
    assert changed == {"experiment_id", "maskidx", "output_root"}
    assert set(candidate) - set(base) == {"dino_target_selection"}


def test_target_follows_frozen_dino_geometry_action_response() -> None:
    candidate = json.loads(CANDIDATE.read_text())
    artifact_hash = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert artifact_hash == candidate["dino_target_selection"]["direction_artifact_sha256"]
    artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    response = artifact["b_star"].abs().mean(dim=0)
    assert response.numel() == 6
    assert int(response.argmax()) == candidate["maskidx"][0]


def test_same_training_core_and_single_4090() -> None:
    wrapper = (ROOT / "VLAAttacker/UADA_OFT_DINO_wrapper.py").read_text()
    trainer = (ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO.py").read_text()
    slurm = (ROOT / "slurm/exploring_oft_dino/14_train_dino_target5_object2000.sbatch").read_text()
    submit = (ROOT / "submit_exploring_oft_dino_target5_object2000.sh").read_text()
    assert 'cfg["experiment_id"] == "EXPLORING-OFT-DINO-TARGET5-OBJECT2000-01"' in wrapper
    assert 'if method == "exploring_oft_uada":\n                        action_loss.backward()' in trainer
    assert '#SBATCH --partition=rtx4090' in slurm
    assert '#SBATCH --gres=gpu:4090:1' in slurm
    assert '#SBATCH --array=' not in slurm
    assert 'CHECK_ONLY' in submit
    assert '--dependency="afterany:${train_id}"' in submit
