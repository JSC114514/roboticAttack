from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/exploring_oft_dino/object_scale500.json"
NEW = ROOT / "configs/exploring_oft_dino/uada_object2000.json"


def test_fresh_2000_baseline_keeps_exploring_oft_uada_objective_and_inputs() -> None:
    old = json.loads(OLD.read_text())
    new = json.loads(NEW.read_text())
    assert new["methods"] == ["exploring_oft_uada"]
    assert new["outer_iterations"] == 2000 and new["inner_loop"] == 50
    assert new["checkpoint_outer_iterations"][-1] == 2000
    assert new["output_root"] != old["output_root"]
    assert "initial_patch" not in new
    exceptions = {"experiment_id", "methods", "output_root", "outer_iterations",
                  "checkpoint_outer_iterations", "mechanism_probe_outer_iterations", "structure_gates"}
    for key in old.keys() - exceptions:
        assert new[key] == old[key], key


def test_same_source_training_core_and_no_new_route_loss() -> None:
    trainer = (ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO.py").read_text()
    wrapper = (ROOT / "VLAAttacker/UADA_OFT_DINO_wrapper.py").read_text()
    assert 'if method == "exploring_oft_uada":\n                        action_loss.backward()' in trainer
    assert 'patch = init_uada_patch(tuple(cfg["patch_shape"]), self.wrapper.device)' in trainer
    assert "build_uada_adamw(" in trainer and "step_uada_patch(" in trainer
    assert "cfg[\"outer_iterations\"] in (200, 500, 2000)" in wrapper


def test_single_gpu_train_and_structure_report_are_manually_submitted() -> None:
    train = (ROOT / "slurm/exploring_oft_dino/11_train_uada_object2000.sbatch").read_text()
    submit = (ROOT / "submit_exploring_oft_uada_object2000.sh").read_text()
    assert "#SBATCH --gres=gpu:3090:1" in train
    assert "#SBATCH --time=16:00:00" in train
    assert "#SBATCH --array=" not in train
    assert "--dependency=\"afterany:${train_id}\"" in submit
    assert "CHECK_ONLY" in submit
