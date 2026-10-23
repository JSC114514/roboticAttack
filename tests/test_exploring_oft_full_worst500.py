from __future__ import annotations

import ast
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs/exploring_oft_dino/object_scale500.json"
CONTROL = ROOT / "configs/exploring_oft_dino/full_worst500.json"
TRAINER = ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO.py"
SBATCH = ROOT / "slurm/exploring_oft_dino/07_train_full_worst500.sbatch"


def test_single_control_differs_from_frozen_scale_only_in_identity_and_destination() -> None:
    frozen = json.loads(FROZEN.read_text())
    control = json.loads(CONTROL.read_text())
    assert control["methods"] == ["exploring_oft_full_worst_chunk"]
    for key in frozen.keys() - {"experiment_id", "methods", "output_root"}:
        assert control[key] == frozen[key], key
    assert control["output_root"] != frozen["output_root"]
    assert not Path(control["output_root"]).is_relative_to(Path(frozen["output_root"]))


def test_full_worst_control_uses_same_temporal_aggregation_and_no_clean_siglip() -> None:
    source = TRAINER.read_text()
    ast.parse(source)
    assert '"exploring_oft_full_worst_chunk": "attacked_full_worst_chunk_boundary_control"' in source
    assert 'None if method == "exploring_oft_full_worst_chunk" else clean_siglip' in source
    assert '"exploring_oft_dino_cf_worst_chunk",\n                                          "exploring_oft_full_worst_chunk"' in source
    assert 'route_error.reshape(route_error.shape[0], route_error.shape[1], -1).mean(-1).max()' in source
    assert 'g_action + float(lambda_dino_cf) * g_dino_cf' in source


def test_worst_chunk_reduction_differs_from_mean_when_error_is_temporally_concentrated() -> None:
    route_error = torch.tensor([[[1.0], [9.0], [1.0], [1.0]]])
    worst = route_error.reshape(route_error.shape[0], route_error.shape[1], -1).mean(-1).max()
    assert float(worst) == 9.0
    assert float(route_error.mean()) == 3.0


def test_single_gpu_non_array_job_and_report_dependency() -> None:
    source = SBATCH.read_text()
    submit = (ROOT / "submit_exploring_oft_full_worst500.sh").read_text()
    assert "#SBATCH --gres=gpu:3090:1" in source
    assert "#SBATCH --time=07:00:00" in source
    assert "#SBATCH --array=" not in source
    assert "--dependency=\"afterany:${train_id}\"" in submit
    assert "CHECK_ONLY" in submit
