"""CPU regression for the new clip boundary, controls, and real RGB update loop."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]
from dino_window_common import METHODS, clip_indices, draw_record, score_and_weights
from test_exploring_oft_ilpd import MockILPD, mock_setup
from white_patch.UADA_OFT_DINO_WINDOW import ExploringOFTDINOWindowAttacker


def test_clip_indices_keep_three_contiguous_decisions_with_boundary_repeat():
    assert clip_indices(0, 1) == (0, 0, 0)
    assert clip_indices(0, 5) == (0, 0, 1)
    assert clip_indices(2, 5) == (1, 2, 3)
    assert clip_indices(4, 5) == (3, 4, 4)


def bank_rows():
    return [{"task_id": task, "state_index": 10 + episode,
             "trajectory_rank": episode, "decision_index": phase_index,
             "phase_proxy": ("early", "middle", "late")[phase_index],
             "attacked_episode_success": episode != 1,
             "dino_score": float(4 - episode + phase_index),
             "observation": {"t": phase_index},
             "observations": [{"t": phase_index + i} for i in range(3)]}
            for task in range(10) for episode in range(3) for phase_index in range(3)]


def test_window_weights_preserve_each_task_phase_control_distribution():
    weighted = score_and_weights(bank_rows(), 1234)
    for task in range(10):
        for phase in ("early", "middle", "late"):
            rows = [r for r in weighted if r["task_id"] == task and r["phase_proxy"] == phase]
            assert sorted(r["dino_weight"] for r in rows) == sorted(r["shuffled_weight"] for r in rows)
            assert {r["dino_weight"] for r in rows} == {1., 2.}


def test_replay_draw_has_same_task_phase_and_clean_schedule():
    clean = [{"task_id": i % 10, "observation": {"t": 0}} for i in range(300)]
    replay = score_and_weights(bank_rows(), 1234)
    for method in METHODS:
        for outer in range(1, 31):
            base_index = outer * 7 % 300
            value, origin, _ = draw_record(clean, replay, method, outer, base_index, 20260929)
            assert value["task_id"] == clean[base_index]["task_id"]
            if method == "clean_continue" or outer % 2:
                assert origin == "clean"
            else:
                assert origin == "replay"
                assert value["phase_proxy"] == ("early", "middle", "late")[(outer // 2 - 1) % 3]


class MockWindow(ExploringOFTDINOWindowAttacker):
    def __init__(self, fail_at=None):
        MockILPD.__init__(self, fail_at)
        self.site = (26, 137)
        self.visited = []

    _base_inputs = MockILPD._base_inputs
    _validate_runtime = MockILPD._validate_runtime

    def _forward(self, inputs, proprio, observation, patch, geometry, **kwargs):
        if patch is not None and patch.requires_grad:
            self.visited.append(observation.get("t", -1))
        return MockILPD._forward(self, inputs, proprio, observation, patch, geometry, **kwargs)


def test_real_update_loop_cycles_three_states_and_resumes(tmp_path):
    cfg, _ = mock_setup(tmp_path)
    cfg.update(base_outer_iterations=2200, outer_iterations=2, inner_loop=6,
               checkpoint_outer_iterations=[0, 1, 2], resume_every_outer_iterations=1,
               output_root=str(tmp_path / "window"), tv_threshold=.4, tv_weight=1.,
               seed=42, data_seed=20260929, clip_length=3)
    clean = [{"task_id": i % 10, "state_index": 10,
              "task_name": "object", "observation": {"t": 0}} for i in range(300)]
    replay = [{"task_id": task, "state_index": 11, "task_name": "object",
               "phase_proxy": phase, "dino_score": 1.,
               "dino_weight": 2., "shuffled_weight": 2.,
               "observations": [{"t": 1}, {"t": 2}, {"t": 3}]}
              for task in range(10) for phase in ("early", "middle", "late")]
    attacker = MockWindow(fail_at=8)
    with pytest.raises(RuntimeError, match="mock interruption"):
        attacker.train(clean, replay, cfg, "window_dino_seed42", {}, {})
    resumed = MockWindow()
    resumed.train(clean, replay, cfg, "window_dino_seed42", {}, {})
    folder = Path(cfg["output_root"]) / "training/window_dino_seed42"
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 12
    assert status["total_optimizer_updates"] == 13212
    assert resumed.visited == [1, 2, 3, 1, 2, 3]
    table = pd.read_parquet(folder / "training.parquet")
    assert table.origin.tolist() == ["clean", "replay"]
    assert (folder / "outer_0002.pt").is_file()
