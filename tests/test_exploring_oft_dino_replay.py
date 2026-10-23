"""CPU-only design and engineering regression for attacked-state replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]
from dino_replay_common import METHODS, draw_record, score_and_weights, selected_decisions
from test_exploring_oft_ilpd import MockILPD, mock_setup
from white_patch.UADA_OFT_DINO_REPLAY import ExploringOFTDINOReplayAttacker


def rows():
    return [{"task_id": task, "state_index": 10 + episode, "trajectory_rank": episode,
             "decision_index": decision, "phase_proxy": ("early", "middle", "late")[decision // 2],
             "attacked_episode_success": episode % 2 == 0,
             "dino_score": float(1 + task + episode + decision), "observation": {}}
            for task in range(10) for episode in range(5) for decision in range(6)]


def test_survivor_only_and_task_local_weights():
    selected = score_and_weights(rows(), 42)
    assert len(selected) == 10 * 3 * 6
    assert all(r["attacked_episode_success"] for r in selected)
    for task in range(10):
        subset = [r for r in selected if r["task_id"] == task]
        assert {r["dino_weight"] for r in subset} == {1., 2.}
        for phase in ("early", "middle", "late"):
            before = sorted(r["dino_weight"] for r in subset if r["phase_proxy"] == phase)
            after = sorted(r["shuffled_weight"] for r in subset if r["phase_proxy"] == phase)
            assert before == after


def test_even_replay_odd_clean_and_matched_task():
    clean = [{"task_id": i % 10} for i in range(300)]
    replay = score_and_weights(rows(), 42)
    for method in METHODS:
        for outer in range(1, 21):
            base_index = outer * 7 % 300
            value, origin, index = draw_record(clean, replay, method, outer, base_index, 20260920)
            assert value["task_id"] == clean[base_index]["task_id"]
            if method == "clean_continue" or outer % 2:
                assert origin == "clean" and index == base_index
            else:
                assert origin == "replay" and value is replay[index]


def test_selected_decisions_are_unique_and_span_episode():
    assert selected_decisions(1) == [0]
    assert selected_decisions(6) == list(range(6))
    assert selected_decisions(31)[0] == 0 and selected_decisions(31)[-1] == 30
    assert len(set(selected_decisions(31))) == 6


class MockReplay(ExploringOFTDINOReplayAttacker):
    def __init__(self, fail_at=None):
        MockILPD.__init__(self, fail_at)
        self.site = (26, 137)

    _forward = MockILPD._forward
    _base_inputs = MockILPD._base_inputs
    _validate_runtime = MockILPD._validate_runtime


def test_train_keeps_original_action_loss_and_resume(tmp_path):
    cfg, _ = mock_setup(tmp_path)
    cfg.update(base_outer_iterations=2000, outer_iterations=2, inner_loop=2,
               checkpoint_outer_iterations=[0, 1, 2], resume_every_outer_iterations=1,
               output_root=str(tmp_path / "replay"), tv_threshold=.4, tv_weight=1.)
    clean = [{"task_id": i % 10, "state_index": 10, "observation": {}} for i in range(300)]
    replay = [{"task_id": i, "state_index": 11, "observation": {},
               "dino_score": 1., "dino_weight": 2., "shuffled_weight": 2.}
              for i in range(10)]
    interrupted = MockReplay(fail_at=3)
    with pytest.raises(RuntimeError, match="mock interruption"):
        interrupted.train(clean, replay, cfg, "replay_dino", {}, {})
    resumed = MockReplay()
    resumed.train(clean, replay, cfg, "replay_dino", {}, {})
    folder = Path(cfg["output_root"]) / "training/replay_dino"
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 4
    assert status["total_optimizer_updates"] == 110000
    import pandas as pd
    table = pd.read_parquet(folder / "training.parquet")
    assert table.origin.tolist() == ["clean", "replay"]
    assert resumed.train_calls == 2 and interrupted.train_calls == 3
    assert (folder / "outer_0200.pt").exists() is False
    assert (folder / "outer_0002.pt").is_file()


def test_fixed_wrist_paste_preserves_pixels_and_gradient():
    class Transform:
        resize_patch = False

        @staticmethod
        def _to_rgb_pil(image):
            from PIL import Image
            return Image.fromarray(image)

    attacker = ExploringOFTDINOReplayAttacker.__new__(ExploringOFTDINOReplayAttacker)
    attacker.wrapper = SimpleNamespace(device="cpu")
    attacker.randomPatchTransform = Transform()
    attacker.site = (26, 137)
    image = np.full((224, 224, 3), 120, dtype=np.uint8)
    patch = torch.full((3, 50, 50), .6, requires_grad=True)
    composed, support = attacker._paste_with_support(image, patch, False)
    assert tuple(composed.shape) == (3, 224, 224)
    assert float(composed[:, 137:187, 26:76].mean()) == pytest.approx(.6)
    assert int(support.sum()) == 2500
    assert float(composed[:, :50, :50].mean()) == pytest.approx(120 / 255)
    composed.sum().backward()
    assert patch.grad is not None and float(patch.grad.sum()) == 7500
