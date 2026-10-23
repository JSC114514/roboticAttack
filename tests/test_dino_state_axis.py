"""Behavior checks for state-axis training; CPU tensors only, no real VLA."""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts"),
               str(ROOT.parent / "vla_internal_attribution_20260812")]

import dino_state_axis_common as common
import white_patch.UADA_OFT_DINO_STATE_AXIS as trainer
from white_patch.UADA_OFT_CORRESPONDENCE import ExploringOFTCorrespondenceAttacker
from white_patch.appply_random_transform import RandomPatchTransform


class LoopMock(trainer.ExploringOFTDINOStateAxisAttacker):
    def __init__(self):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), vla=torch.nn.Linear(1, 1))
        self.site = (26, 137)
        self.forward_log = []
        self.patched_calls = 0
        self.fail_patch_call = None

    def _base_inputs(self, record):
        return {}, None

    def _validate_runtime(self, record, patch, cfg):
        random.random()
        np.random.random()
        torch.rand(1)
        return {"status": "CPU_MOCK_ONLY"}

    def _forward(self, inputs, proprio, observation, patch, geometry, **kwargs):
        assert not kwargs, "Training must use the normal action route"
        self.forward_log.append((patch is not None, bool(geometry),
                                 bool(getattr(self, "transfer_diverse", False)), observation["frame"]))
        value = torch.tensor(observation["frame"] / 100.)
        if patch is not None:
            self.patched_calls += 1
            if self.patched_calls == self.fail_patch_call:
                raise RuntimeError("mock half-outer failure")
            factor = random.uniform(.8, 1.2) if self.transfer_diverse else 1.
            value = value + patch.mean() * factor
        action = value.expand(1, 8, 7) + torch.arange(7).reshape(1, 1, 7) * .02
        return action, torch.ones(1, 256, 4), None

    def _axis_score_record(self, record, displayed, cfg, method, refresh_outer, record_index):
        # Exercise the real refresh/serialization/sampler with changing scores.
        value = float(displayed.mean()) + record_index / 1000.
        scores = [{"axis": axis, "patched_error": value * (axis + 1),
                   "restored_error": value * (axis + 2) if method == "dino_remaining" else None,
                   "action_remaining_score": value * (axis + 1),
                   "dino_remaining_score": value * value * (axis + 1)}
                  for axis in cfg["train_axes"]]
        return {"axis_scores": scores, "score_views": [], "score_view_count": 2,
                "score_full_forward_calls": 5 if method == "dino_remaining" else 3,
                "score_route": "CPU_MOCK_ONLY", "dino_score": value,
                "patched_target_mse": value, "restored_target_mse": value * 2,
                "dino_delta_rms": value}


@pytest.fixture
def tiny_setup(tmp_path):
    torch.set_num_threads(1)
    cfg = {"experiment_id": "CPU-MOCK-ONLY", "output_root": str(tmp_path / "run"),
           "seed": 42, "data_seed": 123, "patch_shape": [3, 50, 50], "inner_loop": 4,
           "bootstrap_outer": 2, "outer_iterations": 8, "warmup_outer_iterations": 2,
           "learning_rate": .002, "weight_decay": 0., "geometry": True,
           "action_head_q01": [-1.] * 7, "action_head_q99": [1.] * 7,
           "train_axes": [0, 1, 2], "clean_axis": 0,
           "replay_probability_uniform_floor": .5, "score_priority": "CPU_MOCK_ONLY",
           "checkpoint_outer_iterations": [0, 2, 4, 6, 8], "refresh_after_outer": [2, 4, 6]}
    clean = [{"task_id": i // 30, "state_index": 10 + i % 5,
              "task_name": f"task{i // 30}", "observation": {"frame": i % 3}}
             for i in range(300)]
    replay = [{"task_id": i % 10, "state_index": 10, "task_name": f"task{i % 10}",
               "phase_proxy": ("early", "middle", "late")[(i // 10) % 3],
               "observation": {"frame": 1},
               "observations": [{"frame": j} for j in range(3)]} for i in range(258)]
    sources = {str(Path(__file__).resolve()): trainer.sha256_file(__file__)}
    return cfg, clean, replay, sources, {"clean_bank": "mock", "window_bank": "mock"}


def saved_state(cfg, method):
    return torch.load(Path(cfg["output_root"]) / "training" / method / "resume.pt", weights_only=False)


@pytest.mark.parametrize("method", common.METHODS)
def test_fresh_training_updates_rgb_and_preserves_budget_eot(method, tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    attacker = LoopMock()
    attacker.train_state_axis(clean, replay, cfg, method, sources, data)
    saved = saved_state(cfg, method)
    trainer.validate_state(saved, cfg)
    assert saved["outer"] == saved["scheduler"]["last_epoch"] == 8
    assert int(next(iter(saved["optimizer"]["state"].values()))["step"]) == 32
    assert saved["costs"]["actual_optimizer_updates"] == 32
    assert torch.equal(saved["initial_raw"], saved["initial"])
    assert not torch.equal(saved["initial"], saved["patch"])
    assert saved["rows"][0]["zero_lr_first_step"]
    assert all(row["rgb_outer_max_abs"] > 0 for row in saved["rows"][1:])
    assert all(row["action_axis"] == 0 for row in saved["rows"] if row["origin"] == "clean")
    if method == "uniform_axis0":
        assert all(row["action_axis"] == 0 for row in saved["rows"])
    patched = [entry for entry in attacker.forward_log if entry[0]]
    assert [entry[1] for entry in patched] == [False, False, True, True] * 8
    assert not any(entry[2] for entry in patched[:8])
    assert [entry[2] for entry in patched[8:]] == [False, False, True, True] * 6
    assert [entry[3] for entry in patched[12:16]] == [0, 1, 2, 0]
    assert [item["outer"] for item in saved["score_refresh_history"]] == [2, 4, 6]
    assert saved["costs"]["score_full_forward_calls"] == 3 * 258 * (5 if method == "dino_remaining" else 3)
    out = Path(cfg["output_root"]) / "training" / method
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert manifest["fresh_start_outer"] == 0 and manifest["initial_optimizer_state_empty"]
    assert not manifest["feature_loss"] and not manifest["appearance_regularization"]
    rows = pd.read_parquet(out / "training.parquet")
    common.validate_training_rows(cfg, saved, rows)
    common.validate_sampling_history(cfg, method, out, saved, rows, clean, replay)
    assert rows.selected_score_available.eq(rows.origin.eq("replay")).all()
    assert rows.selected_restored_error_available.eq(
        rows.origin.eq("replay") & (method == "dino_remaining")).all()
    for checkpoint in cfg["checkpoint_outer_iterations"]:
        path = out / f"outer_{checkpoint:04d}.pt"
        patch = torch.load(path, weights_only=False)["patch"]
        with Image.open(path.with_suffix(".png")) as picture:
            assert np.array_equal(np.asarray(picture), patch.permute(1, 2, 0).mul(255).round().byte().numpy())


@pytest.mark.parametrize("method", common.METHODS)
def test_half_outer_resume_matches_uninterrupted_patch_adam_and_scores(method, tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    reference_cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "reference"))
    LoopMock().train_state_axis(clean, replay, reference_cfg, method, sources, data)
    interrupted = LoopMock()
    interrupted.fail_patch_call = 10
    with pytest.raises(RuntimeError, match="half-outer"):
        interrupted.train_state_axis(clean, replay, cfg, method, sources, data)
    assert saved_state(cfg, method)["outer"] == 2
    LoopMock().train_state_axis(clean, replay, cfg, method, sources, data)
    resumed, expected = saved_state(cfg, method), saved_state(reference_cfg, method)
    assert torch.equal(resumed["patch"], expected["patch"])
    assert resumed["scheduler"] == expected["scheduler"]
    left = next(iter(resumed["optimizer"]["state"].values()))
    right = next(iter(expected["optimizer"]["state"].values()))
    assert all(torch.equal(left[key], right[key]) for key in ("exp_avg", "exp_avg_sq"))
    assert resumed["replay_scoring"] == expected["replay_scoring"]
    assert [r["action_axis"] for r in resumed["rows"]] == [r["action_axis"] for r in expected["rows"]]
    assert torch.equal(resumed["torch_rng"], expected["torch_rng"])
    assert resumed["python_rng"] == expected["python_rng"]
    assert np.array_equal(resumed["numpy_rng"][1], expected["numpy_rng"][1])
    assert resumed["costs"]["actual_optimizer_updates"] == expected["costs"]["actual_optimizer_updates"] + 1


class TinyEncoder(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, pixels):
        return pixels.mean((-2, -1)).unsqueeze(1).expand(-1, 256, -1) * self.scale


class TinyHead(torch.nn.Module):
    def predict_action(self, hidden):
        value = hidden.mean((1, 2))[:, None, None].expand(-1, 8, 7)
        return value * torch.arange(1, 8).reshape(1, 1, 7) / 7


class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Module()
        self.vision_backbone.featurizer = TinyEncoder(.7)
        self.vision_backbone.fused_featurizer = TinyEncoder(.4)
        self.pixel_log = []

    def predict_action(self, pixel_values, **kwargs):
        self.pixel_log.append(pixel_values.detach().clone())
        values = [self.vision_backbone.featurizer(view[:, :3])
                  + self.vision_backbone.fused_featurizer(view[:, 3:6])
                  for view in (pixel_values[:, :6], pixel_values[:, 6:])]
        return None, torch.cat(values, 1)


@pytest.mark.parametrize("method", common.METHODS)
@pytest.mark.parametrize("patch_value", [.01, .6])
def test_actual_scoring_formula_identical_pixels_and_hook_cleanup(method, patch_value, monkeypatch):
    attacker = trainer.ExploringOFTDINOStateAxisAttacker.__new__(trainer.ExploringOFTDINOStateAxisAttacker)
    attacker.wrapper = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
        vla=TinyVLA(), action_head=TinyHead(), gen_cfg=SimpleNamespace(unnorm_key="mock"),
        proprio_projector=None, noisy_action_projector=None)
    attacker.site, attacker.transfer_diverse = (26, 137), False
    monkeypatch.setattr(attacker, "_base_inputs", lambda record: ({}, None))
    render_log = []
    def pixels(self, observation, patch, geometry):
        front = torch.full((1, 3, 224, 224), .2)
        if patch is None:
            wrist = torch.full_like(front, .1)
        else:
            render_log.append((bool(geometry), self.transfer_diverse))
            multiplier = random.uniform(.8, 1.2) if self.transfer_diverse else 1.
            wrist = (patch * multiplier).mean().expand_as(front)
        return torch.cat((front, front, wrist, wrist), 1)
    monkeypatch.setattr(ExploringOFTCorrespondenceAttacker, "_patched_pixels", pixels)
    cfg = {"data_seed": 123, "action_head_q01": [-1., -2., -3.] + [-1.] * 4,
           "action_head_q99": [1., 2., 3.] + [1.] * 4, "train_axes": [0, 1, 2]}
    patch = torch.full((3, 50, 50), patch_value)
    before = trainer._rng_state(patch)
    result = attacker._refresh_axis_scores([{"observation": {}, "task_name": "mock"}], patch, cfg, method, 250)[0]
    after = trainer._rng_state(patch)
    assert torch.equal(before["torch_rng"], after["torch_rng"])
    assert before["python_rng"] == after["python_rng"]
    assert np.array_equal(before["numpy_rng"][1], after["numpy_rng"][1])
    assert render_log == [(False, False), (True, True)]
    logs = attacker.wrapper.vla.pixel_log
    assert len(logs) == (5 if method == "dino_remaining" else 3)
    if method == "dino_remaining":
        assert torch.equal(logs[1], logs[2]) and torch.equal(logs[3], logs[4])
        assert not torch.equal(logs[0][:, 9:], logs[1][:, 9:])
    clean_action = .165 * torch.arange(1, 8).numpy() / 7
    for view in result["score_views"]:
        normal_index = 1 + view["view"] * (2 if method == "dino_remaining" else 1)
        pixels_value = logs[normal_index]
        normal_scalar = sum(
            float(v[:, :3].mean()) * .7 + float(v[:, 3:6].mean()) * .4
            for v in (pixels_value[:, :6], pixels_value[:, 6:])) / 2
        for offset, axis in enumerate(cfg["train_axes"]):
            low, high = cfg["action_head_q01"][axis], cfg["action_head_q99"][axis]
            target = low if clean_action[axis] >= (low + high) / 2 else high
            expected_error = ((normal_scalar * (axis + 1) / 7 - target) / (high - low)) ** 2
            assert view["patched_error"][offset] == pytest.approx(expected_error, rel=1e-6)
    for offset, score in enumerate(result["axis_scores"]):
        errors = [v["patched_error"][offset] for v in result["score_views"]]
        assert score["action_remaining_score"] == pytest.approx(np.mean(errors))
        if method == "dino_remaining":
            restored = [v["restored_error"][offset] for v in result["score_views"]]
            expected = np.mean([max(0., r - p) * p for p, r in zip(errors, restored)])
            assert score["dino_remaining_score"] == pytest.approx(expected)
            assert (score["dino_remaining_score"] > 0) == (patch_value < .1)
        else:
            assert score["dino_remaining_score"] == 0 and score["restored_error"] is None
    assert attacker._scoring_pixels is None and not attacker.transfer_diverse
    assert all(not encoder._forward_hooks for encoder in (
        attacker.wrapper.vla.vision_backbone.featurizer,
        attacker.wrapper.vla.vision_backbone.fused_featurizer))


def test_sampler_really_uses_joint_state_axis_priority(tiny_setup):
    cfg, clean, replay, _, _ = tiny_setup
    for record in replay:
        record["axis_scores"] = [{"axis": axis, "action_remaining_score": 1.,
                                  "dino_remaining_score": float(axis == 2) * 100.}
                                 for axis in cfg["train_axes"]]
    cfg = dict(cfg, outer_iterations=2000)
    choices = [common.draw_record(clean, replay, "dino_remaining", outer, cfg)[4]
               for outer in range(4, 2001, 2)]
    assert choices.count(2) / len(choices) > .60
    assert choices.count(0) > 50 and choices.count(1) > 50  # 50% uniform floor.


@pytest.mark.parametrize("diverse", [False, True])
@pytest.mark.parametrize("geometry", [False, True])
def test_real_wrist_renderer_retains_rgb_gradient_and_site(diverse, geometry):
    attacker = trainer.ExploringOFTDINOStateAxisAttacker.__new__(trainer.ExploringOFTDINOStateAxisAttacker)
    attacker.wrapper = SimpleNamespace(device=torch.device("cpu"))
    attacker.site, attacker.transfer_diverse = (26, 137), diverse
    attacker.randomPatchTransform = RandomPatchTransform("cpu", False)
    patch = torch.full((3, 50, 50), .6, requires_grad=True)
    trainer.seed_transform(123)
    rgb, support = attacker._paste_with_support(np.zeros((224, 224, 3), dtype=np.uint8), patch, geometry)
    rgb.sum().backward()
    assert attacker.site == (26, 137)
    assert rgb.shape == (3, 224, 224) and support.shape == (224, 224)
    assert patch.grad is not None and torch.isfinite(patch.grad).all() and patch.grad.norm() > 0


def test_mock_submit_chain_and_failure_receipt(tmp_path, monkeypatch):
    cfg = {"experiment_id": "CPU-MOCK-ONLY", "output_root": str(tmp_path / "run")}
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        n = sum(c[0] == "sbatch" for c in calls)
        return SimpleNamespace(returncode=0, stdout="" if command[0] == "squeue" else f"{1000+n};cluster\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    report = common.submit_chain(cfg)
    commands = [c for c in calls if c[0] == "sbatch"]
    assert [j["stage"] for j in report["jobs"]] == ["prepare", "train", "validate", "eval", "stats"]
    assert "--array=0,1,2%3" in commands[1] and "--array=0-9%8" in commands[3]
    assert all(f"--dependency=afterok:{1000+i}" in commands[i] for i in range(1, 5))
    calls.clear()
    def fail_eval(command, **kwargs):
        result = fake_run(command, **kwargs)
        if command[0] == "sbatch" and "125_state_axis_eval" in command[-1]:
            result.returncode, result.stdout, result.stderr = 1, "", "mock rejection"
        return result
    monkeypatch.setattr(subprocess, "run", fail_eval)
    with pytest.raises(RuntimeError, match="no downstream"):
        common.submit_chain(cfg)
    receipt = json.loads((Path(cfg["output_root"]) / "reports/SUBMISSION_RECEIPT.json").read_text())
    assert receipt["status"] == "SUBMISSION_FAILED" and receipt["failed_stage"] == "eval"
    assert len(receipt["jobs"]) == 3 and sum(c[0] == "sbatch" for c in calls) == 4
