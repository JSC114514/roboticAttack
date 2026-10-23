"""Actual trainer-loop CPU checks using a tiny two-view DINO/SigLIP model."""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts"),
               str(ROOT.parent / "vla_internal_attribution_20260812")]

import dino_action_counter_common as common
from white_patch.UADA_OFT_DINO import sha256_file
from white_patch.UADA_OFT_DINO_ACTION_COUNTER import (
    ExploringOFTDINOActionCounterAttacker, detach_wrist_siglip,
    patch_tensor_sha256, validate_state,
)
from white_patch.UADA_OFT_SOURCE_COPY import build_uada_adamw, init_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform


class TinyEncoder(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.outputs = []

    def forward(self, pixels):
        value = pixels.mean((2, 3))[:, None, :] * self.scale
        result = value.expand(-1, 4, -1)
        self.outputs.append(result.detach().clone())
        return result


class TinyHead(torch.nn.Module):
    def predict_action(self, hidden):
        return hidden.mean((1, 2))[:, None, None].expand(-1, 8, 7)


class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Module()
        self.vision_backbone.featurizer = TinyEncoder()
        self.vision_backbone.fused_featurizer = TinyEncoder(.7)
        self.fail = False

    def predict_action(self, pixel_values, **kwargs):
        values = []
        for view in (pixel_values[:, :6], pixel_values[:, 6:]):
            values.append(self.vision_backbone.featurizer(view[:, :3])
                          + self.vision_backbone.fused_featurizer(view[:, 3:6]))
        if self.fail:
            raise RuntimeError("mock predictor failure")
        return None, torch.cat(values, 1)


class MockAttacker(ExploringOFTDINOActionCounterAttacker):
    def __init__(self):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
            vla=TinyVLA(), action_head=TinyHead(), gen_cfg=SimpleNamespace(unnorm_key="mock"),
            proprio_projector=None, noisy_action_projector=None)
        self.site = (26, 137)
        self.sample_calls = 0
        self.geometry_calls = []
        self.diverse_calls = []
        self.fail_sample = None

    def _base_inputs(self, record):
        return {}, None

    def _pixels_from_rgb(self, observation, rgb):
        front = torch.as_tensor(observation["full_image"]).permute(2, 0, 1).float()[None] / 255
        return torch.cat((front, front, rgb[None], rgb[None]), 1)

    def _patched_pixels(self, observation, patch, geometry):
        assert patch is None  # No legacy patch-forward or feature substitution.
        wrist = torch.as_tensor(observation["wrist_image"]).permute(2, 0, 1).float() / 255
        return self._pixels_from_rgb(observation, wrist)

    def _paste_with_support(self, image, patch, geometry):
        self.sample_calls += 1
        self.geometry_calls.append(bool(geometry))
        self.diverse_calls.append(bool(self.transfer_diverse))
        if self.sample_calls == self.fail_sample:
            raise RuntimeError("mock half-outer failure")
        wrist = torch.as_tensor(image).permute(2, 0, 1).float() / 255
        if self.transfer_diverse:
            x, y = random.randint(0, 174), random.randint(0, 174)
            brightness, contrast = random.uniform(.8, 1.2), random.uniform(.8, 1.2)
            placed = (((patch - .5) * contrast + .5) * brightness).clamp(0, 1)
        else:
            x, y, placed = 26, 137, patch
        # Consume a reproducible geometry draw even before diverse begins.
        if geometry:
            placed = placed * (.99 + random.random() * .01)
        rgb = wrist.clone()
        rgb[:, y:y + 50, x:x + 50] = placed
        support = torch.zeros(224, 224)
        support[y:y + 50, x:x + 50] = 1
        return rgb, support


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = {"experiment_id": "CPU-MOCK", "output_root": str(tmp_path / "run"), "seed": 42,
           "data_seed": 123, "patch_shape": [3, 50, 50], "inner_loop": 2,
           "outer_iterations": 4, "scheduler_total_outer": 2000, "diverse_after_outer": 250,
           "warmup_outer_iterations": 20, "learning_rate": .002, "weight_decay": 0,
           "action_head_q01": [-1.] * 7, "action_head_q99": [1.] * 7, "maskidx": [0],
           "geometry": True, "counter_steps": 2, "counter_epsilon": 4 / 255,
           "counter_step_size": 2 / 255, "tv_thresholds": {"original": .4, "clear": .3},
           "tv_scale": .4, "tv_weight": 1,
           "checkpoint_outer_iterations": [0, 1, 2, 3, 4], "resume_every_outer_iterations": 1}
    observation = {"full_image": np.full((224, 224, 3), 77, dtype=np.uint8),
                   "wrist_image": np.full((224, 224, 3), 99, dtype=np.uint8)}
    clean = [{"task_id": index // 30, "state_index": index % 30, "task_name": f"task{index // 30}",
              "observation": observation} for index in range(300)]
    replay = [dict(clean[index % 300], observations=[observation] * 3,
                   phase_proxy="middle") for index in range(258)]
    sources = {str(Path(__file__).resolve()): sha256_file(__file__)}
    data = {"clean_bank": "mock", "config": "mock", "window_bank": "mock", "sampling_schedule": "mock"}
    def draw(clean, replay, method, outer, cfg):
        return (clean[0], "clean", 0, 0) if outer % 2 else (replay[0], "replay", 0, 0)
    monkeypatch.setattr(common, "draw_record", draw)
    return cfg, clean, replay, sources, data


@pytest.mark.parametrize("method", common.METHODS)
def test_complete_cpu_training_fresh_adam_warmup_and_cosine(method, tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    attacker = MockAttacker()
    attacker.train_action_counter(clean, replay, cfg, method, sources, data)
    out = Path(cfg["output_root"]) / "training" / method
    saved = torch.load(out / "resume.pt", weights_only=False)
    validate_state(saved, cfg)
    assert saved["outer"] == 4 and saved["scheduler"]["last_epoch"] == 4
    state = next(iter(saved["optimizer"]["state"].values()))
    assert int(state["step"]) == 8 and state["exp_avg"].abs().max() > 0
    assert saved["rows"][0]["learning_rate"] == 0
    assert saved["rows"][0]["rgb_outer_max_abs"] == 0
    assert saved["rows"][1]["learning_rate"] == pytest.approx(.002 / 20)
    assert saved["rows"][-1]["learning_rate"] > 0
    assert saved["scheduler"]["_last_lr"] == pytest.approx([.002 * 4 / 20])
    seed_transform(42)
    assert torch.equal(saved["initial"], init_uada_patch((3, 50, 50), "cpu").detach())
    assert saved["initial_patch_sha256"] == patch_tensor_sha256(saved["initial"])
    assert attacker.geometry_calls[1:] == [False, True] * 4  # One runtime sample plus one per update.
    assert not any(attacker.diverse_calls)
    assert saved["costs"]["eot_draws"] == 8
    assert saved["costs"]["attempted_patch_updates"] == 8
    if method.startswith("none_"):
        assert "counter_forward_calls" not in saved["costs"]
        assert all(row["eta_linf"] == 0 for row in saved["rows"])
    else:
        assert saved["rows"][0]["counter_forward_calls"] == 3
        assert saved["rows"][0]["counter_backward_calls"] == 2
        assert saved["rows"][0]["counter_effective"] == 1
        assert saved["rows"][0]["eta_linf"] > 0
        assert all(row["counter_loss_delta"] >= 0 for row in saved["rows"])
        if method.startswith("random_"):
            assert all(row["random_changed_fraction"] > 0 for row in saved["rows"])
            assert all(row["random_match_rms_error"] <= 1e-10 for row in saved["rows"])
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert manifest["fresh_start_outer"] == 0 and manifest["initial_optimizer_state_empty"] is True
    assert manifest["scheduler_total_outer"] == 2000 and manifest["parameter_count"] == 7500
    assert manifest["tv_threshold"] == (.3 if method.endswith("clear") else .4)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 8
    alignment = json.loads((out / "RUNTIME_ALIGNMENT.json").read_text())
    assert alignment["inner_wrist_siglip_detached"] and alignment["outer_wrist_siglip_restored"]
    assert (out / "source_snapshot.zip").exists() and (out / "outer_0000.png").exists()
    digest = sha256_file(out / "resume.pt")
    attacker.train_action_counter(clean, replay, cfg, method, sources, data)
    assert sha256_file(out / "resume.pt") == digest


def test_half_outer_resume_restores_patch_adam_rng_and_complete_counts(tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    reference_cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "reference"))
    baseline = MockAttacker()
    baseline.train_action_counter(clean, replay, reference_cfg, "random_clear", sources, data)
    interrupted = MockAttacker()
    interrupted.fail_sample = 5  # Failure after runtime, outer1, and the first update of outer2.
    with pytest.raises(RuntimeError, match="half-outer"):
        interrupted.train_action_counter(clean, replay, cfg, "random_clear", sources, data)
    out = Path(cfg["output_root"]) / "training/random_clear"
    partial = torch.load(out / "resume.pt", weights_only=False)
    assert partial["outer"] == 1 and len(partial["rows"]) == 1
    assert int(next(iter(partial["optimizer"]["state"].values()))["step"]) == 2
    assert partial["scheduler"]["last_epoch"] == 1
    assert json.loads((out / "status.json").read_text())["outer"] == 1
    MockAttacker().train_action_counter(clean, replay, cfg, "random_clear", sources, data)
    resumed = torch.load(out / "resume.pt", weights_only=False)
    expected = torch.load(Path(reference_cfg["output_root"]) / "training/random_clear/resume.pt", weights_only=False)
    assert torch.equal(resumed["patch"], expected["patch"])
    assert resumed["scheduler"] == expected["scheduler"]
    assert torch.equal(resumed["torch_rng"], expected["torch_rng"])
    assert resumed["python_rng"] == expected["python_rng"]
    assert np.array_equal(resumed["numpy_rng"][1], expected["numpy_rng"][1])
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(next(iter(resumed["optimizer"]["state"].values()))[key],
                           next(iter(expected["optimizer"]["state"].values()))[key])
    assert resumed["costs"]["attempted_patch_updates"] == expected["costs"]["attempted_patch_updates"] + 1
    for row, reference in zip(resumed["rows"], expected["rows"], strict=True):
        for key in ("main_loss", "counter_loss_delta", "rgb_outer_max_abs", "learning_rate", "eta_rms"):
            assert row[key] == reference[key]


def test_wrist_detach_preserves_forward_front_gradient_and_restores_outer_gradient():
    attacker = MockAttacker()
    vla = attacker.wrapper.vla
    pixels = torch.rand(1, 12, 5, 5, requires_grad=True)
    with detach_wrist_siglip(attacker.wrapper):
        _, hidden = vla.predict_action(pixel_values=pixels)
    inner = attacker.wrapper.action_head.predict_action(hidden).mean()
    inner_gradient = torch.autograd.grad(inner, pixels)[0]
    normal_pixels = pixels.detach().clone().requires_grad_(True)
    _, normal_hidden = vla.predict_action(pixel_values=normal_pixels)
    assert torch.equal(hidden.detach(), normal_hidden.detach())
    outer_gradient = torch.autograd.grad(attacker.wrapper.action_head.predict_action(normal_hidden).mean(), normal_pixels)[0]
    assert torch.equal(inner_gradient[:, :9], outer_gradient[:, :9])
    assert inner_gradient[:, 9:].abs().max() == 0 and outer_gradient[:, 9:].abs().min() > 0
    assert not vla.vision_backbone.fused_featurizer._forward_hooks
    # Fresh SigLIP must change with eta, even though its input derivative is cut.
    changed_pixels = pixels.detach() + .1
    with detach_wrist_siglip(attacker.wrapper):
        vla.predict_action(pixel_values=changed_pixels)
    assert not torch.equal(vla.vision_backbone.fused_featurizer.outputs[-1],
                           vla.vision_backbone.fused_featurizer.outputs[-3])


def test_counter_hook_removed_on_predictor_exception_and_call_order_failure():
    attacker = MockAttacker()
    attacker.wrapper.vla.fail = True
    obs = {"full_image": np.zeros((224, 224, 3), dtype=np.uint8)}
    with pytest.raises(RuntimeError, match="predictor failure"):
        attacker._counter_forward({}, None, obs, torch.zeros(3, 224, 224))
    for module in (attacker.wrapper.vla.vision_backbone.featurizer,
                   attacker.wrapper.vla.vision_backbone.fused_featurizer):
        assert not module._forward_hooks and not module._forward_pre_hooks
    with pytest.raises(RuntimeError, match="call order"):
        with detach_wrist_siglip(attacker.wrapper):
            attacker.wrapper.vla.vision_backbone.fused_featurizer(torch.zeros(1, 3, 4, 4))
    assert not attacker.wrapper.vla.vision_backbone.fused_featurizer._forward_hooks


def test_original_geometry_through_outer250_and_diverse_afterward(tiny_setup):
    cfg, clean, *_ = tiny_setup
    attacker = MockAttacker()
    attacker.diverse_after_outer = 250
    patch = torch.rand(3, 50, 50, requires_grad=True)
    for outer in (1, 249, 250, 251, 500):
        attacker.action_counter_outer = outer
        attacker._sample_rgb(clean[0]["observation"], patch, False)
        attacker._sample_rgb(clean[0]["observation"], patch, True)
        assert attacker.transfer_diverse is False
    assert attacker.diverse_calls == [False, False] * 3 + [False, True] * 2


def test_resume_validator_rejects_partial_adam_and_missing_rng(tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    MockAttacker().train_action_counter(clean, replay, cfg, "none_original", sources, data)
    saved = torch.load(Path(cfg["output_root"]) / "training/none_original/resume.pt", weights_only=False)
    invalid = copy.deepcopy(saved)
    next(iter(invalid["optimizer"]["state"].values()))["step"] = 9
    with pytest.raises(RuntimeError, match="Adam"):
        validate_state(invalid, cfg)
    invalid = copy.deepcopy(saved)
    del invalid["python_rng"]
    with pytest.raises(RuntimeError, match="RNG"):
        validate_state(invalid, cfg)
    seed_transform(42)
    patch = init_uada_patch((3, 50, 50), "cpu")
    optimizer, scheduler = build_uada_adamw(patch, .002, 20, 2000, 0)
    fresh = dict(saved, patch=patch.detach(), initial=patch.detach(), rows=[], outer=0,
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 initial_patch_sha256=patch_tensor_sha256(patch))
    validate_state(fresh, cfg)


def test_startup_failure_commits_empty_outer_zero_adam(tiny_setup):
    cfg, clean, replay, sources, data = tiny_setup
    attacker = MockAttacker()
    attacker.fail_sample = 1
    with pytest.raises(RuntimeError, match="half-outer"):
        attacker.train_action_counter(clean, replay, cfg, "dino_clear", sources, data)
    out = Path(cfg["output_root"]) / "training/dino_clear"
    saved = torch.load(out / "resume.pt", weights_only=False)
    validate_state(saved, cfg)
    assert saved["outer"] == 0 and saved["rows"] == []
    assert saved["optimizer"]["state"] == {} and saved["scheduler"]["last_epoch"] == 0
    assert torch.equal(saved["patch"], saved["initial"])
    assert json.loads((out / "status.json").read_text())["outer"] == 0


def test_runtime_saturated_zero_gradient_is_valid_and_zero_eta(tiny_setup):
    cfg, clean, *_ = tiny_setup
    attacker = MockAttacker()
    class SaturatedHead(TinyHead):
        def predict_action(self, hidden):
            # A connected graph with zero local derivative, as at a BF16 boundary.
            return super().predict_action(hidden) * 0 + 1
    attacker.wrapper.action_head = SaturatedHead()
    report = attacker._validate_action_counter_runtime(clean[0], torch.rand(3, 50, 50),
                                                       cfg, "random_clear", {})
    assert report["status"] == "PASS" and report["outer_gradient_norm"] == 0
    assert report["inner_gradient_norm"] == 0 and report["eta_linf"] == 0
    assert report["counter_effective"] == 0 and report["random_zero_eta"] == 1
    assert report["eta_zero_gradient_relative_error"] == 0


def test_wrapper_dry_run_does_not_load_model_or_access_cuda(tiny_setup, monkeypatch, capsys):
    import UADA_OFT_DINO_ACTION_COUNTER_wrapper as entry
    cfg, clean, replay, sources, data = tiny_setup
    monkeypatch.setattr(entry, "config", lambda: cfg)
    monkeypatch.setattr(entry, "source_records", lambda cfg: clean)
    monkeypatch.setattr(entry, "replay_records", lambda cfg: (replay, "mock"))
    monkeypatch.setattr(entry, "schedule_records", lambda cfg: list(range(500)))
    monkeypatch.setattr(entry, "source_preflight", lambda cfg: {"source": sources, "data": data})
    calls = []
    monkeypatch.setattr(entry, "pipeline_preflight", lambda cfg: calls.append("pipeline"))
    def forbidden():
        raise AssertionError("dry-run must not query CUDA")
    monkeypatch.setattr(entry.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(sys, "argv", ["entry", "--dry-run"])
    entry.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS" and result["model_loaded"] is False
    assert result["fresh_start_outer"] == 0 and result["initial_optimizer_state_empty"] is True
    assert calls == ["pipeline"]
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(sys, "argv", ["entry", "--method", "dino_clear"])
    with pytest.raises(RuntimeError, match="user-submitted Slurm"):
        entry.main()
