"""CPU tensor and complete-loop mocks; never load a VLA checkpoint or a GPU."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts"),
               str(ROOT.parent / "vla_internal_attribution_20260812")]

import dino_repair_common as common
from white_patch.UADA_OFT_DINO_REPAIR import ExploringOFTDINORepairAttacker, frozen_model, validate_state
from white_patch.UADA_OFT_DINO import atomic_json, sha256_file
from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch, build_uada_adamw, step_uada_patch
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.oft_dino_repair_objectives import (charbonnier_tv, high_frequency_energy,
    conservative_token_coverage, outside_dense_error, repair_rgb)


class TinyEncoder(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, pixels):
        dense = F.adaptive_avg_pool2d(pixels.float(), (16, 16)).flatten(2).transpose(1, 2)
        context = dense.mean(1, keepdim=True).expand(-1, 256, -1)
        return torch.cat((dense*self.scale + context*.2, torch.ones_like(dense[..., :1])), -1)


class TinyHead(torch.nn.Module):
    def predict_action(self, hidden):
        scalar = hidden.mean((1, 2))
        return scalar[:, None, None].expand(-1, 8, 7)


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
            dino = self.vision_backbone.featurizer(view[:, :3])
            siglip = self.vision_backbone.fused_featurizer(view[:, 3:6])
            values.append(dino + siglip)
        if self.fail:
            raise RuntimeError("mock predictor failure")
        return None, torch.cat(values, 1)


class MockAttacker(ExploringOFTDINORepairAttacker):
    def __init__(self):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
            vla=TinyVLA(), action_head=TinyHead(), gen_cfg=SimpleNamespace(unnorm_key="mock"),
            proprio_projector=None, noisy_action_projector=None)
        self.site = (26, 137)
        self.sample_calls = 0
        self.geometry_calls = []
        self.fail_sample = None

    def _base_inputs(self, record):
        return {}, None

    def _pixels_from_rgb(self, observation, rgb):
        front = torch.as_tensor(observation["full_image"]).permute(2, 0, 1).float()[None]/255
        return torch.cat((front, front, rgb[None], rgb[None]), 1)

    def _patched_pixels(self, observation, patch, geometry):
        if patch is None:
            wrist = torch.as_tensor(observation["wrist_image"]).permute(2, 0, 1).float()/255
        else:
            wrist, _ = self._sample_rgb(observation, patch, geometry)
        return self._pixels_from_rgb(observation, wrist)

    def _sample_rgb(self, observation, patch, geometry):
        self.sample_calls += 1
        self.geometry_calls.append(bool(geometry))
        if self.sample_calls == self.fail_sample:
            raise RuntimeError("mock half-outer failure")
        wrist = torch.as_tensor(observation["wrist_image"]).permute(2, 0, 1).float()/255
        if geometry:
            x, y = random.randint(0, 174), random.randint(0, 174)
            brightness, contrast = random.uniform(.8, 1.2), random.uniform(.8, 1.2)
            placed = (((patch-.5)*contrast+.5)*brightness).clamp(0, 1)
        else:
            x, y, placed = 26, 137, patch
        rgb = wrist.clone()
        rgb[:, y:y+50, x:x+50] = placed
        support = torch.zeros(224, 224)
        support[y:y+50, x:x+50] = 1
        return rgb, support

    def _refresh_scores(self, replay, patch, cfg):
        return [dict(record, dino_score=float(patch.detach().mean()), dino_weight=2.) for record in replay]


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    cfg = {"experiment_id": "CPU-MOCK", "output_root": str(tmp_path / "run"), "seed": 42,
           "data_seed": 123, "patch_shape": [3, 50, 50], "inner_loop": 2,
           "bootstrap_outer": 1, "outer_iterations": 4, "scheduler_total_outer": 8,
           "warmup_outer_iterations": 0, "learning_rate": .002, "weight_decay": 0,
           "action_head_q01": [-1.]*7, "action_head_q99": [1.]*7, "maskidx": [0],
           "geometry": True, "hf_sigma": .8, "hf_kernel": 5, "repair_steps": 2,
           "repair_epsilon": 4/255, "repair_step_size": 2/255, "appearance_grad_ratio": .15,
           "checkpoint_outer_iterations": [0, 1, 2, 3, 4], "refresh_after_outer": [2],
           "resume_every_outer_iterations": 1}
    observation = {"full_image": np.full((224, 224, 3), 77, dtype=np.uint8),
                   "wrist_image": np.full((224, 224, 3), 99, dtype=np.uint8)}
    clean = [{"task_id": index//30, "state_index": index%30, "task_name": f"task{index//30}",
              "observation": observation} for index in range(300)]
    replay = [dict(clean[0], observations=[observation]*3, dino_score=.1, dino_weight=1.)]
    seed_transform(42)
    patch = init_uada_patch((3, 50, 50), "cpu")
    initial = patch.detach().clone()
    optimizer, scheduler = build_uada_adamw(patch, .002, 0, 8, 0)
    for _ in range(2):
        patch.grad = 2*patch
        step_uada_patch(patch, optimizer, TinyVLA())
    scheduler.step()
    saved = {"config": {"legacy": True}, "outer": 1, "method": "warmup",
             "patch": patch.detach().clone(), "initial": initial, "optimizer": copy.deepcopy(optimizer.state_dict()),
             "scheduler": copy.deepcopy(scheduler.state_dict()), "rows": [{"optimizer_updates": 2,
             "outer_iteration": 1, "method": "warmup"}], "costs": {}, "elapsed_s": .1,
             "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
             "python_rng": random.getstate(), "cuda_rng": []}
    sources = {str(Path(__file__).resolve()): sha256_file(__file__)}
    data = {"prefix_resume": "fixed-mock-prefix", "clean_bank": "mock", "window_bank": "mock"}
    monkeypatch.setattr(common, "prefix_state", lambda cfg: (copy.deepcopy(saved), data["prefix_resume"], tmp_path))
    monkeypatch.setattr(common, "load_calibration", lambda cfg: json.loads((Path(cfg["output_root"])
                        / "reports/APPEARANCE_CALIBRATION.json").read_text()))
    def draw(clean, replay, method, outer, cfg):
        if outer % 2 == 0:
            return replay[0], "replay", 0, 0
        return clean[0], "clean", 0, 0
    monkeypatch.setattr(common, "draw_record", draw)
    def calibration(cfg):
        atomic_json({"status": "COMPLETE", "config": cfg, "source": sources, "data": data,
                     "lambda_tv": .01, "lambda_hf": .01, "calibration_elapsed_s": .1},
                    Path(cfg["output_root"]) / "reports/APPEARANCE_CALIBRATION.json")
    calibration(cfg)
    return cfg, clean, replay, sources, data, saved, calibration


def test_continuous_appearance_keeps_gradient_below_old_threshold():
    patch = torch.full((3, 50, 50), .5)
    patch[:, 20, 20] = .51
    patch.requires_grad_(True)
    tv, hf = charbonnier_tv(patch), high_frequency_energy(patch)
    assert 0 < tv.item() < .4 and hf.item() > 0
    for term in (tv, hf):
        gradient = torch.autograd.grad(term, patch, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    flat = torch.full((3, 50, 50), .5)
    assert charbonnier_tv(flat).item() == 0
    assert high_frequency_energy(flat).item() < 1e-12


def test_any_covered_token_is_excluded():
    support = torch.zeros(224, 224)
    support[70, 80] = 1
    coverage = conservative_token_coverage(support)
    assert (coverage > 0).sum() > 0
    clean = torch.randn(1, 256, 4)
    attacked = clean.clone()
    attacked[:, coverage > 0] = 100
    assert outside_dense_error(attacked, clean, coverage).item() == 0


def test_repair_freezes_patch_reuses_rgb_and_selects_best_with_zero():
    patch = torch.full((3, 224, 224), .4, requires_grad=True)
    support = torch.zeros(224, 224)
    support[100:150, 100:150] = 1
    features = TinyEncoder()
    clean = features(torch.full((1, 3, 224, 224), .38)).detach()
    seen = []
    def feature(rgb):
        seen.append(rgb.detach().clone())
        return features(rgb[None])
    eta, stats = repair_rgb(patch, support, clean, feature)
    assert patch.grad is None and not eta.requires_grad
    assert len(seen) == 3 and stats["repair_pgd_updates"] == 2
    assert stats["repair_error_after"] <= stats["repair_error_before"]
    assert stats["repair_eta_linf"] <= 4/255+2e-7
    assert torch.equal(eta[:, support > 0], torch.zeros_like(eta[:, support > 0]))
    assert all(torch.equal(value[:, support > 0], patch.detach()[:, support > 0]) for value in seen)
    zero, same = repair_rgb(patch, support, features(patch[None]).detach(), feature)
    assert zero.abs().max() == 0 and same["repair_effective"] == 0


@pytest.mark.parametrize("method", common.METHODS)
def test_complete_mock_loop_preserves_prefix_schedule_and_all_eot(method, tiny_setup):
    cfg, clean, replay, sources, data, prefix, _ = tiny_setup
    attacker = MockAttacker()
    attacker.train_repair(clean, replay, cfg, method, sources, data)
    out = Path(cfg["output_root"]) / "training" / method
    saved = torch.load(out / "resume.pt", weights_only=False)
    validate_state(saved, cfg)
    assert saved["outer"] == 4 and len(saved["rows"]) == 4
    assert saved["rows"][0] == prefix["rows"][0]
    assert saved["scheduler"]["last_epoch"] == 4
    assert int(next(iter(saved["optimizer"]["state"].values()))["step"]) == 8
    assert attacker.geometry_calls[-6:] == [False, True]*3
    assert saved["costs"]["new_patch_updates"] == 6
    assert saved["costs"]["repair_pgd_updates"] >= 12 if method == "dino_repair_clear" else True
    assert saved["score_after"] == (1 if method == "uniform_clear" else 2)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 8
    assert (out / "source_snapshot.zip").exists()
    state_hash = sha256_file(out / "resume.pt")
    attacker.train_repair(clean, replay, cfg, method, sources, data)
    assert state_hash == sha256_file(out / "resume.pt")


def test_half_outer_failure_resume_matches_uninterrupted_patch_and_scores(tiny_setup):
    cfg, clean, replay, sources, data, _, calibration = tiny_setup
    reference_cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "reference"))
    calibration(reference_cfg)
    baseline = MockAttacker()
    baseline.train_repair(clean, replay, reference_cfg, "dino_repair_clear", sources, data)
    interrupted = MockAttacker()
    interrupted.fail_sample = 5  # runtime1 + complete outer2 + first inner of outer3
    with pytest.raises(RuntimeError, match="half-outer"):
        interrupted.train_repair(clean, replay, cfg, "dino_repair_clear", sources, data)
    out = Path(cfg["output_root"]) / "training/dino_repair_clear"
    partial = torch.load(out / "resume.pt", weights_only=False)
    assert partial["outer"] == 2 and len(partial["rows"]) == 2
    assert int(next(iter(partial["optimizer"]["state"].values()))["step"]) == 4
    assert json.loads((out / "status.json").read_text())["outer"] == 2
    MockAttacker().train_repair(clean, replay, cfg, "dino_repair_clear", sources, data)
    resumed = torch.load(out / "resume.pt", weights_only=False)
    expected = torch.load(Path(reference_cfg["output_root"]) / "training/dino_repair_clear/resume.pt",
                          weights_only=False)
    assert torch.equal(resumed["patch"], expected["patch"])
    assert resumed["replay_scoring"] == expected["replay_scoring"]
    assert resumed["costs"]["new_patch_updates"] == expected["costs"]["new_patch_updates"]+1
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(next(iter(resumed["optimizer"]["state"].values()))[key],
                           next(iter(expected["optimizer"]["state"].values()))[key])


def test_replacement_hooks_and_model_freeze_restore_on_exception():
    attacker = MockAttacker()
    observation = {"full_image": np.zeros((224, 224, 3), dtype=np.uint8)}
    modules = [attacker.wrapper.vla.vision_backbone.featurizer,
               attacker.wrapper.vla.vision_backbone.fused_featurizer]
    attacker.wrapper.vla.fail = True
    with pytest.raises(RuntimeError, match="predictor failure"):
        attacker._forward_rgb({}, None, observation, torch.zeros(3, 224, 224),
                              clean_siglip=torch.zeros(1, 256, 4))
    assert all(not module._forward_hooks for module in modules)
    with pytest.raises(RuntimeError, match="freeze failure"):
        with frozen_model(attacker.wrapper):
            assert not next(attacker.wrapper.vla.parameters()).requires_grad
            raise RuntimeError("freeze failure")
    assert all(parameter.requires_grad for parameter in attacker.wrapper.vla.parameters())


def test_runtime_exact_reached_boundary_uses_separate_interface_probe(tiny_setup, monkeypatch):
    cfg, clean, _, _, _, prefix, _ = tiny_setup
    attacker = MockAttacker()
    record = clean[0]
    obs = record["observation"]
    patch = prefix["patch"]
    with torch.no_grad():
        _, clean_dino, clean_siglip = attacker._forward({}, None, obs, None, False, capture_siglip=True)
    seed_transform(cfg["seed"]*10_000_000+8_100_000)
    rgb, _ = attacker._sample_rgb(obs, patch, True)
    with torch.no_grad():
        action, _, _ = attacker._forward_rgb({}, None, obs, rgb, clean_siglip=clean_siglip)
    cfg = dict(cfg, action_head_q99=[float(action.flatten()[0])]*7)
    def zero_clean_action(*args, **kwargs):
        return torch.full((1, 8, 7), -2.), clean_dino, clean_siglip
    monkeypatch.setattr(attacker, "_forward", zero_clean_action)
    report = attacker._validate_repair_runtime(record, patch, cfg, "dino_causal_clear", {})
    assert report["training_gradient_norm"] == 0
    assert report["gradient_probe"] == "opposite_boundary_interface_probe"
    assert report["gradient_norm"] > 0 and report["eta_zero_gradient_relative_l2"] < .1


def test_calibration_is_task_balanced_shared_and_immutable(tiny_setup):
    cfg, clean, replay, sources, data, _, _ = tiny_setup
    cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "calibration"))
    attacker = MockAttacker()
    report = attacker.calibrate_appearance(clean, cfg, sources, data)
    assert [row["record_index"] for row in report["gradient_norms"]] == list(range(0, 300, 30))
    med = report["median_gradient_norms"]
    assert report["lambda_tv"]*med["tv"]/med["action"] == pytest.approx(.15)
    assert report["lambda_hf"]*med["hf"]/med["action"] == pytest.approx(.15)
    calls = attacker.sample_calls
    assert attacker.calibrate_appearance(clean, cfg, sources, data) == report
    assert attacker.sample_calls == calls
    with pytest.raises(RuntimeError, match="identity drift"):
        attacker.calibrate_appearance(clean, cfg, {"changed": "source"}, data)
