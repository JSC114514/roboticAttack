"""CPU mocks and tensor contracts; no checkpoint loading, CUDA, or VLA claims."""
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "VLAAttacker"), str(ROOT / "scripts"),
               str(ROOT.parent / "vla_internal_attribution_20260812")]

import dino_sampling_clear_common as common
import white_patch.UADA_OFT_DINO_SAMPLING_CLEAR as trainer
from white_patch.UADA_OFT_CORRESPONDENCE import ExploringOFTCorrespondenceAttacker
from white_patch.UADA_OFT_DINO_TRANSFER import ExploringOFTDINOTransferAttacker
from white_patch.appply_random_transform import RandomPatchTransform
from white_patch.oft_dino_sampling_clear_objectives import constraint_metrics


class LoopMock(trainer.ExploringOFTDINOSamplingClearAttacker):
    def __init__(self):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), vla=torch.nn.Linear(1, 1))
        self.site = (26, 137)
        self.forward_log = []
        self.patched_calls = 0
        self.fail_patch_call = None
        self.score_patch_hashes = []

    def _base_inputs(self, record):
        return {}, None

    def _validate_runtime(self, record, patch, cfg):
        # Intentionally consumes RNG so restoration around startup is tested.
        random.random()
        np.random.random()
        torch.rand(1)
        return {"status": "CPU_MOCK_ONLY", "real_vla_loaded": False}

    def _forward(self, inputs, proprio, observation, patch, geometry, **kwargs):
        assert not kwargs, "Training must use the unchanged normal route"
        self.forward_log.append((patch is not None, bool(geometry), bool(getattr(self, "transfer_diverse", False)),
                                 int(observation["frame"])))
        value = torch.tensor(float(observation["frame"]) / 100)
        if patch is not None:
            self.patched_calls += 1
            if self.patched_calls == self.fail_patch_call:
                raise RuntimeError("mock half-outer failure")
            if getattr(self, "transfer_diverse", False):
                random.randint(0, 174)
                random.randint(0, 174)
                factor = random.uniform(.8, 1.2) * random.uniform(.8, 1.2)
            else:
                factor = 1.
            # Real gradient path, with unequal non-target dimensions.
            value = value + patch.mean() * factor
        action = value.expand(1, 8, 7) + torch.arange(7).reshape(1, 1, 7) * 10
        return action, torch.ones(1, 256, 4), None

    def _stable_score_record(self, record, displayed, cfg, refresh_outer, record_index):
        self.score_patch_hashes.append(trainer.patch_tensor_sha256(displayed))
        scalar = float(displayed.mean()) + record["decision_index"] / 100
        return {"dino_score": scalar, "dino_delta_rms": scalar,
                "patched_target_mse": scalar, "restored_target_mse": 2 * scalar,
                "score_views": [{"raw_delta": scalar}, {"raw_delta": scalar + .1}],
                "score_view_count": 2, "score_full_forward_calls": 5}


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = {"experiment_id": "CPU-MOCK-ONLY", "output_root": str(tmp_path / "run"), "seed": 42,
           "data_seed": 123, "patch_shape": [3, 50, 50], "inner_loop": 4,
           "bootstrap_outer": 2, "outer_iterations": 8, "warmup_outer_iterations": 2,
           "learning_rate": .002, "weight_decay": 0, "action_head_q01": [-1.] * 7,
           "action_head_q99": [1.] * 7, "maskidx": [0], "geometry": True,
           "checkpoint_outer_iterations": [0, 2, 4, 6, 8], "refresh_after_outer": [2, 4, 6],
           "resume_every_outer_iterations": 1, "tv_threshold": .3, "tv_scale": .4, "tv_weight": 1}
    clean = [{"task_id": i // 30, "state_index": i % 30, "task_name": f"task{i // 30}",
              "observation": {"frame": i % 3}} for i in range(300)]
    replay = [dict(clean[0], decision_index=i, phase_proxy="early",
                   observations=[{"frame": j} for j in range(3)], dino_score=0.,
                   dino_weight=1., shuffled_weight=1.) for i in range(4)]
    def draw(clean, replay, method, outer, cfg):
        if outer <= cfg["bootstrap_outer"] or (outer - cfg["bootstrap_outer"]) % 2:
            return clean[0], "clean", 0, 0
        field = "shuffled_weight" if method == "shuffled_hard" else "dino_weight"
        index = random.Random(123 + outer).choices(range(len(replay)), weights=[r[field] for r in replay])[0]
        return replay[index], "replay", index, 0
    monkeypatch.setattr(common, "draw_record", draw)
    legacy_calls = []
    def legacy_score(wrapper, observation, task_name, displayed, cfg):
        legacy_calls.append(displayed.detach().clone())
        random.random()
        np.random.random()
        torch.rand(1)
        scalar = float(displayed.mean())
        return {"dino_score": scalar, "dino_delta_rms": scalar,
                "patched_target_mse": scalar, "restored_target_mse": scalar * 2}
    monkeypatch.setitem(sys.modules, "collect_exploring_oft_dino_replay", SimpleNamespace(score_record=legacy_score))
    sources = {str(Path(__file__).resolve()): trainer.sha256_file(__file__)}
    data = {"clean_bank": "mock", "window_bank": "mock", "config": "mock"}
    return cfg, clean, replay, sources, data, legacy_calls


def load_resume(cfg, method):
    return torch.load(Path(cfg["output_root"]) / "training" / method / "resume.pt", weights_only=False)


@pytest.mark.parametrize("method", common.METHODS)
def test_all_arms_complete_fresh_loop_schedule_and_normal_targets(method, tiny_setup):
    cfg, clean, replay, sources, data, legacy_calls = tiny_setup
    attacker = LoopMock()
    attacker.train_sampling_clear(clean, replay, cfg, method, sources, data)
    saved = load_resume(cfg, method)
    trainer.validate_state(saved, cfg)
    out = Path(cfg["output_root"]) / "training" / method
    manifest = json.loads((out / "RUN_MANIFEST.json").read_text())
    assert manifest["fresh_start_outer"] == 0 and manifest["initial_optimizer_state_empty"]
    assert not manifest["feature_loss"] and not manifest["feature_replacement_hook_during_training"]
    assert manifest["training_read_only_dense_capture"]
    assert saved["outer"] == 8 and saved["scheduler"]["last_epoch"] == 8
    assert int(next(iter(saved["optimizer"]["state"].values()))["step"]) == 32
    assert saved["costs"]["actual_optimizer_updates"] == 32
    assert saved["costs"]["training_action_backward_calls"] == 32
    assert saved["rows"][0]["zero_lr_first_step"]
    assert saved["rows"][0]["rgb_outer_max_abs"] == 0
    assert saved["rows"][0]["main_loss"] < 4  # Dim0 only; other dimensions are far larger.
    assert saved["rows"][-1]["rgb_init_mean_abs"] > 0
    patched = [entry for entry in attacker.forward_log if entry[0]]
    assert [entry[1] for entry in patched] == [False, False, True, True] * 8
    assert not any(entry[2] for entry in patched[:8])
    assert [entry[2] for entry in patched[8:]] == [False, False, True, True] * 6
    replay_outer_four = patched[12:16]
    assert [entry[3] for entry in replay_outer_four] == [0, 1, 2, 0]
    assert [item["outer"] for item in saved["score_refresh_history"]] == [2, 4, 6]
    for history in saved["score_refresh_history"]:
        checkpoint = torch.load(out / f"outer_{history['outer']:04d}.pt", weights_only=False)["patch"]
        assert history["quantized_patch_sha256"] == trainer.patch_tensor_sha256(trainer.canonical_quantized_patch(checkpoint))
    expected_calls = 3 if method in ("dynamic_soft", "dynamic_hard") else 5
    assert saved["costs"]["score_full_forward_calls"] == 3 * len(replay) * expected_calls
    assert bool(legacy_calls) == (method in ("dynamic_soft", "dynamic_hard"))
    assert ("training_appearance_backward_calls" in saved["costs"]) == (method == "dynamic_soft")
    for checkpoint in cfg["checkpoint_outer_iterations"]:
        path = out / f"outer_{checkpoint:04d}.pt"
        value = torch.load(path, weights_only=False)["patch"]
        with Image.open(path.with_suffix(".png")) as image:
            assert np.array_equal(np.asarray(image), value.permute(1, 2, 0).mul(255).round().byte().numpy())
        if method != "dynamic_soft":
            assert constraint_metrics(value)["feasible_internal"]
            assert constraint_metrics(value.mul(255).round().div(255))["feasible_scientific"]
    raw = torch.load(out / "initial_raw.pt", weights_only=False)["patch"]
    assert not torch.equal(raw, saved["initial"])
    assert trainer.patch_tensor_sha256(raw) == manifest["raw_random_patch_sha256"]
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "COMPLETE" and status["optimizer_updates"] == 32
    assert status["score_refresh_outer"] == [2, 4, 6]
    before_hash = trainer.sha256_file(out / "resume.pt")
    attacker.train_sampling_clear(clean, replay, cfg, method, sources, data)
    assert trainer.sha256_file(out / "resume.pt") == before_hash


@pytest.mark.parametrize("method", common.METHODS)
def test_half_outer_resume_bitwise_patch_moments_schedule_scores_rng(method, tiny_setup):
    cfg, clean, replay, sources, data, _ = tiny_setup
    reference_cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "reference"))
    LoopMock().train_sampling_clear(clean, replay, reference_cfg, method, sources, data)
    interrupted = LoopMock()
    interrupted.fail_patch_call = 10  # outer3, second inner; scoring after2 is still uncommitted.
    with pytest.raises(RuntimeError, match="half-outer"):
        interrupted.train_sampling_clear(clean, replay, cfg, method, sources, data)
    partial = load_resume(cfg, method)
    assert partial["outer"] == 2 and len(partial["rows"]) == 2
    assert int(next(iter(partial["optimizer"]["state"].values()))["step"]) == 8
    assert partial["score_refresh_history"] == []
    out = Path(cfg["output_root"]) / "training" / method
    # Mimic a checkpoint written by an uncommitted complete-outer attempt.
    trainer._save_checkpoint(out / "outer_0004.pt", torch.zeros(3, 50, 50), {})
    LoopMock().train_sampling_clear(clean, replay, cfg, method, sources, data)
    resumed, expected = load_resume(cfg, method), load_resume(reference_cfg, method)
    assert torch.equal(resumed["patch"], expected["patch"])
    assert resumed["scheduler"] == expected["scheduler"]
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(next(iter(resumed["optimizer"]["state"].values()))[key],
                           next(iter(expected["optimizer"]["state"].values()))[key])
    assert next(iter(resumed["optimizer"]["state"].values()))["step"] == next(iter(expected["optimizer"]["state"].values()))["step"]
    assert resumed["replay_scoring"] == expected["replay_scoring"]
    assert [r["record_index"] for r in resumed["rows"]] == [r["record_index"] for r in expected["rows"]]
    assert torch.equal(resumed["torch_rng"], expected["torch_rng"])
    assert resumed["python_rng"] == expected["python_rng"]
    assert np.array_equal(resumed["numpy_rng"][1], expected["numpy_rng"][1])
    assert resumed["costs"]["actual_optimizer_updates"] == expected["costs"]["actual_optimizer_updates"] + 1
    assert resumed["costs"]["score_records"] == expected["costs"]["score_records"] + len(replay)
    assert resumed["elapsed_s"] >= partial["elapsed_s"]
    assert list((out / "discarded_attempts").rglob("outer_0004.pt"))
    assert list((out / "discarded_attempts").rglob("scores_after_0002.json"))


def test_refresh_rng_isolation_and_independent_current_patch_quantization(tiny_setup):
    cfg, clean, replay, _, _, legacy_calls = tiny_setup
    attacker = LoopMock()
    patch = torch.rand(3, 50, 50)
    initial_state = trainer._rng_state(patch)
    result = attacker._refresh_sampling_scores(replay, patch, cfg, "dynamic_hard", 2)
    final_state = trainer._rng_state(patch)
    assert torch.equal(initial_state["torch_rng"], final_state["torch_rng"])
    assert initial_state["python_rng"] == final_state["python_rng"]
    assert np.array_equal(initial_state["numpy_rng"][1], final_state["numpy_rng"][1])
    assert torch.equal(legacy_calls[0], patch.mul(255).round().div(255))
    changed = attacker._refresh_sampling_scores(replay, patch + .1, cfg, "dynamic_hard", 2)
    assert changed[0]["dino_score"] != result[0]["dino_score"]
    own = attacker._refresh_sampling_scores(replay, patch, cfg, "shuffled_hard", 2)
    assert len(attacker.score_patch_hashes) == len(replay)
    assert sorted(r["dino_weight"] for r in own) == sorted(r["shuffled_weight"] for r in own)


class TinyEncoder(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, pixels):
        return pixels.mean((-2, -1)).unsqueeze(1).expand(-1, 256, -1) * self.scale


class TinyHead(torch.nn.Module):
    def predict_action(self, hidden):
        return hidden.mean((1, 2))[:, None, None].expand(-1, 8, 7)


class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Module()
        self.vision_backbone.featurizer = TinyEncoder(.7)
        self.vision_backbone.fused_featurizer = TinyEncoder(.4)
        self.pixel_log = []
        self.siglip_log = []

    def predict_action(self, pixel_values, **kwargs):
        self.pixel_log.append(pixel_values.detach().clone())
        values = []
        for view in (pixel_values[:, :6], pixel_values[:, 6:]):
            dino = self.vision_backbone.featurizer(view[:, :3])
            siglip = self.vision_backbone.fused_featurizer(view[:, 3:6])
            self.siglip_log.append(siglip.detach().clone())
            values.append(dino + siglip)
        return None, torch.cat(values, 1)


def test_two_view_scoring_same_pixels_current_siglip_and_raw_min(monkeypatch):
    attacker = trainer.ExploringOFTDINOSamplingClearAttacker.__new__(trainer.ExploringOFTDINOSamplingClearAttacker)
    attacker.wrapper = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
        vla=TinyVLA(), action_head=TinyHead(), gen_cfg=SimpleNamespace(unnorm_key="mock"),
        proprio_projector=None, noisy_action_projector=None)
    attacker.site = (26, 137)
    attacker.transfer_diverse = False
    attacker.randomPatchTransform = RandomPatchTransform("cpu", False)
    monkeypatch.setattr(attacker, "_base_inputs", lambda record: ({}, None))
    render_log = []
    def mock_pixels(self, observation, patch, geometry):
        front = torch.full((1, 3, 224, 224), .2)
        if patch is None:
            wrist = torch.full_like(front, .1)
        else:
            render_log.append((bool(geometry), self.transfer_diverse))
            if self.transfer_diverse:
                multiplier = random.uniform(.8, 1.2)
            else:
                multiplier = 1.
            wrist = (patch * multiplier).mean().expand_as(front)
        return torch.cat((front, front, wrist, wrist), 1)
    monkeypatch.setattr(ExploringOFTCorrespondenceAttacker, "_patched_pixels", mock_pixels)
    cfg = {"data_seed": 123, "action_head_q01": [-1.] * 7,
           "action_head_q99": [1.] * 7, "maskidx": [0]}
    record = {"observation": {}, "task_name": "mock"}
    result = attacker._stable_score_record(record, torch.full((3, 50, 50), .6), cfg, 250, 0)
    assert render_log == [(False, False), (True, True)]  # One cached render per view.
    logs = attacker.wrapper.vla.pixel_log
    assert len(logs) == 5 and torch.equal(logs[1], logs[2]) and torch.equal(logs[3], logs[4])
    siglip = attacker.wrapper.vla.siglip_log
    assert torch.equal(siglip[3], siglip[5]) and torch.equal(siglip[7], siglip[9])
    assert not torch.equal(siglip[1], siglip[3])  # Clean SigLIP was not substituted.
    assert result["dino_score"] == max(0., min(v["raw_delta"] for v in result["score_views"]))
    assert all(view["patched_siglip_retained"] for view in result["score_views"])
    assert attacker._scoring_pixels is None and not attacker.transfer_diverse
    modules = [attacker.wrapper.vla.vision_backbone.featurizer,
               attacker.wrapper.vla.vision_backbone.fused_featurizer]
    assert all(not module._forward_hooks for module in modules)


def test_transfer_renderer_inheritance_preserves_site_and_gradient():
    attacker = trainer.ExploringOFTDINOSamplingClearAttacker.__new__(trainer.ExploringOFTDINOSamplingClearAttacker)
    attacker.wrapper = SimpleNamespace(device=torch.device("cpu"))
    attacker.site = (26, 137)
    attacker.randomPatchTransform = RandomPatchTransform("cpu", False)
    attacker.transfer_diverse = True
    patch = torch.full((3, 50, 50), .6, requires_grad=True)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    random.seed(123)
    np.random.seed(123)
    rgb, support = attacker._paste_with_support(image, patch, True)
    assert rgb.shape == (3, 224, 224) and support.shape == (224, 224)
    assert attacker.site == (26, 137)
    rgb.sum().backward()
    assert patch.grad is not None and torch.isfinite(patch.grad).all() and patch.grad.norm() > 0
    assert trainer.ExploringOFTDINOSamplingClearAttacker._paste_with_support is ExploringOFTDINOTransferAttacker._paste_with_support


def test_resume_rejects_config_source_and_optimizer_drift(tiny_setup):
    cfg, clean, replay, sources, data, _ = tiny_setup
    LoopMock().train_sampling_clear(clean, replay, cfg, "dynamic_hard", sources, data)
    saved = load_resume(cfg, "dynamic_hard")
    bad = copy.deepcopy(saved)
    bad["scheduler"]["last_epoch"] = 0
    with pytest.raises(RuntimeError, match="Scheduler"):
        trainer.validate_state(bad, cfg)
    bad = copy.deepcopy(saved)
    next(iter(bad["optimizer"]["state"].values()))["step"] -= 1
    with pytest.raises(RuntimeError, match="Adam"):
        trainer.validate_state(bad, cfg)
    with pytest.raises(RuntimeError, match="identity"):
        LoopMock().train_sampling_clear(clean, replay, cfg, "dynamic_hard", dict(sources, changed="hash"), data)


def test_zero_outer_resume_has_empty_adam_and_common_seed42_initial(tiny_setup, monkeypatch):
    cfg, clean, replay, sources, data, _ = tiny_setup
    originals = []
    initials = []
    for method in common.METHODS:
        mock = LoopMock()
        mock.fail_patch_call = 1
        with pytest.raises(RuntimeError, match="half-outer"):
            mock.train_sampling_clear(clean, replay, cfg, method, sources, data)
        saved = load_resume(cfg, method)
        assert saved["outer"] == 0 and not saved["optimizer"]["state"]
        assert saved["scheduler"]["last_epoch"] == 0
        originals.append(saved["initial_raw"])
        initials.append(saved["initial"])
    assert all(torch.equal(originals[0], value) for value in originals)
    assert all(torch.equal(initials[0], value) for value in initials)


def test_loaded_hf_cache_code_must_match_frozen_checkpoint(tmp_path):
    original = tmp_path / "checkpoint" / "modeling_prismatic.py"
    original.parent.mkdir()
    original.write_text("class FrozenModel: pass\n")
    cached = tmp_path / "cache" / "modeling_prismatic.py"
    cached.parent.mkdir()
    cached.write_text(original.read_text())
    cfg = {"frozen_eval_assets": {str(original): trainer.sha256_file(original)}}
    trainer.verify_loaded_checkpoint_source(cached, cfg)
    cached.write_text("class ChangedModel: pass\n")
    with pytest.raises(RuntimeError, match="Actually loaded checkpoint source"):
        trainer.verify_loaded_checkpoint_source(cached, cfg)
    unrelated = tmp_path / "cache" / "__init__.py"
    unrelated.write_text("# Unrelated package\n")
    trainer.verify_loaded_checkpoint_source(unrelated, cfg)


def test_wrapper_preflight_does_not_touch_cuda_or_load_model(monkeypatch, capsys):
    import UADA_OFT_DINO_SAMPLING_CLEAR_wrapper as wrapper
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preflight touched CUDA/model startup")
    for name in ("is_available", "device_count", "init", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(wrapper, "config", lambda path: {"outer_iterations": 2000, "inner_loop": 50})
    monkeypatch.setattr(wrapper, "source_records", lambda cfg: [{}] * 300)
    monkeypatch.setattr(wrapper, "replay_records", lambda cfg: ([{}] * 258, "mock"))
    monkeypatch.setattr(wrapper, "pipeline_preflight", lambda cfg: {"status": "PASS", "source": {}, "data": {}})
    monkeypatch.setitem(sys.modules, "UADA_OFT_DINO_wrapper", SimpleNamespace(load_wrapper=forbidden))
    monkeypatch.setattr(sys, "argv", ["wrapper", "--preflight"])
    wrapper.main()
    output = json.loads(capsys.readouterr().out)
    assert output["model_loaded"] is False and output["cuda_initialized"] is False
    assert output["sbatch_submitted"] is False and output["shared_trained_prefix"] is False
    assert output["historical_patch_used_for_initialization"] is False
    assert output["historical_reference_patches_checked"] is True


def test_stable_refresh_exception_restores_training_rng_and_renderer(tiny_setup, monkeypatch):
    cfg, _, replay, _, _, _ = tiny_setup
    attacker = LoopMock()
    patch = torch.rand(3, 50, 50)
    saved = trainer._rng_state(patch)
    def fail(*args, **kwargs):
        attacker.site = (0, 0)
        attacker.transfer_diverse = True
        attacker._scoring_pixels = torch.rand(1)
        random.random()
        np.random.random()
        raise RuntimeError("mock scoring exception")
    monkeypatch.setattr(attacker, "_stable_score_record", fail)
    with pytest.raises(RuntimeError, match="scoring exception"):
        attacker._refresh_sampling_scores(replay, patch, cfg, "stable_hard", 2)
    assert attacker.site == (26, 137) and attacker.transfer_diverse is False
    assert attacker._scoring_pixels is None
    restored = trainer._rng_state(patch)
    assert torch.equal(saved["torch_rng"], restored["torch_rng"])
    assert saved["python_rng"] == restored["python_rng"]
    assert np.array_equal(saved["numpy_rng"][1], restored["numpy_rng"][1])


def test_canonical_cpu_quantization_matches_png_and_every_score_path(tmp_path, tiny_setup, monkeypatch):
    cfg, _, replay, _, _, legacy_calls = tiny_setup
    patch = torch.linspace(-.2, 1.2, 7500, dtype=torch.float64).reshape(3, 50, 50).requires_grad_(True)
    quantized = trainer.canonical_quantized_patch(patch)
    assert quantized.device.type == "cpu" and quantized.dtype == torch.float32
    assert not quantized.requires_grad
    assert torch.equal(quantized, patch.detach().float().clamp(0, 1).mul(255).round().div(255))
    path = tmp_path / "quantization.pt"
    trainer._save_checkpoint(path, patch, {})
    with Image.open(path.with_suffix(".png")) as picture:
        rgb = torch.as_tensor(np.asarray(picture).copy()).permute(2, 0, 1).float().div(255)
    assert torch.equal(rgb, quantized)
    assert trainer.patch_tensor_sha256(rgb) == trainer.patch_tensor_sha256(quantized)
    original = trainer.canonical_quantized_patch
    calls = []
    def canonical_spy(value):
        result = original(value)
        calls.append(result)
        return result
    monkeypatch.setattr(trainer, "canonical_quantized_patch", canonical_spy)
    valid_patch = patch.detach().float().clamp(0, 1)
    trainer._structure(valid_patch, valid_patch, cfg)
    attacker = LoopMock()
    attacker._refresh_sampling_scores(replay, valid_patch, cfg, "dynamic_hard", 2)
    attacker._refresh_sampling_scores(replay, valid_patch, cfg, "stable_hard", 2)
    assert len(calls) == 3 and all(torch.equal(value, quantized) for value in calls)
    assert torch.equal(legacy_calls[0], quantized)
    assert all(digest == trainer.patch_tensor_sha256(quantized) for digest in attacker.score_patch_hashes)
