"""CPU arithmetic/mock contracts ONLY; never model weights, renderer or Slurm."""
from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import METHODS, config, family, method_flags, deployment_conditions
from objectives import (target_action, action_loss, distillation_loss, frequency_filter, precondition_step,
                        scored_weights, draw_record, appearance_loss)
from pipeline import STAGES, existing_gpu_jobs, command_for
from train import save_patch, validate_resume
from analyze import task_casr, paired_bootstrap
from collect import paste_png
from engine import OFTEngine


@pytest.fixture
def cfg():
    return config()


def test_uada_far_endpoint_not_simple_negation(cfg):
    clean = torch.tensor([[[.4, -.3, .2, 0., 0., 0., 0.]]]).expand(1, 8, 7).clone().requires_grad_()
    target = target_action(clean, "uada", {"q01": [-1.] * 7, "q99": [1.] * 7}, cfg)
    assert target[0, 0, 0] == -1 and target[0, 0, 1] == 1
    assert not target.requires_grad


def test_uada_asymmetric_midpoint(cfg):
    clean = torch.full((1, 8, 7), .2)
    target = target_action(clean, "uada", {"q01": [-.9] * 7, "q99": [.4] * 7}, cfg)
    assert torch.allclose(target, torch.full_like(target, -.9))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_tma_xyz_same_target_all_future_slots(cfg, dtype):
    clean = torch.rand(1, 8, 7, dtype=dtype, requires_grad=True)
    target = target_action(clean, "tma", {}, cfg)
    assert torch.equal(target[..., :3], torch.zeros(1, 8, 3))
    assert torch.equal(target[..., 3:], clean[..., 3:]) and not target.requires_grad


@pytest.mark.parametrize("family_name,indices", [("uada", [0]), ("tma", [0, 1, 2])])
def test_only_selected_dof_receives_main_loss_gradient(cfg, family_name, indices):
    action = torch.ones(1, 8, 7, requires_grad=True)
    loss = action_loss(action, torch.zeros_like(action), family_name, cfg)
    loss.backward()
    expected = torch.zeros_like(action).bool(); expected[..., indices] = True
    assert torch.equal(action.grad.ne(0), expected)


def test_teacher_detached_and_dense_sample_specific():
    student = torch.ones(1, 256, 8, requires_grad=True)
    teacher = torch.full_like(student, 2., requires_grad=True)
    clean = torch.zeros_like(student, requires_grad=True)
    loss = distillation_loss(student, teacher, clean, 1e-4)
    loss.backward()
    assert student.grad is not None and teacher.grad is None and clean.grad is None
    assert float(loss.detach()) == pytest.approx(.25)


@pytest.mark.parametrize("shape", [(3, 50, 50), (3, 49, 51)])
def test_frequency_keeps_all_real_dimensions(shape):
    source = torch.randn(shape)
    filtered = frequency_filter(source)
    h, w = shape[-2:]
    fy = torch.fft.fftfreq(h)[:, None]; fx = torch.fft.rfftfreq(w)[None, :]
    gain = (1 + 8 * (fx.square() + fy.square())).pow(-.75)
    recovered = torch.fft.irfft2(torch.fft.rfft2(filtered) / gain, s=(h, w))
    assert (gain > 0).all() and torch.allclose(source, recovered, atol=2e-6)


def test_frequency_initial_actual_clipped_step_and_fixed_scale(cfg):
    before = torch.rand(3, 50, 50)
    proposal = before + torch.randn_like(before) * .01
    new, scale = precondition_step(before, proposal, None, cfg)
    expected = (proposal.clamp(0, 1) - before).norm()
    assert (new - before).norm() == pytest.approx(float(expected), rel=2e-6)
    _, next_scale = precondition_step(new, new + torch.randn_like(new) * .01, scale, cfg)
    assert next_scale == scale and new.min() >= 0 and new.max() <= 1


def test_zero_lr_frequency_does_not_freeze_calibration(cfg):
    before = torch.rand(3, 50, 50)
    value, scale = precondition_step(before, before, None, cfg)
    assert torch.equal(value, before) and scale is None


def test_method_inventory_no_teacher_frequency_confound():
    assert len(METHODS) == 8 and len(deployment_conditions()) == 25
    for method in METHODS:
        f = method_flags(method)
        assert not (f["teacher"] and f["frequency"])
    assert method_flags("uada_dino")["tv"]
    assert not method_flags("uada_original")["tv"]
    assert not method_flags("tma_xyz_dino")["tv"]


def test_dino_weights_no_positive_score_means_no_fake_preference():
    clips = [{"task_id": 0, "phase_proxy": "early"} for _ in range(4)]
    assert scored_weights(clips, [0, 0, 0, 0]) == [1, 1, 1, 1]
    assert scored_weights(clips, [0, 4, 2, 1]) == [1, 2, 2, 1]


def test_source_schedule_and_tma_uniform_only_changes_weights(cfg):
    bank = {"clean": ["clean:0"], "observations": {"clean:0": {"task_id": 0}},
            "clips": [{"task_id": 0, "phase_proxy": "early", "uids": [f"replay:{i}"] * 3} for i in range(3)]}
    assert draw_record(bank, "uada_dino", 250, [1] * 3, cfg)["origin"] == "clean"
    assert draw_record(bank, "uada_dino", 251, [1] * 3, cfg)["origin"] == "clean"
    assert draw_record(bank, "uada_original", 252, [1] * 3, cfg)["origin"] == "clean"
    a = draw_record(bank, "tma_xyz", 252, [1] * 3, cfg)
    b = draw_record(bank, "tma_xyz_dino", 252, [1] * 3, cfg)
    assert a == b and len(a["uids"]) == 3


def test_native_png_paste_wrist_only():
    obs = {"full_image": np.zeros((224, 224, 3), np.uint8), "wrist_image": np.zeros((224, 224, 3), np.uint8)}
    out = paste_png(obs, np.full((50, 50, 3), 128, np.uint8), (26, 137))
    assert np.array_equal(out["full_image"], obs["full_image"])
    assert out["wrist_image"][137:187, 26:76].min() == 128
    assert obs["wrist_image"].max() == 0


def test_png_pt_quantization_is_identical(tmp_path, cfg):
    from PIL import Image
    patch = torch.rand(3, 50, 50)
    save_patch(tmp_path, patch, patch, "uada_original", 500, cfg, "mock")
    value = torch.load(tmp_path / "outer_0500.pt", weights_only=False)
    with Image.open(tmp_path / "outer_0500.png") as image:
        assert np.array_equal(np.asarray(image), value["patch"].permute(1, 2, 0).mul(255).round().byte().numpy())
    assert value["optimizer_updates"] == 25000


def test_resume_requires_adam_clock_and_rng(cfg):
    saved = {"signature": "mock", "method": "uada_original", "config": cfg, "outer": 1, "rows": [{}],
             "scheduler": {"last_epoch": 1}, "optimizer": {"state": {0: {"step": 50}}},
             **{k: None for k in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "weights", "distill_coefficient", "frequency_scale")}}
    validate_resume(saved, "uada_original", cfg, "mock")
    broken = copy.deepcopy(saved); broken["optimizer"]["state"][0]["step"] = 1
    with pytest.raises(RuntimeError, match="Adam"):
        validate_resume(broken, "uada_original", cfg, "mock")
    broken = copy.deepcopy(saved); del broken["cuda_rng"]
    with pytest.raises(RuntimeError, match="Incomplete"):
        validate_resume(broken, "uada_original", cfg, "mock")


def test_fresh_clean_not_frozen_denominator():
    frame = pd.DataFrame([{"task_id": 0, "state_index": i, "condition": "clean", "success": int(i == 0), "failure": int(i != 0)} for i in range(2)]
                         + [{"task_id": 0, "state_index": i, "condition": "attack", "success": 0, "failure": 1} for i in range(2)])
    point, pairs = task_casr(frame, "attack")
    assert point == 1 and pairs.clean_success.sum() == 1
    ci = paired_bootstrap(pairs.assign(value=pairs.failure), draws=20)
    assert ci == [1., 1.]


def test_queue_dependency_and_safe_rejection():
    queue = "100_[0-7%8]|PENDING|gres/gpu:4090:1|Resources\n101|RUNNING|N/A|node\n"
    assert existing_gpu_jobs(queue) == ["100"]
    with pytest.raises(RuntimeError, match="invalid dependency"):
        existing_gpu_jobs("102|PENDING|gres/gpu:3090:1|DependencyNeverSatisfied")
    with pytest.raises(RuntimeError, match="Unknown squeue"):
        existing_gpu_jobs("unparseable")
    assert "--dependency=afterok:200" in command_for("train", "test.sbatch", "200")
    assert "--dependency=afterany:100:101" in command_for("prefix", "test.sbatch", external=["100", "101"])


def test_all_gpu_arrays_capped_and_serial_no_smoke():
    assert len(STAGES) == 10 and not any("smoke" in stage for stage, _ in STAGES)
    for stage, filename in STAGES:
        text = (ROOT / "slurm" / filename).read_text()
        if "--array=" in text:
            assert "%8" in text and "--gres=gpu:" in text
        if "3090" in filename:
            assert "--gres=gpu:3090:1" in text
        if "4090" in filename:
            assert "--gres=gpu:4090:1" in text


def test_evaluation_never_imports_attack_engine_or_teacher():
    tree = ast.parse((ROOT / "code/evaluate.py").read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert "engine" not in imports and "teacher" not in imports
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "register_forward_hook" not in calls


class FakeFeatures(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 256, 8)


def test_restore_keeps_attacked_siglip_and_backprop_to_rgb_mock():
    dino, siglip = FakeFeatures(), FakeFeatures()
    model = SimpleNamespace(vision_backbone=SimpleNamespace(featurizer=dino, fused_featurizer=siglip))
    def predict_action(**values):
        pixels = values["pixel_values"]
        front = dino(pixels[:, :3]); siglip(pixels[:, 3:6])
        wrist = dino(pixels[:, 6:9]); wrist_siglip = siglip(pixels[:, 9:12])
        return None, (front.mean() + wrist.mean() + wrist_siglip.mean()).reshape(1, 1, 1).expand(1, 8, 7)
    model.predict_action = predict_action
    wrapper = SimpleNamespace(vla=model, freeze_model=lambda: None, device="cpu",
                              gen_cfg=SimpleNamespace(unnorm_key="mock"), proprio_projector=None,
                              noisy_action_projector=None, action_head=SimpleNamespace(predict_action=lambda h: h))
    engine = OFTEngine(wrapper)
    pixels = torch.zeros(1, 12, 4, 4, requires_grad=True)
    with torch.no_grad():
        _, clean = engine.forward({"pixel_values": pixels.detach()}, np.zeros(8))
    attacked_pixels = pixels + torch.cat([torch.zeros(1, 6, 4, 4), torch.ones(1, 6, 4, 4)], dim=1)
    action, caps = engine.forward({"pixel_values": attacked_pixels}, np.zeros(8))
    restored, restored_caps = engine.forward({"pixel_values": attacked_pixels}, np.zeros(8), replacement=clean["dino"])
    assert torch.equal(caps["siglip"], restored_caps["siglip"]) and restored.mean() < action.mean()
    action.sum().backward()
    assert pixels.grad[:, 6:9].abs().sum() > 0 and len(dino._forward_hooks) == 0


def test_entire_submission_chain_mock_afterok_and_external_wait(tmp_path, monkeypatch):
    import pipeline
    commands = []
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda: {"signature": "mock"})
    monkeypatch.setattr(pipeline, "run_signature", lambda: "mock")
    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="900|RUNNING|gres/gpu:4090:1|node\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=str(1000 + len(commands) - 2) + "\n", stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.submit()
    from common import load_json
    receipt = load_json(tmp_path / "reports/SUBMISSION_RECEIPT.json")
    assert receipt["status"] == "SUBMITTED" and len(receipt["jobs"]) == 10
    assert "--dependency=afterany:900" in commands[1]
    for index in range(2, 11):
        assert "--dependency=afterok:" + str(1000 + index - 2) in commands[index]
    assert all(command[0] != "scancel" for command in commands)


def test_submission_failure_preserves_first_job_no_cancel(tmp_path, monkeypatch):
    import pipeline
    calls = []
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda: {"signature": "mock"})
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if len(calls) == 2:
            return SimpleNamespace(returncode=0, stdout="1000\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="mock resource error")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Earlier jobs retained"):
        pipeline.submit()
    from common import load_json
    receipt = load_json(tmp_path / "reports/SUBMISSION_RECEIPT.json")
    assert receipt["status"] == "SUBMISSION_FAILED" and receipt["failed_stage"] == "prefix"
    assert len(receipt["jobs"]) == 1 and receipt["jobs"][0]["job_id"] == "1000"
    assert all(command[0] != "scancel" for command in calls)


def test_successful_renderer_runtime_reused_without_algorithm_changes():
    current = ast.parse((ROOT / "code/model_runtime.py").read_text())
    old_path = ROOT.parent / "wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/code/model_runtime_fixed.py"
    original = ast.parse(old_path.read_text())
    def function(tree, name):
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    for name in ("observation", "load_wrapper", "reset_wait", "bind_environment_context", "reset_environment", "make_env"):
        assert function(current, name) == function(original, name)
