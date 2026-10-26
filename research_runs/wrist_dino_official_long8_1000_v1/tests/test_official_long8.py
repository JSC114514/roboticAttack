"""CPU/mock checks only; never load a checkpoint, render, or submit Slurm."""
from __future__ import annotations
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "code"))
import common
import data
import deployment
import objectives as obj
import pipeline
import train as training
import analyze
from engine import OFTEngine
from readonly_model import AUTO_MAP, install_readonly_loader, verify_checkpoint


def cfg():
    return copy.deepcopy(common.config())


def test_config_and_eight_distinct_methods():
    settings = cfg()
    assert settings["outer_iterations"] * settings["inner_loop"] == 50000
    assert len(common.METHODS) == 8
    assert settings["checkpoint_outer_iterations"] == [0, 250, 500, 750, 1000]
    assert common.flags("tma_xyz_geometry") == {
        "family": "tma", "sampling": False, "frequency": False, "geometry": True, "attention": False}
    assert common.flags("tma_xyz_attention")["attention"]
    assert not common.flags("tma_xyz_attention")["sampling"]


def test_expert_transform_bounds_and_no_mutation():
    raw = np.array([[2, -2, .2, .3, .4, .5, -1], [-.2, .2, 0, .3, .4, .5, 1]], np.float32)
    before = raw.copy()
    stats = {"q01": [-1] * 6 + [0], "q99": [1] * 7,
             "min": [-1] * 6 + [0], "max": [1] * 7, "mask": [True] * 6 + [False]}
    norm = data.normalized_expert(raw, stats)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(norm[:, 0], [1, -.2], atol=1e-6)
    np.testing.assert_array_equal(norm[:, 6], [1, 0])
    stats["min"][4] = stats["max"][4]
    assert not data.normalized_expert(raw, stats)[:, 4].any()


@pytest.mark.parametrize("family", ["uada", "tma"])
def test_targets_use_expert_labels_and_fixed_mask(family):
    expert = torch.linspace(-.7, .7, 56).reshape(1, 8, 7)
    before = expert.clone()
    target = obj.target_action(expert, family, cfg())
    assert torch.equal(expert, before)
    if family == "uada":
        assert torch.equal(target[..., 0], torch.where(expert[..., 0] >= 0, -1., 1.))
    else:
        assert not target[..., :3].any()
        assert torch.equal(target[..., 3:], expert[..., 3:])
    prediction = expert.clone().requires_grad_()
    obj.action_loss(prediction, target, family, cfg()).backward()
    assert not prediction.grad[..., 1 if family == "uada" else 3:].any()
    assert prediction.grad[..., 0].abs().sum() > 0


def test_candidate_windows_do_not_cross_episodes():
    episodes = [{"episode_id": "a", "eligible_windows": 2}, {"episode_id": "b", "eligible_windows": 3}]
    table = data.candidate_schedule(episodes, outer=100, seed=42)
    assert table == data.candidate_schedule(episodes, outer=100, seed=42)
    assert all(len(row) == 4 for row in table)
    assert {x for row in table for x in row} == {"a:0", "a:1", "b:0", "b:1", "b:2"}


def test_complete_official_preparation_schema_uses_expert_windows_only(monkeypatch, tmp_path):
    class TensorStub:
        def __init__(self, value):
            self.value = np.asarray(value)
        @property
        def shape(self):
            return self.value.shape
        def numpy(self):
            return self.value.item() if self.value.ndim == 0 else self.value
        def __getitem__(self, index):
            return TensorStub(self.value[index])
    settings = cfg()
    raw = np.tile(np.arange(7, dtype=np.float32)[None] / 20, (8, 1))
    raw[:, 0] = np.arange(8, dtype=np.float32) / 10
    episodes = [{"tfds_id": TensorStub(f"demo-{i}".encode()),
                 "episode_metadata": {"file_path": TensorStub(f"expert-{i}.hdf5".encode())},
                 "steps": {"action": TensorStub(raw.copy()),
                           "language_instruction": TensorStub([f"task {i % 10:02d}".encode()] * 8),
                           "observation": {"state": TensorStub(np.ones((8, 8), np.float32)),
                                           "image": TensorStub([b"jpeg"] * 8), "wrist_image": TensorStub([b"jpeg"] * 8)}}}
                for i in range(379)]
    shard = tmp_path / "mock_shard"
    shard.write_bytes(b"not_an_actual_RLDS_shard")
    image = np.zeros((224, 224, 3), np.uint8)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "config", lambda: settings)
    monkeypatch.setattr(data, "run_signature", lambda: "cpu_mock_only")
    monkeypatch.setattr(data, "preflight", lambda: {"status": "CPU_MOCK_ONLY", "signature": "cpu_mock_only"})
    monkeypatch.setattr(data, "dataset_files", lambda: [shard])
    monkeypatch.setattr(data, "episode_iterator", lambda: iter(episodes))
    monkeypatch.setattr(data, "resize_policy_jpeg", lambda encoded: image)
    data.prepare()
    bank = torch.load(tmp_path / "data/CANDIDATE_BANK.pt", weights_only=False)
    status = json.loads((tmp_path / "data/DATA_STATUS.json").read_text())
    assert status["status"] == "COMPLETE" and status["official_episodes"] == 379
    assert len(bank["candidates"]) == 1000 and all(len(row) == 4 for row in bank["candidates"])
    assert {r["task_id"] for r in bank["records"].values()} == set(range(10))
    expected = torch.from_numpy(data.normalized_expert(raw, bank["statistics"])).unsqueeze(0)
    for record in bank["records"].values():
        assert record["step_index"] == 0 and record["source_split"] == "train"
        assert "clean_action" not in record and "native_action" not in record
        torch.testing.assert_close(record["expert_actions"], expected)
    before = (tmp_path / "data/CANDIDATE_BANK.pt").stat().st_mtime_ns
    data.prepare()
    assert (tmp_path / "data/CANDIDATE_BANK.pt").stat().st_mtime_ns == before


@pytest.mark.parametrize("scores,expected", [([4, 3, 0, 1], [2, 2, 1, 1]),
                                            ([0, 0, 0, 0], [1, 1, 1, 1]),
                                            ([0, 0, 3, 0], [1, 1, 2, 1])])
def test_dino_sampling_weights(scores, expected):
    index, weights = obj.sampled_index(scores, 123)
    assert weights == expected
    assert (index, weights) == obj.sampled_index(scores, 123)


def test_full_rank_frequency_and_clipped_step_match():
    settings = cfg()
    before = torch.rand(3, 50, 50, generator=torch.Generator().manual_seed(8)) * .98 + .01
    proposal = before + torch.randn(3, 50, 50, generator=torch.Generator().manual_seed(9)) * .03
    value, scale = obj.precondition_step(before, proposal, None, settings)
    torch.testing.assert_close((value - before).norm(), (proposal.clamp(0, 1) - before).norm(), rtol=1e-5, atol=1e-6)
    assert scale > 0 and torch.isfinite(value).all()
    assert bool(((value >= 0) & (value <= 1)).all())
    checker = ((torch.arange(50)[:, None] + torch.arange(50)[None, :]) % 2).float() * 2 - 1
    filtered = obj.frequency_filter(checker[None])
    assert 0 < filtered.norm() < checker.norm()
    _, next_scale = obj.precondition_step(value, value + .01, scale, settings)
    assert next_scale == scale


def test_zero_frequency_step_and_dense_support():
    zeros = torch.zeros(3, 50, 50)
    value, scale = obj.precondition_step(zeros, zeros, None, cfg())
    assert torch.equal(value, zeros) and scale is None
    mask = torch.zeros(224, 224, dtype=torch.bool)
    mask[137:187, 26:76] = True
    footprint = obj.token_footprint(mask)
    assert footprint.shape == (256,) and 0 < (footprint > 0).sum() < 256
    assert footprint.max() <= 1 and not footprint.requires_grad


def test_geometry_selection_sign_and_projection_gradient():
    clean = torch.zeros(1, 256, 8)
    clean[..., 0] = 1
    references = []
    for shift in [-.1, -.2, .1, .2]:
        value = clean.clone()
        value[..., 1] += shift
        references.append(value)
    gradient = torch.zeros_like(clean)
    gradient[..., 1] = 1
    direction, scores, chosen = obj.geometry_reference(clean, references, gradient)
    assert direction is not None and scores[chosen] > 0 and chosen in (0, 1)
    weights = obj.action_token_weights(gradient)
    footprint = torch.zeros(256)
    current = references[chosen].clone().requires_grad_()
    loss, metrics = obj.geometry_loss(current, clean, direction, weights, footprint)
    assert loss < 0 and metrics["geometry_projection"] > 0
    assert torch.isfinite(torch.autograd.grad(loss, current)[0]).all()
    inactive, _ = obj.geometry_loss(current, clean, direction, weights, torch.ones(256))
    assert inactive is None
    none, _, _ = obj.geometry_reference(clean, [clean] * 4, gradient)
    assert none is None


def attention_module():
    return SimpleNamespace(num_heads=2, q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(), scale=.5)


def test_attention_uses_dense_native_qk_excludes_prefix_and_has_gradient():
    values = torch.randn(1, 261, 24, generator=torch.Generator().manual_seed(3), requires_grad=True)
    attn = attention_module()
    footprint = torch.zeros(256)
    footprint[4:9] = 1
    weights = torch.arange(1, 257).float()
    weights /= weights.sum()
    loss, _ = obj.attention_loss({8: values}, {8: attn}, weights, footprint, {8: 2.})
    q, k = obj.split_qkv(values, attn)
    w = weights * (footprint == 0)
    w /= w.sum()
    expected = ((q[:, :, 5:] * attn.scale) @ k[:, :, 5:].transpose(-2, -1)
                * w[None, None, :, None] * (footprint / footprint.sum())[None, None, None, :]).sum((-1, -2)).mean() / 2
    torch.testing.assert_close(loss, -expected)
    grad = torch.autograd.grad(loss, values)[0]
    assert torch.isfinite(grad).all() and grad[:, 5:].norm() > 0
    assert not grad[:, :5].any()
    changed = values.detach().clone()
    changed[:, :5] *= 10000
    again, _ = obj.attention_loss({8: changed}, {8: attn}, weights, footprint, {8: 2.})
    torch.testing.assert_close(loss.detach(), again)


@pytest.mark.parametrize("ratio", [.1, .3])
def test_auxiliary_norm_ratio_without_discarding_main(ratio):
    main = torch.tensor([2., 1.])
    auxiliary = torch.tensor([-100., 20.])
    mixed, metrics = obj.combine_gradients(main, auxiliary, ratio)
    torch.testing.assert_close((mixed - main).norm() / main.norm(), torch.tensor(ratio))
    assert metrics["effective_aux_ratio"] == pytest.approx(ratio)
    untouched, metrics = obj.combine_gradients(main, torch.zeros_like(main))
    assert torch.equal(untouched, main) and metrics["effective_aux_ratio"] == 0


def test_deployment_only_wrist_and_never_mutates_input():
    obs = {"full_image": np.ones((224, 224, 3), np.uint8), "wrist_image": np.zeros((224, 224, 3), np.uint8),
           "state": np.arange(8)}
    result = deployment.paste_png(obs, np.full((50, 50, 3), 123, np.uint8))
    assert result["full_image"] is obs["full_image"] and result["state"] is obs["state"]
    assert obs["wrist_image"].sum() == 0
    assert np.all(result["wrist_image"][137:187, 26:76] == 123)
    assert np.count_nonzero(result["wrist_image"]) == 50 * 50 * 3


def test_submission_commands_are_serial_and_resource_caps():
    assert pipeline.existing_gpu_jobs("123_[0-7%8]|PENDING|gres:gpu:4090:1|Resources\n124|RUNNING|N/A|None") == ["123"]
    with pytest.raises(RuntimeError, match="invalid dependency"):
        pipeline.existing_gpu_jobs("123|PENDING|gres:gpu:4090:1|DependencyNeverSatisfied")
    first = pipeline.command_for("01_prepare_cpu.sbatch", external=["123"])
    following = pipeline.command_for("02_train_4090.sbatch", dependency="456", external=["123"])
    assert "--dependency=afterany:123" in first and "--dependency=afterok:456" in following
    assert "--kill-on-invalid-dep=yes" in following
    for filename, array in [("02_train_4090.sbatch", "0-7%8"), ("04_eval_4090.sbatch", "0-9%8")]:
        script = (EXPERIMENT / "slurm" / filename).read_text()
        assert "--gres=gpu:4090:1" in script and f"--array={array}" in script
    assert not any("smoke" in filename for _, filename in pipeline.STAGES)


def test_native_episode_ast_and_renderer_safety_are_inherited():
    old = EXPERIMENT.parent / "wrist_dino_long_final8_v2/evaluation_4090_v1/code"
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert functions(EXPERIMENT / "code/evaluate.py")["episode"] == functions(old / "evaluate.py")["episode"]
    current = functions(EXPERIMENT / "code/model_runtime.py")
    previous = functions(old / "model_runtime_4090.py")
    assert {k: v for k, v in current.items() if k != "load_wrapper"} == {
        k: v for k, v in previous.items() if k != "load_wrapper"}


def test_per_suite_paired_statistics_and_no_overall_row():
    rows = []
    for suite in common.SUITES:
        for task in range(10):
            for state in range(40, 50):
                for i, condition in enumerate(common.deployment_conditions()):
                    failure = 0 if condition == "clean" else int(state - 40 < i)
                    rows.append({"suite": suite, "task_id": task, "state_index": state, "condition": condition,
                                 "rollout_seed": task * 100 + state, "failure": failure, "success": 1 - failure})
    raw = pd.DataFrame(rows)
    summary, contrasts, tasks = analyze.summarize(raw, draws=20)
    assert (len(summary), len(contrasts), len(tasks)) == (36, 32, 320)
    assert set(summary.suite) == set(common.SUITES)
    assert np.allclose(summary.loc[summary.method.eq("uada_dino"), "task_equal_casr"], .2)
    assert np.allclose(contrasts.loc[contrasts.left.eq("uada_dino"), "casr_difference"], .1)
    raw.loc[0, "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Unpaired"):
        analyze.summarize(raw, draws=10)


def test_conditioned_casr_excludes_clean_failure():
    frame = pd.DataFrame([
        {"task_id": 0, "state_index": s, "condition": c, "success": success, "failure": 1-success}
        for s, clean, attack in [(40, 0, 0), (41, 1, 0), (42, 1, 1)]
        for c, success in [("clean", clean), ("m", attack)]])
    point, pairs = analyze.task_casr(frame, "m")
    assert point == .5 and len(pairs) == 3
    assert analyze.paired_bootstrap(pairs.assign(value=pairs.failure), draws=20) == analyze.paired_bootstrap(pairs.assign(value=pairs.failure), draws=20)


class MockAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.scale = 2, .5
        self.q_norm, self.k_norm = torch.nn.Identity(), torch.nn.Identity()
        self.qkv = torch.nn.Linear(8, 24)


class MockEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 5
        self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(17)])
        for block in self.blocks:
            block.attn = MockAttention()
    def forward(self, rgb):
        tokens = rgb.mean() + torch.linspace(.1, 1., 8)[None, None].expand(1, 261, 8)
        for layer in (8, 16):
            self.blocks[layer].attn.qkv(tokens)
        return tokens[:, 5:]


class MockNativeVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = torch.nn.Module()
        self.vision_backbone.featurizer = MockEncoder()
        self.vision_backbone.fused_featurizer = MockEncoder()
    def predict_action(self, pixel_values, **kwargs):
        views = []
        for start in (0, 6):
            dino = self.vision_backbone.featurizer(pixel_values[:, start:start + 3])
            siglip = self.vision_backbone.fused_featurizer(pixel_values[:, start + 3:start + 6])
            views.append(dino.mean() + siglip.mean())
        return None, views[0] * .2 + views[1] * .8


class MockHead(torch.nn.Module):
    def predict_action(self, hidden):
        return hidden * torch.ones(1, 8, 7)


def native_wrapper():
    wrapper = SimpleNamespace(vla=MockNativeVLA(), action_head=MockHead(), proprio_projector=None,
                              noisy_action_projector=None, device=torch.device("cpu"),
                              gen_cfg=SimpleNamespace(unnorm_key="mock"))
    def freeze():
        for module in (wrapper.vla, wrapper.action_head):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    wrapper.freeze_model = freeze
    return wrapper


def test_actual_engine_hooks_wrist_restore_leaf_qkv_and_rgb_gradients():
    wrapper = native_wrapper()
    engine = OFTEngine(wrapper)
    clean_pixels = torch.zeros(1, 12, 2, 2)
    clean_action, clean, _ = engine.forward({"pixel_values": clean_pixels}, None)
    pixels = clean_pixels.clone()
    pixels[:, 6:] = .25
    pixels.requires_grad_()
    action, caps, qkv = engine.forward({"pixel_values": clean_pixels}, None, pixels, attention=True)
    gradient = torch.autograd.grad(action.square().mean(), pixels)[0]
    assert gradient[:, 6:9].norm() > 0 and torch.isfinite(gradient).all()
    assert set(qkv) == {8, 16} and qkv[8].shape == (1, 261, 24)
    _, restored, _ = engine.forward({"pixel_values": clean_pixels}, None, pixels, replacement=clean["dino"])
    assert torch.equal(restored["dino"], clean["dino"]) and torch.equal(restored["siglip"], caps["siglip"])
    restored_action, _, _ = engine.forward({"pixel_values": clean_pixels}, None, replacement=clean["dino"])
    torch.testing.assert_close(restored_action, clean_action)
    leaf_action, leaf, _ = engine.forward({"pixel_values": clean_pixels}, None, leaf=True)
    leaf_gradient = torch.autograd.grad(leaf_action.square().mean(), leaf["dino"])[0]
    assert leaf["dino"].is_leaf and leaf_gradient.norm() > 0
    assert not engine.dino._forward_hooks and not engine.siglip._forward_hooks
    assert not any(parameter.requires_grad or parameter.grad is not None for parameter in wrapper.vla.parameters())


def test_engine_hooks_cleanup_on_failure():
    engine = OFTEngine(native_wrapper())
    with pytest.raises(ValueError, match="synthetic"):
        with engine.hooks(), engine.qkv_hooks(True):
            raise ValueError("synthetic")
    assert not engine.dino._forward_hooks and not engine.siglip._forward_hooks
    assert all(not module.qkv._forward_hooks for module in engine.attentions.values())


def test_readonly_native_loader_checks_without_writing_checkpoint(tmp_path):
    local, native = tmp_path / "checkpoint", tmp_path / "oft/prismatic/extern/hf"
    local.mkdir()
    native.mkdir(parents=True)
    (local / "config.json").write_text(json.dumps({"auto_map": AUTO_MAP}))
    for name in ("modeling_prismatic.py", "configuration_prismatic.py"):
        (local / name).write_text("# mock identity\n")
        (native / name).write_text("# mock identity\n")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in local.iterdir()}
    utils = SimpleNamespace(update_auto_map=lambda _: pytest.fail("native sync must not run"),
                            check_model_logic_mismatch=lambda _: pytest.fail("native sync must not run"))
    wrapper = SimpleNamespace(_try_import_official=lambda: None, _official_openvla_utils=utils)
    install_readonly_loader(wrapper, tmp_path / "oft")
    utils.update_auto_map(local)
    utils.check_model_logic_mismatch(local)
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in local.iterdir()}
    assert before == after
    (local / "modeling_prismatic.py").write_text("# wrong\n")
    with pytest.raises(RuntimeError, match="logic mismatch"):
        verify_checkpoint(local, tmp_path / "oft")


class MockEngine:
    """Small CPU computation graph; NOT an OFT or scientific result."""
    fail_after = None
    def __init__(self, wrapper):
        self.calls = 0
        self.attentions = {8: attention_module(), 16: attention_module()}
    def validate(self, record, patch):
        return {"status": "CPU_MOCK_ONLY", "scientific_result": "NOT_TESTED"}
    def base(self, record):
        return {"offset": record["offset"]}, None
    def pixels(self, record, patch, seed, diverse=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic interruption")
        footprint = torch.zeros(256)
        footprint[0:16] = 1
        return patch, footprint
    def forward(self, inputs, proprio, pixels=None, replacement=None, **kwargs):
        offset = inputs["offset"]
        value = torch.tensor(0.) if pixels is None or replacement is not None else pixels.mean()
        action = torch.ones(1, 8, 7) * (.8 + offset - .3 * value)
        clean = torch.linspace(.1, 1, 8)[None, None].expand(1, 256, 8)
        pattern = torch.linspace(-.5, .5, 8)[None, None]
        dino = clean + value * pattern
        qkv = {layer: torch.linspace(.1, 1., 24)[None, None].expand(1, 261, 24) + value
               for layer in self.attentions} if kwargs.get("attention") else {}
        return action, {"dino": dino, "siglip": torch.ones_like(clean)}, qkv
    def mechanism_reference(self, record, inputs, proprio, target, cfg, **kwargs):
        _, captures, _ = self.forward(inputs, proprio)
        clean = captures["dino"]
        direction = F.normalize(clean + torch.linspace(-.5, .5, 8), dim=-1) - F.normalize(clean, dim=-1)
        return {"clean": clean, "weights": torch.ones(256) / 256, "direction": direction,
                "geometry_selected": 0, "geometry_scores": [1., 0., 0., 0.], "attention_scales": {8: 1., 16: 1.}}


@pytest.fixture
def mock_training(monkeypatch, tmp_path):
    settings = cfg()
    settings.update(outer_iterations=3, inner_loop=2, warmup_outer_iterations=1, resume_every_outer=1,
                    checkpoint_outer_iterations=[0, 1, 2, 3])
    settings["eot"]["fixed_inner_count"] = 1
    records = {f"e:{i}": {"uid": f"e:{i}", "episode_id": "e", "step_index": i, "task_id": 0,
                           "expert_actions": torch.ones(1, 8, 7) * .2, "offset": i * .01} for i in range(4)}
    bank = {"candidates": [[f"e:{i}" for i in range(4)]] * 3, "records": records}
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "config", lambda: settings)
    monkeypatch.setattr(training, "prepared", lambda: None)
    monkeypatch.setattr(training, "require_gpu", lambda _: "CPU_MOCK_ONLY")
    monkeypatch.setattr(training, "patch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training, "load_bank", lambda: bank)
    monkeypatch.setattr(training, "run_signature", lambda: "mock_not_scientific")
    monkeypatch.setattr(training, "runtime_config", lambda: {})
    monkeypatch.setattr(training, "load_wrapper", lambda *args: object())
    monkeypatch.setattr(training, "OFTEngine", MockEngine)
    monkeypatch.setattr(MockEngine, "fail_after", None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.ones(16, dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    return settings, bank, tmp_path


@pytest.mark.parametrize("method", common.METHODS)
def test_all_eight_cpu_mock_update_paths_and_complete_state(mock_training, method):
    settings, bank, root = mock_training
    training.train(method)
    folder = root / "training" / method
    saved = torch.load(folder / "resume.pt", weights_only=False)
    training.validate_resume(saved, method, settings, "mock_not_scientific")
    assert saved["outer"] == 3 and len(saved["rows"]) == 3
    assert saved["rows"][-1]["optimizer_updates"] == 6
    assert not torch.equal(saved["patch"], saved["initial"])
    assert json.loads((folder / "status.json").read_text())["status"] == "COMPLETE"
    assert (folder / "outer_0003.png").exists()
    if common.flags(method)["frequency"]:
        assert saved["frequency_scale"] > 0
    if common.flags(method)["sampling"]:
        assert sum(saved["rows"][-1]["candidate_weights"]) == 6
    if common.flags(method)["geometry"] or common.flags(method)["attention"]:
        assert saved["rows"][-1]["effective_aux_ratio"] == pytest.approx(.1, abs=1e-6)
    training.train(method)  # Completed methods skip safely; no new optimizer run.


def test_mock_interruption_restores_adam_scheduler_rng_and_rgb(mock_training, monkeypatch):
    settings, bank, root = mock_training
    reference = root / "reference"
    monkeypatch.setattr(training, "ROOT", reference)
    training.train("uada_original")
    expected = torch.load(reference / "training/uada_original/resume.pt", weights_only=False)
    interrupted = root / "interrupted"
    monkeypatch.setattr(training, "ROOT", interrupted)
    monkeypatch.setattr(MockEngine, "fail_after", 2)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        training.train("uada_original")
    checkpoint = torch.load(interrupted / "training/uada_original/resume.pt", weights_only=False)
    assert checkpoint["outer"] == 1
    monkeypatch.setattr(MockEngine, "fail_after", None)
    training.train("uada_original")
    resumed = torch.load(interrupted / "training/uada_original/resume.pt", weights_only=False)
    assert torch.equal(expected["patch"], resumed["patch"])
    assert expected["scheduler"] == resumed["scheduler"]
    assert torch.equal(expected["torch_rng"], resumed["torch_rng"])
    for a, b in zip(expected["optimizer"]["state"].values(), resumed["optimizer"]["state"].values()):
        assert a["step"] == b["step"] == 6
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(a["exp_avg_sq"], b["exp_avg_sq"], rtol=0, atol=0)
    drift = copy.deepcopy(resumed)
    drift["scheduler"]["last_epoch"] -= 1
    with pytest.raises(RuntimeError, match="scheduler clock"):
        training.validate_resume(drift, "uada_original", settings, "mock_not_scientific")
    drift = copy.deepcopy(resumed)
    next(iter(drift["optimizer"]["state"].values())).pop("exp_avg")
    with pytest.raises(RuntimeError, match="moments missing"):
        training.validate_resume(drift, "uada_original", settings, "mock_not_scientific")
    drift = copy.deepcopy(resumed)
    drift["cuda_rng"] = []
    with pytest.raises(RuntimeError, match="RNG state"):
        training.validate_resume(drift, "uada_original", settings, "mock_not_scientific")
