"""CPU-only regression checks. Synthetic results are NOT experiment results."""
from __future__ import annotations

import ast
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import dino_register_content_common as common
import run_exploring_oft_dino_register_content_cross_suite as evaluation
import analyze_exploring_oft_dino_register_content_cross_suite as statistics
from oft_dino_register_content_objectives import auxiliary_active, combine_gradients
from UADA_OFT_DINO_REGISTER_CONTENT import ExploringOFTDINORegisterContentAttacker, validate_state


def test_new_python_sources_parse():
    root = common.ROOT
    for name in (
        "scripts/dino_register_content_common.py",
        "scripts/run_exploring_oft_dino_register_content_cross_suite.py",
        "scripts/analyze_exploring_oft_dino_register_content_cross_suite.py",
        "research_runs/wrist_dino_register_content_v1/code/UADA_OFT_DINO_REGISTER_CONTENT_wrapper.py",
        "research_runs/wrist_dino_register_content_v1/code/UADA_OFT_DINO_REGISTER_CONTENT.py",
        "research_runs/wrist_dino_register_content_v1/code/oft_dino_register_content_objectives.py",
    ):
        ast.parse((root / name).read_text(), filename=name)


@pytest.mark.parametrize("auxiliary", [[1., 1.], [-1., 1.], [-1., 0.], [0., 0.]])
@pytest.mark.parametrize("project", [False, True])
def test_auxiliary_cap_and_main_preservation(auxiliary, project):
    main = torch.tensor([1., 0.])
    value, metrics = combine_gradients(main, torch.tensor(auxiliary), .1, project)
    assert value.isfinite().all()
    assert metrics["effective_aux_gradient_ratio"] <= .100001
    if project:
        assert (main * (value - main)).sum() >= -1e-7


def test_delayed_onset_and_no_projection_renormalization():
    cfg = {"auxiliary_start_outer": 251}
    assert not auxiliary_active(250, cfg) and auxiliary_active(251, cfg)
    value, metrics = combine_gradients(torch.tensor([1., 0.]), torch.tensor([-1., 1.]), .1, True)
    assert torch.allclose(value, torch.tensor([1., .1 / 2 ** .5]))
    assert metrics["effective_aux_gradient_ratio"] == pytest.approx(.1 / 2 ** .5)
    for bad in (torch.tensor([float("nan"), 0.]), torch.tensor([1., 2., 3.])):
        with pytest.raises(ValueError):
            combine_gradients(torch.tensor([1., 0.]), bad, .1, True)


def test_closeout_report_and_mock_submission_dependencies(tmp_path, monkeypatch):
    cfg = common.config()
    endpoints = [*common.DENOMINATORS, "spatial_goal_macro"]
    summary = pd.DataFrame([
        {"checkpoint_outer": outer, "condition": method, "suite_or_endpoint": endpoint,
         "task_equal_casr": .5}
        for outer in common.CHECKPOINTS for method in common.CONDITIONS for endpoint in endpoints
    ])
    contrasts = pd.DataFrame([{
        "checkpoint_outer": 500, "suite_or_endpoint": "spatial_goal_macro",
        "contrast": "content_minus_action", "difference_left_minus_right": .1,
        "ci95_low": -.1, "ci95_high": .2,
        "positive_tasks": 15, "negative_tasks": 10, "tied_tasks": 5}])
    metrics = pd.DataFrame([{
        "checkpoint_outer": 500, "condition": "register_content_g10",
        "mean_raw_gradient_cosine": -.1, "conflict_fraction": .3,
        "mean_effective_aux_ratio": .1, "rgb_mae_from_outer250": .02,
        "mean_cf_clean_action_mae": .01, "mean_donor_kv_change": .02}])
    path = statistics.write_closeout(tmp_path, cfg, summary, contrasts, metrics, [])
    assert "500轮" in path.read_text() and "Spatial/Goal" in path.read_text()
    assert "Long（辅助）" in path.read_text() and "+10.00" in path.read_text()
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="" if command[0] == "squeue" else str(99000 + len(commands)), stderr="")
    monkeypatch.setattr(common.subprocess, "run", fake_run)
    cfg["output_root"] = str(tmp_path / "mock_submission")
    receipt = common.submit_chain(cfg)
    assert receipt["status"] == "SUBMITTED"
    assert [job["stage"] for job in receipt["jobs"]] == ["prepare", "train", "validate", "eval", "stats"]
    assert "--array=0,1,2,3%4" in commands[2]
    assert "--array=0-9%8" in commands[4]
    for prior, current in zip(receipt["jobs"], receipt["jobs"][1:]):
        assert f"--dependency=afterok:{prior['job_id']}" in current["command"]
        assert "--kill-on-invalid-dep=yes" in current["command"]
    monkeypatch.setattr(common.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="98001|oft_regcontent_train|gres:gpu:4090:1|node1", stderr=""))
    with pytest.raises(RuntimeError, match="remains queued"):
        common.submit_chain(cfg)


@pytest.fixture
def synthetic_evaluation(tmp_path, monkeypatch):
    train = common.config()
    cfg = evaluation.load_config(train)
    cfg["output_root"] = str(tmp_path / "evaluation")
    calls = []
    interrupt = {"enabled": True}
    hashes = {str(outer): {name: {"pt": f"{outer}-{name}", "png": f"png-{outer}-{name}"}
                          for name in common.CONDITIONS} for outer in common.CHECKPOINTS}

    def loader(_cfg, device):
        return {outer: {name: (torch.tensor([outer], device=device), hashes[str(outer)][name]["pt"])
                        for name in common.CONDITIONS} for outer in common.CHECKPOINTS}, hashes

    def clean(_cfg, suite):
        return pd.DataFrame([
            {"task_id": task, "state_index": state, "rollout_seed": 100 * task + state,
             "success": int((state - 40) * 10 + task >= 100 - common.DENOMINATORS[suite])}
            for task in range(10) for state in range(40, 50)
        ])

    def episode(env, wrapper, task, state, seed, condition, patch, digest, runtime, renderer, vendor, order):
        if interrupt["enabled"] and len(calls) == 5:
            interrupt["enabled"] = False
            raise RuntimeError("intentional interruption after one committed paired block")
        outer = int(patch.item())
        calls.append((runtime["suite"], task.task_id, state, condition, outer))
        failed = condition == "register_content_g10"
        row = {"success": int(not failed), "failure": int(failed), "rollout_seed": seed,
               "patch_sha256": digest, "condition": condition, "condition_order": order,
               "renderer_vendor": vendor, "decision_blocks": 1, "elapsed_s": 1.}
        trace = {"condition": condition, "decision_block": 0, "value": 1.}
        return row, [trace]

    monkeypatch.setattr(evaluation, "frozen_clean", clean)
    monkeypatch.setattr(evaluation, "runtime_config", lambda _cfg, suite: {"suite": suite})
    monkeypatch.setattr(evaluation, "load_wrapper", lambda runtime: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(evaluation, "make_env", lambda runtime, task: (
        SimpleNamespace(close=lambda: None), SimpleNamespace(task_id=task, language="synthetic"),
        list(range(50))))
    monkeypatch.setattr(evaluation, "renderer_metadata", lambda runtime: ("mock-egl", "NVIDIA synthetic"))
    monkeypatch.setattr(evaluation, "run_episode", episode)
    monkeypatch.setattr(common, "source_manifest", lambda *_args, **_kwargs: {})
    return cfg, hashes, calls, loader


def test_one_budget_evaluation_resume_and_bootstrap(synthetic_evaluation):
    cfg, hashes, calls, loader = synthetic_evaluation
    with pytest.raises(RuntimeError, match="intentional interruption"):
        evaluation.run_shard(0, cfg, patch_loader=loader)
    saved = torch.load(Path(cfg["output_root"]) / "libero_object/task_00_resume.pt",
                       weights_only=False)
    assert len(saved["rows"]) == 5
    assert {row["checkpoint_outer"] for row in saved["rows"]} == {500}
    frames = []
    for index in range(40):
        evaluation.run_shard(index, cfg, patch_loader=loader)
        suite, task = cfg["suites"][index // 10], index % 10
        rows = pd.read_parquet(Path(cfg["output_root"]) / suite / f"task_{task:02d}_rollouts.parquet")
        traces = pd.read_parquet(Path(cfg["output_root"]) / suite / f"task_{task:02d}_traces.parquet")
        evaluation.validate_new_shard(rows, traces, suite, task, cfg, hashes)
        evaluation.validate_runtime(rows, traces, cfg, suite, task)
        frames.append(rows)
    assert len(calls) == len(set(calls)) == 2000
    raw = pd.concat(frames, ignore_index=True)
    eligible = statistics.validate_raw(raw, cfg)
    assert len(eligible) == 1910
    for outer in common.CHECKPOINTS:
        current = eligible.loc[eligible.checkpoint_outer.eq(outer)]
        summary, contrasts = statistics.paired_statistics(current, draws=300, checkpoint_outer=outer)
        assert summary.checkpoint_outer.eq(outer).all()
        assert set(summary.suite_or_endpoint) == {
            *common.DENOMINATORS, "object_spatial_goal_macro", "spatial_goal_macro", "four_suite_macro"}
        target = contrasts.loc[contrasts.contrast.eq("content_minus_action")]
        assert target.difference_left_minus_right.eq(1.).all()
        assert np.allclose(target.ci95_low, 1.) and np.allclose(target.ci95_high, 1.)
    with pytest.raises(RuntimeError, match="one matched budget"):
        statistics.paired_statistics(eligible.assign(checkpoint_outer=501), draws=10)
    corrupted = raw.copy()
    corrupted.loc[0, "checkpoint_outer"] = 501
    with pytest.raises(RuntimeError):
        statistics.validate_raw(corrupted, cfg)


class SyntheticTrainer(ExploringOFTDINORegisterContentAttacker):
    """Production outer loop with deterministic small differentiable objectives."""
    def __init__(self, *, interrupt_after_warmup=False):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), vla=SimpleNamespace(zero_grad=lambda **kw: None))
        self.interrupt_after_warmup = interrupt_after_warmup
        self.step_calls = 0
        self.runtime_alignment = {"status": "PASS"}

    def _base_inputs(self, record):
        return {}, None

    def _forward(self, inputs, proprio, observation, patch, geometry):
        return torch.zeros(1, 8, 7), None, None

    def _timed(self, costs, name, fn):
        return fn()

    def _validate_runtime(self, record, patch, cfg):
        return {"status": "PASS", "synthetic": True}

    def _objective_step(self, inputs, proprio, observation, target, clean_action,
                        patch, geometry, axis, cfg, spec, enabled, costs):
        if self.interrupt_after_warmup and self.step_calls == 500:
            self.interrupt_after_warmup = False
            raise RuntimeError("intentional training interruption at auxiliary onset")
        self.step_calls += 1
        main = (patch - .8).square().mean()
        main_grad = torch.autograd.grad(main, patch)[0]
        aux_grad, aux_value = None, 0.
        if enabled:
            aux = (patch - .3).square().mean()
            aux_grad = torch.autograd.grad(aux, patch)[0]
            aux_value = float(aux.detach())
        return main_grad, aux_grad, float(main.detach()), aux_value, {
            "register_attention_mass": .1 if enabled else 0.,
            "spatial_attention_mass": .1 if enabled else 0.,
            "patch_token_count": 16 if enabled else 0.,
            "target_query_count": 4 if enabled else 0.,
            "content_token_count": 4 if enabled and spec["target"].endswith("_content") else 0.,
            "cf_clean_action_mae": .01 if enabled else 0.,
            "cf_patched_action_mae": .01 if enabled else 0.,
            "donor_kv_change": .01 if enabled else 0.}


@pytest.mark.parametrize("method", common.METHODS)
def test_training_loop_onset_optimizer_resume(tmp_path, monkeypatch, method):
    cfg = copy.deepcopy(common.config())
    cfg.update(outer_iterations=252, inner_loop=2, checkpoint_outer_iterations=[0, 250, 252])
    clean = [{"task_id": i % 10, "state_index": i // 10, "task_name": "synthetic",
              "observation": {}} for i in range(300)]
    replay = [{"task_id": i % 10, "state_index": i // 10, "task_name": "synthetic",
               "phase_proxy": ("early", "middle", "late")[(i // 10) % 3],
               "observations": [{}, {}, {}]} for i in range(258)]
    cfg["output_root"] = str(tmp_path / "uninterrupted")
    full = SyntheticTrainer()
    full.train_register(clean, replay, cfg, method, {}, {})
    full_path = Path(cfg["output_root"]) / "training" / method
    full_saved = torch.load(full_path / "resume.pt", weights_only=False)
    cfg["output_root"] = str(tmp_path / "resumed")
    interrupted = SyntheticTrainer(interrupt_after_warmup=True)
    with pytest.raises(RuntimeError, match="intentional training interruption"):
        interrupted.train_register(clean, replay, cfg, method, {}, {})
    folder = Path(cfg["output_root"]) / "training" / method
    commit = torch.load(folder / "resume.pt", weights_only=False)
    assert commit["outer"] == 250
    validate_state(commit, cfg)
    resumed = SyntheticTrainer()
    resumed.train_register(clean, replay, cfg, method, {}, {})
    saved = torch.load(folder / "resume.pt", weights_only=False)
    validate_state(saved, cfg)
    assert saved["outer"] == 252
    assert torch.equal(saved["patch"], full_saved["patch"])
    assert saved["optimizer"]["state"][0]["step"] == 504
    rows = pd.read_parquet(folder / "training.parquet")
    assert not rows.iloc[:250].auxiliary_enabled.any()
    assert rows.iloc[250:].auxiliary_enabled.eq(method != "uniform_action").all()
    assert rows.iloc[:250].aux_grad_norm.eq(0).all()
    assert (rows.iloc[250:].aux_grad_norm.gt(0).all() if method != "uniform_action"
            else rows.aux_grad_norm.eq(0).all())
    assert rows.iloc[250:].effective_aux_gradient_ratio.le(.10001).all()
    assert rows.iloc[250:].rgb_outer_max_abs.gt(0).all()
    assert json.loads((folder / "status.json").read_text())["status"] == "COMPLETE"
    # Exercise the real final-artifact validator on this shorter SYNTHETIC run.
    monkeypatch.setattr(common, "FINAL_OUTER", 252)
    monkeypatch.setattr(common, "CHECKPOINTS", (252,))
    monkeypatch.setattr(common, "require_preparation", lambda *_a, **_k: {})
    monkeypatch.setattr(common, "reference_patches", lambda *_a, **_k: {252: {}})
    monkeypatch.setattr(common, "data_identity", lambda *_a, **_k: {})
    result = common.validate_training(cfg, (method,), write_report=False)
    assert result["status"] == "COMPLETE" and result["evaluation_ready"]

def test_source_contract_and_native_content_preflight():
    from oft_dino_register_content_objectives import cpu_preflight
    cfg = common.config()
    result = cpu_preflight(cfg)
    assert result["status"] == "PASS"
    assert all(row["identity_exact"] and row["front_exact"] and row["chain_rule_close"]
               for row in result["content"])
    assert not result["cuda_initialized"]


def test_kv_exact_isolation_hook_cleanup_and_dense_footprint():
    from oft_dino_register_content_objectives import content_indices, extract_kv, replace_kv, replace_wrist_kv
    cfg = common.config()
    coverage = torch.zeros(256)
    coverage[:60] = 1
    for target in ("register_content", "dense_content"):
        indices = content_indices(target, coverage, cfg)
        if target == "dense_content":
            assert indices.min() >= 5 and coverage[indices - 5].eq(0).all()
        qkv = torch.randn(1, 261, 96)
        donor = extract_kv(qkv, indices) + .2
        mixed = replace_kv(qkv, indices, donor).reshape(1, 261, 3, 32)
        native = qkv.reshape(1, 261, 3, 32)
        assert torch.equal(mixed[:, :, 0], native[:, :, 0])
        unselected = torch.ones(261, dtype=torch.bool)
        unselected[indices] = False
        assert torch.equal(mixed[:, unselected], native[:, unselected])
        assert torch.equal(mixed[:, indices, 1:], donor)
    with pytest.raises(ValueError):
        content_indices("dense_content", torch.ones(256), cfg)
    module = torch.nn.Linear(32, 96)
    x = torch.randn(1, 261, 32)
    donor = extract_kv(module(x), indices).detach().requires_grad_(True)
    with pytest.raises(RuntimeError, match="injected"):
        with replace_wrist_kv(module, indices, donor):
            raise RuntimeError("injected")
    assert not module._forward_hooks
    with pytest.raises(ValueError, match="count"):
        with replace_wrist_kv(module, indices, donor):
            module(x)
    assert not module._forward_hooks


def test_submission_serializes_existing_chains_and_queue_failure(tmp_path, monkeypatch):
    cfg = common.config()
    cfg["output_root"] = str(tmp_path)
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        queue = "137659_0|oft_reglate_train|gres:gpu:4090:1|node1\n137662|oft_reglate_stats|N/A|Dependency\n"
        return SimpleNamespace(returncode=0, stdout=queue if command[0] == "squeue"
                               else str(98000 + len(commands)), stderr="")
    monkeypatch.setattr(common.subprocess, "run", fake_run)
    receipt = common.submit_chain(cfg)
    assert receipt["external_wait_jobs"] == ["137659", "137662"]
    assert "--dependency=afterany:137659:137662" in receipt["jobs"][0]["command"]
    assert receipt["training_gpu_limit"] == 4 and receipt["evaluation_gpu_limit"] == 8
    commands.clear()
    def failed_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="controller unavailable")
    monkeypatch.setattr(common.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="resource occupancy"):
        common.submit_chain(cfg)
    assert [cmd[0] for cmd in commands] == ["squeue"]


class NativeSyntheticTrainer(ExploringOFTDINORegisterContentAttacker):
    """Real production inner objective, native timm Q/K normalization/attention."""
    def __init__(self):
        from timm.models.vision_transformer import Attention
        self.blocks = [SimpleNamespace(attn=Attention(32, num_heads=4, qk_norm=True).eval())
                       for _ in range(17)]
        for block in self.blocks:
            for parameter in block.attn.parameters():
                parameter.requires_grad_(False)
        self.wrapper = SimpleNamespace(device=torch.device("cpu"),
            vla=SimpleNamespace(vision_backbone=SimpleNamespace(
                featurizer=SimpleNamespace(blocks=self.blocks))))
        self.front = torch.randn(1, 261, 32)
        self.wrist = torch.randn(1, 261, 32)
        self.support = torch.zeros(1, 261, 1)
        self.support[:, 85:101] = 1
        self.direction = torch.randn(1, 1, 32)

    def _timed(self, costs, name, fn):
        return fn()

    def _forward(self, inputs, proprio, observation, patch, geometry):
        wrist = self.wrist if patch is None else self.wrist + self.support * self.direction * patch.mean()
        self.patch_tokens = torch.zeros(256)
        if patch is not None:
            self.patch_tokens[80:96] = 1
        results = []
        for value in (self.front, wrist):
            for layer in (8, 16):
                value = value + self.blocks[layer].attn(value)
            results.append(value)
        dense = results[1][:, 5:, :]
        action = dense.mean().expand(1, 8, 7)
        return action, dense, None


@pytest.mark.parametrize("method", common.METHODS)
def test_production_objective_native_forward_and_gradients(method):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(120)
        attacker = NativeSyntheticTrainer()
        cfg = common.config()
        patch = torch.rand(3, 50, 50, requires_grad=True)
        clean, _, _ = attacker._forward({}, None, {}, None, False)
        target = torch.full((1, 8, 7), .75)
        costs = {}
        main_grad, aux_grad, main_loss, aux_loss, stats = attacker._objective_step(
            {}, None, {}, target, clean.detach(), patch, False, 0,
            cfg, cfg["method_specs"][method], method != "uniform_action", costs)
        assert main_grad.shape == patch.shape and main_grad.isfinite().all()
        assert main_grad.norm() > 0 and np.isfinite([main_loss, aux_loss]).all()
        if method != "uniform_action":
            assert aux_grad is not None and aux_grad.isfinite().all() and aux_grad.norm() > 0
        else:
            assert aux_grad is None and aux_loss == 0.
        if method.endswith("content_g10"):
            assert stats["content_token_count"] == 4 and stats["donor_kv_change"] > 0
            assert costs["training_content_forward_calls"] == 1
        assert all(not block.attn.qkv._forward_hooks for block in attacker.blocks)
        native_action, _, _ = attacker._forward({}, None, {}, patch, False)
        expected = torch.autograd.grad(torch.nn.functional.mse_loss(native_action[..., [0]], target[..., [0]]), patch)[0]
        assert torch.allclose(main_grad, expected, atol=1e-8, rtol=1e-5)
