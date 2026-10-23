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

import dino_register_delayed_common as common
import run_exploring_oft_dino_register_delayed_cross_suite as evaluation
import analyze_exploring_oft_dino_register_delayed_cross_suite as statistics
from white_patch.oft_dino_register_delayed_objectives import auxiliary_active, combine_gradients
from white_patch.UADA_OFT_DINO_REGISTER_DELAYED import ExploringOFTDINORegisterDelayedAttacker, validate_state


def test_new_python_sources_parse():
    root = common.ROOT
    for name in (
        "scripts/dino_register_delayed_common.py",
        "scripts/run_exploring_oft_dino_register_delayed_cross_suite.py",
        "scripts/analyze_exploring_oft_dino_register_delayed_cross_suite.py",
        "VLAAttacker/UADA_OFT_DINO_REGISTER_DELAYED_wrapper.py",
        "VLAAttacker/white_patch/UADA_OFT_DINO_REGISTER_DELAYED.py",
        "VLAAttacker/white_patch/oft_dino_register_delayed_objectives.py",
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
    endpoints = [*common.DENOMINATORS, "object_spatial_goal_macro"]
    summary = pd.DataFrame([
        {"checkpoint_outer": outer, "condition": method, "suite_or_endpoint": endpoint,
         "task_equal_casr": .5}
        for outer in common.CHECKPOINTS for method in common.CONDITIONS for endpoint in endpoints
    ])
    contrasts = pd.DataFrame([{
        "checkpoint_outer": 2000, "suite_or_endpoint": "object_spatial_goal_macro",
        "contrast": "synthetic", "difference_left_minus_right": .1,
        "ci95_low": -.1, "ci95_high": .2,
        "positive_tasks": 15, "negative_tasks": 10, "tied_tasks": 5}])
    metrics = pd.DataFrame([{
        "checkpoint_outer": 2000, "condition": "register_late_g10",
        "mean_raw_gradient_cosine": -.1, "conflict_fraction": .3,
        "mean_effective_aux_ratio": .1, "rgb_mae_from_outer500": .02}])
    path = statistics.write_closeout(tmp_path, cfg, summary, contrasts, metrics, [])
    assert "1000" in path.read_text() and "2000" in path.read_text()
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
    assert "--array=0,1,2%3" in commands[2]
    assert "--array=0-9%8" in commands[4]
    for prior, current in zip(receipt["jobs"], receipt["jobs"][1:]):
        assert f"--dependency=afterok:{prior['job_id']}" in current["command"]
        assert "--kill-on-invalid-dep=yes" in current["command"]
    monkeypatch.setattr(common.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="a mock active job", stderr=""))
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
        failed = condition == "register_late_g10" or (
            outer == 2000 and condition == "register_late_projected_g10")
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


def test_two_budget_evaluation_resume_and_bootstrap(synthetic_evaluation):
    cfg, hashes, calls, loader = synthetic_evaluation
    with pytest.raises(RuntimeError, match="intentional interruption"):
        evaluation.run_shard(0, cfg, patch_loader=loader)
    saved = torch.load(Path(cfg["output_root"]) / "libero_object/task_00_resume.pt",
                       weights_only=False)
    assert len(saved["rows"]) == 5
    assert {row["checkpoint_outer"] for row in saved["rows"]} == {1000}
    frames = []
    for index in range(40):
        evaluation.run_shard(index, cfg, patch_loader=loader)
        suite, task = cfg["suites"][index // 10], index % 10
        rows = pd.read_parquet(Path(cfg["output_root"]) / suite / f"task_{task:02d}_rollouts.parquet")
        traces = pd.read_parquet(Path(cfg["output_root"]) / suite / f"task_{task:02d}_traces.parquet")
        evaluation.validate_new_shard(rows, traces, suite, task, cfg, hashes)
        evaluation.validate_runtime(rows, traces, cfg, suite, task)
        frames.append(rows)
    assert len(calls) == len(set(calls)) == 4000
    raw = pd.concat(frames, ignore_index=True)
    eligible = statistics.validate_raw(raw, cfg)
    assert len(eligible) == 3820
    for outer in common.CHECKPOINTS:
        current = eligible.loc[eligible.checkpoint_outer.eq(outer)]
        summary, contrasts = statistics.paired_statistics(current, draws=300, checkpoint_outer=outer)
        assert summary.checkpoint_outer.eq(outer).all()
        assert set(summary.suite_or_endpoint) == {
            *common.DENOMINATORS, "object_spatial_goal_macro", "spatial_goal_macro", "four_suite_macro"}
        target = contrasts.loc[contrasts.contrast.eq("register_late_g10_minus_uniform_axis0")]
        assert target.difference_left_minus_right.eq(1.).all()
        assert np.allclose(target.ci95_low, 1.) and np.allclose(target.ci95_high, 1.)
    with pytest.raises(RuntimeError, match="one matched budget"):
        statistics.paired_statistics(eligible, draws=10)
    corrupted = raw.copy()
    corrupted.loc[0, "checkpoint_outer"] = 500
    with pytest.raises(RuntimeError):
        statistics.validate_raw(corrupted, cfg)


class SyntheticTrainer(ExploringOFTDINORegisterDelayedAttacker):
    """Exercises the production loop; replaces only real model/vision computations."""
    def __init__(self, *, interrupt_after_warmup=False):
        self.wrapper = SimpleNamespace(device=torch.device("cpu"), vla=SimpleNamespace(zero_grad=lambda: None))
        self.interrupt_after_warmup = interrupt_after_warmup
        self.auxiliary_calls = 0
        self.runtime_alignment = {"status": "PASS"}

    def _base_inputs(self, record):
        return {}, None

    def _forward(self, inputs, proprio, observation, patch, geometry):
        if patch is None:
            return torch.zeros(1, 8, 7), None, None
        if self.interrupt_after_warmup and self.auxiliary_calls == 1:
            self.interrupt_after_warmup = False
            raise RuntimeError("intentional training interruption at auxiliary onset")
        value = (patch - .5).square().mean()
        return value.expand(1, 8, 7), None, None

    def _timed(self, costs, name, fn):
        return fn()

    def _validate_runtime(self, record, patch, cfg):
        return {"status": "PASS", "synthetic": True}

    @contextmanager
    def _capture_dino_qkv(self, cfg):
        self.auxiliary_calls += 1
        yield {}

    def _attention_masses(self, captures, cfg, target):
        # Inspect the active patch through the real autograd graph.
        patch = self.current_patch
        return (patch - .3).square().mean(), {
            "register_attention_mass": .1, "spatial_attention_mass": .1,
            "patch_token_count": 16, "target_query_count": 4 if target == "register" else 240}

@pytest.mark.parametrize("method", common.METHODS)
def test_training_loop_onset_optimizer_resume(tmp_path, monkeypatch, method):
    cfg = copy.deepcopy(common.config())
    cfg.update(outer_iterations=252, inner_loop=2, checkpoint_outer_iterations=[0, 250, 252])
    clean = [{"task_id": i % 10, "state_index": i // 10, "task_name": "synthetic",
              "observation": {}} for i in range(300)]
    replay = [{"task_id": i % 10, "state_index": i // 10, "task_name": "synthetic",
               "phase_proxy": ("early", "middle", "late")[(i // 10) % 3],
               "observations": [{}, {}, {}]} for i in range(258)]
    # Production forward delegates to the mock and captures its active RGB leaf.
    base_forward = SyntheticTrainer._forward
    def captured_forward(self, inputs, proprio, observation, patch, geometry):
        if patch is not None:
            self.current_patch = patch
        return base_forward(self, inputs, proprio, observation, patch, geometry)
    monkeypatch.setattr(SyntheticTrainer, "_forward", captured_forward)
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
    assert rows.iloc[250:].auxiliary_enabled.all()
    assert rows.iloc[:250].aux_grad_norm.eq(0).all()
    assert rows.iloc[250:].aux_grad_norm.gt(0).all()
    assert rows.iloc[250:].effective_aux_gradient_ratio.le(.10001).all()
    assert rows.iloc[250:].rgb_outer_max_abs.gt(0).all()
    assert json.loads((folder / "status.json").read_text())["status"] == "COMPLETE"
