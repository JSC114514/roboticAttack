"""CPU and mock contracts only: no victim loading, rendering or real scheduler."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace, ModuleType
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("OFT_FACTORIAL_ARM", "legacy_freq")
import common
import objectives
import monitor
import train
import analyze
import pipeline
import evaluate
from UADA_SOURCE_COPY import build_uada_adamw

@pytest.mark.parametrize("arm,horizon,sampling,freq", [
    ("legacy_freq", 1000, False, True), ("original_dino", 2000, True, False),
    ("original_freq", 2000, False, True)])
def test_only_approved_factors_change_and_horizon_survives(arm, horizon, sampling, freq):
    cfg = common.config(arm); parent = common.parent_config(common.arm_info(arm)["protocol"])
    allowed = {"experiment_id", "methods", "outer_iterations", "schedule_outer_iterations", "checkpoint_outer_iterations",
               "evaluation_checkpoints", "arm", "protocol", "resources"}
    assert all(cfg[k] == v for k, v in parent.items() if k not in allowed)
    assert cfg["outer_iterations"] * cfg["inner_loop"] == 50000
    assert cfg["schedule_outer_iterations"] == horizon
    assert common.flags(arm) == dict(family="uada", sampling=sampling, frequency=freq, geometry=False, attention=False)
    assert common.scoring_diverse(cfg) == (cfg["protocol"] == "original")
    assert [common.training_diverse(cfg, i) for i in range(50)] == ([False]*25+[True]*25 if horizon == 1000 else [True]*50)

def test_wrong_axis_target_and_descent_have_native_meaning():
    cfg = common.config("original_dino")
    expert = torch.linspace(-1, 1, 56).reshape(1, 8, 7)
    before = expert.clone(); action = expert.clone().requires_grad_()
    target = objectives.target_action(expert, "uada", cfg)
    loss = objectives.action_loss(action, target, "uada", cfg)
    gradient = torch.autograd.grad(loss, action)[0]
    assert torch.equal(expert, before) and gradient[..., 0].abs().sum() > 0
    assert gradient[..., 1:].abs().sum() == 0
    assert objectives.action_loss(action - .01 * gradient, target, "uada", cfg) < loss

def test_scheduler_midpoint_is_not_end():
    learning_rates = []
    for horizon in [1000, 2000]:
        parameter = torch.zeros(1, requires_grad=True)
        optimizer, schedule = build_uada_adamw(parameter, .002, 20, horizon, 0)
        for _ in range(1000):
            parameter.grad = torch.ones_like(parameter); optimizer.step(); schedule.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] == 0 and learning_rates[1] > .0009

def test_rng_guard_preserves_future_random_draws():
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    a = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with monitor.preserve_rng():
        random.random(); np.random.rand(); torch.rand(5)
    assert random.getstate() == a[0]
    np.testing.assert_array_equal(np.random.get_state()[1], a[1][1])
    assert torch.equal(torch.get_rng_state(), a[2])

class MockEngine:
    failure_seed = None
    def __init__(self, wrapper): pass
    def validate(self, record, patch): return {"status": "CPU_MOCK_ONLY"}
    def base(self, record): return {}, None
    def pixels(self, record, patch, seed, diverse=False):
        if seed == self.failure_seed:
            raise RuntimeError("synthetic interruption")
        random.seed(seed); np.random.seed(seed % (2**32-1))
        noise = .001 * (random.random() + np.random.rand() + torch.rand(()))
        return patch + noise, torch.zeros(256)
    def forward(self, inputs, proprio, pixels=None, replacement=None, **kwargs):
        value = torch.tensor(0.) if pixels is None or replacement is not None else pixels.mean()
        action = torch.ones(1, 8, 7) * (.8 - .3 * value)
        dino = torch.ones(1, 256, 4) * (1 + value)
        return action, {"dino": dino, "siglip": torch.ones_like(dino)}, {}

@pytest.fixture
def mock_training(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    monkeypatch.setattr(train, "require_gpu", lambda _: "CPU_MOCK_ONLY")
    monkeypatch.setattr(train, "patch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "prepared", lambda: None)
    monkeypatch.setattr(train, "run_signature", lambda: "CPU_MOCK_NOT_RESEARCH")
    monkeypatch.setattr(train, "load_wrapper", lambda *_: None)
    monkeypatch.setattr(train, "OFTEngine", MockEngine)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.tensor([1], dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_: None)
    record = {"uid": "source", "episode_id": "episode", "step_index": 0, "task_id": 0,
              "expert_actions": torch.ones(1, 8, 7)*.2, "observation": {"state": np.zeros(8)}}
    bank = {"records": {"source": record}, "candidates": [["source"]*4]*3}
    monkeypatch.setattr(train, "load_bank", lambda: copy.deepcopy(bank))
    monkeypatch.setattr(MockEngine, "failure_seed", None)
    def run(arm, folder=None):
        os.environ["OFT_FACTORIAL_ARM"] = arm
        cfg = common.config(arm)
        cfg.update(outer_iterations=3, inner_loop=2, warmup_outer_iterations=1,
                   checkpoint_outer_iterations=[0, 1, 2, 3], resume_every_outer=1)
        monkeypatch.setattr(train, "config", lambda: copy.deepcopy(cfg))
        if folder is not None:
            monkeypatch.setattr(train, "ROOT", folder); monkeypatch.setattr(monitor, "ROOT", folder)
        train.train(arm)
        return cfg, torch.load(train.ROOT / "training" / arm / "resume.pt", weights_only=False)
    return run, tmp_path

@pytest.mark.parametrize("arm", common.METHODS)
def test_actual_three_update_paths_and_probe_outputs(mock_training, arm):
    run, root = mock_training
    cfg, saved = run(arm)
    train.validate_resume(saved, arm, cfg, "CPU_MOCK_NOT_RESEARCH")
    assert saved["outer"] == 3 and not torch.equal(saved["patch"], saved["initial"])
    assert saved["scheduler"]["last_epoch"] == 3
    assert (saved["frequency_scale"] is not None) == common.flags(arm)["frequency"]
    folder = root / "training" / arm
    steps = monitor.collect_steps(folder)
    assert list(zip(steps.outer, steps.inner)) == [(o, i) for o in [1, 2, 3] for i in [0, 1]]
    assert list((folder / "attempts").glob("*/probe_0001_00.pt"))

def test_monitoring_does_not_change_rgb_adam_or_rng(mock_training, monkeypatch):
    run, root = mock_training
    _, measured = run("original_freq", root / "measured")
    class NoMonitor:
        def __init__(self, *_): pass
        def step(self, *_): pass
        def should_probe(self, *_): return False
        def flush(self): pass
        def close(self): pass
    monkeypatch.setattr(train, "Monitor", NoMonitor)
    _, plain = run("original_freq", root / "plain")
    assert torch.equal(measured["patch"], plain["patch"])
    assert torch.equal(measured["torch_rng"], plain["torch_rng"])
    assert measured["scheduler"] == plain["scheduler"]
    for a, b in zip(measured["optimizer"]["state"].values(), plain["optimizer"]["state"].values()):
        assert a["step"] == b["step"]
        for key in ("exp_avg", "exp_avg_sq"): assert torch.equal(a[key], b[key])

def test_interruption_resumes_exact_adam_rng_and_preserves_attempts(mock_training, monkeypatch):
    run, root = mock_training
    _, reference = run("original_freq", root / "reference")
    monkeypatch.setattr(MockEngine, "failure_seed", 20261011 * 1000000 + 2 * 10000 + 1)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run("original_freq", root / "recovered")
    monkeypatch.setattr(MockEngine, "failure_seed", None)
    _, recovered = run("original_freq", root / "recovered")
    assert torch.equal(reference["patch"], recovered["patch"])
    assert torch.equal(reference["torch_rng"], recovered["torch_rng"])
    assert len(list((root / "recovered/training/original_freq/attempts").iterdir())) == 2
    assert len(monitor.collect_steps(root / "recovered/training/original_freq")) == 6
    monkeypatch.setattr(train, "require_gpu", lambda _: pytest.fail("completed training must not allocate a GPU"))
    run("original_freq", root / "recovered")

def synthetic_raw():
    rows = []
    for protocol in ["legacy", "original"]:
        for suite in common.SUITES:
            for task in range(10):
                for state in range(40, 50):
                    for cell in ("clean", *common.CELLS):
                        reused = cell != "uada_freq" and not (protocol == "original" and cell == "uada_dino")
                        n = {"clean": 0, "uada_original": 2, "uada_dino": 3, "uada_freq": 4, "uada_dino_freq": 6}[cell]
                        failure = int(state - 40 < n)
                        rows.append({"protocol": protocol, "suite": suite, "task_id": task, "state_index": state,
                                     "condition": cell, "rollout_seed": task*100 + state, "reused": reused,
                                     "failure": failure, "success": 1-failure})
    return pd.DataFrame(rows)

def test_factorial_main_effects_interaction_pairing_and_recommendation():
    summary, contrasts, tasks = analyze.summarize(synthetic_raw(), draws=30)
    assert len(summary) == 40 and len(contrasts) == 60 and len(tasks) == 320
    assert np.allclose(contrasts.loc[contrasts.contrast.eq("dino_with_frequency"), "difference"], .2)
    assert np.allclose(contrasts.loc[contrasts.contrast.eq("interaction"), "difference"], .1)
    assert analyze.recommendation(summary) == "SECOND_SEED_COMPARISON_WORTHWHILE"
    summary.loc[summary.method.eq("uada_freq"), "casr"] = .8
    assert analyze.recommendation(summary) == "KEEP_FREQUENCY_ONLY_STOP_DINO_ENHANCEMENT"

def test_clean_failure_and_zero_denominator_tasks_are_not_artificial_attack_wins():
    raw = synthetic_raw()
    # Task0 has no qualified states. Task1 loses one state equally in every cell.
    mask = raw.condition.eq("clean") & (raw.task_id.eq(0) | (raw.task_id.eq(1) & raw.state_index.eq(40)))
    raw.loc[mask, ["success", "failure"]] = [0, 1]
    summary, _, tasks = analyze.summarize(raw, draws=20)
    assert summary.loc[summary.suite.isin(common.SUITES), "qualified_tasks"].eq(9).all()
    assert tasks.loc[tasks.task_id.eq(0), "eligible_states"].eq(0).all()
    assert tasks.loc[tasks.task_id.eq(0), "casr"].isna().all()

@pytest.mark.parametrize("kind", ["duplicate", "seed", "reuse"])
def test_invalid_result_grid_is_rejected(kind):
    raw = synthetic_raw()
    if kind == "duplicate": raw = pd.concat([raw.iloc[:-1], raw.iloc[[0]]], ignore_index=True)
    elif kind == "seed": raw.loc[1, "rollout_seed"] += 1
    else: raw.loc[0, "reused"] = False
    with pytest.raises(RuntimeError): analyze.summarize(raw, draws=10)

def test_real_positive_gain_is_not_assigned_to_dino_when_frequency_wins():
    summary, _, _ = analyze.summarize(synthetic_raw(), draws=10)
    summary.loc[summary.method.eq("uada_freq"), "casr"] = .9
    assert analyze.recommendation(summary) == "KEEP_FREQUENCY_ONLY_STOP_DINO_ENHANCEMENT"
    summary.loc[summary.method.isin(["uada_freq", "uada_dino_freq", "uada_dino"]), "casr"] = 0
    assert analyze.recommendation(summary) == "STOP_THIS_ENHANCEMENT_ROUTE"

def test_read_only_dino_observation_leaves_native_action_identical():
    encoder = torch.nn.Identity()
    wrapper = SimpleNamespace(vla=SimpleNamespace(vision_backbone=SimpleNamespace(featurizer=encoder)))
    value = torch.ones(1, 256, 3)
    with evaluate.observe_dino(wrapper, True) as observed:
        first = encoder(value); second = encoder(value*2)
    assert torch.equal(observed["value"], value*2) and torch.equal(first, value) and torch.equal(second, value*2)
    assert not encoder._forward_hooks
    with pytest.raises(ValueError):
        with evaluate.observe_dino(wrapper, True): raise ValueError("test cleanup")
    assert not encoder._forward_hooks

def test_pipeline_mock_dependencies_caps_and_ambiguous_response(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "preflight", lambda **_: {"signature": "test_signature"})
    monkeypatch.setattr(pipeline, "pending_indices", lambda _: ([0, 1, 2], list(range(12))))
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue": return SimpleNamespace(returncode=0, stdout="5|RUNNING|gres/gpu:4090:1|node\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=str(100+len(calls)), stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    pipeline.submit()
    jobs = [c for c in calls if c[0] == "sbatch"]
    assert len(jobs) == 5 and "--dependency=afterany:5" in jobs[0]
    assert any(c == "--array=0,1,2%3" for c in jobs[1])
    assert any(c.endswith("%4") and c.startswith("--array=") for c in jobs[3])
    assert all(any(c.startswith("--dependency=afterok:") for c in job) for job in jobs[1:])
    receipt = json.loads((tmp_path / "reports/SUBMISSION_RECEIPT.json").read_text())
    receipt["status"] = "AMBIGUOUS_SBATCH_RESPONSE"
    (tmp_path / "reports/SUBMISSION_RECEIPT.json").write_text(json.dumps(receipt))
    before = len(calls)
    with pytest.raises(RuntimeError, match="unresolved"): pipeline.submit()
    assert len(calls) == before

def test_gpu_array_parser_and_completed_entries_need_no_resubmission():
    assert pipeline.existing_gpu_jobs("123_[1-3%2]|PENDING|gres/gpu:4090:1|Resources\n") == ["123"]
    with pytest.raises(RuntimeError): pipeline.existing_gpu_jobs("broken")
    with pytest.raises(RuntimeError): pipeline.existing_gpu_jobs("123|PENDING|gpu:4090:1|DependencyNeverSatisfied")

def test_duplicate_bootstrap_tasks_have_independent_state_draws():
    class FrozenDraws:
        calls = 0
        def integers(self, lo, hi, size):
            self.calls += 1
            if self.calls == 1:
                assert size == (2, 2)
                return np.zeros(size, dtype=int)
            assert size == (4, 2)
            return np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
    groups = [np.array([[0.]*4, [1.]*4]), np.ones((1, 4))]
    draws = analyze.bootstrap_cells(groups, 2, FrozenDraws())
    np.testing.assert_array_equal(draws, np.full((2, 4), .5))

def test_parent_candidate_prefix_is_read_only(monkeypatch, tmp_path):
    candidates = [[f"row{i}"]*4 for i in range(2000)]
    records = {f"row{i}": {"expert_actions": torch.ones(1, 8, 7)*i} for i in range(2000)}
    path = tmp_path / "bank.pt"
    torch.save({"signature": "parent", "candidates": candidates, "records": records}, path)
    checksum = common.sha256_file(path); before = path.stat().st_mtime_ns
    monkeypatch.setattr(common, "lineage", lambda: {"original": {"bank_path": str(path), "bank_sha256": checksum, "signature": "parent"}})
    monkeypatch.setenv("OFT_FACTORIAL_ARM", "original_dino")
    bank = common.load_bank()
    assert bank["candidates"] == candidates[:1000] and len(bank["records"]) == 1000
    assert torch.equal(bank["records"]["row999"]["expert_actions"], records["row999"]["expert_actions"])
    assert path.stat().st_mtime_ns == before and common.sha256_file(path) == checksum

def test_monitor_ignores_only_an_incomplete_final_line(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    folder = tmp_path / "arm"; path = folder / "attempts/1/INNER_STEPS.jsonl"
    path.parent.mkdir(parents=True)
    row = json.dumps({"outer": 1, "inner": 0})
    path.write_text(row + '\n{"outer":')
    assert len(monitor.collect_steps(folder)) == 1
    path.write_text('broken\n' + row + '\n')
    with pytest.raises(RuntimeError, match="Corrupted"): monitor.collect_steps(folder)
