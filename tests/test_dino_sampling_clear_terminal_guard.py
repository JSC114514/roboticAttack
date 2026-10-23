"""CPU-only regression for terminal Adam float32 rounding and atomic resume."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_dino_sampling_clear_trainer import LoopMock, load_resume, tiny_setup
import white_patch.UADA_OFT_DINO_SAMPLING_CLEAR as trainer


@pytest.fixture
def terminal_setup(tiny_setup, monkeypatch):
    """Short CPU loop, retaining real Adam but mocking the final tiny LR."""
    cfg, clean, replay, sources, data, _ = tiny_setup
    original_build = trainer.build_uada_adamw
    schedule = {"first_tiny_epoch": cfg["outer_iterations"] - 2, "terminal_lr": 5.034994e-9}

    def build(patch, lr, warmup, total, weight_decay):
        optimizer, scheduler = original_build(patch, lr, warmup, total, weight_decay)
        original_lambda = scheduler.lr_lambdas[0]
        scheduler.lr_lambdas[0] = lambda epoch: (
            schedule["terminal_lr"] / lr
            if schedule["first_tiny_epoch"] <= epoch < total else original_lambda(epoch))
        return optimizer, scheduler

    monkeypatch.setattr(trainer, "build_uada_adamw", build)
    return cfg, clean, replay, sources, data, schedule


def status(cfg, method):
    return json.loads((Path(cfg["output_root"]) / "training" / method / "status.json").read_text())


def assert_same_state(actual, expected):
    assert torch.equal(actual["patch"], expected["patch"])
    assert actual["scheduler"] == expected["scheduler"]
    actual_adam = next(iter(actual["optimizer"]["state"].values()))
    expected_adam = next(iter(expected["optimizer"]["state"].values()))
    assert actual_adam["step"] == expected_adam["step"]
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(actual_adam[key], expected_adam[key])
    assert torch.equal(actual["torch_rng"], expected["torch_rng"])
    assert actual["python_rng"] == expected["python_rng"]
    assert np.array_equal(actual["numpy_rng"][1], expected["numpy_rng"][1])
    assert actual["replay_scoring"] == expected["replay_scoring"]
    assert actual["terminal_rounding_history"] == expected["terminal_rounding_history"]


@pytest.mark.parametrize("method", ("dynamic_hard", "stable_hard", "shuffled_hard"))
@pytest.mark.parametrize("terminal_lr", (5.034994e-9, 1.258749e-9))
def test_terminal_tiny_lr_completes_without_forcing_rgb_or_changing_row_schema(
        method, terminal_lr, terminal_setup):
    cfg, clean, replay, sources, data, schedule = terminal_setup
    schedule["terminal_lr"] = terminal_lr
    LoopMock().train_sampling_clear(clean, replay, cfg, method, sources, data)
    saved = load_resume(cfg, method)
    trainer.validate_state(saved, cfg)
    assert saved["outer"] == cfg["outer_iterations"]
    assert saved["costs"]["actual_optimizer_updates"] == cfg["outer_iterations"] * cfg["inner_loop"]
    assert int(next(iter(saved["optimizer"]["state"].values()))["step"]) == 32
    out = Path(cfg["output_root"]) / "training" / method
    previous = torch.load(out / "outer_0006.pt", weights_only=False)["patch"]
    assert torch.equal(previous, saved["patch"])
    assert [entry["outer"] for entry in saved["terminal_rounding_history"]] == [7, 8]
    for entry in saved["terminal_rounding_history"]:
        assert entry["reason"] == "terminal_float32_rounding"
        assert entry["learning_rate"] == saved["rows"][entry["outer"] - 1]["learning_rate"]
        assert entry["learning_rate"] == pytest.approx(terminal_lr, rel=1e-15, abs=0.)
        assert entry["evidence_count"] == cfg["inner_loop"]
        assert entry["effective_rgb_updates"] == entry["correction_updates"] == 0
        assert np.isfinite(entry["max_intended_update_abs"]) and entry["max_intended_update_abs"] > 0
    rows = saved["rows"]
    assert all(row.keys() == rows[0].keys() for row in rows)
    assert all(row["rgb_outer_max_abs"] == 0 and row["effective_rgb_updates"] == 0 for row in rows[-2:])
    assert all(row["action_gradient_norm"] > 0 and row["patch_gradient_norm"] > 0 for row in rows)
    table = pd.read_parquet(out / "training.parquet")
    assert np.isfinite(table.select_dtypes(include="number").to_numpy()).all()
    assert all(row["feasible_internal"] and row["png_feasible_scientific"] for row in rows)
    final = status(cfg, method)
    assert final["status"] == "COMPLETE" and final["terminal_rounding_history"] == saved["terminal_rounding_history"]


@pytest.mark.parametrize("fail_patch_call", (26, 30))
def test_terminal_resume_is_bitwise_and_old_optional_history_defaults_empty(fail_patch_call, terminal_setup):
    cfg, clean, replay, sources, data, _ = terminal_setup
    method = "stable_hard"
    reference_cfg = dict(cfg, output_root=str(Path(cfg["output_root"]).parent / "reference"))
    LoopMock().train_sampling_clear(clean, replay, reference_cfg, method, sources, data)
    failed = LoopMock()
    failed.fail_patch_call = fail_patch_call
    with pytest.raises(RuntimeError, match="half-outer"):
        failed.train_sampling_clear(clean, replay, cfg, method, sources, data)
    partial = load_resume(cfg, method)
    if fail_patch_call == 26:
        assert partial["outer"] == 6 and partial["terminal_rounding_history"] == []
        # Simulate the already committed pre-fix format in a fresh test output.
        partial.pop("terminal_rounding_history")
        trainer.atomic_torch(partial, Path(cfg["output_root"]) / "training" / method / "resume.pt")
    else:
        assert partial["outer"] == 7 and [item["outer"] for item in partial["terminal_rounding_history"]] == [7]
    committed_rows = copy.deepcopy(partial["rows"])
    LoopMock().train_sampling_clear(clean, replay, cfg, method, sources, data)
    resumed, expected = load_resume(cfg, method), load_resume(reference_cfg, method)
    assert_same_state(resumed, expected)
    assert resumed["rows"][:len(committed_rows)] == committed_rows
    assert len(resumed["rows"]) == 8 and int(next(iter(resumed["optimizer"]["state"].values()))["step"]) == 32


def test_tiny_lr_early_stall_is_still_rejected(terminal_setup):
    cfg, clean, replay, sources, data, schedule = terminal_setup
    schedule["first_tiny_epoch"] = 2
    with pytest.raises(RuntimeError, match="Nonzero LR but unchanged complete outer RGB"):
        LoopMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    saved = load_resume(cfg, "stable_hard")
    assert saved["outer"] == 2 and saved["terminal_rounding_history"] == []
    assert status(cfg, "stable_hard")["attempted_outer"] == 3


@pytest.mark.parametrize("corruption", ("missing", "count", "nonfinite"))
def test_terminal_resume_requires_matching_finite_rounding_evidence(corruption, terminal_setup):
    cfg, clean, replay, sources, data, _ = terminal_setup
    LoopMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    saved = load_resume(cfg, "stable_hard")
    if corruption == "missing":
        saved.pop("terminal_rounding_history")
    elif corruption == "count":
        saved["terminal_rounding_history"][0]["evidence_count"] -= 1
    else:
        saved["terminal_rounding_history"][0]["max_intended_update_abs"] = float("nan")
    with pytest.raises(RuntimeError, match="terminal-rounding evidence"):
        trainer.validate_state(saved, cfg)


def test_large_lr_terminal_stall_is_still_rejected(terminal_setup, monkeypatch):
    cfg, clean, replay, sources, data, schedule = terminal_setup
    schedule["terminal_lr"] = .01
    original_step = trainer.step_uada_patch

    def stall(patch, optimizer, model):
        before = patch.detach().clone()
        original_step(patch, optimizer, model)
        if int(optimizer.state[patch]["step"]) > 24:
            with torch.no_grad():
                patch.copy_(before)

    monkeypatch.setattr(trainer, "step_uada_patch", stall)
    with pytest.raises(RuntimeError, match="Nonzero LR but unchanged complete outer RGB"):
        LoopMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    assert load_resume(cfg, "stable_hard")["outer"] == 6
    assert status(cfg, "stable_hard")["attempted_outer"] == 7


class NonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


@pytest.mark.parametrize("bad_gradient", ("zero", "nonfinite"))
def test_terminal_zero_or_nonfinite_action_gradient_is_rejected(bad_gradient, terminal_setup):
    cfg, clean, replay, sources, data, _ = terminal_setup

    class BadGradientMock(LoopMock):
        def _forward(self, inputs, proprio, observation, patch, geometry, **kwargs):
            result = super()._forward(inputs, proprio, observation, patch, geometry, **kwargs)
            if patch is None or self.patched_calls <= 24:
                return result
            value = patch.mean() * 0. if bad_gradient == "zero" else NonfiniteBackward.apply(patch.mean())
            return value.expand(1, 8, 7), result[1], result[2]

    message = "unchanged complete outer RGB" if bad_gradient == "zero" else "Nonfinite/zero real-model"
    with pytest.raises(RuntimeError, match=message):
        BadGradientMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    assert load_resume(cfg, "stable_hard")["outer"] == 6
    assert status(cfg, "stable_hard")["terminal_rounding_history"] == []


def test_terminal_noop_with_any_constraint_correction_is_rejected(terminal_setup, monkeypatch):
    cfg, clean, replay, sources, data, _ = terminal_setup
    original_constraints = trainer.enforce_constraints
    calls = 0

    def correction(patch, configuration):
        nonlocal calls
        calls += 1
        result, info = original_constraints(patch, configuration)
        if calls > 25:  # Common initialization plus the first six outer loops.
            info = dict(info, correction_applied=True)
        return result, info

    monkeypatch.setattr(trainer, "enforce_constraints", correction)
    with pytest.raises(RuntimeError, match="Nonzero LR but unchanged complete outer RGB"):
        LoopMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    assert load_resume(cfg, "stable_hard")["outer"] == 6


def test_terminal_guard_cannot_bypass_real_hard_rgb_tv_hf_budgets(terminal_setup, monkeypatch):
    cfg, clean, replay, sources, data, _ = terminal_setup
    original_constraints = trainer.enforce_constraints
    calls = 0
    checker = torch.tensor([[.1, .9] * 25, [.9, .1] * 25] * 25).expand(3, -1, -1).clone()

    def broken_constraints(patch, configuration):
        nonlocal calls
        calls += 1
        result, info = original_constraints(patch, configuration)
        return (checker.to(patch.device), info) if calls > 25 else (result, info)

    monkeypatch.setattr(trainer, "enforce_constraints", broken_constraints)
    with pytest.raises(RuntimeError, match="Hard arm failed continuous/deployment feasibility"):
        LoopMock().train_sampling_clear(clean, replay, cfg, "stable_hard", sources, data)
    assert load_resume(cfg, "stable_hard")["outer"] == 6
    assert status(cfg, "stable_hard")["terminal_rounding_history"] == []


def test_rounding_evidence_is_read_only_and_uses_post_step_adam_moments():
    patch = torch.full((3, 2, 2), .5, requires_grad=True)
    optimizer, _ = trainer.build_uada_adamw(patch, .002, 0, 2000, 0.)
    optimizer.param_groups[0]["lr"] = 5.034994e-9
    gradient = torch.full_like(patch, .1)
    patch.grad = gradient.clone()
    before = patch.detach().clone()
    optimizer.step()
    assert torch.equal(patch, before)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = torch.get_rng_state().clone()
    evidence = trainer._terminal_float32_rounding_evidence(before, optimizer, gradient, gradient)
    assert evidence is not None and 0 < evidence["max_intended_update_abs"] < 5.034994e-9
    assert torch.equal(patch, before) and torch.equal(torch.get_rng_state(), saved_rng)
    actual_state = optimizer.state_dict()["state"][0]
    for key, value in saved_optimizer["state"][0].items():
        assert torch.equal(actual_state[key], value) if isinstance(value, torch.Tensor) else actual_state[key] == value
    assert optimizer.state_dict()["param_groups"] == saved_optimizer["param_groups"]
