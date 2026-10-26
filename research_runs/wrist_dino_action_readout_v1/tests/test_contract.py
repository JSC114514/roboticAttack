"""CPU checks of scientific contracts and submission behavior; no GPU execution."""
import copy
import sys
from pathlib import Path
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from readout import ActionReadout, energy_logits, patch_objective, response_loss
from run_core import ARMS, BASE, BASE_CONDITION, ROOT, SUITES, native, signature, split_episodes
from submit_run import STAGES, ensure_no_active, job_command
from summarize import compare


def test_wrong_energy_target_gradient_moves_toward_wrong_action():
    action = torch.zeros(1, 8, requires_grad=True)
    clean, wrong = torch.zeros_like(action), torch.ones_like(action)
    loss, _ = patch_objective(action, None, clean, wrong, ARMS[0], 1)
    grad, = torch.autograd.grad(loss, action)
    updated = action.detach() - .1 * grad
    assert F.mse_loss(updated, wrong) < F.mse_loss(action.detach(), wrong)


def test_helper_gate_uses_closeness_and_final100_is_exact_uada():
    action = torch.zeros(1, 8, requires_grad=True)
    clean, wrong = torch.zeros_like(action), torch.ones_like(action)
    _, early = patch_objective(action, clean, clean, wrong, ARMS[1], 400)
    _, fooled = patch_objective(action, wrong, clean, wrong, ARMS[1], 400)
    last, detail = patch_objective(action, None, clean, wrong, ARMS[1], 401)
    assert early["beta"] == .5 and fooled["beta"] == 0
    assert detail["beta"] == 0 and torch.equal(last, F.mse_loss(action, wrong))


def test_beta_zero_c_equals_native_control():
    action = torch.full((1, 8), .2, requires_grad=True)
    clean, wrong = torch.zeros_like(action), torch.ones_like(action)
    b, _ = patch_objective(action, None, clean, wrong, ARMS[0], 200)
    c, detail = patch_objective(action, wrong, clean, wrong, ARMS[1], 200)
    assert detail["beta"] == 0 and torch.equal(b, c)


def test_frozen_readout_preserves_input_gradient():
    torch.manual_seed(8)
    model = ActionReadout(feature_dim=6, language_dim=10, width=16, tokens=4).eval().requires_grad_(False)
    features = torch.randn(1, 2, 4, 6, requires_grad=True)
    prediction = model(features, torch.randn(1, 10), torch.randn(1, 8))
    prediction.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())


def test_response_supervision_detects_wrong_finite_step_direction():
    teacher = torch.tensor([0., .2, .3, .1]).reshape(1, 4, 1).expand(2, 4, 8)
    scales = {k: torch.ones(8) for k in ["action", "delta", "response"]}
    correct, terms = response_loss(teacher, teacher, scales, .1)
    wrong = teacher.clone(); wrong[:, 2], wrong[:, 3] = teacher[:, 3], teacher[:, 2]
    incorrect, wrong_terms = response_loss(wrong, teacher, scales, .1)
    assert correct == 0 and all(v == 0 for v in terms.values())
    assert incorrect > 0 and wrong_terms["response"] > 0


def test_episode_split_protects_all_patch_episodes_and_is_reproducible():
    records = {f"u{i}_{j}": {"episode_id": f"e{i}"} for i in range(12) for j in range(3)}
    source = {"records": records, "candidates": [("u0_0",), ("u1_2",)]}
    split = split_episodes(source, 17, validation_count=3)
    assert split == split_episodes(source, 17, validation_count=3)
    assert not set(split["training_episodes"]) & set(split["validation_episodes"])
    assert {"u0_0", "u1_2"} <= set(split["train_uids"])
    assert all(records[u]["episode_id"] in split["validation_episodes"] for u in split["validation_uids"])


def test_old_initial_float_and_native_eot_schedule_are_exact():
    saved = torch.load(BASE / "training/m01_uada_original_b1/outer_0000.pt", map_location="cpu", weights_only=False)
    expected = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))
    assert torch.equal(saved["patch"], expected)
    assert native().eot_seeds(125, 49, 1) == [20261013 * 1000000 + 125 * 10000 + 49]


def test_native_observer_returns_identical_features_and_removes_hooks():
    from native_bridge import ReadoutEngine
    engine = object.__new__(ReadoutEngine)
    engine.dino, engine.siglip = torch.nn.Identity(), torch.nn.Identity()
    features = torch.randn(1, 256, 6, requires_grad=True)
    with engine.hooks() as captured:
        first = engine.dino(features); engine.siglip(features)
        second = engine.dino(features * 2); engine.siglip(features * 3)
    assert first is features
    assert torch.equal(captured["dino_all"][:, 0], first)
    assert torch.equal(captured["dino_all"][:, 1], second)
    captured["dino_all"].sum().backward()
    assert torch.equal(features.grad, torch.ones_like(features) * 3)
    assert len(engine.dino._forward_hooks) == len(engine.siglip._forward_hooks) == 0


def test_resume_checks_real_adam_clock_and_rejects_head_change():
    from train_patch import validate_resume
    initial = torch.rand(3, 50, 50)
    patch = initial.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([patch], lr=.001)
    for _ in range(50):
        optimizer.zero_grad(); patch.square().mean().backward(); optimizer.step()
    saved = {"signature": signature(), "arm": ARMS[1], "head_sha256": "head1", "initial": initial,
             "outer": 1, "patch": patch.detach(), "optimizer": optimizer.state_dict(), "steps": [{}] * 50, "history": [{}]}
    validate_resume(saved, ARMS[1], initial, "head1")
    with pytest.raises(RuntimeError, match="identity"):
        validate_resume(saved, ARMS[1], initial, "head2")
    broken = copy.deepcopy(saved); broken["optimizer"]["state"][0]["step"] = torch.tensor(49.)
    with pytest.raises(RuntimeError, match="clock"):
        validate_resume(broken, ARMS[1], initial, "head1")


def test_native_and_helper_path_gradients_sum_to_actual_mixed_gradient():
    from train_patch import component_gradients
    patch = torch.tensor([.1, -.2], requires_grad=True)
    clean, wrong = torch.zeros(1, 8), torch.ones(1, 8)
    native_action = (patch[0] + .2 * patch[1]).expand(1, 8)
    helper = (.3 * patch[0] - patch[1]).expand(1, 8)
    gn, gh = component_gradients(native_action, helper, clean, wrong, ARMS[1], 1, patch)
    loss, detail = patch_objective(native_action, helper, clean, wrong, ARMS[1], 1)
    actual, = torch.autograd.grad(loss, patch)
    assert detail["beta"] == .5
    assert torch.allclose(actual, gn + gh, rtol=1e-6, atol=1e-7)
    assert gh.norm() > 0


def test_inactive_helper_has_zero_actual_component_gradient():
    from train_patch import component_gradients
    patch = torch.tensor([.1], requires_grad=True)
    native_action, helper = patch.expand(1, 8), (1 + patch).expand(1, 8)
    clean, wrong = torch.zeros(1, 8), torch.ones(1, 8)
    _, gh = component_gradients(native_action, helper, clean, wrong, ARMS[1], 1, patch)
    assert torch.equal(gh, torch.zeros_like(patch))


def test_original_optimizer_and_scheduler_resume_matches_continuation():
    native()
    from UADA_SOURCE_COPY import build_uada_adamw
    initial = torch.tensor([.2, .7])
    def create():
        p = initial.clone().requires_grad_(True)
        optimizer, scheduler = build_uada_adamw(p, .002, 20, 2000, weight_decay=0.)
        return p, optimizer, scheduler
    def outer_step(p, optimizer, scheduler):
        for _ in range(50):
            optimizer.zero_grad(); (p - 1).square().mean().backward(); optimizer.step()
            with torch.no_grad(): p.clamp_(0, 1)
        scheduler.step()
    p, opt, sched = create()
    outer_step(p, opt, sched)
    saved_p, saved_opt, saved_sched = p.detach().clone(), copy.deepcopy(opt.state_dict()), copy.deepcopy(sched.state_dict())
    outer_step(p, opt, sched)
    resumed, opt2, sched2 = create()
    with torch.no_grad(): resumed.copy_(saved_p)
    opt2.load_state_dict(saved_opt); sched2.load_state_dict(saved_sched)
    outer_step(resumed, opt2, sched2)
    assert torch.equal(p, resumed)
    assert opt.param_groups[0]["lr"] == opt2.param_groups[0]["lr"]


def test_gpu_work_is_rejected_before_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU probed without allocation"))
    with pytest.raises(RuntimeError, match="Allocated Slurm GPU"):
        native().require_gpu("training")


def paired_frame():
    rows = []
    for suite in SUITES:
        for state in [40, 41]:
            for condition in ["clean", BASE_CONDITION, *ARMS]:
                success = {"clean": int(state == 40), BASE_CONDITION: int(state == 40),
                           ARMS[0]: int(state == 40), ARMS[1]: int(state == 41)}[condition]
                rows.append({"suite": suite, "task_id": 0, "state_index": state, "rollout_seed": state,
                             "condition": condition, "success": success})
    return pd.DataFrame(rows)


def test_casr_excludes_clean_failures_and_pairs_every_condition():
    metrics, _, pairs, decision = compare(paired_frame())
    assert metrics.loc[metrics.condition.eq(ARMS[1]), "casr"].eq(1).all()
    assert metrics.loc[metrics.condition.eq(BASE_CONDITION), "casr"].eq(0).all()
    assert decision["decision"] == "C_ADVANCES_TO_MANUAL_SECOND_SEED"
    assert pairs.C_only_causes_failure.eq(1).all()
    incomplete = paired_frame().iloc[:-1]
    with pytest.raises(RuntimeError, match="Unpaired"):
        compare(incomplete)


def test_duplicate_paired_outcomes_fail_closed():
    frame = paired_frame()
    with pytest.raises(RuntimeError, match="Duplicate"):
        compare(pd.concat([frame, frame.iloc[:1]]))


def test_submission_dag_limits_overlap_and_blocks_duplicate_active_jobs():
    dependencies = {name: upstream for name, _, upstream in STAGES}
    assert dependencies["B_train"] == dependencies["teacher_head"] == []
    assert dependencies["C_train"] == ["teacher_head"]
    assert dependencies["C_eval"] == ["C_train", "B_eval"]
    cmd = job_command("C_eval", "05_C_eval.sbatch", ["123", "456"], "prefix_")
    assert "--dependency=afterok:123:456" in cmd
    ensure_no_active("other_job|55|RUNNING", "dar1_abc_")
    with pytest.raises(RuntimeError, match="active"):
        ensure_no_active("dar1_abc_C_train|56|PENDING", "dar1_abc_")


def test_submit_is_only_explicit_entry(monkeypatch):
    import submit_run
    from types import SimpleNamespace
    monkeypatch.setattr(submit_run, "preflight", lambda: {"signature": signature()})
    monkeypatch.setattr(submit_run, "locked", lambda path: __import__("contextlib").nullcontext())
    monkeypatch.setattr(submit_run, "complete", lambda stage: False)
    calls, writes = [], []
    def runner(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="" if cmd[0] == "squeue" else str(1000 + len(calls)), stderr="")
    monkeypatch.setattr(submit_run.subprocess, "run", runner)
    monkeypatch.setattr(submit_run, "write_json", lambda value, path: writes.append(str(path)))
    submit_run.submit()
    sbatch = [c for c in calls if c[0] == "sbatch"]
    assert len(sbatch) == 6
    assert "--dependency=afterok:1003" in sbatch[2]
    assert "--dependency=afterok:1004:1005" in sbatch[4]
    assert not any(c[0] in ["scancel", "srun", "salloc"] for c in calls)
