"""Analytic CPU checks for fidelity, state isolation and paired comparison."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import common
import meta
import pipeline
from data import pair_schedule
from train import epochs, validate_resume
from monitor import rng_state, restore_rng, collect_steps
from UADA_SOURCE_COPY import build_uada_adamw

def adam(p, lr=.01):
    from transformers import AdamW
    return AdamW([p], lr=lr, weight_decay=0., no_deprecation_warning=True)

def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: state_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): state_equal(x, y)
    else: assert a == b

def test_visual_vjp_equals_direct_chain_rule():
    p = torch.tensor([.2, .4, .7], dtype=torch.float64, requires_grad=True)
    dino, siglip = p.square(), torch.sin(p)
    loss = (2*dino.sum()+3*siglip.sum()-1).square()
    direct = torch.autograd.grad(loss, p, retain_graph=True)[0]
    expected_d = (2*(2*dino.sum()+3*siglip.sum()-1)*4*p).detach()
    full, d = meta.split_rgb_gradient(loss, dino, siglip, p)
    torch.testing.assert_close(full, direct, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(d, expected_d, rtol=1e-13, atol=1e-13)
    assert not full.requires_grad and not d.requires_grad
    assert not torch.allclose(full, d)

@pytest.mark.parametrize("warm_steps", [0, 1, 4])
def test_adam_preview_matches_real_step_and_preserves_moments(warm_steps):
    p = torch.tensor([0., .4, 1.], requires_grad=True); opt = adam(p)
    for _ in range(warm_steps):
        p.grad = torch.tensor([.3, -.2, .4]); opt.step(); p.data.clamp_(0, 1)
    before = copy.deepcopy(opt.state_dict()); value = p.detach().clone()
    grad = torch.tensor([-.2, .5, -.9])
    preview = meta.preview_adamw(p, grad, opt)
    state_equal(before, opt.state_dict()); assert torch.equal(p, value)
    p.grad = grad; opt.step(); p.data.clamp_(0, 1)
    assert torch.equal(p, preview)

def test_virtual_norm_matches_after_clipping():
    p = torch.tensor([0., .9, .5]); gd = torch.tensor([.8, -1., .2])
    q, info = meta.virtual_patch(p, gd, .3)
    assert info["virtual_status"] == "MATCHED"
    assert float((q-p).norm()) == pytest.approx(.3, rel=2e-5)
    assert q.min() >= 0 and q.max() <= 1 and q[0] == 0

def test_virtual_impossible_and_zero_cases_are_explicit():
    p = torch.zeros(3)
    q, m = meta.virtual_patch(p, torch.ones(3), .1)
    assert q is None and m["virtual_status"] == "UNATTAINABLE_CLIPPED_NORM"
    q, m = meta.virtual_patch(p, torch.zeros(3), .1)
    assert q is None and m["virtual_status"] == "ZERO_DINO_GRADIENT"
    q, m = meta.virtual_patch(p, torch.ones(3), 0.)
    assert torch.equal(q, p) and m["virtual_status"] == "ZERO_ADAM_STEP"

class ToyEngine:
    def __init__(self, noise=False): self.noise = noise; self.calls = []
    def gradients(self, record, patch, seed, split):
        random.seed(seed); np.random.seed(seed % (2**32-1))
        factor = record["factor"]
        if self.noise: factor += .01*(random.random()+np.random.rand()+float(torch.rand(())))
        d, s = patch.square()*factor, torch.sin(patch)*(.4-factor)
        loss = (d.sum()+s.sum()-record["target"]).square()
        if split: full, gd = meta.split_rgb_gradient(loss, d, s, patch)
        else: full, gd = torch.autograd.grad(loss, patch)[0].detach(), None
        self.calls.append((record["uid"], seed, split, patch.detach().clone()))
        return {"full": full, "dino": gd, "loss": float(loss.detach())}
    def clear_bases(self): pass

def test_actual_update_is_first_order_average_at_virtual_point():
    cur = {"uid": "e:0", "factor": .7, "target": 2.}
    nxt = {"uid": "e:8", "factor": -.2, "target": -.5}
    p = torch.tensor([.2, .6], requires_grad=True); expected = p.detach().clone().requires_grad_(True)
    opt, ref = adam(p), adam(expected); manual = ToyEngine()
    a = manual.gradients(cur, expected, 8, True)
    preview = meta.preview_adamw(expected, a["full"], ref)
    q, _ = meta.virtual_patch(expected, a["dino"], float((preview-expected).norm()))
    b = manual.gradients(nxt, q.detach().requires_grad_(True), 8, False)
    expected.grad = (a["full"]+b["full"])*.5; ref.step(); expected.data.clamp_(0, 1)
    engine = ToyEngine(); row, tensors = meta.update(engine, cur, nxt, p, opt, 8)
    assert torch.equal(p, expected) and row["used_query"]
    assert int(opt.state[p]["step"]) == 1 and len(engine.calls) == 2
    assert [c[1] for c in engine.calls] == [8, 8]
    assert not tensors["combined_gradient"].requires_grad
    state_equal(opt.state_dict(), ref.state_dict())
    assert not torch.equal(p, tensors["virtual"])

def test_missing_successor_uses_exact_native_update_once():
    record = {"uid": "e:9", "factor": .5, "target": 2.}
    p = torch.tensor([.3, .4], requires_grad=True); refp = p.detach().clone().requires_grad_(True)
    opt, ref = adam(p), adam(refp); e = ToyEngine()
    refp.grad = e.gradients(record, refp, 19, True)["full"]; ref.step(); refp.data.clamp_(0, 1)
    row, _ = meta.update(e, record, None, p, opt, 19)
    assert torch.equal(p, refp) and row["virtual_status"] == "NO_EXPERT_SUCCESSOR" and not row["used_query"]
    assert int(opt.state[p]["step"]) == 1

def test_unattainable_dino_step_falls_back_without_query():
    class Conflict:
        def gradients(self, r, p, seed, split):
            assert split
            return {"full": -torch.ones_like(p), "dino": torch.ones_like(p), "loss": 1.}
    p = torch.zeros(3, requires_grad=True); opt = adam(p)
    row, _ = meta.update(Conflict(), {}, {}, p, opt, 1)
    assert row["virtual_status"] == "UNATTAINABLE_CLIPPED_NORM" and not row["used_query"]
    assert (p > 0).all() and int(opt.state[p]["step"]) == 1

def test_pair_gap_is_one_eight_action_chunk_and_keeps_boundary_rows():
    eps = [{"episode_id": "e", "length": 24}]
    pairs = pair_schedule([["e:0"], ["e:8"], ["e:16"]], eps, count=3)
    assert [p["query_uid"] for p in pairs] == ["e:8", "e:16", None]
    assert [p["uid"] for p in pairs] == ["e:0", "e:8", "e:16"]

def toy_setup():
    cfg = dict(outer_iterations=4, inner_loop=3, data_seed=17, diagnostic_outers=[1, 2, 3, 4], diagnostic_inners=[0, 2], patch_shape=[3, 2, 2])
    initial = torch.rand((3, 2, 2), generator=torch.Generator().manual_seed(42))
    p = initial.clone().requires_grad_(True)
    opt, scheduler = build_uada_adamw(p, .002, 2, 2000, weight_decay=0.)
    records = {f"e:{i}": {"uid": f"e:{i}", "factor": .2+i*.01, "target": .4+i*.03} for i in range(5)}
    bank = {"pairs": [{"uid": f"e:{i}", "query_uid": f"e:{i+1}" if i < 3 else None} for i in range(4)], "records": records, "initial": initial}
    return cfg, p, opt, scheduler, bank

def test_monitor_on_off_preserves_patch_moments_and_rng():
    cfg, p, opt, scheduler, bank = toy_setup(); random.seed(5); np.random.seed(5); torch.manual_seed(5)
    state = rng_state()
    epochs(ToyEngine(True), bank, p, opt, scheduler, cfg)
    expected = p.detach().clone(); opt_expected = copy.deepcopy(opt.state_dict()); rng_expected = rng_state()
    cfg, p, opt, scheduler, bank = toy_setup(); restore_rng(state)
    def noisy_probe(*args):
        random.random(); np.random.rand(); torch.rand(71)
    epochs(ToyEngine(True), bank, p, opt, scheduler, cfg, diagnostic=noisy_probe)
    assert torch.equal(p, expected); state_equal(opt_expected, opt.state_dict())
    assert torch.equal(rng_state()["torch"], rng_expected["torch"])
    assert rng_state()["python"] == rng_expected["python"]
    np.testing.assert_equal(rng_state()["numpy"], rng_expected["numpy"])

def test_interrupt_resume_matches_uninterrupted_training():
    cfg, p, opt, scheduler, bank = toy_setup(); torch.manual_seed(9); random.seed(9); np.random.seed(9)
    initial_rng = rng_state(); epochs(ToyEngine(True), bank, p, opt, scheduler, cfg)
    expected = p.detach().clone(); expected_opt = copy.deepcopy(opt.state_dict()); expected_scheduler = scheduler.state_dict()
    cfg, p, opt, scheduler, bank = toy_setup(); restore_rng(initial_rng)
    rows = epochs(ToyEngine(True), bank, p, opt, scheduler, cfg, stop_after=2)
    saved = {"signature": "test", "config": cfg, "method": common.METHOD, "outer": 2, "rows": rows,
             "patch": p.detach().clone(), "initial": bank["initial"], "optimizer": copy.deepcopy(opt.state_dict()),
             "scheduler": copy.deepcopy(scheduler.state_dict()), "rng": rng_state()}
    validate_resume(saved, cfg, "test")
    cfg, p, opt, scheduler, bank = toy_setup(); p.data.copy_(saved["patch"])
    opt.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); restore_rng(saved["rng"])
    epochs(ToyEngine(True), bank, p, opt, scheduler, cfg, start=2, rows=saved["rows"])
    assert torch.equal(p, expected); state_equal(opt.state_dict(), expected_opt); state_equal(scheduler.state_dict(), expected_scheduler)

def make_grid(conditions=("clean", "uada1000")):
    return pd.DataFrame([{"suite": s, "task_id": t, "state_index": i, "condition": c,
                          "success": 1 if c == "clean" else int(t % 2 == 0),
                          "failure": 0 if c == "clean" else int(t % 2 != 0), "rollout_seed": 1000+t*10+i}
        for s in common.SUITES for t in range(10) for i in (40, 41) for c in conditions])

def test_historical_grid_detects_duplicates_and_wrong_reset_seeds(monkeypatch):
    f = make_grid(); seeds = f.loc[f.suite.eq(common.SUITES[0]) & f.condition.eq("clean"), ["task_id", "state_index", "rollout_seed"]]
    monkeypatch.setattr(common, "reset_seeds", lambda suite: seeds)
    common.validate_grid(f, ("clean", "uada1000"))
    with pytest.raises(RuntimeError, match="Missing/duplicated"):
        common.validate_grid(pd.concat([f, f.iloc[:1]]), ("clean", "uada1000"))
    wrong = f.copy(); wrong.loc[wrong.task_id.eq(0) & wrong.state_index.eq(40), "rollout_seed"] += 1
    with pytest.raises(RuntimeError, match="Reset grid differs"):
        common.validate_grid(wrong, ("clean", "uada1000"))

def test_summary_is_task_equal_paired_and_compares_both_baselines():
    from analyze import summarize
    f = make_grid(("clean", "uada500", "uada1000", common.METHOD))
    f.loc[f.condition.eq(common.METHOD), ["success", "failure"]] = [0, 1]
    summary, contrasts = summarize(f, draws=30)
    assert len(summary) == 12 and len(contrasts) == 10
    assert summary.loc[summary.condition.eq(common.METHOD), "casr"].eq(1).all()
    assert summary.loc[summary.condition.eq("uada1000"), "casr"].eq(.5).all()
    assert contrasts.delta_casr.eq(.5).all()
    assert set(contrasts.comparison) == {common.METHOD+"-uada500", common.METHOD+"-uada1000"}

def test_summary_excludes_and_counts_tasks_without_clean_success():
    from analyze import summarize
    f = make_grid(("clean", "uada500", "uada1000", common.METHOD))
    f.loc[f.task_id.eq(0) & f.condition.eq("clean"), ["success", "failure"]] = [0, 1]
    summary, _ = summarize(f, draws=5)
    assert summary.eligible_tasks.eq(9).all() and summary.excluded_tasks.eq(1).all()
    assert summary.eligible_episodes.eq(18).all()

def test_update_log_resume_uses_latest_attempt_and_rejects_complete_corruption(tmp_path):
    for n in ("1", "2"):
        path = tmp_path / "attempts" / n; path.mkdir(parents=True)
    a = {"outer": 1, "inner": 0, "current_uada_loss": 1.}
    b = {**a, "current_uada_loss": 2.}
    (tmp_path / "attempts/1/INNER_STEPS.jsonl").write_text(json.dumps(a)+"\n"+'{"outer":')
    later = tmp_path / "attempts/2/INNER_STEPS.jsonl"; later.write_text(json.dumps(b)+"\n")
    assert collect_steps(tmp_path).iloc[0].current_uada_loss == 2.
    later.write_text('broken\n')
    with pytest.raises(RuntimeError, match="Corrupt completed"):
        collect_steps(tmp_path)

def test_native_target_is_unchanged_wrong_x_not_action_l2():
    from objectives import target_action, action_loss
    expert = torch.ones(1, 8, 7); expert[:, :4, 0] = -1
    cfg = common.config(); target = target_action(expert, "uada", cfg)
    assert target[0, :4, 0].eq(1).all() and target[0, 4:, 0].eq(-1).all()
    action = target.clone(); action[..., 1:] += 100
    assert action_loss(action, target, "uada", cfg) == 0

def test_resource_contract_and_fixed_gap_not_a_sweep():
    cfg = common.config(); assert cfg["pair_gap"] == 8 and cfg["methods"] == [common.METHOD]
    assert cfg["schedule_outer_iterations"] == 2000 and cfg["outer_iterations"] == 500
    scripts = CODE.parent / "slurm"
    for name, memory, cpus in [("01_prepare_cpu.sbatch", 12, 4), ("02_train_4090.sbatch", 16, 4),
                                ("03_validate_cpu.sbatch", 4, 2), ("04_eval_4090.sbatch", 12, 4), ("05_stats_cpu.sbatch", 4, 2)]:
        text = (scripts / name).read_text()
        assert f"#SBATCH --mem={memory}G\n" in text and f"#SBATCH --cpus-per-task={cpus}\n" in text
    assert "#SBATCH --array=0-7%4" in (scripts / "04_eval_4090.sbatch").read_text()
    assert "--array" not in (scripts / "02_train_4090.sbatch").read_text()

def test_submission_dependencies_and_no_full_gradient_training_arm():
    signature = "a"*64
    prepare = pipeline.command_for("prepare", "prep", signature, existing=("10",))
    assert not any(x.startswith("--dependency=") for x in prepare)
    train = pipeline.command_for("train", "train", signature, dependency="100", existing=("10", "11"))
    assert "--dependency=afterok:100,afterany:10:11" in train and not any(x.startswith("--array") for x in train)
    evaluate = pipeline.command_for("eval", "eval", signature, dependency="102", indices=[0, 4])
    assert "--array=0,4%4" in evaluate
    assert pipeline.existing_gpu_jobs("12_[0-7%4]|PENDING|gres/gpu:4090:1|Dependency\n11|RUNNING|gres/gpu:4090:1|node\n13|RUNNING|N/A|node") == ["11", "12"]

def test_ambiguous_submission_receipt_rejects_before_queue_query(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/SUBMISSION_RECEIPT.json").write_text(json.dumps({"signature": "s", "status": "AMBIGUOUS_SBATCH_RESPONSE"}))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path); monkeypatch.setattr(pipeline, "preflight", lambda **k: {"signature": "s"})
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected queue/submission call"))
    with pytest.raises(RuntimeError, match="unresolved"):
        pipeline.submit()

def test_active_batch_rejects_before_any_sbatch(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/SUBMISSION_RECEIPT.json").write_text(json.dumps({"signature": "s", "status": "SUBMITTED", "jobs": [{"job_id": "55"}]}))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path); monkeypatch.setattr(pipeline, "preflight", lambda **k: {"signature": "s"})
    def query(command, **kw):
        assert command[0] == "squeue"
        return SimpleNamespace(returncode=0, stdout="55_0|RUNNING|gres/gpu:4090:1|node\n", stderr="")
    monkeypatch.setattr(pipeline.subprocess, "run", query)
    with pytest.raises(RuntimeError, match="already has active"):
        pipeline.submit()

def test_dry_run_never_queries_or_submits(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pipeline", "--action", "dry-run"])
    monkeypatch.setattr(pipeline, "run_signature", lambda: "s")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected scheduler call"))
    pipeline.main()
    assert "DRY_RUN_ONLY" in capsys.readouterr().out
