from __future__ import annotations
import copy
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from common import ROOT, METHODS
from key_math import close_slots, open_loss, feature_loss, project_rms, transactional_step
from key_runtime import spatial_key, KeyEngine
from data import split_records

def same_state(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same_state(x,y) for x,y in zip(a,b))
    return a == b

def test_strict_dual_close_and_open_threshold():
    e = torch.zeros(8, 7); c = e.clone(); c[0, 6] = .5; e[1, 6] = .5
    mask = close_slots(e, c)
    assert mask.tolist() == [False, False]+[True]*6
    a = e.clone(); a[:, 6] = .6
    assert open_loss(a, mask) == 0
    from key_math import action_metrics
    a[:, 6] = .5
    assert action_metrics(a, mask)["wrong_open_slot_fraction"] == 0.

def test_direction_allows_orthogonal_freedom_but_point_penalizes_it():
    z = torch.zeros(1, 1, 1, 2); t = torch.tensor([[[[1., 0.]]]]); s = torch.ones(1, 1, 1, 1)
    x = torch.tensor([[[[1., 7.]]]], requires_grad=True)
    assert feature_loss(x, z, t, s, "key_direction").item() == 0.
    assert feature_loss(x, z, t, s, "key_point").item() > 20.
    p = feature_loss(z.clone().requires_grad_(True), z, t, s, "key_direction")
    assert p.item() == 1.

def test_rms_bound_is_per_head():
    z = torch.tensor([[[[4.,4.]], [[.1,.1]]]], requires_grad=True)
    project_rms(z, .2)
    assert torch.allclose(z[0,0], torch.full_like(z[0,0], .2))
    assert torch.allclose(z[0,1], torch.full_like(z[0,1], .1))

@pytest.mark.parametrize("seed_moments", [False, True])
def test_all_rejections_restore_pixels_and_complete_adam_state(seed_moments):
    p = torch.tensor([.5], requires_grad=True); opt = torch.optim.AdamW([p], lr=.1, weight_decay=0)
    if seed_moments:
        p.grad = torch.ones_like(p); opt.step()
    before = p.detach().clone(); state = copy.deepcopy(opt.state_dict()); p.grad = torch.ones_like(p)
    r = transactional_step(p, opt, lambda q: (0., 2.), 1., 1.)
    assert not r["accepted"] and len(r["trials"]) == 3
    assert torch.equal(p, before) and same_state(opt.state_dict(), state)

def test_backtracking_accepts_half_step_and_advances_moments_once():
    p = torch.tensor([.5], requires_grad=True); opt = torch.optim.AdamW([p], lr=.2, weight_decay=0)
    p.grad = torch.ones_like(p)
    def check(q): return float(q.detach()), 2. if q.item() < .35 else .9
    r = transactional_step(p, opt, check, .5, 1.)
    assert r["accepted"] and r["alpha"] == .5 and p.item() == pytest.approx(.4, abs=1e-5)
    assert int(opt.state[p]["step"]) == 1

def test_key_hook_keeps_front_and_prefix_and_removes_itself_on_failure():
    module = torch.nn.Identity(); front = torch.randn(1, 2, 261, 3); wrist = front+1
    target = torch.randn(1, 2, 256, 3, requires_grad=True)
    with spatial_key(module, 2, target) as result:
        f = module(front); w = module(wrist)
    assert torch.equal(f, front) and torch.equal(w[:,:,:5], wrist[:,:,:5])
    assert torch.equal(w[:,:,5:], target)
    result["key"].sum().backward(); assert torch.equal(target.grad, torch.ones_like(target))
    assert not module._forward_hooks
    with pytest.raises(RuntimeError):
        with spatial_key(module, 2, target): module(front)
    assert not module._forward_hooks

def test_key_teacher_can_reach_margin_and_records_zero_response():
    class Toy:
        wrapper = SimpleNamespace(device="cpu")
        def __init__(self, responsive): self.responsive = responsive
        def native(self, record, replacement=None):
            k = torch.ones(1, 2, 256, 2) if replacement is None else replacement
            g = .4 + (2*(k-1).mean() if self.responsive else 0.*k.mean())
            a = torch.zeros(1, 8, 7); a[:,:,6] = g
            return a, k
    cfg = {"key_lr": .02, "key_steps": 20, "key_regularizer": .1, "key_rms_limit": .2}
    record = {"uid": "toy", "close_slots": torch.ones(8, dtype=torch.bool)}
    good = KeyEngine.make_target(Toy(True), record, Toy(True).native(record)[0], cfg)
    assert good["outcome"] == "MARGIN_REACHED" and good["nonzero"]
    failed = KeyEngine.make_target(Toy(False), record, Toy(False).native(record)[0], cfg)
    assert failed["outcome"] == "NO_IMPROVEMENT" and not failed["nonzero"]

def test_train_validation_are_task_balanced_and_episode_disjoint():
    records = [{"uid": str(i), "task_id": i%3, "episode_id": i//2} for i in range(60)]
    train, validation = split_records(records, 12, 12)
    assert len(train) == len(validation) == 12
    assert not {r["episode_id"] for r in train} & {r["episode_id"] for r in validation}
    assert [r["task_id"] for r in train] == [0,1,2]*4

def test_only_executed_slots_count_and_half_is_not_open(monkeypatch):
    import evaluate as ev
    fake = SimpleNamespace(process_libero_action=lambda a: a)
    monkeypatch.setitem(sys.modules, "VLAAttacker.multiview.openvla_oft_experiments", fake)
    raw = {"robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    monkeypatch.setattr(ev, "reset_wait", lambda *a: raw)
    monkeypatch.setattr(ev, "contact_objects", lambda *a: [])
    monkeypatch.setattr(ev, "observation", lambda *a: {"wrist_image": np.zeros((224,224,3),dtype=np.uint8)})
    def predict(w, obs, task):
        a = np.zeros((8,7)); a[:,6] = 0.
        if obs["wrist_image"].any(): a[:,6] = [.6,.5,.4,1,1,1,1,1]
        return a, a.copy()
    monkeypatch.setattr(ev, "native_prediction", predict)
    class Env:
        n = 0
        def step(self, a): self.n += 1; return raw, 0, self.n==3, {}
    w = SimpleNamespace(gen_cfg=SimpleNamespace(task_suite_name="libero_spatial"))
    cfg = {"evaluation": {"num_steps_wait":10,"max_steps_by_suite":{"libero_spatial":20},"patch_xy":[26,137]}}
    result, traces = ev.episode(Env(), w, SimpleNamespace(language="test"), None, 42,
        np.ones((50,50,3),dtype=np.uint8), {}, cfg)
    assert result["executed_steps"] == result["eligible_executed_close_slots"] == 3
    assert result["wrong_open_executed_slots"] == 1 and traces[0]["executed_slots"] == 3

def test_eval_submission_stops_before_scheduler_when_training_invalid(tmp_path, monkeypatch):
    import pipeline as p
    (tmp_path/"reports").mkdir()
    (tmp_path/"reports/AUTO_VALIDATION.json").write_text(json.dumps({"status":"COMPLETE","signature":"s","evaluation_allowed":False}))
    monkeypatch.setattr(p, "ROOT", tmp_path); monkeypatch.setattr(p, "preflight", lambda **kw: None)
    monkeypatch.setattr(p, "prepared", lambda: None); monkeypatch.setattr(p, "run_signature", lambda: "s")
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected scheduler call"))
    with pytest.raises(RuntimeError, match="validated"): p.submit(True)

def test_submit_commands_use_two_suite_workers_and_dependencies():
    from pipeline import command_for, gpu_dependencies
    c = command_for("eval", "s", "123", ["4","5"], [0,1,2,3])
    assert "--array=0,1,2,3%2" in c and "--dependency=afterok:123,afterany:4:5" in c
    assert gpu_dependencies("14_[0-7%4]|PENDING|gres/gpu:4090:1|Dependency\n") == ["14"]

def test_resource_sizes_and_native_copies():
    from common import lineage, sha256_file
    for rel, v in lineage()["native_copies"].items(): assert sha256_file(ROOT/rel) == v["sha256"]
    for name, memory in [("01_prepare_cpu.sbatch",4),("02_train_4090.sbatch",16),
                         ("03_validate_cpu.sbatch",4),("04_eval_4090.sbatch",12),("05_stats_cpu.sbatch",4)]:
        assert f"#SBATCH --mem={memory}G\n" in (ROOT/"slurm"/name).read_text()

def test_candidate_log_resume_uses_latest_attempt_and_ignores_uncommitted_tail(tmp_path):
    from analyze import collect_steps
    for name, rows in [("001",[{"index":0,"v":1},{"index":1,"v":1},{"index":2,"v":99}]),
                       ("002",[{"index":1,"v":2}])]:
        d=tmp_path/"attempts"/name;d.mkdir(parents=True)
        (d/"INNER_STEPS.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    assert [r["v"] for r in collect_steps(tmp_path,2)] == [1,2]

def test_fit_interrupt_and_resume_preserves_candidate_count_and_adam(tmp_path, monkeypatch):
    import train as tr
    from analyze import collect_steps
    cfg = {"learning_rate":.002, "data_seed":42, "inner_loop":2, "single_candidates":5, "backtracking":[1.,.5,.25]}
    monkeypatch.setattr(tr, "config", lambda: cfg); monkeypatch.setattr(tr, "run_signature", lambda: "test")
    monkeypatch.setattr(tr, "snapshot", lambda: {})
    class Engine:
        wrapper = SimpleNamespace(device="cpu")
        n = 0; crash = True
        def rgb_key(self, r, p, seed, fixed):
            self.n += 1
            if self.crash and self.n == 3: raise RuntimeError("injected interruption")
            return p.mean((1,2)).reshape(1,1,1,3)
        def native(self, r, p, seed, fixed):
            q = p.mul(255).round().div(255)
            a = torch.zeros(1,8,7);a[:,:,6]=.4+.1*q.mean()
            return a,q.mean((1,2)).reshape(1,1,1,3)
    record = {"uid":"r", "task_id":0, "pilot_split":"train"}
    t = {"uid":"r","mask":torch.ones(8,dtype=torch.bool),"nonzero":True,"outcome":"PARTIAL",
         "clean_key":torch.zeros(1,1,1,3),"target_key":torch.ones(1,1,1,3),"scale":torch.ones(1,1,1,1)}
    e=Engine()
    with pytest.raises(RuntimeError,match="interruption"): tr.fit(e,[record],{"r":t},"key_point",tmp_path,False,[record])
    assert torch.load(tmp_path/"resume.pt",weights_only=False)["completed"] == 2
    e.crash=False
    tr.fit(e,[record],{"r":t},"key_point",tmp_path,False,[record])
    result=json.loads((tmp_path/"status.json").read_text()); rows=collect_steps(tmp_path,5)
    assert result["status"] == "COMPLETE" and sum(result["counts"].values()) == len(rows) == 5
    saved=torch.load(tmp_path/"resume.pt",weights_only=False)
    assert int(next(iter(saved["optimizer"]["state"].values()))["step"]) == result["counts"]["accepted"]

def test_paired_statistics_keep_task_equal_casr_and_failure_switches():
    import pandas as pd
    from common import SUITES
    from analyze import paired_summary
    rows=[]
    for s in SUITES:
        for t in range(10):
            for state in [40,41]:
                for c in ["clean","spatial_uada500","long_uada1000",*METHODS]:
                    fail=0 if c=="clean" else 1 if c=="key_direction" else int(state==40)
                    rows.append({"suite":s,"task_id":t,"state_index":state,"condition":c,"success":1-fail,"failure":fail})
    summary,contrasts,tasks=paired_summary(pd.DataFrame(rows),draws=100)
    point=summary[(summary.suite=="equal_suite_mean")&(summary.condition=="key_point")].iloc[0]
    assert point.task_equal_casr == .5
    x=contrasts[(contrasts.suite=="libero_spatial")&(contrasts.left=="key_direction")&(contrasts.right=="key_point")].iloc[0]
    assert x.delta_casr==.5 and x.new_failures==10 and x.lost_failures==0
