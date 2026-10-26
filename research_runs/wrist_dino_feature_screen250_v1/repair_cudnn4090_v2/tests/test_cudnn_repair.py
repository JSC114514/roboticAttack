"""Reproduce initialization-order failure and dependency repair with CPU mocks."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

REPAIR_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPAIR_ROOT/"code"))
import repair_common as rc
import repair_run as launch
import repair_submit as submitter
import feature_objectives as feature

torch.set_num_threads(2)
JOBS={"prepare":"139713","train":"139714","validate":"139715","eval":"139716","stats":"139717"}


def test_generator_cudnn_is_disabled_before_first_forward_not_after_model_load(monkeypatch):
    original=launch.original_train
    settings=rc.base.config()
    real_rgb=original.generated_rgb
    monkeypatch.setattr(torch.backends.cudnn,"enabled",True)
    def rgb(generator,noise):
        if torch.backends.cudnn.enabled:
            raise RuntimeError("simulated CUDNN_STATUS_NOT_INITIALIZED")
        return real_rgb(generator,noise)
    monkeypatch.setattr(original,"generated_rgb",rgb)
    monkeypatch.setattr(original,"new_generator",lambda cfg,seed,device: feature.new_generator(cfg,seed,device,filters=(8,8,4,4)))
    with pytest.raises(RuntimeError,match="CUDNN_STATUS_NOT_INITIALIZED"):
        original.initialize_attack("advencoder_cos",settings,"cpu")
    calls=[]
    def train(method):
        calls.append(method)
        _,generator,noise,initial,optimizer,scheduler=original.initialize_attack(method,settings,"cpu")
        assert not torch.backends.cudnn.enabled and initial.shape==(3,50,50) and scheduler is None
        output=feature.generated_rgb(generator,noise)
        output.mean().backward(); optimizer.step()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
    monkeypatch.setattr(original,"train",train)
    launch.execute_generator()
    assert calls==["advencoder_cos"]


@pytest.mark.parametrize("fail",[False,True])
def test_launcher_records_success_or_early_initialization_failure(monkeypatch,tmp_path,fail):
    root,repair=tmp_path/"base",tmp_path/"repair"
    manifest={"base_signature":rc.BASE_SIGNATURE,"repair_signature":"MOCK_RUNTIME","method":"advencoder_cos"}
    monkeypatch.setattr(launch,"BASE_ROOT",root)
    monkeypatch.setattr(launch,"REPAIR_ROOT",repair)
    monkeypatch.setattr(launch,"preflight",lambda: None)
    monkeypatch.setattr(launch,"identity",lambda: manifest)
    monkeypatch.setattr(launch.base,"require_gpu",lambda stage: "MOCK_ONLY")
    def execute():
        if fail: raise RuntimeError("simulated constructor failure")
        rc.base.atomic_json({"status":"COMPLETE","signature":rc.BASE_SIGNATURE,"outer":250},root/"training/advencoder_cos/status.json")
    monkeypatch.setattr(launch,"execute_generator",execute)
    if fail:
        with pytest.raises(RuntimeError,match="constructor failure"): launch.run_generator()
    else: launch.run_generator()
    report=rc.base.load_json(repair/"reports/GENERATOR_STATUS.json")
    assert report["status"]==("FAILED" if fail else "COMPLETE")
    assert rc.base.load_json(root/"training/advencoder_cos/RUNTIME_REPAIR.json")==manifest
    if fail:
        status=rc.base.load_json(root/"training/advencoder_cos/status.json")
        assert status["outer"]==0 and status["phase"]=="generator_initialization"


def test_queue_preserves_running_uda_ignores_failed_array_downstream():
    queue="139714_0|RUNNING|gres:gpu:4090:1|node\n139715|PENDING|N/A|Dependency\n139716_[0-9%4]|PENDING|gres:gpu:4090:1|DependencyNeverSatisfied\n200|RUNNING|gpu:2|node"
    external,baseline=submitter.queue_plan(queue,JOBS,"RUNNING")
    assert external==["200"] and baseline=="139714_0"


@pytest.mark.parametrize("queue,status",[("139714_3|RUNNING|gpu:1|node","RUNNING"),
                                        ("139716_0|RUNNING|gpu:1|node","COMPLETE"),
                                        ("","RUNNING")])
def test_queue_refuses_duplicate_original_work_or_stale_status(queue,status):
    with pytest.raises(RuntimeError): submitter.queue_plan(queue,JOBS,status)


def test_complete_baseline_needs_no_old_array_dependency():
    assert submitter.queue_plan("",JOBS,"COMPLETE")==([],None)


@pytest.mark.parametrize("queue_ok",[True,False])
def test_submit_only_generator_then_joins_uda_child_before_eval(monkeypatch,tmp_path,queue_ok):
    root,repair=tmp_path/"base",tmp_path/"repair"
    report={"repair_signature":"MOCK_RUNTIME","base_signature":rc.BASE_SIGNATURE,"original_jobs":JOBS,"uda_status":{"status":"RUNNING"}}
    monkeypatch.setattr(submitter,"BASE_ROOT",root)
    monkeypatch.setattr(submitter,"REPAIR_ROOT",repair)
    monkeypatch.setattr(submitter,"preflight",lambda: report)
    rc.base.atomic_json({"status":"RUNNING","signature":rc.BASE_SIGNATURE},root/"training/uada_original/status.json")
    calls=[]
    def command(args,**kwargs):
        calls.append(args)
        if args[0]=="squeue":
            return SimpleNamespace(returncode=0 if queue_ok else 1,
                                   stdout="139714_0|RUNNING|gres:gpu:4090:1|node\n139716_[0-9%4]|PENDING|gpu:1|Dependency",stderr="mock queue error" if not queue_ok else "")
        assert args[0]=="sbatch"  # No scancel/scontrol/old training array.
        return SimpleNamespace(returncode=0,stdout=str(200+len(calls)),stderr="")
    monkeypatch.setattr(submitter.subprocess,"run",command)
    if not queue_ok:
        with pytest.raises(RuntimeError,match="no jobs submitted"): submitter.submit()
        assert len(calls)==1 and not (repair/"reports/SUBMISSION_RECEIPT.json").exists()
        return
    submitter.submit()
    assert len(calls)==5
    assert calls[1][-1].endswith("01_generator_4090.sbatch")
    assert not any("02_train_4090.sbatch" in str(c) for c in calls)
    assert "--dependency=afterok:202:139714_0" in calls[2]
    assert "--dependency=afterok:203" in calls[3] and "--dependency=afterok:204" in calls[4]
    assert not any("--dependency=afterok:139714"==arg for call in calls for arg in call)
    receipt=rc.base.load_json(repair/"reports/SUBMISSION_RECEIPT.json")
    assert receipt["retrained_methods"]==["advencoder_cos"] and receipt["evaluation_gpu_cap"]==4


def test_base_frozen_source_signature_stays_unchanged():
    assert rc.base.run_signature()==rc.BASE_SIGNATURE
    assert "--array" not in (REPAIR_ROOT/"slurm/01_generator_4090.sbatch").read_text()
