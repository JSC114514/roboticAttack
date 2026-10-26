"""CPU/toy verification only; no real policy forward, GPU, renderer or Slurm."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common
import remaining_objectives as target
import feature_objectives as features
import train
import analyze
import evaluate
import pipeline

torch.set_num_threads(2)


def test_contract_and_provenance_are_real_readonly():
    cfg = common.config()
    assert common.METHODS == ("dino_value", "advencoder_cos_b8", "advencoder_nce_b8")
    assert cfg["value_layers"] == [8,16] and cfg["batch_sizes"] == dict(zip(common.METHODS,[1,8,8]))
    assert cfg["outer_iterations"]*cfg["inner_loop"] == 12500
    assert cfg["generator"] == common.base.config()["generator"]
    assert cfg["eot"] == common.base.config()["eot"] and cfg["evaluation"] == common.base.config()["evaluation"]
    assert len(common.deployment_conditions())*80 == 240
    assert len(common.analysis_conditions())*80 == 640
    assert common.verify_references()["reference_signature"] == common.REFERENCE_SIGNATURE


def test_b8_schedule_paired_distinct_and_same_source_pool():
    schedule = [[str(i)] for i in range(250)]
    a = [common.batch_uids(schedule,i,"dino_value") for i in range(1,251)]
    b = [common.batch_uids(schedule,i,"advencoder_cos_b8") for i in range(1,251)]
    c = [common.batch_uids(schedule,i,"advencoder_nce_b8") for i in range(1,251)]
    assert a == schedule and b == c and all(len(set(row)) == 8 for row in b)
    assert pd.Series([uid for row in b for uid in row]).value_counts().eq(8).all()


def test_native_value_is_not_qk_prefix_and_hooks_are_removed():
    class QKV(torch.nn.Module):
        def forward(self,x):
            return torch.cat([x+100,x+200,x*2],dim=-1)
    qkv = QKV()
    attn = SimpleNamespace(qkv=qkv,num_heads=2)
    dino = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)],num_prefix_tokens=5)
    engine = SimpleNamespace(dino=dino,direct_dino=lambda x:qkv(x)[:,5:])
    pixels = torch.randn(1,261,4,requires_grad=True)
    captured = target.value_forward(engine,pixels,[0])[0]
    assert captured.shape == (1,2,256,2)
    expected = (pixels*2).reshape(1,261,2,2).permute(0,2,1,3)[:,:,5:]
    assert torch.equal(captured,expected) and not qkv._forward_hooks
    captured.sum().backward()
    assert pixels.grad[:,:5].eq(0).all() and pixels.grad[:,5:].eq(2).all()
    def broken(x):
        qkv(x)
        raise ValueError("toy failure")
    engine.direct_dino = broken
    with pytest.raises(ValueError):
        target.value_forward(engine,pixels,[0])
    assert not qkv._forward_hooks


def test_value_target_prefers_rotation_away_not_scaling():
    a = torch.tensor([[[[1.,0.],[0.,1.]]]])
    same,_ = target.value_loss({8:a},{8:a.clone().requires_grad_()})
    reverse,_ = target.value_loss({8:a},{8:-a})
    scaled,_ = target.value_loss({8:a},{8:3*a})
    assert float(reverse) < float(same) and torch.allclose(same,scaled)


def test_nce_has_negative_sign_and_cross_image_coupling():
    clean = torch.eye(3)
    matched = clean.clone().requires_grad_()
    permuted = clean.roll(1,0).clone().requires_grad_()
    good,_ = target.batch_feature_loss(clean,matched,"nce",.1)
    wrong,_ = target.batch_feature_loss(clean,permuted,"nce",.1)
    assert wrong < good
    wrong.backward()
    assert torch.isfinite(permuted.grad).all() and permuted.grad.norm() > 0
    # Each attacked row also serves as a negative for other clean queries.
    queries = torch.tensor([[1.,.2,.1],[.2,1.,.3],[.1,.3,1.]])
    values = torch.tensor([[.4,.5,.3],[.2,.8,.6],[.7,.2,.9]],requires_grad=True)
    loss,_ = target.batch_feature_loss(queries,values,"nce",.1)
    full_grad = torch.autograd.grad(loss,values)[0]
    singleton = values[0:1].detach().requires_grad_()
    logits = F.normalize(queries[:1],dim=1) @ F.normalize(singleton,dim=1).T / .1
    degenerate = -F.cross_entropy(logits,torch.zeros(1,dtype=torch.long))
    assert torch.autograd.grad(degenerate,singleton)[0].eq(0).all() and full_grad[0].norm() > 0
    with pytest.raises(RuntimeError,match="actual batch negatives"):
        target.batch_feature_loss(clean[:1],matched[:1],"nce",.1)


class ToyEngine:
    def __init__(self,wrapper=None):
        self.wrapper = wrapper
        self.dino = torch.nn.Identity().eval()
    def base(self,record):
        rgb = torch.tensor(record["observation"]["wrist_image"]).permute(2,0,1)[None].float()/255
        return {"pixel_values":rgb},None
    def pixels(self,record,rgb,seed,diverse=False,return_rgb=False):
        original = torch.tensor(record["observation"]["wrist_image"]).permute(2,0,1).float()/255
        rendered = (.35*rgb+.65*original)[None]
        return (rendered,torch.zeros(256),rendered) if return_rgb else (rendered,torch.zeros(256))
    def direct_dino(self,pixels):
        pooled = F.adaptive_avg_pool2d(pixels,(2,2)).flatten(2).transpose(1,2)
        return torch.cat([pooled,pooled.square()+.1],dim=-1)
    def validate(self,*args):
        return {"status":"TOY_ONLY"}


def records(count=8):
    result=[]
    for i in range(count):
        im=np.zeros((50,50,3),dtype=np.uint8)
        im[:]=[20+i%70,90+i%80,150-i%100]
        result.append({"uid":str(i),"episode_id":i,"step_index":1,"task_id":i%10,
                       "observation":{"wrist_image":im}})
    return result


@pytest.mark.parametrize("criterion",["cos","nce"])
def test_cached_full_b8_gradient_matches_direct(criterion):
    cfg=common.config()
    engine=ToyEngine()
    rec=records()
    clean=torch.cat([engine.direct_dino(engine.base(r)[0]["pixel_values"]) for r in rec]).detach()
    originals=[engine.base(r)[0]["pixel_values"] for r in rec]
    rgb=torch.rand(3,50,50,requires_grad=True)
    direct=rgb.detach().clone().requires_grad_()
    adv=[]; aux=[]
    for r,original in zip(rec,originals):
        pixels,_,rendered=engine.pixels(r,direct,42,True,True)
        adv.append(engine.direct_dino(pixels))
        aux.append(target.rgb_terms(original,rendered,cfg["generator"])[0])
    feature,_=target.batch_feature_loss(clean,torch.cat(adv),criterion,cfg["nce_temperature"])
    full=5*feature+torch.stack(aux).mean()
    full.backward()
    value,detail=target.cached_generator_backward(engine,rec,clean,originals,rgb,[42]*8,criterion,cfg)
    assert np.isclose(value,float(full.detach()),atol=1e-6)
    assert torch.allclose(rgb.grad,direct.grad,atol=2e-7,rtol=2e-5)
    assert detail["attacked_dino_forwards_per_update"]==16 and detail["recompute_relative_max_error"]==0


def test_cache_refuses_nonrepeatable_model():
    class BadEngine(ToyEngine):
        counter=0
        def direct_dino(self,pixels):
            self.counter+=1
            return super().direct_dino(pixels)+self.counter*.1
    engine=BadEngine(); rec=records(2)
    originals=[engine.base(r)[0]["pixel_values"] for r in rec]
    clean=torch.cat([ToyEngine().direct_dino(v) for v in originals])
    with pytest.raises(RuntimeError,match="recompute changed"):
        target.cached_generator_backward(engine,rec,clean,originals,torch.rand(3,50,50,requires_grad=True),[1,2],"nce",common.config())


@pytest.mark.parametrize("method",common.METHODS)
def test_full_toy_loop_resume_and_final_receipt(method,tmp_path,monkeypatch):
    cfg=copy.deepcopy(common.config())
    cfg.update(outer_iterations=4,inner_loop=2,resume_every_outer=1,checkpoint_outer_iterations=[0,2,4],evaluation_checkpoints=[4])
    rec=records(250); bank={"records":{r["uid"]:r for r in rec},"candidates":[[r["uid"]] for r in rec]}
    monkeypatch.setattr(train,"config",lambda:cfg)
    monkeypatch.setattr(train,"run_signature",lambda:"TOY-SIGNATURE")
    monkeypatch.setattr(train,"prepared",lambda:None)
    monkeypatch.setattr(train,"require_gpu",lambda _stage:"TOY-NO-GPU")
    monkeypatch.setattr(train,"patch_device",lambda:torch.device("cpu"))
    monkeypatch.setattr(train,"load_bank",lambda:bank)
    monkeypatch.setattr(train,"new_generator",lambda settings,seed,device:features.new_generator(settings,seed,device,filters=(4,4,4,4)))
    policy=torch.nn.Linear(1,1).requires_grad_(False)
    wrapper=SimpleNamespace(vla=policy,action_head=None,proprio_projector=None)
    monkeypatch.setattr(train,"load_wrapper",lambda *_args:wrapper)
    monkeypatch.setattr(train,"OFTEngine",ToyEngine)
    monkeypatch.setattr(train,"runtime_config",lambda *_args:None)
    monkeypatch.setattr(train,"value_forward",lambda engine,pixels,layers:{i:engine.direct_dino(pixels) for i in layers})
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",lambda:[torch.get_rng_state().clone()])
    monkeypatch.setattr(torch.cuda,"set_rng_state_all",lambda _states:None)
    full_root=tmp_path/"full"; resumed_root=tmp_path/"resumed"
    monkeypatch.setattr(train,"ROOT",full_root)
    train.train(method)
    full=torch.load(full_root/"training"/method/"resume.pt",weights_only=False)
    atomic=train.atomic_torch
    def interrupt(value,path):
        atomic(value,path)
        if Path(path).name=="resume.pt" and value["outer"]==2:
            raise RuntimeError("TOY power-loss after committed checkpoint")
    monkeypatch.setattr(train,"ROOT",resumed_root)
    monkeypatch.setattr(train,"atomic_torch",interrupt)
    with pytest.raises(RuntimeError,match="TOY power-loss"):
        train.train(method)
    monkeypatch.setattr(train,"atomic_torch",atomic)
    train.train(method)
    folder=resumed_root/"training"/method
    resumed=torch.load(folder/"resume.pt",weights_only=False)
    assert torch.equal(full["patch"],resumed["patch"])
    assert full["parameter_shapes"]==resumed["parameter_shapes"] and full["scheduler"]==resumed["scheduler"]
    for left,right in zip(full["optimizer"]["state"].values(),resumed["optimizer"]["state"].values()):
        assert all(torch.equal(left[k],right[k]) if torch.is_tensor(left[k]) else left[k]==right[k] for k in left)
    if full["generator"] is not None:
        assert all(torch.equal(full["generator"][k],resumed["generator"][k]) for k in full["generator"])
    assert sum(r["effective_rgb_updates"] for r in resumed["rows"])>0
    assert all(len(r["uids"])==cfg["batch_sizes"][method] for r in resumed["rows"])
    # Final commit recovery MUST NOT load a model or perform any further update.
    (folder/"status.json").unlink()
    monkeypatch.setattr(train,"load_wrapper",lambda *_args:pytest.fail("model loaded during final receipt recovery"))
    train.train(method)
    assert json.loads((folder/"status.json").read_text())["phase"]=="recovered_final_receipt"
    train.train(method)


def test_reference_rows_keep_original_two_signatures():
    raw,trace=common.reference_frames()
    assert len(raw)==400 and raw.reused.all() and trace.reused.all()
    assert raw.signature.value_counts().to_dict()=={common.BASE_SIGNATURE:320,common.REFERENCE_SIGNATURE:80}
    assert set(raw.source_run)=={common.BASE_ROOT.name,common.REFERENCE_ROOT.name}


def test_merge_metric_and_objective_comparison_without_gpu():
    old,_=common.reference_frames()
    seed=old.loc[old.condition.eq("clean")].copy()
    new=[]
    for m in common.METHODS:
        rows=seed.copy(); rows["condition"]=m+"@250"; rows["signature"]="TOY-NEW"; rows["reused"]=False
        rows["success"]=1; rows["failure"]=0
        if m=="advencoder_nce_b8":
            rows.loc[rows.task_id.eq(0),["success","failure"]]=[0,1]
        new.append(rows)
    raw=pd.concat([old,*new],ignore_index=True)
    summary,contrasts,tasks=analyze.summarize(raw,draws=20)
    assert len(raw)==640 and len(summary)==32 and len(contrasts)==64 and len(tasks)==280
    matched=contrasts.loc[contrasts.contrast_type.eq("matched_B8_objective")]
    assert len(matched)==4 and matched.casr_difference.eq(.1).all() and matched.positive_tasks.eq(1).all()
    assert contrasts.loc[contrasts.left.eq("advencoder_nce_b8")&contrasts.right.eq("uada_original")].contrast_type.eq("expanded_image_budget").all()


def test_normal_evaluation_ast_identical_and_manual_resource_caps():
    new=ast.parse((ROOT/"code/evaluate.py").read_text())
    old=ast.parse((common.REFERENCE_ROOT/"code/evaluate.py").read_text())
    assert ast.dump(new)==ast.dump(old)
    training=(ROOT/"slurm/02_train_4090.sbatch").read_text()
    evaluation=(ROOT/"slurm/04_eval_4090.sbatch").read_text()
    assert "--array=0-2%3" in training and "--array=0-9%4" in evaluation
    assert "--gres=gpu:4090:1" in training and "--gres=gpu:4090:1" in evaluation
    assert "afterok:123" in " ".join(pipeline.command_for("04_eval_4090.sbatch","123"))
    assert pipeline.existing_gpu_jobs("99_1|PENDING|gres:gpu:4090:1|DependencyNeverSatisfied\n100_0|RUNNING|gres:gpu:4090:1|None")==["100"]
    assert "if args.submit:" in (ROOT/"code/pipeline.py").read_text()
