from __future__ import annotations
import ast,random,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
import common,monitor
from engine import OFTEngine
from batch_runtime import prepare_references,pixel_batch_backward,generator_batch_backward

class Block(nn.Module):
    def __init__(self,n):super().__init__();self.linear=nn.Linear(n,n)
    def forward(self,x):
        x=x+.01*torch.tanh(self.linear(x))
        return self.attn(x) if hasattr(self,"attn") else x

class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(3,9);self.num_heads=1;self.scale=3**-.5
        self.q_norm=nn.Identity();self.k_norm=nn.Identity()
    def forward(self,x):return x+.01*self.qkv(x)[...,6:]

class Encoder(nn.Module):
    num_prefix_tokens=5
    def __init__(self):super().__init__();self.blocks=nn.ModuleList([Block(3) for _ in range(24)])
    def forward(self,x):
        x=F.adaptive_avg_pool2d(x,(16,16)).flatten(2).transpose(1,2)
        x=torch.cat([x[:,:5]*0,x],1)
        for b in self.blocks:x=b(x)
        return x[:,5:]

class LM(nn.Module):
    def __init__(self):super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([Block(8) for _ in range(3)])
    def forward(self,x):
        for layer in self.model.layers:x=layer(x)
        return x.mean(1,keepdim=True).expand(-1,56,-1)

class Head(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(8,1)
    def predict_action(self,x):return self.linear(x)

class VLA(nn.Module):
    def __init__(self):
        super().__init__();self.vision_backbone=nn.Module()
        self.vision_backbone.featurizer=Encoder();self.vision_backbone.fused_featurizer=Encoder()
        # OFTEngine's attention registry is not used by these readout tests.
        for b in self.vision_backbone.featurizer.blocks:b.attn=Attention()
        self.projector=nn.Linear(6,8);self.language_model=LM()
    def predict_action(self,pixel_values,action_head,**_):
        xs=[]
        for i in (0,6):
            a=self.vision_backbone.featurizer(pixel_values[:,i:i+3])
            b=self.vision_backbone.fused_featurizer(pixel_values[:,i+3:i+6])
            xs.append(torch.cat([a,b],-1))
        hidden=self.language_model(self.projector(torch.cat(xs,1)))
        return None,hidden

class Wrapper:
    def __init__(self):
        self.vla=VLA();self.action_head=Head();self.proprio_projector=None;self.noisy_action_projector=None
        self.gen_cfg=SimpleNamespace(unnorm_key="toy");self.device=torch.device("cpu");self.dtype=torch.float32
    def freeze_model(self):
        for m in (self.vla,self.action_head):m.eval();m.requires_grad_(False)

class Engine(OFTEngine):
    def base(self,record):return record["inputs"],np.zeros(8,np.float32)
    def pixels(self,record,patch,seed,diverse=False,return_rgb=False):
        rng=np.random.default_rng(seed);gain=float(rng.uniform(.8,1)) if diverse else 1.
        rgb=F.interpolate(patch[None],size=(16,16),mode="bilinear",align_corners=False)*gain
        clean=record["inputs"]["pixel_values"]
        pixels=torch.cat([clean[:,:6],rgb,rgb],1)
        footprint=torch.zeros(256);footprint[:4]=1
        return (pixels,footprint,rgb) if return_rgb else (pixels,footprint)

@pytest.fixture
def setup():
    torch.manual_seed(74);torch.set_num_threads(1)
    engine=Engine(Wrapper());records=[]
    for i in range(2):
        image=np.random.default_rng(i).integers(0,256,(16,16,3),dtype=np.uint8)
        records.append({"uid":str(i),"inputs":{"pixel_values":torch.rand(1,12,16,16)},
                        "expert_actions":torch.randn(1,8,7).clamp(-1,1),
                        "observation":{"wrist_image":image}})
    return engine,records

def test_rng_preserved():
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    expected=(random.random(),np.random.rand(),torch.rand(3))
    random.seed(3);np.random.seed(3);torch.manual_seed(3)
    with monitor.preserve_rng():random.random();np.random.rand(10);torch.rand(12)
    actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])

@pytest.mark.parametrize("family",list(common.FAMILIES.values()))
@pytest.mark.parametrize("batch",[1,2])
def test_observers_preserve_actual_gradient_and_optimizer_step(setup,family,batch,monkeypatch):
    engine,records=setup;records=records[:batch];cfg=common.config();seeds=[101+i for i in range(batch)]
    refs=prepare_references(engine,records,family,cfg);center=torch.zeros_like(refs[0].get("clean",torch.zeros(1,256,3)))
    rgb=torch.rand(3,50,50,requires_grad=True);before=rgb.detach().clone()
    engine._training_sample_gradients=[]
    if family.startswith("advencoder_"):
        loss,_=generator_batch_backward(engine,records,refs,rgb,seeds,family,cfg)
    else:loss,_=pixel_batch_backward(engine,records,refs,rgb,seeds,family,cfg,center)
    expected_gradient=rgb.grad.detach().clone();parts=engine._training_sample_gradients or [expected_gradient]
    result=monitor.before_probe(engine,records,refs,before,seeds,family,cfg,center,expected_gradient,parts)
    result["actual_loss_before"]=loss
    assert torch.equal(rgb.grad,expected_gradient)
    expected_parameter=before.clone().requires_grad_(True);expected_parameter.grad=expected_gradient.clone()
    expected_opt=torch.optim.Adam([expected_parameter],lr=.002);expected_opt.step()
    opt=torch.optim.Adam([rgb],lr=.002);opt.step()
    assert torch.equal(rgb,expected_parameter)
    after=rgb.detach().clamp(0,1)
    assert len(result["rows"])==batch
    assert set(result["grad_uada"].shape)=={3,50}
    assert all(torch.isfinite(result[k]).all() for k in ["grad_uada","grad_tma"])
    rows=monitor.after_probe(engine,records,refs,before,after,seeds,cfg,result,"toy",25,49,expected_gradient)
    assert len(rows)==batch
    assert all("uada_same_input_loss_change" in r and "action_hidden_cosine" in r for r in rows)
    assert all(np.isfinite(r["method_same_input_loss_change"]) for r in rows)
    assert engine._diagnostic_observer is None
    assert all(len(b._forward_hooks)==0 for b in engine.dino.blocks)

def test_update_alignment_and_clipping_metrics():
    before=torch.tensor([.1,.9]);g=torch.tensor([1.,-1.]);proposal=torch.tensor([-.1,1.1]);after=proposal.clamp(0,1)
    r=monitor.scalar_row(1,0,1.,g,before,proposal,after,{})
    assert r["gradient_dot_actual_step"]<0 and r["descent_actual_step_cosine"]==pytest.approx(1.)
    assert r["clipped_channel_fraction"]==1 and r["clipping_removed_step_l2"]>0
    assert monitor.batch_cancellation([g,-g])==0

def test_schedule_budget_and_distinct_batch():
    assert len(common.METHODS)==16 and len(common.EVAL_TASKS)==40
    schedule=[[str(i)] for i in range(250)]
    for m in common.METHODS:
        ids=common.batch_uids(schedule,500,m)
        assert len(set(ids))==common.arm_info(m)["batch"]
    assigned=sum(common.worker_schedule(),[])
    assert sorted(assigned)==sorted(common.METHODS)
    assert common.config()["evaluation"]["state_indices"]==[40,41]

def test_formal_episode_inference_and_execution_unchanged():
    def essential(path):
        tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="episode")
        # Compare native prediction and simulator stepping expressions exactly.
        return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr in {"predict_action","step"}]
    assert essential(ROOT/"code/evaluate.py")==essential(common.PARENT/"code/evaluate.py")

def test_complete_grid_and_missing_data_not_silently_accepted():
    from analyze import summarize
    rows=[]
    for s in common.SUITES:
        for t in range(10):
            for state in (40,41):
                for c in common.deployment_conditions():
                    clean=c=="clean";rows.append(dict(suite=s,task_id=t,state_index=state,condition=c,success=int(clean),failure=int(not clean),rollout_seed=t*100+state))
    raw=pd.DataFrame(rows)
    a,b,c=summarize(raw,draws=25)
    assert len(raw)==1360 and len(a)==68 and b.casr_difference.notna().all()
    with pytest.raises(RuntimeError):summarize(raw.iloc[:-1],draws=25)

def test_same_state_replay_matches_formal_action_without_modifying_inputs(setup,tmp_path,monkeypatch):
    import replay
    from deployment import paste_png
    engine,_=setup;wrapper=engine.wrapper
    def prepare(obs,language):
        front=torch.as_tensor(obs["full_image"].copy()).permute(2,0,1)[None].float()/255
        wrist=torch.as_tensor(obs["wrist_image"].copy()).permute(2,0,1)[None].float()/255
        return {"pixel_values":torch.cat([front,front,wrist,wrist],1)}
    wrapper.prepare_hf_inputs=prepare
    wrapper.vla._unnormalize_actions=lambda x,key:np.asarray(x)
    class ReplayEngine(Engine):
        def base(self,record):return prepare(record["observation"],record["task_name"]),np.zeros(8,np.float32)
    engine=ReplayEngine(wrapper);monkeypatch.setattr(replay,"OFTEngine",lambda w:engine)
    rng=np.random.default_rng(45)
    obs={"full_image":rng.integers(0,256,(224,224,3),dtype=np.uint8),
         "wrist_image":rng.integers(0,256,(224,224,3),dtype=np.uint8),"state":np.zeros(8,np.float32)}
    original=obs["wrist_image"].copy();cases=[]
    for condition,rgb in [("clean",None),("toy@500",np.full((50,50,3),127,np.uint8))]:
        patched=obs if rgb is None else paste_png(obs,rgb)
        with torch.no_grad():a,_,_=engine.forward(prepare(patched,"test"),np.zeros(8))
        cases.append(dict(suite="libero_spatial",task_id=0,state_index=40,condition=condition,decision_block=0,
                          observation=obs,task_name="test",patch_rgb=rgb,formal_action_chunk=a.numpy()[0]))
    replay.diagnose(wrapper,cases,tmp_path,0)
    df=pd.read_parquet(tmp_path/"task_00_REPLAY.parquet")
    assert len(df)==2 and df.replay_environment_action_max_error.max()==0
    assert df.identity_action_max_error.max()==0
    assert np.array_equal(obs["wrist_image"],original)
