"""Replay order and passive-hook contracts, without MuJoCo, weights or GPUs."""
import hashlib
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
import atlas_capture as capture
from atlas_common import sha256_file
from atlas_features import Encoder


class QKV(nn.Module):
    def forward(self,x): return x.repeat(1,1,3)


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.attn=nn.Module(); self.attn.qkv=QKV()
        self.attn.num_heads=16; self.attn.q_norm=nn.Identity(); self.attn.k_norm=nn.Identity()
    def forward(self,x):
        self.attn.qkv(x)
        return x+1


class Dino(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=nn.ModuleList([Block() for _ in range(24)])
        self.num_prefix_tokens=5; self.scale=nn.Parameter(torch.ones(()))
    def forward(self,view):
        x=torch.ones(1,261,1024)*view*self.scale
        for i,b in enumerate(self.blocks):
            x=b(x)
            if i==22: consumed=x[:,5:]
        return consumed


class Vision(nn.Module):
    def __init__(self): super().__init__(); self.featurizer=Dino()
    def forward(self,inputs):
        return torch.cat([self.featurizer(v) for v in inputs.flatten()],dim=1)


def wrapper():
    vla=nn.Module(); vla.vision_backbone=Vision()
    def freeze():
        vla.eval()
        for p in vla.parameters(): p.requires_grad_(False)
    return SimpleNamespace(vla=vla,freeze_model=freeze,
        prepare_hf_inputs=lambda *args:{'pixel_values':torch.tensor([[1.,2.]])},
        predict_action=lambda *args:vla.vision_backbone(torch.tensor([[1.,2.]])).mean().reshape(1,1).detach().numpy())


def test_hooks_capture_all_layers_but_consume_22_without_changing_actions():
    torch.set_num_threads(2); w=wrapper(); engine=Encoder(w)
    features,dense,qa=engine.extract({},'instruction',validate=True)
    assert qa['native_hook_action_max_abs']==0 and not qa['block23_consumed_by_policy']
    assert len(features)==170 and len(dense)==8
    assert features['front__b22_spatial'].shape==(4*4*1024,)
    assert np.array_equal(dense['front__dense_b22'],dense['front__dense_consumed'])
    assert not np.array_equal(features['front__b23_spatial'],features['front__consumed'])
    assert not any(m._forward_hooks for m in w.vla.modules())


def test_hooks_are_removed_after_failed_capture():
    w=wrapper(); e=Encoder(w)
    with pytest.raises(RuntimeError):
        with e.hooks(): raise RuntimeError('fake renderer interruption')
    assert not any(m._forward_hooks for m in w.vla.modules())


def test_replay_uses_saved_actions_and_captures_before_action(monkeypatch,tmp_path):
    class Env:
        def __init__(self): self.state=0; self.actions=[]; self.env=self
        def seed(self,s): assert s==9
        def set_init_state(self,s): self.state=int(s[0]); return self.raw()
        def raw(self):
            return {'rgb':np.full((224,224,3),self.state,np.uint8),'proprio':np.zeros(8,np.float32)}
        def step(self,a):
            self.actions.append(a)
            if a[0]: self.state+=1
            return self.raw(),0,False,{}
        def get_sim_state(self): return np.array([self.state])
        def _check_success(self): return self.state==5
    env=Env(); closed=[]
    monkeypatch.setitem(sys.modules,'atlas_runtime',SimpleNamespace(bind_environment_context=lambda e:None,
        close_environment=lambda e:closed.append(e),renderer_metadata=lambda e:('test','test'),
        reset_environment=lambda e:e.raw()))
    monkeypatch.setattr(capture,'native_observation',lambda raw:{'full_image':raw['rgb'],'wrist_image':raw['rgb'],'state':raw['proprio']})
    monkeypatch.setattr(capture,'make_env',lambda b:env)
    monkeypatch.setattr(capture,'pose_bundle',lambda e:{'state':e.state})
    monkeypatch.setattr(capture,'progress',lambda *a,**k:None)
    class Sampler:
        entities=[]
        def __init__(self,*a): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def arm(self,tag): self.tag=tag
        def take(self,raw):
            assert self.tag==f's{env.state}'
            return {},{'contact':0.,'grasp':0.},{'state':env.state},{}
    monkeypatch.setattr(capture,'CameraLabels',Sampler)
    initial=tmp_path/'initial.npz'; trajectory=tmp_path/'actions.npz'
    np.savez(initial,sim_state=np.array([0])); np.savez(trajectory,executed_actions=np.tile([1.,0,0,0,0,0,0],(5,1)))
    episode={'initial_state_file':str(initial),'initial_state_file_sha256':sha256_file(initial),
             'trajectory_file':str(trajectory),'trajectory_file_sha256':sha256_file(trajectory),
             'episode_id':'test','rollout_seed':9}
    states=[]
    for step in range(5):
        p=tmp_path/f's{step}.npz'
        np.savez(p,front_rgb=np.full((224,224,3),step,np.uint8),wrist_rgb=np.full((224,224,3),step,np.uint8),
                 proprio=np.zeros(8,np.float32),sim_state=np.array([step]),language='test')
        states.append({'step_id':step,'state_id':f's{step}','episode_id':'test','task_id':0,'split':'discovery',
                       'state_file':str(p),'state_file_sha256':sha256_file(p),'normalized_progress':step/4})
    rows,summary=capture.replay_episode(episode,states,{'bddl_path':'test'},tmp_path,0,{})
    assert [r['step_id'] for r in rows]==[0,1,2,3,4]
    assert len(env.actions)==15 and summary['success'] and closed==[env]


@pytest.mark.parametrize('labels_fail',[False,True])
def test_no_vla_load_before_all_labels_are_valid(monkeypatch,labels_fail):
    events=[]
    monkeypatch.setenv('SLURM_JOB_ID','TEST_ONLY_NO_REAL_JOB')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'FAKE 4090')
    monkeypatch.setattr(capture,'prepared',lambda:{'signature':'abc'})
    monkeypatch.setattr(capture,'runtime_config',lambda:{})
    monkeypatch.setattr(capture,'progress',lambda *a,**k:None)
    monkeypatch.setattr(capture,'receipt_valid',lambda path,*a:path.name=='LABELS_COMPLETE.json')
    def labels(prep):
        events.append('labels')
        if labels_fail: raise RuntimeError('unaligned labels')
    monkeypatch.setattr(capture,'capture_labels',labels)
    monkeypatch.setattr(capture,'capture_features',lambda *a:events.append('features'))
    monkeypatch.setitem(sys.modules,'atlas_runtime',SimpleNamespace(add_paths=lambda *a:None,
        prepare_graphics_backend=lambda:None,load_wrapper=lambda *a:events.append('VLA') or object()))
    monkeypatch.setitem(sys.modules,'atlas_features',SimpleNamespace(Encoder=lambda w:w))
    if labels_fail:
        with pytest.raises(RuntimeError,match='unaligned'): capture.capture()
        assert events==['labels']
    else:
        capture.capture(); assert events==['labels','VLA','features']
