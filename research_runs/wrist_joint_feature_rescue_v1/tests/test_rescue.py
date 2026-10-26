"""CPU target selection, oracle hook scope, paired accounting and native-loop mocks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import fr_common as C
import fr_math as M
from fr_vision import Vision

def observation(wrist=2,front=1):
    return dict(full_image=np.full((224,224,3),front,np.uint8),wrist_image=np.full((224,224,3),wrist,np.uint8),state=np.zeros(8))

class Dino(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
    def forward(self,x):return self.blocks[22](x)[:,5:]

class FakeWrapper:
    def __init__(self):
        self.vla=torch.nn.Module();self.vla.vision_backbone=torch.nn.Module()
        self.vla.vision_backbone.featurizer=Dino();self.vla.vision_backbone.fused_featurizer=torch.nn.Identity()
    def predict_action(self,obs,instruction):
        out=0.
        for index,key in enumerate(('full_image','wrist_image')):
            n=float(obs[key].mean());d=self.vla.vision_backbone.featurizer(torch.full((1,261,3),n))
            s=self.vla.vision_backbone.fused_featurizer(torch.full((1,256,2),n*2+1))
            out+=(index+1)*(float(d.mean())+float(s.mean()))
        return np.full((8,7),out,np.float32)

@pytest.mark.parametrize('mode,selected',[('native',()),('dino',('dino',)),('siglip',('siglip',)),('both',('dino','siglip'))])
def test_only_requested_wrist_branch_is_restored(mode,selected):
    v=Vision(FakeWrapper());clean,donor=v.action(observation(), 'task')
    action,capture=v.action(observation(wrist=20),'task',mode,donor)
    assert torch.equal(capture['dino_front'],donor['dino_front'])
    assert torch.equal(capture['siglip_front'],donor['siglip_front'])
    metrics=v.metrics(capture,donor,mode)
    for name in ('dino','siglip'):
        assert metrics[name+'_native_delta_rms']>0
        assert (metrics[name+'_used_delta_rms']==0)==(name in selected)
    if mode=='both':np.testing.assert_array_equal(action,clean)
    else:assert not np.array_equal(action,clean)

def test_zero_donor_identity_and_exception_cleanup():
    w=FakeWrapper();v=Vision(w);action,features=v.action(observation(),'task')
    restored,_=v.action(observation(),'task','both',features)
    np.testing.assert_array_equal(action,restored)
    with pytest.raises(ValueError,match='deliberate'):
        with v.hooks():raise ValueError('deliberate')
    assert not v.dino._forward_hooks and not v.siglip._forward_hooks and not v.dino.blocks[22]._forward_hooks

def test_front_mismatch_and_bad_view_count_rejected():
    w=FakeWrapper();v=Vision(w);_,donor=v.action(observation(),'task')
    with pytest.raises(RuntimeError,match='Front features'):
        v.action(observation(front=3),'task','both',donor)
    with pytest.raises(RuntimeError,match='exactly once'):
        with v.hooks():pass
    assert not v.dino._forward_hooks and not v.dino.blocks[22]._forward_hooks

def test_donor_is_required_and_source_not_mutated():
    obs=observation();before={k:v.copy() for k,v in obs.items()};v=Vision(FakeWrapper())
    with pytest.raises(RuntimeError,match='requires'):v.action(obs,'task','dino')
    v.action(obs,'task')
    assert all(np.array_equal(obs[k],before[k]) for k in before)

def test_rescue_denominators_missing_and_new_harm():
    value=M.paired_outcomes({'a':1,'b':1,'c':0,'missing':1},{'a':0,'b':1,'c':0},{'a':1,'b':0,'c':1})
    assert value['pairs']==3 and value['clean_success_pairs']==2
    assert value['patch_induced_failures']==1 and value['rescued_induced_failures']==1
    assert value['raw_failures_rescued']==2 and value['raw_successes_lost']==1
    assert value['rescue_fraction']==1 and value['SR_difference']==pytest.approx(1/3)
    assert M.paired_outcomes({}, {}, {})['rescue_fraction'] is None

def test_fixed_budget_source_model_and_resources():
    import fr_cli as cli
    c=C.config();rows=M.conditions(c)
    assert len(rows)==9 and c['screen_states']==[10] and c['confirmation_states']==[11]
    assert sum(map(len,c['tasks'].values()))==16 and 16*len(rows)==144
    assert [r['name'] for r in rows]==c['conditions']
    assert [r['name'] for r in cli.plans(2,True)]==['evaluate','report','videos']
    assert cli.plans(2,True)[0]['array']=='0-1%2'
    with pytest.raises(RuntimeError):cli.plans(3)
    for name in ('clean','evaluate'):
        s=(C.ROOT/'slurm'/(name+'.sbatch')).read_text()
        assert '--gres=gpu:4090:1' in s and '--mem=12G' in s and '--cpus-per-task=4' in s
    assert '--gres' not in (C.ROOT/'slurm/cpu.sbatch').read_text()

def test_source_compatibility_fails_on_seed_or_model():
    from fr_prepare import compatible
    c=C.config();case=dict(suite='libero_goal',task_id=0,state_index=10,rollout_seed=123,start='test',task_name='test')
    meta={**case,'checkpoint':c['model_dir'],'revision':c['revision'],'condition':'clean','num_open_loop_steps':8,'max_steps':300}
    compatible(meta,case,c)
    with pytest.raises(RuntimeError):compatible({**meta,'rollout_seed':124},case,c)
    with pytest.raises(RuntimeError):compatible({**meta,'checkpoint':'other'},case,c)

def test_bootstrap_uses_task_start_units():
    rows=[dict(suite=s,task_id=t,value=1) for s in ('a','b') for t in range(3)]
    assert M.interval(rows,50)==[1.,1.]
    assert M.interval([]) is None

def test_all_four_block22_hooks_restore_after_model_exception():
    w=FakeWrapper();v=Vision(w)
    def broken(*_):raise ValueError('policy failed')
    w.predict_action=broken
    with pytest.raises(ValueError,match='policy failed'):v.action(observation(),'task')
    assert not v.dino._forward_hooks and not v.siglip._forward_hooks and not v.dino.blocks[22]._forward_hooks

def test_completed_receipt_protects_copied_asset(tmp_path,monkeypatch):
    monkeypatch.setattr(C.common,'ROOT',tmp_path);monkeypatch.setattr(C.common,'signature',lambda:'test')
    f=tmp_path/'asset.json';f.write_text('{}');receipt=tmp_path/'COMPLETE.json'
    C.finish(receipt,[f]);assert C.complete(receipt)
    f.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='Changed artifact'):C.complete(receipt)

def test_partial_report_retains_missing_not_success(tmp_path,monkeypatch):
    import fr_report as report
    cfg=C.config();cfg['bootstrap_draws']=20
    m=dict(cases=[dict(suite='libero_goal',task_id=0,start='s')],conditions=M.conditions(cfg),checkpoint=cfg['model_dir'])
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'config',lambda:cfg)
    monkeypatch.setattr(report,'verify',lambda:m);monkeypatch.setattr(report,'complete',lambda p:False)
    status=report.run(SimpleNamespace(update=lambda **kwargs:None))
    assert status['status']=='PARTIAL' and status['completed']==0 and status['missing']==144
    assert (tmp_path/'results/RESULTS.md').is_file() and not (tmp_path/'results/COMPLETE.json').exists()

def test_full_closedloop_mock_uses_current_donor_and_clean_pair(tmp_path,monkeypatch):
    import json
    from collections import deque
    from contextlib import contextmanager
    import pandas as pd
    import fr_rollout as R
    from jr_rollout import assert_start
    cfg=C.config();cfg['max_steps']['libero_goal']=10
    monkeypatch.setattr(C.common,'ROOT',tmp_path);monkeypatch.setattr(C.common,'signature',lambda:'test')
    monkeypatch.setattr(R,'ROOT',tmp_path);monkeypatch.setattr(R,'config',lambda:cfg)
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',SimpleNamespace(process_libero_action=lambda a:a))
    class Env:
        def __init__(self):self.n=0;self.env=SimpleNamespace(control_timestep=.05)
        def get_sim_state(self):return np.array([self.n],float)
        def step(self,a):self.n+=1;return {},0,self.n>=10,{}
    def obs(env):
        value=observation(wrist=env.n,front=env.n);value['state']=np.full(8,env.n,float);return value
    def fp(env):return dict(time=env.n*.05,n=env.n)
    def snap(raw,samples,env,step):
        return dict(control_step=step,observation=obs(env),sensors={'wrist':dict(frame_id=env.n,sensor_timestamp_s=env.n*.05)})
    @contextmanager
    def warm(env,initial,seed):yield {},deque([snap({},None,env,0)],maxlen=3),None
    def reference(case):
        e=Env();return R.start_arrays(e,obs(e)),dict(fingerprint=fp(e),rollout_seed=case['rollout_seed'])
    class Writer:
        def __init__(self,path,fps):self.path=path
        def __enter__(self):return self
        def append(self,obs):pass
        def __exit__(self,*args):
            for n in ('front.mp4','wrist.mp4'):(self.path/n).write_bytes(b'mock_video')
    utils=SimpleNamespace(switch_suite=lambda *a:None,fingerprint=fp,assert_start=assert_start,CameraWriters=Writer)
    monkeypatch.setattr(R,'native',lambda:utils)
    monkeypatch.setattr(R,'runtime',lambda:SimpleNamespace(close_environment=lambda e:None))
    monkeypatch.setattr(R,'create',lambda case:(Env(),SimpleNamespace(language='task'),np.zeros(1)))
    monkeypatch.setattr(R,'warm_history',warm);monkeypatch.setattr(R,'snapshot',snap);monkeypatch.setattr(R,'native_start_reference',reference)
    def truth(env,target):return dict(contact=env.n>=1,grasp=env.n>=2,eef_xyz=[env.n*.001,0,0],target_xyz=[.1,0,0],distance_m=.1-env.n*.001)
    monkeypatch.setattr(R,'geometry',lambda:dict(anchor=lambda *a:None,truth=truth))
    class MockVision:
        def action(self,observation,language,mode='native',donor=None):
            n=observation['state'][0]
            if donor is not None:assert donor['donor_step']==n
            out=np.full((8,7),n*.01,np.float32)
            return out,dict(donor_step=n)
        def metrics(self,*args):return dict(dino_native_delta_rms=1,dino_used_delta_rms=0,siglip_native_delta_rms=1,siglip_used_delta_rms=0)
    wrapper=SimpleNamespace(vla=SimpleNamespace(norm_stats={'test':dict(action=dict(q01=[-1]*7,q99=[1]*7))}),gen_cfg=SimpleNamespace(unnorm_key='test'))
    monitor=SimpleNamespace(update=lambda **kwargs:None)
    case=dict(suite='libero_goal',task_id=0,state_index=10,rollout_seed=42,start='test',task_name='task')
    vision=MockVision()
    R.episode(wrapper,vision,case,dict(name='clean',mode='native',patch=None,sha256=None),None,monitor,False)
    (tmp_path/'data').mkdir();(tmp_path/'data/REUSE_PLAN.json').write_text(json.dumps(dict(source_signature='test_source')))
    R.episode(wrapper,vision,case,dict(name='test__both',mode='both',patch='test',sha256='hash'),np.full((50,50,3),255,np.uint8),monitor,False)
    receipt=C.episode_receipt(case,'test__both');assert C.complete(receipt)
    files=C.episode_files(receipt);meta=C.read(files['EPISODE.json']);trace=pd.read_parquet(files['TRACE.parquet'])
    assert meta['success']==1 and meta['executed_steps']==10 and meta['first_target_contact_step']==1
    assert meta['oracle_feature_restoration'] and not meta['native_PNG_only']
    assert trace.donor_control_step.tolist()==[0,8] and trace.donor_wrist_frame_id.tolist()==[0,8]
    assert trace.executed_slots.tolist()==[8,2] and trace.both_restore_max_abs_error.max()==0
