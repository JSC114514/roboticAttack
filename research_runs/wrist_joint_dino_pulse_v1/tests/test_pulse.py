"""CPU contracts exercise the real branch loop, boundaries, provenance and partial reports."""
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import dp_common as C
import dp_math as M


@pytest.mark.parametrize('contact,expected',[(1,0),(8,0),(9,8),(16,8),(45,40),(64,56)])
def test_contact_onset_strictly_precedes_first_contact(contact,expected):
    assert M.contact_onset(contact)==(expected,'ELIGIBLE')
    assert expected < contact and expected % 8 == 0


def test_absent_contact_is_retained_unavailable():
    assert M.contact_onset(None)==(None,'NO_TARGET_CONTACT')
    assert M.contact_onset(0)==(None,'CONTACT_AT_RESET')


def test_pulse_half_open_boundaries_and_budget():
    assert [M.pulse_active(i,8,8) for i in (7,8,15,16)]==[False,True,True,False]
    value=M.pulse_outcome(False,56,8,32,56)
    assert value['pulse_window_steps']==32 and value['post_removal_steps']==16
    assert value['remaining_budget_at_onset']==48 and value['remaining_budget_at_nominal_removal']==16
    assert value['removal_reached'] and value['terminal_phase']=='clean_continuation'
    early=M.pulse_outcome(True,12,8,32,56)
    assert not early['removal_reached'] and early['pulse_window_steps']==4 and early['post_removal_steps']==0


def test_fixed_design_scheduler_and_resources():
    import dp_cli as cli
    c=C.config(); rows=M.conditions(c)
    assert len(rows)==8 and c['state_index']==11 and sum(map(len,c['tasks'].values()))==16
    assert c['durations']==[8,32] and c['patch_xy']==[26,137] and not c['training']
    assert [r['name'] for r in cli.plans(2)]==['evaluate','report','videos']
    assert cli.plans(2)[0]['array']=='0-1%2'
    assert cli.plans(1)[1]['after']=='afterany' and cli.plans(1)[1]['parents']==['evaluate']
    assert cli.plans(2,True,True)==[dict(name='videos',launcher='cpu.sbatch',after='afterany',parents=[])]
    assert cli.plans(2,True,True,True)==[]
    with pytest.raises(RuntimeError):cli.plans(3)
    gpu=(C.ROOT/'slurm/evaluate.sbatch').read_text()
    assert all(x in gpu for x in ('--gres=gpu:4090:1','--mem=12G','--cpus-per-task=4','--partition=rtx4090'))
    cpu=(C.ROOT/'slurm/cpu.sbatch').read_text()
    assert '--gres' not in cpu and '--mem=4G' in cpu and '--cpus-per-task=2' in cpu


@pytest.fixture
def loop(tmp_path,monkeypatch):
    import dp_rollout as R
    from jr_rollout import assert_start
    cfg=copy.deepcopy(C.config());cfg['max_steps']['libero_goal']=56
    monkeypatch.setattr(C.common,'ROOT',tmp_path);monkeypatch.setattr(C.common,'signature',lambda:'test')
    monkeypatch.setattr(R,'ROOT',tmp_path);monkeypatch.setattr(R,'config',lambda:cfg)
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',
                        SimpleNamespace(process_libero_action=lambda action:action))
    class Env:
        def __init__(self):self.n=0;self.x=0.;self.success_at=56;self.closed=False;self.env=SimpleNamespace(control_timestep=.05)
        def get_sim_state(self):return np.array([self.n,self.x],float)
        def step(self,action):
            self.n+=1;self.x+=float(action[0])*.001
            return {},0,self.n>=self.success_at,{}
    def observation(env):
        return dict(full_image=np.full((224,224,3),env.n,np.uint8),
                    wrist_image=np.full((224,224,3),env.n,np.uint8),state=np.array([env.n,env.x],float))
    def fingerprint(env):return dict(time=env.n*.05,n=env.n,x=env.x)
    def snapshot(raw,samples,env,step):
        return dict(control_step=step,observation=observation(env),
                    sensors={'wrist':dict(frame_id=env.n,sensor_timestamp_s=env.n*.05)})
    @contextmanager
    def warm(env,initial,seed):yield {},deque([snapshot({},None,env,0)],maxlen=3),None
    class Writer:
        records=[]
        def __init__(self,path,fps):self.path=path;self.frames=[];Writer.records.append(self)
        def __enter__(self):return self
        def append(self,obs):self.frames.append(bool(np.any(obs['wrist_image']==255)))
        def __exit__(self,*args):
            for name in ('front.mp4','wrist.mp4'):(self.path/name).write_bytes(b'mock_video')
    utils=SimpleNamespace(switch_suite=lambda *args:None,fingerprint=fingerprint,assert_start=assert_start,CameraWriters=Writer)
    envs=[]
    def create(case):
        env=Env();env.success_at=case.get('mock_success_at',56);envs.append(env)
        return env,SimpleNamespace(language='task'),np.zeros(2)
    monkeypatch.setattr(R,'native',lambda:utils)
    monkeypatch.setattr(R,'runtime',lambda:SimpleNamespace(close_environment=lambda env:setattr(env,'closed',True)))
    monkeypatch.setattr(R,'create',create);monkeypatch.setattr(R,'warm_history',warm);monkeypatch.setattr(R,'snapshot',snapshot)
    def truth(env,target):
        return dict(contact=env.n>=12,grasp=env.n>=24,eef_xyz=[env.x,0.,0.],target_xyz=[.1,0.,0.],distance_m=.1-env.x)
    monkeypatch.setattr(R,'geometry',lambda:dict(anchor=lambda *args:None,truth=truth))
    class Vision:
        calls=[]
        def action(self,obs,language,mode='native',donor=None):
            step=int(obs['state'][0]);patched=bool(np.any(obs['wrist_image']==255))
            if donor is not None:assert donor['step']==step and donor['x']==obs['state'][1]
            value=.1 if not patched or mode=='both' else {'native':.2,'dino':.12,'siglip':.16}[mode]
            self.calls.append((step,patched,mode))
            return np.full((8,7),value,np.float32),dict(step=step,x=obs['state'][1])
        def metrics(self,*args):return dict(dino_native_delta_rms=1.,dino_used_delta_rms=0.,siglip_native_delta_rms=1.,siglip_used_delta_rms=0.)
    source=tmp_path/'source';source.mkdir()
    reference=Env();obs=observation(reference)
    np.savez_compressed(source/'START.npz',**R.start_arrays(reference,obs))
    C.atomic(dict(fingerprint=fingerprint(reference),rollout_seed=42),source/'START.json')
    action=np.full((8,7),.1,np.float32);np.save(source/'FIRST_ACTION.npy',action)
    pd.DataFrame([dict(step=i,controller_command=action[0].tolist()) for i in range(1,57)]).to_parquet(source/'STEPS.parquet',index=False)
    pd.DataFrame([dict(step_start=i,predicted_actions=action.tolist()) for i in range(0,56,8)]).to_parquet(source/'TRACE.parquet',index=False)
    case=dict(start='test',suite='libero_goal',task_id=0,state_index=11,rollout_seed=42,task_name='task',onset=8,
              eligibility='ELIGIBLE',clean_success=1,clean_executed_steps=56,
              reference_files={p.name:str(p) for p in source.iterdir()})
    wrapper=SimpleNamespace(vla=SimpleNamespace(norm_stats={'test':dict(action=dict(q01=[-1]*7,q99=[1]*7))}),gen_cfg=SimpleNamespace(unnorm_key='test'))
    return SimpleNamespace(R=R,case=case,wrapper=wrapper,vision=Vision(),patch=np.full((50,50,3),255,np.uint8),
                           monitor=SimpleNamespace(update=lambda **kwargs:None),envs=envs,writers=Writer.records,root=tmp_path)


@pytest.mark.parametrize('mode',['clean','png','dino','siglip'])
@pytest.mark.parametrize('duration',[8,32])
def test_native_loop_current_donors_pulse_removal_and_total_horizon(loop,mode,duration):
    x=loop;condition=dict(name=f'{mode}_{duration:02d}',mode=mode,duration=duration)
    x.R.episode(x.wrapper,x.vision,x.case,condition,x.patch,x.monitor,False)
    receipt=C.episode_receipt(x.case,condition['name']);assert C.complete(receipt)
    files=C.episode_files(receipt);meta=C.read(files['EPISODE.json']);trace=pd.read_parquet(files['TRACE.parquet'])
    assert meta['executed_steps']==56 and meta['remaining_budget_at_onset']==48
    assert meta['post_removal_steps']==56-8-duration and meta['removal_snapshot_saved']
    assert meta['actual_png_steps']==(0 if mode=='clean' else duration)
    active=trace[trace.png_applied]
    assert active.step_start.tolist()==([] if mode=='clean' else list(range(8,8+duration,8)))
    assert active.donor_control_step.tolist()==active.step_start.tolist()
    assert meta['both_restore_max_abs_error']==0 and all(e.closed for e in x.envs)
    if mode!='clean':
        assert x.writers[-1].frames[8:8+duration]==[True]*duration
        assert not any(x.writers[-1].frames[8+duration:])
    with np.load(files['ONSET.npz']) as onset, np.load(files['REMOVAL.npz']) as removal:
        assert onset['sim_state'][0]==8 and removal['sim_state'][0]==8+duration


def test_early_success_does_not_claim_clean_continuation_or_full_exposure(loop):
    x=loop;x.case['mock_success_at']=12
    row=dict(name='png_32',mode='png',duration=32)
    x.R.episode(x.wrapper,x.vision,x.case,row,x.patch,x.monitor,False)
    files=C.episode_files(C.episode_receipt(x.case,row['name']));meta=C.read(files['EPISODE.json'])
    assert meta['success']==1 and meta['actual_png_steps']==4 and not meta['removal_reached']
    assert not meta['removal_snapshot_saved'] and meta['post_removal_steps']==0
    assert 'REMOVAL.npz' not in files


def test_all_conditions_strictly_share_the_same_onset(loop):
    x=loop
    for mode in ('png','dino','siglip','clean'):
        row=dict(name=f'{mode}_08',mode=mode,duration=8)
        x.R.episode(x.wrapper,x.vision,x.case,row,x.patch,x.monitor,False)
    assert len(list((x.root/'paired_onsets').glob('*/START.npz')))==1
    assert all(e.closed for e in x.envs)


def test_onset_mismatch_stops_before_intervention_and_retains_failure(loop):
    x=loop;path=Path(x.case['reference_files']['TRACE.parquet']);trace=pd.read_parquet(path)
    trace.at[1,'predicted_actions']=[list(np.full(7,.9)) for _ in range(8)]
    trace.to_parquet(path,index=False)
    row=dict(name='png_08',mode='png',duration=8)
    with pytest.raises(RuntimeError,match='onset clean action differs'):
        x.R.episode(x.wrapper,x.vision,x.case,row,x.patch,x.monitor,False)
    assert list((x.root/'episodes').glob('*/*/attempt_*/FAILURE.json'))
    assert not C.episode_receipt(x.case,row['name']).exists() and all(e.closed for e in x.envs)


def test_empty_report_preserves_missing_and_unavailable(tmp_path,monkeypatch):
    import dp_report as report
    cfg=C.config();cfg['bootstrap_draws']=20
    cases=[dict(start='available',suite='libero_goal',task_id=0,eligibility='ELIGIBLE'),
           dict(start='absent',suite='libero_goal',task_id=1,eligibility='NO_TARGET_CONTACT')]
    m=dict(cases=cases,conditions=M.conditions(cfg),checkpoint=cfg['model_dir'],runnable=8,expected=16)
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'config',lambda:cfg)
    monkeypatch.setattr(report,'verify',lambda:m);monkeypatch.setattr(report,'complete',lambda path:False)
    status=report.run(SimpleNamespace(update=lambda **kwargs:None))
    assert status['status']=='PARTIAL' and status['completed']==0 and status['missing']==8 and status['unavailable']==8
    assert not (tmp_path/'results/COMPLETE.json').exists()
    rows=pd.read_csv(tmp_path/'results/SUMMARY.csv')
    assert rows.task_SR.isna().all() and rows.CASR.isna().all()


def test_partial_pair_point_and_interval_use_same_suite_weights(tmp_path,monkeypatch):
    import dp_report as report
    cfg=copy.deepcopy(C.config());cfg['bootstrap_draws']=30
    cases=[dict(start=f's{i}',suite='libero_goal' if i==0 else 'libero_object',task_id=i,eligibility='ELIGIBLE',
                rollout_seed=42,state_index=11) for i in range(5)]
    m=dict(cases=cases,conditions=M.conditions(cfg),checkpoint=cfg['model_dir'],runnable=40,expected=40)
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'config',lambda:cfg)
    monkeypatch.setattr(report,'verify',lambda:m)
    store={}
    for case in cases:
        for mode in ('clean','png','dino'):
            name=f'{mode}_08';key=(case['start'],name)
            folder=tmp_path/case['start']/name;folder.mkdir(parents=True)
            success=1 if mode=='clean' else int((mode=='dino')==(case['task_id']==0))
            C.atomic(dict(**case,condition=name,mode=mode,pulse_duration=8,checkpoint=cfg['model_dir'],
                success=success,removal_reached=True,post_removal_steps=16,actual_png_steps=0 if mode=='clean' else 8),folder/'EPISODE.json')
            pd.DataFrame([dict(pulse_active=True,executed_slots=8,executed_xyz_normalized_rmse=.1,
                executed_rotation_normalized_rmse=.1,dino_used_delta_rms=.1,siglip_used_delta_rms=.1)]).to_parquet(folder/'TRACE.parquet',index=False)
            store[key]={name:folder/name for name in ('EPISODE.json','TRACE.parquet')}
    monkeypatch.setattr(report,'episode_receipt',lambda case,name:(case['start'],name))
    monkeypatch.setattr(report,'complete',lambda receipt:receipt in store)
    monkeypatch.setattr(report,'episode_files',lambda receipt:store[receipt])
    report.run(SimpleNamespace(update=lambda **kwargs:None))
    effects=pd.read_csv(tmp_path/'results/DINO_EFFECTS.csv')
    row=effects[(effects.suite=='ALL_SUITES')&(effects.restoration=='dino')&(effects.pulse_duration==8)].iloc[0]
    assert row.SR_difference==pytest.approx(0.) and row.SR_difference_pooled==pytest.approx(-.6)
    assert row.difference_CI95=='[0.0, 0.0]' and row.suites_with_pairs==2


@pytest.mark.parametrize('mode,expected',[('native',126.),('dino',90.),('siglip',54.),('both',18.)])
def test_consumed_block22_restore_changes_only_requested_wrist_branch(mode,expected):
    import torch
    from fr_vision import Vision
    class Dino(torch.nn.Module):
        def __init__(self):super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
        def forward(self,x):return self.blocks[22](x)[:,5:]
    class Wrapper:
        def __init__(self):
            self.vla=torch.nn.Module();self.vla.vision_backbone=torch.nn.Module()
            self.vla.vision_backbone.featurizer=Dino();self.vla.vision_backbone.fused_featurizer=torch.nn.Identity()
        def predict_action(self,obs,language):
            value=0.
            for i,key in enumerate(('full_image','wrist_image')):
                n=float(obs[key].mean())
                d=self.vla.vision_backbone.featurizer(torch.full((1,261,3),n))
                s=self.vla.vision_backbone.fused_featurizer(torch.full((1,256,2),n*2+1))
                value+=(i+1)*(float(d.mean())+float(s.mean()))
            return np.full((8,7),value,np.float32)
    vision=Vision(Wrapper());clean=dict(full_image=np.ones((224,224,3),np.uint8),wrist_image=np.full((224,224,3),2,np.uint8))
    _,donor=vision.action(clean,'task');patched={**clean,'wrist_image':np.full((224,224,3),20,np.uint8)}
    action,features=vision.action(patched,'task',mode,donor)
    np.testing.assert_array_equal(action,np.full((8,7),expected,np.float32))
    assert torch.equal(features['dino_front'],donor['dino_front']) and torch.equal(features['siglip_front'],donor['siglip_front'])
    assert not vision.dino._forward_hooks and not vision.siglip._forward_hooks and not vision.dino.blocks[22]._forward_hooks


def test_native_camera_mp4_and_ide_webm_full_decode(tmp_path):
    from jr_rollout import CameraWriters
    from dp_common import module,PROJECT
    exporter=module('pulse_test_video_exporter',PROJECT/'research_tools/export_singleview_videos.py')
    source=tmp_path/'source';source.mkdir()
    with CameraWriters(source,20) as writer:
        for i in range(12):
            writer.append(dict(full_image=np.full((224,224,3),i*8,np.uint8),
                               wrist_image=np.full((224,224,3),255-i*8,np.uint8)))
    for view in ('front','wrist'):
        for extension in ('.mp4','.webm'):
            result=exporter.encode(source/(view+'.mp4'),tmp_path/(view+extension))
            assert result['decoded_complete'] and tuple(result['metadata']['size'])==(224,224)
            assert result['metadata']['fps']==20 and result['metadata']['duration']==pytest.approx(.6,abs=.05)
