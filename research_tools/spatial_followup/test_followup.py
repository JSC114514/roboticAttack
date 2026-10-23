"""CPU checks for scientific matching, planning, real encoder and episode IO."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('FOLLOWUP_ROOT',str(Path(__file__).resolve().parents[2]/'research_runs/wrist_spatial_relation_expand_v1'))
from sf_math_targets import rotation_between, matched_targets, donor_map


@pytest.mark.parametrize('a,b', [([1,0,0],[1,0,0]),([1,0,0],[-1,0,0]),([1,0,0],[0,0,1])])
def test_rotation_preserves_geometry(a,b):
    r=rotation_between(a,b)
    assert np.allclose(r@a,b) and np.allclose(r.T@r,np.eye(3))
    assert np.isclose(np.linalg.det(r),1)


def test_random_matched_lengths_and_native_box():
    rng=np.random.default_rng(42)
    for _ in range(120):
        clean=rng.uniform(-.05,.05,(1,8,3));target=rng.uniform(-.05,.05,(1,8,3))
        a=rng.normal(size=3);a/=np.linalg.norm(a);b=rng.normal(size=3);b/=np.linalg.norm(b)
        correct,shuffled,alpha=matched_targets(clean,target,a,b)
        assert 0<=alpha<=1 and abs(correct).max()<=.05000001 and abs(shuffled).max()<=.05000001
        assert np.allclose(np.linalg.norm(correct-clean,axis=-1),np.linalg.norm(shuffled-clean,axis=-1),atol=7e-9)


def test_saturated_degeneracy_is_explicit_and_never_clipped_differently():
    c=np.full((1,8,3),.05);t=c.copy();t[...,0]=.04
    a,b,alpha=matched_targets(c,t,[1,0,0],[-1,0,0])
    assert alpha==0 and np.allclose(a,c) and np.allclose(b,c)
    with pytest.raises(ValueError):matched_targets(c+.01,t,[1,0,0],[0,1,0])


def test_donors_are_reproducible_cross_task_and_trajectory():
    rows=[{'suite':suite,'task_id':t,'episode_id':f'{suite}_{t}_{e}','frame_rank':f,'state_id':f'{suite}_{t}_{e}_{f}'}
          for suite in ['a','b'] for t in range(6) for e in range(2) for f in range(3)]
    first=donor_map(rows);second=donor_map(rows)
    for row in rows:
        donor=first[row['state_id']]
        assert donor==second[row['state_id']]
        assert donor['suite']==row['suite'] and donor['task_id']!=row['task_id'] and donor['episode_id']!=row['episode_id']
        assert donor['frame_rank']==row['frame_rank']


def test_dependency_plan_and_two_gpu_limit(monkeypatch,tmp_path):
    import sf_cli as cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'complete',lambda p:False)
    monkeypatch.setattr(cli,'read',lambda p:{'cases':[{'start':'s0'},{'start':'s1'}]})
    monkeypatch.setattr(cli,'config',lambda:{'scope':'control','seeds':[42,43]})
    monkeypatch.setattr(cli,'candidates',lambda:[(f'{m}{s}',m,s) for s in [42,43] for m in ['a','b']])
    calls=[]
    def one(stage,deps=(),after='afterok'):
        calls.append((stage,list(deps),after));return stage+'_JOB'
    cli.stages(one,['BASE_REPORT_JOB'])
    assert calls[:2]==[('train0',['BASE_REPORT_JOB'],'afterok'),('train1',['BASE_REPORT_JOB'],'afterok')]
    assert calls[2]==('select',['train0_JOB','train1_JOB'],'afterok')
    assert calls[3:5]==[('evaluate0',['select_JOB'],'afterok'),('evaluate1',['select_JOB'],'afterok')]
    assert calls[-1]==('report',['evaluate0_JOB','evaluate1_JOB'],'afterany')


def test_completed_episodes_do_not_request_another_gpu(monkeypatch,tmp_path):
    import sf_cli as cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'config',lambda:{'scope':'expansion'})
    monkeypatch.setattr(cli,'read',lambda p:{'cases':[{'start':'s0'},{'start':'s1'}],'condition_order':['clean','patch']})
    sealed=set()
    monkeypatch.setattr(cli,'complete',lambda p:'/episodes/' in str(p) or str(p) in sealed)
    monkeypatch.setattr(cli,'finish',lambda p,files:sealed.add(str(p)))
    calls=[]
    cli.stages(lambda stage,deps=(),after='afterok':calls.append((stage,deps)) or stage)
    assert calls==[('report',[])]


def test_slurm_gpu_resources_and_cpu_stages():
    from sf_common import PROJECT
    for name in ['wrist_spatial_relation_expand_v1','wrist_spatial_relation_shuffle_v1']:
        for path in (PROJECT/'research_runs'/name/'slurm').glob('*.sbatch'):
            text=path.read_text();gpu=path.stem.startswith(('train','evaluate'))
            if gpu:
                assert '#SBATCH --gres=gpu:4090:1' in text and '#SBATCH --cpus-per-task=4' in text
                assert ('#SBATCH --mem=16G' if path.stem.startswith('train') else '#SBATCH --mem=12G') in text
            else:assert '--gres' not in text and '#SBATCH --mem=4G' in text


def test_native_episode_contract_without_simulator(monkeypatch,tmp_path):
    import sf_evaluate as e
    class Env:
        def __init__(self):self.env=NS(control_timestep=.05);self.step_count=0
        def step(self,action):
            self.step_count+=1
            return {'robot0_eef_pos':np.array([self.step_count*.001,0,0])},0,self.step_count==2,{}
    env=Env()
    obs={'full_image':np.full((224,224,3),80,np.uint8),'wrist_image':np.full((224,224,3),90,np.uint8),'state':np.zeros(8)}
    runtime=NS(reset_wait=lambda *args:{'robot0_eef_pos':np.zeros(3)},observation=lambda *args:obs)
    monkeypatch.setattr(e.native,'runtime_module',lambda:runtime)
    monkeypatch.setattr(e,'ROOT',tmp_path)
    monkeypatch.setattr(e,'paired_start',lambda *a,**kw:{'sim_state':np.zeros(3)})
    monkeypatch.setattr(e,'anchor',lambda *a:{'entity':'fixed_target'})
    monkeypatch.setattr(e,'controller',lambda *a:None)
    monkeypatch.setattr(e,'truth',lambda *a:{'target_xyz':[.1,0,0],'eef_xyz':[env.step_count*.001,0,0],
                                         'distance_m':.1-env.step_count*.001,'contact':False,'grasp':False})
    class Capture:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def arm(self,key):self.key=key
        def take(self,*a):return {'mask':np.ones((256,256),bool),'sample_time_s':env.step_count*.05}
        def __exit__(self,*a):pass
    class Logger(Capture):
        def arm(self):pass
        def take(self):return {'tracking_error_peak_m':0.,'peak_robot_contact_force_n':0.}
    monkeypatch.setattr(e,'SensorCapture',Capture);monkeypatch.setattr(e,'PhysicsLogger',Logger)
    attempt=tmp_path/'attempt';attempt.mkdir()
    wrapper=NS(predict_action=lambda *a:np.zeros((8,7),np.float32))
    case={'start':'libero_goal_t00_s02','suite':'libero_goal','task_id':0,'rollout_seed':1}
    e.episode(env,NS(language='test'),np.zeros(1),case,'clean',None,wrapper,attempt,NS(update=lambda **k:None))
    result=json.loads((attempt/'EPISODE.json').read_text())
    assert result['success']==1 and result['executed_steps']==2
    assert len(pd.read_parquet(attempt/'STEPS.parquet'))==2
    assert pd.read_parquet(attempt/'TRACE.parquet').iloc[0].executed_slots==2
    import imageio.v2 as imageio
    with imageio.get_reader(attempt/'video.mp4',ffmpeg_params=['-threads','1']) as reader:
        assert len(list(reader.iter_data()))==3


def test_train_adapter_uses_new_outputs_without_mutating_source(monkeypatch,tmp_path):
    import sf_common as c
    source_root=c.source.ROOT
    monkeypatch.setattr(c,'ROOT',tmp_path)
    module=c.training_module()
    assert module.ROOT==tmp_path and module.complete is c.complete
    assert c.source.ROOT==source_root and module.__name__!='sr_train'


def test_pairing_rejects_changed_physics_or_images(monkeypatch,tmp_path):
    import sf_evaluate as e
    images={'full_image':np.zeros((2,2,3),np.uint8),'wrist_image':np.zeros((2,2,3),np.uint8),'state':np.zeros(8)}
    env=NS(get_sim_state=lambda:np.zeros(3));runtime=NS(observation=lambda *a:images)
    monkeypatch.setattr(e.native,'runtime_module',lambda:runtime);monkeypatch.setattr(e,'fingerprint',lambda env:{'time':0})
    e.paired_start(env,{},1,tmp_path/'paired')
    images['wrist_image'][0,0]=1
    with pytest.raises(RuntimeError,match='array/RGB'):e.paired_start(env,{},1,tmp_path/'paired')


def test_report_preserves_clean_failures_and_missing_pairs(monkeypatch,tmp_path):
    import sf_report as r
    from contextlib import nullcontext
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'lock',lambda name:nullcontext())
    monkeypatch.setattr(r,'config',lambda:{'scope':'expansion'})
    monkeypatch.setattr(r,'conditions',lambda:{'clean':None,'patch':'x.png'})
    monkeypatch.setattr(r,'complete',lambda path:False)
    (tmp_path/'results').mkdir();(tmp_path/'data').mkdir()
    cases=[{'start':f's{i}','suite':'libero_goal','task_id':i,'state_index':2} for i in range(3)]
    (tmp_path/'data/MANIFEST.json').write_text(json.dumps({'cases':cases}))
    mapping={}
    for i in range(2):
        p=tmp_path/f'clean{i}';p.mkdir()
        (p/'EPISODE.json').write_text(json.dumps({'success':i,'executed_steps':2,'events':{}}))
        mapping[f's{i}']={'clean':p}
    mapping['s2']={}
    monkeypatch.setattr(r,'lookup',lambda case:mapping[case['start']])
    r.run(NS(update=lambda **kw:None))
    status=json.loads((tmp_path/'results/STATUS.json').read_text())
    assert status['status']=='PARTIAL' and status['clean_successes']==1 and status['clean_starts_complete']==2
    patch=next(row for row in status['summary'] if row['condition']=='patch')
    assert patch['paired_completed']==0 and patch['CASR'] is None


def test_comparison_holds_ended_frames_and_uses_common_clock(tmp_path):
    import imageio.v2 as imageio
    from sf_report import compose
    paths={}
    for index,name in enumerate(['clean','patch']):
        p=tmp_path/name;p.mkdir();paths[name]=p
        (p/'EPISODE.json').write_text(json.dumps({'fps':20,'executed_steps':index+1,'success':1-index}))
        with imageio.get_writer(p/'video.mp4',fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as writer:
            for _ in range(index+2):writer.append_data(np.full((288,448,3),100,np.uint8))
    destination=tmp_path/'output';destination.mkdir()
    compose({'start':'unit'},paths,destination)
    with imageio.get_reader(destination/'comparison.mp4',ffmpeg_params=['-threads','1']) as reader:
        frames=list(reader.iter_data())
    assert len(frames)==3 and frames[-1].shape==(320,1792,3)
    assert frames[-1][100:200,20:200].mean()>90
