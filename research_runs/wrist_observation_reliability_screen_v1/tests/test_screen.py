"""CPU contracts; no model or simulator imports on login nodes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import sc_common as C
import sc_math as M
import sc_cli as CLI


def history():
    return [dict(control_step=k,time_s=k*.05,observation=dict(full_image=np.full((224,224,3),k+10,np.uint8),
        wrist_image=np.full((224,224,3),k+40,np.uint8),state=np.arange(8)+k)) for k in (18,19,20)]


@pytest.mark.parametrize('view',['front','wrist'])
@pytest.mark.parametrize('lag',[1,2])
def test_real_control_steps_not_decision_blocks(view,lag):
    h=history();out,source,active=M.deploy(h,f'{view}_delay{lag}',20,[])
    key=M.KEYS[view];other=M.KEYS['wrist' if view=='front' else 'front']
    assert source['control_step']==20-lag and np.array_equal(out[key],h[-1-lag]['observation'][key])
    assert np.array_equal(out[other],h[-1]['observation'][other]) and np.array_equal(out['state'],h[-1]['observation']['state'])
    out[key][0,0]=0;assert h[-1-lag]['observation'][key][0,0,0]>0 and active


def test_no_synthetic_frames_or_missing_history():
    with pytest.raises(RuntimeError):M.deploy(history()[-1:],'wrist_delay2',20,[])
    h=history();out,_,active=M.deploy(h,'clean',20,[])
    assert not active and all(np.array_equal(out[k],h[-1]['observation'][k]) for k in out)


def test_paired_occlusion_schedule_and_actual_area():
    onset=M.schedule(160);assert onset==[32,80,128]
    assert M.active(32,onset) and M.active(39,onset) and not M.active(40,onset)
    for view in ('front','wrist'):
        out,_,active=M.deploy(history(),view+'_occlusion',32,onset)
        key=M.KEYS[view];assert active
        assert np.all(out[key][84:140,84:140]==128)
        assert np.count_nonzero(out[key]!=history()[-1]['observation'][key])==56*56*3
        out,_,active=M.deploy(history(),view+'_occlusion',40,onset)
        assert not active and np.array_equal(out[key],history()[-1]['observation'][key])


def test_pose_and_suite_normalized_action_metrics():
    a=dict(xyz=[.02,0,0],rotation=np.eye(3).ravel().tolist());b=dict(xyz=[0,0,0],rotation=np.eye(3).ravel().tolist())
    shift,rot=M.pose_error(a,b);assert shift==.02 and rot==0
    clean=np.zeros((8,7));bad=clean.copy();bad[7]=1;stats=dict(q01=[-1]*7,q99=[1]*7,mask=[True]*6+[False])
    value=M.action_difference(bad,clean,stats,2)
    assert value['executed_xyz_normalized_rmse']==0 and value['all8_xyz_normalized_rmse']>0


def test_missing_and_clean_failures_are_distinct():
    value=M.paired_counts({'a':True,'b':False,'c':True},{'a':False,'b':True})
    assert value==dict(pairs=2,new_failures=1,rescued=1,SR_difference=0)


def test_budget_model_selection_resources_and_no_training():
    c=C.config();assert sum(map(len,c['tasks'].values()))==16 and c['screen_states']==[10,11]
    assert len(c['conditions'])*16*2==224 and c['max_workers']==2 and 'spatial-object-goal-10' in c['model_dir']
    plan=CLI.plans(2);assert [v['name'] for v in plan]==['clean','evaluate','report','videos']
    assert plan[1]['parents']==['clean'] and plan[1]['array']=='0-1%2'
    with pytest.raises(RuntimeError):CLI.plans(3)
    assert CLI.plans(2,True,True,True,True)==[]
    for name in ('clean','evaluate'):
        text=(C.ROOT/'slurm'/(name+'.sbatch')).read_text()
        assert '--gres=gpu:4090:1' in text and '--mem=12G' in text and '--cpus-per-task=4' in text
    assert '--gres' not in (C.ROOT/'slurm/cpu.sbatch').read_text()


def test_blocked_bootstrap_and_no_frame_pseudoreplication():
    from sc_report import interval,rank_correlation
    rows=[dict(suite=s,task_id=t,value=-1) for s in ('a','b') for t in range(4) for _ in range(2)]
    assert interval(rows,100)==[-1,-1]
    assert rank_correlation([0,0,0],[1,2,3]) is None


def test_passive_samples_restore_hooks_after_error():
    from sc_rollout import Samples
    cameras={k:SimpleNamespace(_sensor=lambda cache:np.full((4,4,3),20,np.uint8)) for k in ('agentview_image','robot0_eye_in_hand_image')}
    originals={k:v._sensor for k,v in cameras.items()}
    env=SimpleNamespace(env=SimpleNamespace(_observables=cameras),sim=SimpleNamespace(
        model=SimpleNamespace(camera_name2id=lambda n:0 if n=='agentview' else 1),
        data=SimpleNamespace(time=1.,cam_xpos=np.zeros((2,3)),cam_xmat=np.tile(np.eye(3).ravel(),(2,1)))))
    with pytest.raises(ValueError):
        with Samples(env) as samples:
            raw={k:v._sensor({}) for k,v in cameras.items()};values=samples.take(raw)
            assert values['front']['sensor_timestamp_s']==1 and values['wrist']['frame_id']==1
            raw['agentview_image']=np.zeros((4,4,3),np.uint8)
            with pytest.raises(RuntimeError,match='timestamp mismatch'):samples.take(raw)
            raise ValueError('test hook restoration')
    assert all(cameras[k]._sensor is originals[k] for k in cameras)


def test_full_episode_integration_without_simulator(tmp_path,monkeypatch):
    import pandas as pd
    import sc_rollout as R
    cfg=C.config();monkeypatch.setattr(C.common,'ROOT',tmp_path);monkeypatch.setattr(C.common,'signature',lambda:'test')
    monkeypatch.setattr(R,'ROOT',tmp_path);monkeypatch.setattr(R,'config',lambda:cfg)
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',SimpleNamespace(
        process_libero_action=lambda a:a,get_libero_dummy_action=lambda:np.zeros(7)))
    class Env:
        def __init__(self):
            self.n=0;self.env=SimpleNamespace(control_timestep=.05,_observables={})
            self.sim=SimpleNamespace(model=SimpleNamespace(camera_name2id=lambda n:0 if n=='agentview' else 1),
                data=SimpleNamespace(time=0.,cam_xpos=np.zeros((2,3)),cam_xmat=np.tile(np.eye(3).ravel(),(2,1))))
            for i,name in enumerate(('agentview_image','robot0_eye_in_hand_image')):
                self.env._observables[name]=SimpleNamespace(_sensor=lambda cache,i=i:np.full((224,224,3),self.n+i*10,np.uint8))
        def seed(self,s):pass
        def set_init_state(self,initial):self.n=0;return self.raw()
        def get_sim_state(self):return np.array([self.n],float)
        def raw(self):return {k:o._sensor({}) for k,o in self.env._observables.items()}
        def step(self,a):
            self.n+=1;self.sim.data.time=self.n*.05;self.sim.data.cam_xpos[1,0]=self.n*.001
            return self.raw(),0,self.n==94,{}
    closed=[]
    rt=SimpleNamespace(bind_environment_context=lambda e:None,reset_environment=lambda e:None,renderer_metadata=lambda e:None,
        validate_camera_frames=lambda r:None,close_environment=lambda e:closed.append(e),
        observation=lambda raw,cfg:dict(full_image=raw['agentview_image'].copy(),wrist_image=raw['robot0_eye_in_hand_image'].copy(),state=np.zeros(8)))
    monkeypatch.setattr(R,'runtime',lambda:rt);monkeypatch.setattr(R,'runtime_config',lambda:dict(num_steps_wait=10))
    case=dict(start='libero_goal_t00_s10',suite='libero_goal',task_id=0,state_index=10,task_name='task',rollout_seed=20281030)
    monkeypatch.setattr(R,'create',lambda c:(Env(),SimpleNamespace(language='task'),np.array([0.])))
    def fingerprint(e):return dict(n=e.n,time=e.sim.data.time)
    def reference(c):
        e=Env()
        for _ in range(10):raw,_,_,_=e.step(np.zeros(7))
        return R.start_arrays(e,rt.observation(raw,None)),dict(fingerprint=fingerprint(e),rollout_seed=c['rollout_seed'])
    monkeypatch.setattr(R,'native_start_reference',reference)
    def assert_start(arrays,current,folder):
        assert current==C.read(folder/'START.json')
        with np.load(folder/'START.npz') as saved:
            assert all(np.array_equal(arrays[k],saved[k]) for k in arrays)
    frames=[]
    class Writer:
        def __init__(self,folder,fps):self.folder=folder;self.n=0
        def __enter__(self):
            for v in ('front','wrist'):(self.folder/(v+'.mp4')).write_bytes(b'test')
            return self
        def append(self,obs):self.n+=1
        def __exit__(self,*a):frames.append(self.n)
    fake=SimpleNamespace(fingerprint=fingerprint,assert_start=assert_start,CameraWriters=Writer,
        switch_suite=lambda *a:None,interface_gate=lambda *a:dict(status='PASS'),
        predict=lambda w,obs,l:np.full((8,7),obs['wrist_image'].mean()/255,dtype=np.float32))
    monkeypatch.setattr(R,'native',lambda:fake)
    def truth(env,target):return dict(eef_xyz=[env.n*.001,0,0],target_xyz=[.2,0,0],distance_m=.2-env.n*.001,contact=env.n>=16,grasp=env.n>=18)
    monkeypatch.setattr(R,'geometry',lambda:dict(anchor=lambda *a:{},truth=truth))
    stats=dict(q01=[-1]*6+[0],q99=[1]*7,mask=[True]*6+[False])
    wrapper=SimpleNamespace(gen_cfg=SimpleNamespace(unnorm_key='key'),vla=SimpleNamespace(norm_stats={'key':{'action':stats}}))
    monitor=SimpleNamespace(update=lambda **kw:None)
    R.episode(wrapper,case,'clean',monitor,True);R.episode(wrapper,case,'wrist_delay2',monitor,False)
    for name in ('clean','wrist_delay2'):
        receipt=C.episode_receipt(case,name);assert C.complete(receipt)
        files=C.episode_files(receipt);meta=C.read(files['EPISODE.json']);trace=pd.read_parquet(files['TRACE.parquet'])
        assert meta['success']==1 and meta['executed_steps']==84 and list(trace.iloc[-1]['predicted_actions'][0])
        assert trace.iloc[-1]['executed_slots']==4 and meta['first_target_contact_step']==6
        if name=='clean':assert len(pd.read_parquet(files['PROBES.parquet']))==12
        else:
            assert np.allclose(trace.wrist_added_age_s,.1) and np.allclose(trace.front_added_age_s,0)
            assert np.all(trace.wrist_source_control_step==trace.step_start-2)
    assert frames==[85,85] and len(closed)==2


def test_cpu_imports_do_not_load_simulator_or_native_policy():
    assert 'mujoco' not in sys.modules and 'libero.libero.envs' not in sys.modules
    assert 'joint_native_runtime' not in sys.modules


def test_partial_report_does_not_turn_missing_runs_into_success(tmp_path,monkeypatch):
    import sc_report as R
    cfg=C.config();cases=[dict(start=f'{s}_{t}_{i}',suite=s,task_id=t,state_index=i)
        for s in cfg['suites'] for t in range(4) for i in (10,11)]
    m=dict(cases=cases,conditions=[dict(name=n) for n in cfg['conditions']],checkpoint=cfg['model_dir'])
    monkeypatch.setattr(R,'ROOT',tmp_path);monkeypatch.setattr(R,'verify',lambda:m);monkeypatch.setattr(R,'config',lambda:cfg)
    monkeypatch.setattr(R,'complete',lambda p:False)
    value=R.run(SimpleNamespace(update=lambda **kw:None))
    assert value['status']=='INCOMPLETE' and value['missing']==224 and value['completed']==0
    import csv
    rows=list(csv.DictReader((tmp_path/'results/SUMMARY.csv').open()))
    total=next(r for r in rows if r['condition']=='clean' and r['suite']=='ALL_SUITES')
    assert total['completed']=='0' and total['missing']=='32' and total['task_SR']==''
    assert not (tmp_path/'results/COMPLETE.json').exists()
