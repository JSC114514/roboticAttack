"""CPU-only scientific/interface checks; never import MuJoCo on login."""
from pathlib import Path
import sys
import json
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from rr_math import radial_projection,restore

def test_random_radial_equalities_box_and_exact_identity():
    rng=np.random.default_rng(42)
    for _ in range(25):
        p=rng.uniform(-3,3,(8,3));c=rng.uniform(-3,3,(8,3));u=rng.normal(size=3);u/=np.linalg.norm(u)
        r=radial_projection(p,c,u)
        assert np.max(np.abs((r-np.clip(c,-1,1))@u))<2e-13
        assert np.max(np.abs(r))<=1
        assert np.array_equal(radial_projection(p,p,u),np.clip(p,-1,1))

def test_projection_matches_independent_constrained_optimizer():
    from scipy.optimize import minimize
    rng=np.random.default_rng(7)
    for _ in range(12):
        p=rng.uniform(-1,1,3);c=rng.uniform(-1,1,3);u=rng.normal(size=3);u/=np.linalg.norm(u)
        solution=radial_projection(p,c,u)
        numerical=minimize(lambda x:np.sum((x-p)**2),c,bounds=[(-1,1)]*3,
            constraints={'type':'eq','fun':lambda x:u@(x-c)},method='SLSQP',options={'ftol':1e-12,'maxiter':200})
        assert numerical.success
        assert np.linalg.norm(solution-numerical.x)<2e-6

def test_unbounded_radial_formula_and_other_channels_unchanged():
    rng=np.random.default_rng(3);p=rng.uniform(-.2,.2,(8,7));c=rng.uniform(-.2,.2,(8,7));u=np.array([1,2,3.]);u/=np.linalg.norm(u)
    out,meta=restore(p,c,u,'restore_radial')
    expected=p[:,:3]+np.outer((c[:,:3]-p[:,:3])@u,u)
    assert np.allclose(out[:,:3],expected,atol=1e-13)
    assert np.array_equal(out[:,3:],p[:,3:])
    assert meta['tangential_change_m']<1e-14

def test_saturated_box_preserves_equality_and_reports_collateral_change():
    p=np.tile([-1,1,1,0,0,0,.7],(8,1));c=np.tile([1,-1,-1,0,0,0,.1],(8,1));u=np.array([.2,.8,.5])
    out,meta=restore(p,c,u,'restore_radial');un=u/np.linalg.norm(u)
    assert np.max(np.abs((out[:,:3]-c[:,:3])@un))<1e-12
    assert meta['radial_error_m']<1e-12 and meta['tangential_change_m']>0

@pytest.mark.parametrize('method',['restore_radial','restore_xyz','restore_pose_grip'])
def test_identity_and_selected_channel_preservation(method):
    p=np.arange(56,dtype=float).reshape(8,7)/100;c=-p
    out,_=restore(p,p,[1,0,0],method)
    assert np.array_equal(out,p)
    out,_=restore(p,c,[1,0,0],method)
    if method=='restore_pose_grip':assert np.array_equal(out[:,:3],p[:,:3]) and np.array_equal(out[:,3:],c[:,3:])
    else:assert np.array_equal(out[:,3:],p[:,3:])

def test_degenerate_direction_keeps_full_patch_and_nonfinite_rejected():
    p=np.ones((8,7));c=-p
    out,meta=restore(p,c,[0,0,0],'restore_radial')
    assert np.array_equal(out,p) and meta['degenerate_direction']
    p[0,1]=np.nan
    with pytest.raises(ValueError):restore(p,c,[1,0,0],'restore_xyz')

def test_actual_native_float32_processor_respects_radial_roundoff_bound():
    from rr_math import executed_radial_error
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    rng=np.random.default_rng(29)
    for _ in range(30):
        p=rng.uniform(-2,2,(8,7)).astype(np.float32);c=rng.uniform(-2,2,(8,7)).astype(np.float32);u=rng.normal(size=3)
        out,_=restore(p,c,u,'restore_radial')
        for slot in range(8):
            native=process_libero_action(out[slot]);assert native.dtype==np.float32
            error,bound=executed_radial_error(native,c[slot],u)
            assert error<=bound
            assert np.array_equal(native[3:6],process_libero_action(p[slot])[3:6])
            assert native[6]==process_libero_action(p[slot])[6]

class FakeModel:
    nbody=3;body_parentid=np.array([0,0,1]);geom_bodyid=np.array([0,2]);_model='model'
    actuator_ctrllimited=np.array([1,1]);actuator_ctrlrange=np.array([[-1,1],[-1,1]])
    jnt_limited=np.array([1]);jnt_qposadr=np.array([0]);jnt_range=np.array([[-1,1]]);jnt_type=np.array([3])
    def body_name2id(self,name):return 1
    def body_id2name(self,i):return ['world','base','finger'][i]
    def geom_id2name(self,i):return ['table','finger_geom'][i]
    def joint_id2name(self,i):return 'joint0'

class FakeSim:
    def __init__(self):
        self.model=FakeModel();self.calls=0
        self.data=NS(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9),ctrl=np.array([1.,-.3]),
            qpos=np.array([.9]),ncon=1,contact=[NS(geom1=0,geom2=1)],time=0.,_data='data')
    def step(self,*args,**kwargs):self.calls+=1;self.data.time+=.002

def fake_environment():
    sim=FakeSim();controller=NS(goal_pos=np.array([.1,0,0]),goal_ori=np.eye(3),torques=np.array([2.,0]))
    robot=NS(controller=controller,torques=np.array([1.,0]),eef_site_id=0,robot_model=NS(root_body='base'))
    return NS(sim=sim,env=NS(robots=[robot]))

def test_substep_logger_aggregates_real_steps_and_restores_method(tmp_path):
    from rr_logger import PhysicsLogger
    env=fake_environment();before=env.sim.step
    def force(m,d,i,out):out[0]=env.sim.calls;out[1]=2
    with PhysicsLogger(env,force) as logger:
        logger.arm();env.sim.step();env.sim.step();result=logger.take()
    assert env.sim.step==before and 'step' not in vars(env.sim)
    assert env.sim.calls==2 and env.sim.data.time==.004
    assert result['physics_substeps']==2 and result['contacts'][0]['samples']==2
    assert result['contacts'][0]['involves_robot']
    assert result['peak_robot_contact_force_n']==np.sqrt(8)
    assert result['actuator_bound_samples']==2 and result['torque_clip_peak_nm']==1
    assert np.allclose(result['joint_min_margin_by_joint'],[.1])
    # Nested contacts/controllers survive the same Parquet representation used by production.
    p=tmp_path/'steps.parquet';pd.DataFrame([{'controller':result}]).to_parquet(p,index=False)
    assert pd.read_parquet(p).iloc[0].controller['contacts'][0]['samples']==2

def test_substep_logger_restores_on_failure_without_retry():
    from rr_logger import PhysicsLogger
    env=fake_environment();original=env.sim.step
    def fail(*args):raise RuntimeError('synthetic contact API error')
    with pytest.raises(RuntimeError,match='synthetic contact API error'):
        with PhysicsLogger(env,fail) as logger:
            logger.arm();env.sim.step()
    assert env.sim.step==original and env.sim.calls==1

def test_recovery_statistics_retain_regressions_and_do_not_count_missing(monkeypatch):
    import rr_report as report
    monkeypatch.setattr(report,'config',lambda:{'methods':['restore_radial'],'bootstrap_draws':100})
    cases=[{'start':str(i),'episode':{'task_id':i,'success':s}} for i,s in enumerate([0,1,0])]
    lookup={('0','restore_radial'):{'success':1},('1','restore_radial'):{'success':0}}
    result=report.restoration_summary(cases,lookup)[0]
    assert result['paired_completed']==2 and result['expected']==3
    assert result['rescued_original_failures']==1 and result['regressed_original_successes']==1
    assert result['success_rate_gain']==0

def test_all_episode_receipts_skip_gpu_when_final_seal_missing(monkeypatch,tmp_path):
    import rr_cli as cli
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli,'config',lambda:{'methods':['A','B','C']})
    monkeypatch.setattr(cli,'read',lambda p:{'cases':[{'start':'s0'},{'start':'s1'}]})
    sealed=set()
    monkeypatch.setattr(cli,'complete',lambda p:'/episodes/' in str(p) or str(p) in sealed)
    monkeypatch.setattr(cli,'finish',lambda p,artifacts:sealed.add(str(p)))
    calls=[];cli.stages(lambda stage,deps:calls.append((stage,deps)) or '123')
    assert calls==[('report',[])]

def test_comparison_stream_preserves_common_clock_and_end_frames(monkeypatch,tmp_path):
    import imageio.v2 as imageio
    import rr_report as report
    labels=['clean','reference_replay','restore_radial','restore_xyz','restore_pose_grip'];paths={}
    monkeypatch.setattr(report,'config',lambda:{'methods':labels[2:]})
    for index,label in enumerate(labels):
        p=tmp_path/label;p.mkdir();paths[label]=p;length=index+2
        (p/'EPISODE.json').write_text(json.dumps({'fps':20,'executed_steps':length-1,'success':index%2}))
        with imageio.get_writer(p/'video.mp4',fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as writer:
            for _ in range(length):writer.append_data(np.full((288,448,3),30+index*30,np.uint8))
    case={'start':'synthetic','baseline_videos':{'clean':str(paths['clean']/'video.mp4')},
          'baseline_episodes':{'clean':{'executed_steps':1,'success':1}}}
    out=tmp_path/'output';out.mkdir();report.compose(case,paths,out)
    with imageio.get_reader(out/'comparison.mp4',ffmpeg_params=['-threads','1']) as reader:frames=list(reader.iter_data())
    assert len(frames)==6 and frames[-1].shape==(576,1344,3)
    assert frames[-1][100:200,20:200].mean()>10
