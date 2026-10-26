"""CPU regressions; do not import or initialize MuJoCo, EGL or the VLA."""
from pathlib import Path
import ast
import importlib
import json
import sys
import types
import numpy as np
import pytest

CODE=Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0,str(CODE))
from va_math import all_conditions,nonattack_conditions,patch_conditions,condition_spec,random_pose,rotation_angle,rotation_axis,blend,grid,blur,residual,bootstrap

def test_branch_budget_and_shared_clean():
    a=set(nonattack_conditions());b=set(patch_conditions())
    assert len(a)==13 and len(b)==12 and a&b=={'clean'}
    assert len(all_conditions())*12==288
    assert len(all_conditions())*12*64==18432
    assert all(condition_spec(n)['route'] in ('shared','nonattack','patch') for n in all_conditions())

@pytest.mark.parametrize('patch',['relation','uada7'])
def test_random_camera_and_clean_have_identical_donor(patch):
    a=condition_spec('b_'+patch+'_random');b=condition_spec('b_'+patch+'_random_clean')
    assert a['camera_source']==b['camera_source']==patch
    assert a['camera']==b['camera']=='random' and a['patch']==patch and b['patch'] is None

@pytest.mark.parametrize('theta',[0.,.001,.5,np.pi-1e-7])
def test_so3_and_translation_match_amplitudes(theta):
    base=rotation_axis([1,2,3],.4);p=np.array([.2,-.3,.7]);q,r=random_pose(p,base,.023,theta,42)
    np.testing.assert_allclose(np.linalg.norm(q-p),.023,atol=1e-12)
    np.testing.assert_allclose(rotation_angle(base,r),theta,atol=2e-8)
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
    assert np.linalg.det(r)==pytest.approx(1.)
    a,b=random_pose(p,base,.023,theta,42)
    np.testing.assert_array_equal(a,q);np.testing.assert_array_equal(b,r)

def test_gate_is_centered_on_fit_mean():
    x=np.array([[1.,3.]]);mu=np.array([[5.,1.]])
    np.testing.assert_array_equal(blend(x,mu,.5),[[3,2]])
    np.testing.assert_array_equal(blend(x,mu,1),x)
    np.testing.assert_array_equal(blend(x,mu,0),mu)
    with pytest.raises(ValueError):blend(x,mu,-.5)
    with pytest.raises(ValueError):blend(x,mu[:,:1],.5)

def test_pooled_grid_keeps_location_and_channels():
    x=np.arange(256*3).reshape(256,3)
    result=grid(x).reshape(4,4,3)
    np.testing.assert_equal(result[0,0],x.reshape(16,16,3)[:4,:4].mean((0,1)))
    assert not np.array_equal(result[0,0],result[3,3])

def test_blur_preserves_constant_and_channel_independence():
    x=np.full((224,224,3),73,np.uint8);np.testing.assert_array_equal(blur(x),x)
    x[:]=0;x[110:114,110:114,0]=255;y=blur(x)
    assert y[:,:,1:].max()==0 and 0<y[:,:,0].max()<255
    assert y.dtype==np.uint8 and y.shape==x.shape

def test_delay_uses_true_eight_control_steps_not_prior_decision_alias(monkeypatch):
    import va_rollout
    history=[dict(full_image=np.full((224,224,3),i,np.uint8),wrist_image=np.full((224,224,3),i,np.uint8)) for i in range(10)]
    sensors=types.SimpleNamespace(observation=lambda raw:dict(full_image=history[-1]['full_image'],wrist_image=history[-1]['wrist_image'],state=np.zeros(8)))
    monkeypatch.setattr(va_rollout,'adapters',lambda:(None,None))
    spec=condition_spec('a_wrist_lag_raw')
    got=va_rollout.deployed_observation({},sensors,history,spec,0,None)
    assert got['wrist_image'][0,0,0]==1 and got['full_image'][0,0,0]==9
    assert history[-1]['wrist_image'][0,0,0]==9
    with pytest.raises(RuntimeError):va_rollout.deployed_observation({},sensors,history[-8:],spec,0,None)

def test_temporal_training_does_not_cross_trajectory_or_gap():
    from va_calibrate import transitions
    meta=[dict(episode_id='a',decision_block=i) for i in (4,5)]+[dict(episode_id='b',decision_block=i) for i in (0,1)]
    assert set(map(tuple,transitions(meta,range(4))))=={(0,1),(2,3)}
    meta[1]['decision_block']=6
    with pytest.raises(ValueError):transitions(meta,range(4))

def test_group_folds_exclude_entire_trajectories():
    from va_calibrate import group_folds
    groups=np.repeat(np.arange(12),5);seen=[]
    for train,valid in group_folds(groups):
        assert not set(groups[train])&set(groups[valid]);seen.extend(valid.tolist())
    assert sorted(seen)==list(range(60))

def test_auc_ties_and_undefined_classes():
    from va_math import auc
    assert auc([0,0,1,1],[0.,1.,1.,2.])==pytest.approx(.875)
    assert auc([0,0],[1.,2.]) is None

def test_validation_changes_cannot_change_fitted_pca_or_predictor():
    from va_calibrate import fit_model
    rng=np.random.default_rng(42);x=rng.normal(size=(60,24));p=rng.normal(size=(60,8))
    meta=[dict(episode_id=str(i//5),decision_block=i%5) for i in range(60)]
    a=fit_model(x,p,meta,range(40),1.)
    x[40:]=1e8;p[40:]=1e6;b=fit_model(x,p,meta,range(40),1.)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    a['residual_scale']=np.ones(8)
    assert np.isfinite(residual(x[0],x[1],p[1]-p[0],a))

@pytest.mark.parametrize('fail',[False,True])
def test_shadow_uses_snapshot_and_restores_native_context_even_on_exception(monkeypatch,fail):
    import va_camera
    calls=[];live={'value':10};sample={'value':4};copied={}
    data=types.SimpleNamespace(_data=copied,cam_xpos=np.zeros((1,3)),cam_xmat=np.zeros((1,9)))
    def render(**kwargs):
        calls.append('render')
        if fail:raise ValueError('synthetic render failure')
        return np.zeros((2,2,3),np.uint8)
    sim=types.SimpleNamespace(data=data,model=types.SimpleNamespace(_model='copy-model',camera_name2id=lambda n:0),render=render)
    obj=va_camera.ShadowRenderer.__new__(va_camera.ShadowRenderer);obj.sim=sim
    obj.env=types.SimpleNamespace(sim=types.SimpleNamespace(data=types.SimpleNamespace(_data=live)))
    obj.context=types.SimpleNamespace(gl_ctx=types.SimpleNamespace(make_current=lambda:calls.append('shadow_context')))
    obj.mujoco=types.SimpleNamespace(mj_copyData=lambda dst,model,src:dst.update(src))
    native=types.SimpleNamespace(runtime_module=lambda:types.SimpleNamespace(bind_environment_context=lambda env:calls.append('native_context')))
    monkeypatch.setattr(va_camera,'adapters',lambda:(None,native))
    monkeypatch.setitem(sys.modules,'audit_replay',types.SimpleNamespace(fingerprint=lambda env:dict(live)))
    if fail:
        with pytest.raises(ValueError):obj.render([1,2,3],np.eye(3),state=sample)
    else:obj.render([1,2,3],np.eye(3),state=sample)
    assert copied['value']==4 and live['value']==10 and calls[-1]=='native_context'
    np.testing.assert_array_equal(data.cam_xpos[0],[1,2,3])

def test_tampered_completion_never_silently_resumes(tmp_path,monkeypatch):
    import va_common
    monkeypatch.setattr(va_common,'ROOT',tmp_path);monkeypatch.setattr(va_common,'signature',lambda:'fixed')
    asset=tmp_path/'asset';asset.write_text('one');receipt=tmp_path/'COMPLETE.json'
    va_common.finish(receipt,[asset]);assert va_common.complete(receipt)
    asset.write_text('two')
    with pytest.raises(RuntimeError):va_common.complete(receipt)

def test_task_clustered_interval_does_not_treat_frames_as_units():
    rows=[dict(task_id=t,difference=1.) for t in range(6) for _ in range(2)]
    assert bootstrap(rows,draws=50)==[1.,1.]
    assert bootstrap([]) is None

def test_dependency_graph_is_serial_then_at_most_two_gpus(monkeypatch,capsys):
    import va_cli
    from contextlib import nullcontext
    monkeypatch.setattr(va_cli,'check_launchers',lambda:None);monkeypatch.setattr(va_cli,'done',lambda s:False)
    monkeypatch.setattr(va_cli,'signature',lambda:'a'*64);monkeypatch.setattr(va_cli,'lock',lambda n:nullcontext())
    monkeypatch.setattr(va_cli,'queue',lambda:(_ for _ in ()).throw(AssertionError('dry-run must not contact Slurm')))
    va_cli.submit(dry=True);lines=capsys.readouterr().out.splitlines()
    assert len(lines)==6
    assert '--dependency=afterok:DRY3' in lines[3] and '--dependency=afterok:DRY3' in lines[4]
    assert '--dependency=afterany:DRY4:DRY5' in lines[5]
    assert sum('/gpu.sbatch' in line for line in lines)==3

def test_active_other_revision_prevents_duplicate_submission(monkeypatch,capsys):
    import va_cli
    from contextlib import nullcontext
    monkeypatch.setattr(va_cli,'check_launchers',lambda:None);monkeypatch.setattr(va_cli,'lock',lambda n:nullcontext())
    monkeypatch.setattr(va_cli,'queue',lambda:[['123','PENDING','wasym1_older_shared','Priority']])
    monkeypatch.setattr(va_cli.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not submit')))
    va_cli.submit();assert 'ALREADY_ACTIVE' in capsys.readouterr().out

def test_prepare_and_status_do_not_import_simulator():
    for filename in ('va_common.py','va_prepare.py','va_cli.py','va_math.py','va_calibrate.py'):
        nodes=ast.parse((CODE/filename).read_text()).body
        for n in nodes:
            if isinstance(n,ast.Import):assert all(not a.name.startswith(('mujoco','robosuite','libero')) for a in n.names)
            if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith(('mujoco','robosuite','libero'))

def test_resource_and_no_training_contract():
    root=CODE.parent;cfg=json.loads((root/'config.json').read_text());gpu=(root/'slurm/gpu.sbatch').read_text();cpu=(root/'slurm/cpu.sbatch').read_text()
    assert cfg['tasks']==[0,1,3,5,6,7] and cfg['max_concurrent_gpus']==2
    assert '--gres=gpu:4090:1' in gpu and '--cpus-per-task=4' in gpu and '--mem=12G' in gpu
    assert '--mem=4G' in cpu and '--gres' not in cpu
    for p in CODE.glob('*.py'):
        assert 'optimizer.step(' not in p.read_text() and '.backward(' not in p.read_text()

def test_short_rollout_executes_eight_slots_and_removes_patch_after_first_chunk(tmp_path,monkeypatch):
    """Exercise the actual episode writer/loop with a toy controller, not MuJoCo."""
    from contextlib import contextmanager
    import copy
    import torch
    import pandas as pd
    from PIL import Image
    import va_rollout as r
    import va_common as c
    (tmp_path/'config.json').write_text((CODE.parent/'config.json').read_text())
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(c,'signature',lambda:'test')
    patch=tmp_path/'patch.png';Image.fromarray(np.full((50,50,3),255,np.uint8)).save(patch)
    (tmp_path/'data').mkdir();(tmp_path/'data/MANIFEST.json').write_text(json.dumps({'patches':{'relation':str(patch)}}))
    base=dict(full_image=np.zeros((224,224,3),np.uint8),wrist_image=np.zeros((224,224,3),np.uint8),state=np.zeros(8,np.float32))
    case=dict(episode_id='libero_goal_t00_s42',task_id=0,state_index=42)
    class Env:
        def __init__(self):self.step_id=0;self.env=types.SimpleNamespace(control_timestep=.05)
        def get_sim_state(self):return np.zeros(4)
        def step(self,action):self.step_id+=1;return copy.deepcopy(base),0.,False,{}
    env=Env()
    class Sensor:
        def __init__(self):self.zero_checks=0;self.zero_max=0.;self.update()
        def update(self):
            self.latest={'front':{'mask':np.ones((256,256),bool)},'wrist':dict(position=np.zeros(3),rotation=np.eye(3),sample_time_s=env.step_id*.05,
                mask=np.ones((256,256),bool),translation_m=0.,rotation_rad=0.,fingerprint={'sample':0})}
        def activate(self,*args):pass
        def observation(self,raw):return copy.deepcopy(raw)
        def arm(self,*args):pass
        def take(self,raw):self.update()
    sensor=Sensor();history=[copy.deepcopy(base) for _ in range(10)]
    row=dict(observation=copy.deepcopy(base),teacher={'clean':torch.zeros(1,8,7)},geometry={'target_xyz':[1.,0.,0.],'eef_xyz':[0.,0.,0.]})
    @contextmanager
    def replay(*args):yield env,types.SimpleNamespace(language='task'),copy.deepcopy(base),sensor,None,history,row
    monkeypatch.setattr(r,'replay',replay)
    def paste(obs,png):
        obs=copy.deepcopy(obs);obs['wrist_image'][137:187,26:76]=png;return obs
    native=types.SimpleNamespace(runtime_module=lambda:types.SimpleNamespace(observation=lambda raw,cfg:copy.deepcopy(raw)),runtime_config=lambda:{},paste_numpy=paste)
    monkeypatch.setattr(r,'adapters',lambda:(None,native));monkeypatch.setattr(r,'pooled',lambda caps:np.zeros((2,8),np.float32))
    monkeypatch.setitem(sys.modules,'sr_geometry',types.SimpleNamespace(truth=lambda env,target:dict(contact=env.step_id>=3,grasp=False,distance_m=1-env.step_id*.001,eef_xyz=[env.step_id*.001,0.,0.],target_xyz=[1.,0.,0.])))
    monkeypatch.setitem(sys.modules,'audit_math',types.SimpleNamespace(visibility=lambda *args:dict(policy_fov_pixels=1,policy_uncovered_pixels=1)))
    monkeypatch.setitem(sys.modules,'audit_replay',types.SimpleNamespace(fingerprint=lambda env:{'control':0}))
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',types.SimpleNamespace(process_libero_action=lambda a:a))
    class Video:
        def __init__(self,folder,fps):self.folder=folder
        def __enter__(self):return self
        def append(self,obs):pass
        def __exit__(self,*args):
            for view in ('front','wrist'):(self.folder/(view+'.mp4')).write_bytes(b'fake video')
    monkeypatch.setattr(r,'Videos',Video)
    class Vision:
        wrapper=types.SimpleNamespace(vla=types.SimpleNamespace(_unnormalize_actions=lambda a,k:a),gen_cfg=types.SimpleNamespace(unnorm_key='goal'))
        def action(self,obs,*args):return np.full((8,7),float(obs['wrist_image'][137,26,0]>0),np.float32),{}
        def features(self,*args):return {}
    arrays=dict(sim_state=np.zeros(4),**base)
    reference=dict(horizon=16,start=dict(fingerprint={'control':0},sensor_fingerprint={'sample':0}),start_arrays=arrays,camera_path={},constructor_rng=None)
    detail=r.run_episode(case,'b_relation_first_chunk',Vision(),None,types.SimpleNamespace(update=lambda **v:None),None,reference)
    receipt=c.branch_receipt(case,'b_relation_first_chunk');assert c.complete(receipt)
    files=c.branch_files(receipt);decisions=pd.read_parquet(files['DECISIONS.parquet'])
    assert list(decisions.step)==[0,8] and detail['executed_steps']==16
    assert np.stack(decisions.iloc[0].actions).sum()==56 and np.stack(decisions.iloc[1].actions).sum()==0
    assert detail['first_contact_step']==3 and len(pd.read_parquet(files['EVENTS.parquet']))==17
    assert not np.any(base['wrist_image'])

def test_sensor_hooks_restore_on_exception():
    from va_camera import Sensors
    front=lambda cache:'front';wrist=lambda cache:'wrist'
    observables={'agentview_image':types.SimpleNamespace(_sensor=front),'robot0_eye_in_hand_image':types.SimpleNamespace(_sensor=wrist)}
    env=types.SimpleNamespace(env=types.SimpleNamespace(_observables=observables))
    with pytest.raises(ValueError):
        with Sensors(env,None):
            assert observables['agentview_image']._sensor is not front
            raise ValueError('interrupted')
    assert observables['agentview_image']._sensor is front and observables['robot0_eye_in_hand_image']._sensor is wrist

@pytest.mark.parametrize('prepared',[False,True])
def test_partial_report_does_not_recommend_a_scientific_direction(tmp_path,monkeypatch,prepared):
    import va_common as c
    import va_report as r
    (tmp_path/'config.json').write_text((CODE.parent/'config.json').read_text())
    if prepared:
        (tmp_path/'data').mkdir()
        (tmp_path/'data/MANIFEST.json').write_text(json.dumps({'cases':[]}))
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(c,'signature',lambda:'test');monkeypatch.setattr(r,'signature',lambda:'test')
    r.run(types.SimpleNamespace(update=lambda **values:None),export=True)
    assert json.loads((tmp_path/'results/STATUS.json').read_text())['status']=='PARTIAL'
    assert '等待完整配对' in json.loads((tmp_path/'results/RECOMMENDATION.json').read_text())['direction']
