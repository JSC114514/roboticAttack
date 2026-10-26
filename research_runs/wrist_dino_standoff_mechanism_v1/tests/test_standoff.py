"""Scientific failure modes and execution contracts; no MuJoCo import or GPU."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from sd_math import radial_delta,choose,aabb_center,trajectory_metrics,paired_arrays,hierarchical_interval

def test_radial_rule_is_rotation_equivariant_and_crosses_eef_without_clamping():
    eef=np.array([.2,.1,.3]);r=np.array([.03,.04,0.]);delta=radial_delta(eef+r,eef)
    np.testing.assert_allclose(delta,[-.03,-.04,0.],atol=1e-12)
    q=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_allclose(radial_delta(q@(eef+r),q@eef),q@delta,atol=1e-12)
    # A 3cm real range shifted by 5cm yields a point 2cm behind the EEF;
    # silently clipping at zero would test a different hypothesis.
    np.testing.assert_allclose(radial_delta([.03,0,0],[0,0,0]),[-.05,0,0])

def test_degenerate_geometry_is_explicit():
    with pytest.raises(ValueError):radial_delta([0,0,0],[0,0,0])
    with pytest.raises(ValueError):radial_delta([np.nan,0,0],[0,0,0])

def test_selection_uses_visibility_contact_bins_and_earliest_tie_not_actions():
    bins=json.loads((ROOT/'config.json').read_text())['bins']
    def r(d,i,**extra):return {'distance_m':d,'decision_block':i,'visible':True,'contact':False,'grasp':False,**extra}
    candidates=[r(.08,0,visible=False),r(.08,1,contact=True),r(.08,2,grasp=True),r(.08,4),r(.08,3),r(.03,5,task_done=True)]
    selected,missing=choose(candidates,bins)
    assert selected['far']['decision_block']==3
    assert 'near' in missing and 'near' not in selected

def test_anchor_bbox_uses_geometry_not_body_origin():
    v=np.array([[1,2,3],[5,8,7],[-1,1,2]])
    np.testing.assert_allclose(aabb_center(v),[2,4.5,4.5])

def test_dwell_does_not_count_initial_frame_and_reports_orbiting_path():
    rows=[{'distance_m':.05,'eef_xyz':[.05,0,0]},
          {'distance_m':.05,'eef_xyz':[0,.05,0]},
          {'distance_m':.05,'eef_xyz':[-.05,0,0]},
          {'distance_m':.07,'eef_xyz':[-.07,0,0]}]
    m=trajectory_metrics(rows)
    assert m['longest_band_dwell_steps']==2 and m['executed_steps']==3
    assert m['band_dwell_path_m']>.14
    assert m['band_dwell_net_m']==pytest.approx(.1)

def test_physical_pairing_is_strict_while_rgb_has_fixed_audited_tolerance():
    ref={'sim_state':np.zeros(3),'front':np.zeros((256,256,3),dtype=np.uint8)}
    cur={k:v.copy() for k,v in ref.items()};cur['sim_state'][0]=1e-14;cur['front'][0,0,0]=1
    result=paired_arrays(cur,ref)
    assert not result['sim_state']['pass'] and result['front']['pass']
    assert not result['front']['exact']

def test_hierarchical_interval_does_not_weight_frames_as_trajectories():
    f=pd.DataFrame([{'suite':'s','task_id':0,'episode_id':'a','m':1.}]*100+
                   [{'suite':'s','task_id':0,'episode_id':'b','m':3.}]+
                   [{'suite':'s','task_id':1,'episode_id':'c','m':10.}])
    out=hierarchical_interval(f,'m',50)
    assert out['mean']==pytest.approx(6) and out['trajectories']==3

def test_renderer_exception_restores_positions_and_global_hook(monkeypatch):
    from sd_geometry import SceneTranslation
    geom=SimpleNamespace(objtype=5,objid=8,pos=np.array([1.,2.,3.]))
    scene=SimpleNamespace(ngeom=1,geoms=[geom])
    def fail(*args):
        np.testing.assert_allclose(geom.pos,[2,2,3]);raise RuntimeError('renderer failed')
    fake=SimpleNamespace(mjr_render=fail);monkeypatch.setitem(sys.modules,'mujoco',fake)
    with pytest.raises(RuntimeError,match='renderer failed'):
        with SceneTranslation(scene,[8],[1,0,0]):fake.mjr_render(None,scene,None)
    np.testing.assert_array_equal(geom.pos,[1,2,3]);assert fake.mjr_render is fail

def test_sampler_exception_restores_sensor(monkeypatch):
    import sd_geometry as g
    original=lambda cache:np.zeros((256,256,3),dtype=np.uint8)
    observable=SimpleNamespace(_sensor=original);env=SimpleNamespace(env=SimpleNamespace(_observables={'robot0_eye_in_hand_image':observable}))
    monkeypatch.setattr(g,'fixed_anchor',lambda *args:{'entity':'fixed'})
    def fail(*args):raise RuntimeError('render gate')
    monkeypatch.setattr(g,'fingerprint',fail)
    with pytest.raises(RuntimeError,match='render gate'):
        with g.Camera(env,{},full=False) as camera:
            camera.arm(0);observable._sensor({})
    assert observable._sensor is original

class FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)]);self.num_prefix_tokens=5
    def forward(self,x):
        for b in self.blocks:x=b(x)
        return x[:,5:]

def fake_wrapper():
    dino=FakeDino();sig=torch.nn.Identity();vla=torch.nn.Module()
    vla.vision_backbone=SimpleNamespace(featurizer=dino,fused_featurizer=sig);vla.eval()
    def predict(obs,language):
        total=0.
        for i in range(2):
            x=torch.full((1,261,1024),float(i+1),dtype=torch.bfloat16)
            d=dino(x);s=sig(x[:,5:]);total+=float(d.float().mean()+s.float().mean())
        return np.full((8,7),total,dtype=np.float32)
    return SimpleNamespace(vla=vla,action_head=None,proprio_projector=None,noisy_action_projector=None,predict_action=predict)

def test_native_hook_changes_only_wrist_spatial_branch_and_keeps_prefixes():
    from sd_model import Engine,verify_isolation
    e=Engine(fake_wrapper());a,f=e.forward({},'x');donor=f['dino_1']+1
    b,g=e.forward({},'x',{'dino':donor});verify_isolation(f,g)
    np.testing.assert_array_equal(g['dino_1'],donor);assert not np.array_equal(a,b)
    identity,_=e.forward({},'x',{'dino':f['dino_1']});np.testing.assert_array_equal(a,identity)

def test_native_hooks_removed_after_action_exception():
    from sd_model import Engine
    w=fake_wrapper();e=Engine(w)
    def fail(*args):raise RuntimeError('policy failure')
    w.predict_action=fail
    with pytest.raises(RuntimeError):e.forward({},'x')
    assert not e.dino._forward_hooks and not e.siglip._forward_hooks and not e.dino.blocks[22]._forward_hooks

def test_dry_run_schedules_bounded_arrays_and_no_training(monkeypatch):
    import sd_cli as cli
    monkeypatch.setattr(cli,'worker_complete',lambda *args:False)
    monkeypatch.setattr(cli,'complete',lambda *args:False)
    monkeypatch.setattr(cli,'signature',lambda:'abc123')
    stages=cli.commands();assert [s for s,_ in stages]==['capture','mechanism','report']
    assert '--array=0,1%2' in stages[0][1]
    assert '--dependency=afterany:<CAPTURE_JOB_ID>' in stages[1][1]
    assert '--dependency=afterany:<MECHANISM_JOB_ID>' in stages[2][1]
    assert all('train' not in ' '.join(cmd) for _,cmd in stages)

def test_resume_only_requests_incomplete_shards(monkeypatch):
    import sd_cli as cli
    monkeypatch.setattr(cli,'worker_complete',lambda stage,i:stage=='capture' or i==0)
    monkeypatch.setattr(cli,'complete',lambda *args:False)
    monkeypatch.setattr(cli,'signature',lambda:'x')
    stages=cli.commands();assert stages[0][0]=='mechanism' and '--array=1%2' in stages[0][1]

def test_gpu_resources_and_no_gpu_report():
    for stage in ('capture','mechanism'):
        s=(ROOT/f'slurm/{stage}.sbatch').read_text()
        assert '--mem=12G' in s and '--cpus-per-task=4' in s and '--gres=gpu:4090:1' in s
    s=(ROOT/'slurm/report.sbatch').read_text()
    assert '--mem=4G' in s and '--cpus-per-task=2' in s and '--gres' not in s

def test_virtual_collision_uses_scratch_data_and_does_not_mutate_live_positions(monkeypatch):
    import sd_geometry as g
    model=SimpleNamespace(ngeom=2,geom_contype=np.array([1,1]),geom_conaffinity=np.array([1,1]),
        geom_type=np.array([6,6]),geom_rbound=np.array([1.,1.]))
    live=SimpleNamespace(geom_xpos=np.array([[0.,0,0],[.1,0,0]]),geom_xmat=np.tile(np.eye(3).reshape(9),(2,1)))
    scratch=SimpleNamespace(geom_xpos=np.zeros_like(live.geom_xpos),geom_xmat=np.zeros_like(live.geom_xmat))
    def distance(m,d,a,b,maximum,segment):
        assert d is scratch;return float(d.geom_xpos[b,0]-d.geom_xpos[a,0]-.05)
    fake=SimpleNamespace(MjData=lambda m:scratch,mj_geomDistance=distance)
    monkeypatch.setitem(sys.modules,'mujoco',fake)
    env=SimpleNamespace(sim=SimpleNamespace(model=model,data=live))
    before=live.geom_xpos.copy();check=g.CollisionCheck(env,{'render_ids':[0],'contact_ids':[0]})
    assert check.check([.08,0,0])['new_or_deeper_penetration']
    np.testing.assert_array_equal(before,live.geom_xpos)

@pytest.mark.parametrize('condition',['clean','dino_near'])
def test_rollout_stops_at_first_contact_inside_action_chunk(monkeypatch,tmp_path,condition):
    from contextlib import contextmanager
    import sd_mechanism as m
    env=SimpleNamespace(step_count=0)
    def step(action):env.step_count+=1;return {},0,False,{}
    env.step=step
    def t():return {'eef_xyz':[0,0,0],'target_xyz':[.08-.01*env.step_count,0,0],
        'distance_m':.08-.01*env.step_count,'physics_time_s':env.step_count*.05,
        'camera_target_distance_m':.2,'camera_optical_depth_m':.2,
        'contact':env.step_count>=3,'grasp':False}
    payload=lambda:{'visible':True,'truth':t(),'conditions':{'near':{'valid':False,'reasons':['virtual_geometry_penetration']}}}
    camera=SimpleNamespace(anchor={},take=lambda *a:payload(),arm=lambda *a,**kw:None)
    @contextmanager
    def restore(*a,**kw):yield {},camera
    monkeypatch.setattr(m,'restore_case',restore);monkeypatch.setattr(m,'truth',lambda *a:t())
    def verify(*a):
        np.savez_compressed(tmp_path/'START.npz',state=np.zeros(1));(tmp_path/'PAIRING.json').write_text('{}')
    monkeypatch.setattr(m,'verify_start',verify);monkeypatch.setattr(m,'ROOT',tmp_path)
    obs={'full_image':np.zeros((224,224,3),dtype=np.uint8),'wrist_image':np.zeros((224,224,3),dtype=np.uint8)}
    monkeypatch.setattr(m,'runtime',lambda:SimpleNamespace(observation=lambda *a:obs))
    monkeypatch.setattr(m,'runtime_config',lambda:{})
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',SimpleNamespace(process_libero_action=lambda a:a))
    engine=SimpleNamespace(forward=lambda *a:(np.zeros((8,7),dtype=np.float32),{}),frozen=lambda:None)
    case={'case_id':'x','episode_id':'e','suite':'s','task_id':0,'state_index':40,'cohort':'validation','bin':'far','decision_block':0,'language':'x'}
    m.rollout(env,None,{},case,condition,engine,SimpleNamespace(update=lambda **kw:None),tmp_path)
    result=json.loads((tmp_path/'EPISODE.json').read_text())
    assert result['executed_steps']==3 and result['stop_reason']=='first_target_contact'
    assert result['applied_steps']==0
    rows=pd.read_parquet(tmp_path/'STEPS.parquet');assert len(rows)==4
    chunks=pd.read_parquet(tmp_path/'DECISIONS.parquet');assert chunks.executed_slots.tolist()==[3]
