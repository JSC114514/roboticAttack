"""CPU contracts for provenance, matched starts, scheduling, accounting and video."""
import copy
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import jr_common as common
import jr_cli as cli
import jr_math as math
import jr_prepare as prepare
from jr_assets import inventory, confirmation_conditions


def test_inventory_exact_unique_and_confirmations():
    rows = inventory()
    assert len(rows) == 26 and sum(r['kind']=='trained' for r in rows) == 22
    assert len({r['sha256'] for r in rows if r['path']}) == 25
    assert len(confirmation_conditions(rows)) == 14
    assert all(r['source_model'] in ('libero_goal','libero_spatial') for r in rows if r['kind']=='trained')
    assert sorted(r['updates'] for r in rows if r['name'].startswith('geometry_action_')) == [19000,19000,20000,20000]


def test_pngs_have_native_rgb_shape():
    from PIL import Image
    for r in inventory():
        if r['path']:
            with Image.open(r['path']) as img: assert img.size == (50,50) and img.mode=='RGB'


def test_no_patch_front_or_state_mutation():
    obs = dict(full_image=np.zeros((224,224,3),np.uint8), wrist_image=np.ones((224,224,3),np.uint8), state=np.arange(8,dtype=float))
    patch = np.full((50,50,3),197,np.uint8); out = math.paste(obs,patch)
    assert np.all(out['wrist_image'][137:187,26:76]==197)
    assert (out['wrist_image']!=obs['wrist_image']).sum()==50*50*3
    assert np.array_equal(out['full_image'],obs['full_image']) and np.array_equal(out['state'],obs['state'])
    assert np.all(obs['wrist_image']==1)
    zero = math.paste(obs,None)
    assert all(np.array_equal(obs[k],zero[k]) and not np.shares_memory(obs[k],zero[k]) for k in obs)


def test_clean_failure_and_missing_do_not_become_attack_success():
    clean = {'a':dict(success=1),'b':dict(success=0),'c':dict(success=1)}
    value = math.paired_summary([dict(start='a',success=0),dict(start='b',success=0)],clean,3)
    assert value['CASR']==1 and value['clean_success_pairs']==1 and value['missing']==1
    assert value['task_SR']==0 and value['induced_failures']==1
    assert math.paired_summary([],clean,3)['CASR'] is None
    assert math.suite_equal({'s':None},['s']) is None


def test_task_start_resampling_keeps_conditions_paired():
    from jr_report import estimates
    samples = [dict(task_id=t,num=1,den=1) for t in range(6)]
    ci, values = estimates(samples,100,42)
    assert ci == [1,1] and np.all(values==1)
    assert estimates([],100,42)==(None,None)


def test_resources_array_and_dependencies():
    for workers in (1,2,4):
        plans = cli.plans(workers)
        assert [p['name'] for p in plans]==['clean','evaluate','report','videos']
        assert plans[1]['parents']==['clean'] and plans[1]['after']=='afterok'
        assert plans[1]['array']==f'0-{workers-1}%{workers}'
        assert plans[2]['after']=='afterany'
        assert all('train' not in p['name'] and 'confirmation' not in p['name'] for p in plans)
    assert cli.plans(4,True,True,True,True)==[]
    assert cli.plans(2,True,False)[0]['name']=='evaluate'
    with pytest.raises(RuntimeError): cli.plans(5)
    for name in ('clean','evaluate'):
        shell = (common.ROOT/'slurm'/(name+'.sbatch')).read_text()
        assert '--gres=gpu:4090:1' in shell and '--mem=12G' in shell and '--cpus-per-task=4' in shell
    cpu = (common.ROOT/'slurm/cpu.sbatch').read_text()
    assert '--mem=4G' in cpu and '--cpus-per-task=2' in cpu and '--gres' not in cpu


def test_weighted_shards_cover_each_start_once():
    cfg = common.config(); cases=[dict(start=f'{s}_{t}',suite=s) for s in cfg['suites'] for t in range(10)]
    for count in (1,2,4):
        shards=math.weighted_shards(cases,count,cfg['max_steps'])
        assert sorted(c['start'] for g in shards for c in g)==sorted(c['start'] for c in cases)
        costs=[sum(cfg['max_steps'][c['suite']] for c in g) for g in shards]
        assert max(costs)-min(costs)<=520


def test_native_statistics_keys_and_bad_statistics():
    cfg=common.config(); p=Path(cfg['model_dir'])/'dataset_statistics.json'
    stats=json.loads(p.read_text()); prepare.validate_statistics(stats,cfg['suites'])
    bad=copy.deepcopy(stats); del bad['libero_10_no_noops']
    with pytest.raises(RuntimeError,match='Missing joint normalization'): prepare.validate_statistics(bad,cfg['suites'])
    bad=copy.deepcopy(stats); bad['libero_goal_no_noops']['action']['q01'][0]=float('nan')
    with pytest.raises(RuntimeError,match='Invalid'): prepare.validate_statistics(bad,cfg['suites'])


def test_truncated_weight_rejected_without_loading(tmp_path):
    p=tmp_path/'weights.safetensors'; h=json.dumps({'x':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
    p.write_bytes(struct.pack('<Q',len(h))+h+b'1234'); prepare.validate_safetensor(p)
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(RuntimeError,match='Truncated'): prepare.validate_safetensor(p)


def test_exact_start_rejects_clock_and_rgb_changes(tmp_path):
    from jr_rollout import assert_start
    data=dict(sim_state=np.arange(4),proprio=np.arange(8),front=np.zeros((2,2,3),np.uint8),wrist=np.zeros((2,2,3),np.uint8))
    current={'fingerprint':{'clock':.5},'rollout_seed':20261022}
    np.savez(tmp_path/'START.npz',**data); (tmp_path/'START.json').write_text(json.dumps(current))
    assert_start(data,current,tmp_path)
    with pytest.raises(RuntimeError): assert_start(data,{'fingerprint':{'clock':.6}},tmp_path)
    data['wrist'][0,0,0]=1
    with pytest.raises(RuntimeError,match='wrist'): assert_start(data,current,tmp_path)


def test_receipt_resume_only_sealed_unchanged_results(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path); monkeypatch.setattr(common,'signature',lambda:'test')
    p=tmp_path/'EPISODE.json'; p.write_text('{"success":0}'); receipt=tmp_path/'COMPLETE.json'
    assert not common.complete(receipt)
    common.finish(receipt,[p]); assert common.complete(receipt)
    p.write_text('{"success":1}')
    with pytest.raises(RuntimeError,match='Changed artifact'): common.complete(receipt)


def test_two_camera_export_full_decode_and_resume(tmp_path):
    import imageio.v2 as imageio
    exporter=common.module('joint_test_exporter',common.PROJECT/'research_tools/export_singleview_videos.py')
    root=tmp_path/'run'; attempt=root/'episodes/libero_goal_t00_s02/clean/attempt_1'; attempt.mkdir(parents=True)
    (root/'data').mkdir(); (root/'data/MANIFEST.json').write_text(json.dumps({'signature':'test'}))
    meta=attempt/'EPISODE.json'; meta.write_text(json.dumps(dict(condition='clean',success=1)))
    for view,color in [('front',20),('wrist',210)]:
        with imageio.get_writer(attempt/(view+'.mp4'),fps=20,codec='libx264',ffmpeg_params=['-threads','1','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
            for _ in range(4): writer.append_data(np.full((224,224,3),color,np.uint8))
    assets=[meta,attempt/'front.mp4',attempt/'wrist.mp4']
    (attempt.parent/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',signature='test',
        artifacts={str(p.relative_to(root)):common.sha(p) for p in assets})))
    output=tmp_path/'videos'; first=exporter.export_run(root,output); second=exporter.export_run(root,output)
    assert first['episodes']==1 and len(first['files'])==4 and second['reused_files']==4
    assert len(list(output.glob('MP4/clean/*/*.mp4')))==2
    assert len(list(output.glob('IDE_WebM/clean/*/*.webm')))==2
    frames={}
    for v in ('front','wrist'):
        with imageio.get_reader(output/f'MP4/clean/{v}/libero_goal_t00_s02__task_success_1.mp4') as reader:
            frames[v]=reader.get_data(0).mean()
    assert frames['wrist']-frames['front']>150


def test_no_simulator_import_for_cpu_commands():
    # Imports above include rollout definitions, never the dynamic simulator runtime.
    assert 'mujoco' not in sys.modules and 'libero.libero.envs' not in sys.modules
    assert 'joint_native_runtime' not in sys.modules


def test_rollout_pairing_executed_slots_and_failure_preservation(tmp_path,monkeypatch):
    import torch
    import pandas as pd
    import jr_rollout as rollout
    cfg=common.config(); monkeypatch.setattr(common,'ROOT',tmp_path); monkeypatch.setattr(common,'signature',lambda:'test')
    monkeypatch.setattr(rollout,'ROOT',tmp_path); monkeypatch.setattr(rollout,'config',lambda:cfg)
    monkeypatch.setattr(rollout,'runtime_config',lambda:dict(num_steps_wait=10))
    monkeypatch.setattr(rollout,'switch_suite',lambda *a:None); seeds=[]
    monkeypatch.setattr(rollout,'seed_all',seeds.append)
    monkeypatch.setattr(rollout,'fingerprint',lambda e:dict(time=0,arrays='same-reset',observable_clocks={}))
    monkeypatch.setattr(rollout,'interface_gate',lambda *a:dict(status='PASS'))
    monkeypatch.setitem(sys.modules,'VLAAttacker.multiview.openvla_oft_experiments',SimpleNamespace(process_libero_action=lambda x:x))
    initial=np.arange(5,dtype=np.float64); initial_file=tmp_path/'initial.pt'; torch.save([initial]*4,initial_file)
    case=dict(start='libero_goal_t00_s02',suite='libero_goal',task_id=0,state_index=2,rollout_seed=20281022,
              task_name='task',initial_file=str(initial_file))
    changed=[False]; closed=[]; frame_counts=[]
    class Env:
        def __init__(self): self.n=0; self.env=SimpleNamespace(control_timestep=.05)
        def get_sim_state(self): return initial.copy()
        def raw(self): return dict(robot0_eef_pos=np.array([self.n*.01,0,0]),robot0_gripper_qpos=np.zeros(2))
        def step(self,command): self.n+=1; return self.raw(),0,self.n==2,{}
    def observation(raw,cfg):
        return dict(full_image=np.full((224,224,3),int(changed[0]),np.uint8),wrist_image=np.zeros((224,224,3),np.uint8),state=np.zeros(8))
    rt=SimpleNamespace(make_env=lambda *a:(Env(),SimpleNamespace(name='task',language='task'),[initial]*4),
        reset_wait=lambda env,*a:env.raw(),observation=observation,close_environment=lambda e:closed.append(e))
    monkeypatch.setattr(rollout,'runtime',lambda:rt)
    class Writer:
        def __init__(self,folder,fps): self.folder=folder; self.n=0
        def __enter__(self):
            for v in ('front','wrist'): (self.folder/(v+'.mp4')).write_bytes(b'test-encoded-elsewhere')
            return self
        def append(self,obs): self.n+=1
        def __exit__(self,*args): frame_counts.append(self.n)
    monkeypatch.setattr(rollout,'CameraWriters',Writer)
    wrapper=SimpleNamespace(gen_cfg=SimpleNamespace(unnorm_key='libero_goal_no_noops'),
        predict_action=lambda obs,instruction:np.full((8,7),.2 if obs['wrist_image'][137,26,0] else 0,dtype=np.float32))
    monitor=SimpleNamespace(stage='test',update=lambda **kw:None)
    clean=dict(name='clean',sha256=None); patch=dict(name='initial_s42',sha256='test')
    rollout.episode(wrapper,case,clean,None,monitor,True)
    rollout.episode(wrapper,case,patch,np.full((50,50,3),255,np.uint8),monitor,False)
    for row in (clean,patch):
        receipt=common.episode_receipt(case,row['name']); assert common.complete(receipt)
        files=common.episode_files(receipt); info=common.read(files['EPISODE.json'])
        trace=pd.read_parquet(files['TRACE.parquet']); steps=pd.read_parquet(files['STEPS.parquet'])
        assert info['success']==1 and info['executed_steps']==2 and info['max_steps']==300
        assert trace.iloc[0]['executed_slots']==2 and list(steps['executed_slot'])==[0,1]
        predicted=np.asarray(trace.iloc[0]['predicted_actions'].tolist()).reshape(8,7)
        assert np.allclose(predicted,0 if row['name']=='clean' else .2)
    assert seeds==[case['rollout_seed']]*2 and frame_counts==[3,3] and len(closed)==2
    changed[0]=True
    with pytest.raises(RuntimeError,match='front'):
        rollout.episode(wrapper,case,dict(name='bad',sha256='test'),None,monitor,False)
    assert not common.episode_receipt(case,'bad').exists() and len(list((tmp_path/'episodes'/case['start']/'bad').glob('attempt_*/FAILURE.json')))==1
    assert len(closed)==3


def test_status_never_submits_and_counts_partial_receipts(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli,'signature',lambda:'test')
    called=[]
    def fail(*args,**kwargs): called.append(args); raise AssertionError('Status attempted external process')
    monkeypatch.setattr(cli.subprocess,'run',fail)
    result=cli.status(False)
    assert result['completed']==0 and result['expected']==1040 and not result['prepared'] and called==[]


def test_joint_checkpoint_native_L1_code_is_identical():
    from jr_loading import code_compatibility
    cfg=common.runtime_config(); result=code_compatibility(common.config()['model_dir'],cfg['paths']['openvla_oft_root'])
    assert result['configuration_prismatic.py']['comparison']=='byte_identical'
    assert result['modeling_prismatic.py']['comparison'] in ('byte_identical','L1_AST_identical')


def test_only_exact_inactive_diffusion_delta_is_allowed(tmp_path):
    from jr_loading import code_compatibility,AUTO_MAP
    checkpoint=tmp_path/'checkpoint'; official=tmp_path/'native'; native=official/'prismatic/extern/hf'
    checkpoint.mkdir();native.mkdir(parents=True); (checkpoint/'config.json').write_text(json.dumps(dict(auto_map=AUTO_MAP)))
    for p in (checkpoint,native): (p/'configuration_prismatic.py').write_text('x=1\n')
    base='class Model:\n def active(self):\n  return 1\n def _run_diffusion_prediction(self,action_head):\n  return 2\n'
    changed=base.replace('  return 2','  action_head.noise_scheduler.set_timesteps(action_head.num_diffusion_steps)\n  return 2')
    (native/'modeling_prismatic.py').write_text(base); (checkpoint/'modeling_prismatic.py').write_text(changed)
    assert code_compatibility(checkpoint,official)['modeling_prismatic.py']['comparison']=='L1_AST_identical'
    (checkpoint/'modeling_prismatic.py').write_text(changed.replace('return 1','return 3'))
    with pytest.raises(RuntimeError,match='active model logic differs'): code_compatibility(checkpoint,official)


def test_suite_switches_both_native_normalization_keys(monkeypatch):
    import jr_rollout as rollout
    cfg=common.runtime_config(); w=SimpleNamespace(cfg=SimpleNamespace(),gen_cfg=SimpleNamespace(),
        vla=SimpleNamespace(norm_stats={k:{} for k in cfg['unnorm_keys'].values()}))
    for suite in common.config()['suites']:
        rollout.switch_suite(w,suite)
        assert w.cfg.task_suite_name==w.gen_cfg.task_suite_name==suite
        assert w.cfg.unnorm_key==w.gen_cfg.unnorm_key==suite+'_no_noops'
