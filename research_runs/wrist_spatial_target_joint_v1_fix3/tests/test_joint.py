"""CPU regression tests for branch state, budget, resume, hierarchy and renderer safety."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from jt_math import translations,hierarchy_weights,choose,counts,feature_loss,realization
import jt_train as train

def test_relative_displacements_and_degenerate_direction():
    delta=translations([1,0,5],[0,0,2])
    np.testing.assert_allclose(delta,[[-.02,0,0],[.02,0,0],[0,.02,0],[0,-.02,0]],atol=1e-14)
    assert translations([0,0,5],[0,0,2]) is None
    np.testing.assert_allclose(np.linalg.norm(delta,axis=1),.02)

def test_balancing_does_not_overweight_tasks_with_two_trajectories():
    rows=[{'suite':'a','task_id':0},{'suite':'a','task_id':0},{'suite':'a','task_id':1},{'suite':'b','task_id':0}]
    np.testing.assert_allclose(hierarchy_weights(rows),[.125,.125,.25,.5])
    harm=np.array([[1,-1,0,0],[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]])
    q=np.ones((4,4));assert choose(harm,q,hierarchy_weights(rows),'joint')[0]==0
    assert choose(harm,q,hierarchy_weights(rows),'realizability_only')[0]==0
    np.testing.assert_allclose(realization([0,1,2],[1,1,1]),[1,0,0])

def test_feature_loss_penalizes_orthogonal_errors_and_keeps_gradient():
    goal=torch.tensor([[1.,0.],[1.,0.]])
    feature=torch.tensor([[1.,2.],[1.,2.]],requires_grad=True)
    loss=feature_loss(feature,goal,torch.tensor([.5,.5]),1.)
    assert float(loss)==2.
    gradient=torch.autograd.grad(loss,feature)[0]
    assert torch.isfinite(gradient).all() and float(gradient[:,1].norm())>0

def test_adam_snapshot_does_not_alias_rejected_branch():
    p=torch.nn.Parameter(torch.tensor([.2,.5]));o=torch.optim.Adam([p],lr=.01)
    p.grad=torch.ones_like(p);o.step();saved=train.snapshot(p,o)
    saved_moment=saved['optimizer']['state'][0]['exp_avg'].clone()
    for _ in range(3):
        train.restore(saved,p,o);p.grad=torch.tensor([-3.,2.]);o.step()
        assert torch.equal(saved['optimizer']['state'][0]['exp_avg'],saved_moment)
    train.restore(saved,p,o)
    assert torch.equal(p.detach(),saved['patch'])
    assert int(o.state[p]['step'])==1

def test_scene_hook_restores_positions_and_function_on_error(monkeypatch):
    from jt_cache import SceneTranslation
    geom=SimpleNamespace(objtype=5,objid=3,pos=np.array([1.,2.,3.]))
    scene=SimpleNamespace(ngeom=1,geoms=[geom]);original=geom.pos.copy()
    def fail(*args,**kwargs):raise RuntimeError('renderer failed')
    fake=SimpleNamespace(mjr_render=fail);monkeypatch.setitem(sys.modules,'mujoco',fake)
    with pytest.raises(RuntimeError,match='renderer failed'):
        with SceneTranslation(scene,[3],[.02,0,0]):fake.mjr_render(None,scene,None)
    assert fake.mjr_render is fail
    np.testing.assert_array_equal(geom.pos,original)

class ToyEngine:
    def __init__(self,fail_at=None):
        self.wrapper=SimpleNamespace(device='cpu',vla=SimpleNamespace(language_model=torch.nn.Identity()))
        self.dino=torch.nn.Identity();self.siglip=torch.nn.Identity();self.calls=0;self.fail_at=fail_at
    def verify_patch_path(self,row):pass
    def spatial_patch(self,obs,patch):
        if torch.is_grad_enabled():
            self.calls+=1
            if self.calls==self.fail_at:raise RuntimeError('simulated interruption')
        x=patch[:,::25,::25].mean(0)
        return self.dino(x)[None],patch

def setup_toy(monkeypatch,root,fail_at=None):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    (root/'data').mkdir(parents=True,exist_ok=True)
    torch.save({'token_weights':torch.tensor([.5,.5]),'scale':1.},root/'data/FEATURE_STATS.pt')
    rows=[{'suite':s,'task_id':0,'state_id':s} for s in cfg['suites']]
    pool={r['state_id']:{'row':{'state_id':r['state_id'],'observation':{},'teacher':{}},'clean_feature':torch.zeros(2,2),
          'goals':torch.stack([torch.full((2,2),v) for v in (.2,.4,.7,.9)]),'harm':np.array([.1,.2,.3,.4])} for r in rows}
    engine=ToyEngine(fail_at)
    monkeypatch.setattr(train,'ROOT',root);monkeypatch.setattr(train,'config',lambda:cfg)
    monkeypatch.setattr(train,'signature',lambda:'toy');monkeypatch.setattr(train,'eligible',lambda cohort:rows)
    monkeypatch.setattr(train,'cached',lambda r:pool[r['state_id']])
    monkeypatch.setattr(train,'read',lambda p:{'proceed':True,'fixed_rule':3})
    monkeypatch.setattr(train,'make_engine',lambda checkpointing=False:engine)
    monkeypatch.setattr(train,'complete',lambda p:p.exists() or p.name=='TARGETS_COMPLETE.json')
    monkeypatch.setattr(train,'finish',lambda p,files,**fields:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(fields))))
    monkeypatch.setattr(train.native,'require_gpu',lambda stage:None)
    monkeypatch.setattr(train.native,'load_png',lambda p,device:torch.full((3,50,50),.42))
    monitor=SimpleNamespace(update=lambda **kwargs:None)
    return cfg,monitor,engine

def test_full_joint_budget_and_interrupted_resume_are_identical(monkeypatch,tmp_path):
    full=tmp_path/'full';cfg,monitor,_=setup_toy(monkeypatch,full)
    train.run('joint',monitor)
    result=torch.load(full/'training/joint/LAST.pt',map_location='cpu',weights_only=False)
    assert result['counts']=={'charged_backwards':4000,'accepted_updates':3775,'discarded_probe_updates':225}
    assert len(result['selection'])==15
    resumed=tmp_path/'resumed';setup_toy(monkeypatch,resumed,fail_at=613)
    with pytest.raises(RuntimeError,match='simulated interruption'):train.run('joint',monitor)
    saved=torch.load(resumed/'training/joint/LAST.pt',map_location='cpu',weights_only=False)
    assert saved['budget']==600
    setup_toy(monkeypatch,resumed)
    train.run('joint',monitor)
    actual=torch.load(resumed/'training/joint/LAST.pt',map_location='cpu',weights_only=False)
    assert torch.equal(actual['state']['patch'],result['state']['patch'])
    assert actual['selection']==result['selection']
    assert actual['counts']==result['counts']
    for key in ('exp_avg','exp_avg_sq','step'):
        assert torch.equal(actual['state']['optimizer']['state'][0][key],result['state']['optimizer']['state'][0][key])

def test_fixed_and_adaptive_budget_counts():
    cfg={'adaptive_budget':3000,'cycle_budget':200,'probe_updates':5}
    assert counts('fixed_harm',4000,cfg)['accepted_updates']==4000
    assert counts('joint',200,cfg)['accepted_updates']==185
    assert counts('joint',3000,cfg)['accepted_updates']==2775

def test_submission_has_two_dependency_lanes_and_never_executes_sbatch(monkeypatch,tmp_path,capsys):
    import jt_cli as cli
    monkeypatch.setattr(cli,'check_launchers',lambda:None)
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli,'signature',lambda:'a'*64)
    monkeypatch.setattr(cli,'complete',lambda path:False);monkeypatch.setattr(cli,'done',lambda stage:False)
    monkeypatch.setattr(cli,'lock',lambda name:nullcontext())
    monkeypatch.setattr(cli,'queue',lambda:pytest.fail('dry run queried live queue'))
    cli.submit(dry=True);lines=capsys.readouterr().out.splitlines()
    assert len(lines)==13
    assert 'afterok:DRY2' in next(line for line in lines if line.endswith('cache0'))
    assert 'afterok:DRY2' in next(line for line in lines if line.endswith('cache1'))
    gate=next(line for line in lines if line.endswith('replay_check'))
    assert '/cache.sbatch' in gate and 'afterok:DRY1' in gate
    assert 'afterok:DRY6' in next(line for line in lines if line.endswith('train2'))
    assert 'afterok:DRY7' in next(line for line in lines if line.endswith('train3'))
    assert 'afterok:DRY8:DRY9' in next(line for line in lines if line.endswith('select'))
    assert 'afterany:DRY11:DRY12' in lines[-1]

def test_declared_resources_and_no_automatic_real_robot_or_cancel():
    root=Path(__file__).resolve().parents[1]
    for stage,memory in [('cache',12),('train',16),('evaluate',12)]:
        text=(root/'slurm'/f'{stage}.sbatch').read_text()
        assert '#SBATCH --gres=gpu:4090:1' in text and '#SBATCH --cpus-per-task=4' in text
        assert f'#SBATCH --mem={memory}G' in text
    assert '#SBATCH --gres' not in (root/'slurm/cpu.sbatch').read_text()

def test_validation_cannot_change_feature_scale_or_fixed_target(monkeypatch,tmp_path):
    import jt_common as common
    import jt_prepare as prep
    entries=[];payloads={}
    for cohort in ('fit','verify'):
        for i,suite in enumerate(common.config()['suites']):
            row={'state_id':cohort+str(i),'suite':suite,'task_id':0,'cohort':cohort};entries.append(row)
            factor=1 if cohort=='fit' else 100
            payloads[row['state_id']]={'valid':True,'invalid_reasons':[],'clean_feature':torch.zeros(256,2),
                'goals':torch.stack([torch.full((256,2),factor*v) for v in (1.,2.,3.,4.)]),
                'harm':np.array([0,.1,.2,.3]) if cohort=='fit' else np.array([100,0,0,0])}
    monkeypatch.setattr(common,'selected',lambda:entries)
    monkeypatch.setattr(common,'cached',lambda row:payloads[row['state_id']])
    monkeypatch.setattr(prep,'ROOT',tmp_path);monkeypatch.setattr(prep,'signature',lambda:'test')
    monkeypatch.setattr(prep,'complete',lambda p:False)
    monkeypatch.setattr(prep,'finish',lambda *args,**kwargs:None)
    prep.finalize(SimpleNamespace(update=lambda **kwargs:None))
    result=json.loads((tmp_path/'data/TARGET_STATS.json').read_text())
    assert result['fixed_rule']==3 and result['fit_only_statistics']
    assert result['scale']==pytest.approx(7.5)

def test_partial_report_keeps_clean_failures_out_of_casr(monkeypatch,tmp_path):
    import jt_report as report
    (tmp_path/'data').mkdir();(tmp_path/'results').mkdir();baseline=[];cases=[]
    for i,clean in enumerate((1,0)):
        case={'start':f't{i}','suite':'libero_goal','task_id':i,'state_index':2};cases.append(case)
        path=tmp_path/f'clean{i}.json';path.write_text(json.dumps({'success':clean}))
        baseline.append({'start':case['start'],'condition':'clean','episode':str(path)})
        folder=tmp_path/'episodes'/case['start']/'joint';folder.mkdir(parents=True)
        episode=folder/'EPISODE.json';episode.write_text(json.dumps({'success':1-clean}))
        (folder/'COMPLETE.json').write_text(json.dumps({'artifacts':{str(episode.relative_to(tmp_path)):'test'}}))
    (tmp_path/'data/MANIFEST.json').write_text(json.dumps({'cases':cases,'baselines':baseline}))
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'complete',lambda p:p.exists())
    monkeypatch.setattr(report,'lock',lambda name:nullcontext())
    report.run(SimpleNamespace(update=lambda **kwargs:None),export=False)
    result=json.loads((tmp_path/'results/STATUS.json').read_text())
    joint=next(r for r in result['summary'] if r['condition']=='joint')
    assert result['status']=='PARTIAL'
    assert joint['CASR']==1 and joint['clean_success_pairs']==1
    assert joint['robot_task_SR']==.5 and joint['episodes_completed']==2

def test_physical_conditions_use_fresh_environments_and_close_each():
    from jt_cache import execute_paired_branches
    created=[];closed=[];expected={'time':1,'arrays':{'state':'same'},'observable_clocks':{}}
    def start():
        env=SimpleNamespace(identity=len(created),hidden_contact_memory=0)
        created.append(env);return env,{},None,copy.deepcopy(expected)
    def execute(env,raw,target,command):
        assert env.hidden_contact_memory==0
        env.hidden_contact_memory=command+1
        return {'distance_to_original_anchor_m':float(command),'executed_slots':8}
    def close(env):closed.append(env.identity)
    first=start()
    result=execute_paired_branches(list(range(5)),first,start,execute,close,expected)
    assert len(created)==5 and closed==list(range(5))
    assert len(result)==5 and all(v['environment_isolation']=='fresh_environment_for_each_condition' for v in result)

def test_physical_environment_closed_when_execution_or_alignment_fails():
    from jt_cache import execute_paired_branches
    closed=[];expected={'arrays':{'state':'expected'}}
    env=object()
    def fail(*args):raise RuntimeError('controller failed')
    with pytest.raises(RuntimeError,match='controller failed'):
        execute_paired_branches([0],(env,{},None,expected),lambda:None,fail,closed.append,expected)
    assert closed==[env]
    closed.clear()
    with pytest.raises(RuntimeError,match='physics/camera/clock'):
        execute_paired_branches([0],(env,{},None,{'arrays':{'state':'different'}}),lambda:None,fail,closed.append,expected)
    assert closed==[env]

def test_fingerprint_diagnostics_do_not_relax_equality():
    from jt_cache import fingerprint_difference
    expected={'arrays':{'sim_state':'a','cam_xpos':'same'},'time':1.,'observable_clocks':{'wrist':{'enabled':True}}}
    actual={'arrays':{'sim_state':'b','cam_xpos':'same'},'time':1.,'observable_clocks':{'wrist':{'enabled':False}}}
    diff=fingerprint_difference(expected,actual)
    assert diff['array_hashes_changed']==['sim_state'] and 'wrist' in diff['observable_clocks_changed']
    assert expected!=actual

def test_environment_constructor_random_stream_is_replayed():
    import random
    from jt_cache import constructor_rng,restore_constructor_rng
    saved=constructor_rng()
    first=(np.random.rand(3),random.random(),torch.rand(3))
    np.random.rand(50);random.random();torch.rand(50)
    restore_constructor_rng(saved)
    actual=(np.random.rand(3),random.random(),torch.rand(3))
    np.testing.assert_array_equal(actual[0],first[0]);assert actual[1]==first[1]
    assert torch.equal(actual[2],first[2])

def test_real_worker_report_lock_is_not_reentered(monkeypatch,tmp_path):
    import jt_common as common
    import jt_cli as cli
    import jt_report as report
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'signature',lambda:'test')
    monkeypatch.setattr(cli,'signature',lambda:'test');monkeypatch.setenv('JT_EXPECTED_SIGNATURE','test')
    reached=[]
    def actual_report(monitor):
        with common.lock('report'):reached.append(True)
    monkeypatch.setattr(report,'run',actual_report)
    cli.worker('report')
    assert reached==[True]
    assert (tmp_path/'reports/locks/worker_report.lock').exists() and (tmp_path/'reports/locks/report.lock').exists()

def test_repair_reuses_only_visuals_and_keeps_parent_immutable(monkeypatch,tmp_path):
    import jt_prepare as prepare
    parent=tmp_path/'parent';(parent/'data/cache').mkdir(parents=True);(parent/'reports/COSTS').mkdir(parents=True)
    rows=[{'state_id':'state0'}];model={'weights':[100,123]};old_cfg={'scientific_parameter':1}
    cfg={**old_cfg,'repair_parent_run':'parent','repair_parent_signature':'old'}
    (parent/'config.json').write_text(json.dumps(old_cfg))
    (parent/'data/MANIFEST.json').write_text(json.dumps({'signature':'old','selected':rows,'model_file_stats':model}))
    path=parent/'data/cache/state0.pt'
    torch.save({'valid':True,'outcomes':[{'steps':[]} for _ in range(5)]},path)
    receipt=path.with_suffix('.json');receipt.write_text('{}')
    hashes={str(p):prepare.sha(p) for p in (path,receipt,parent/'config.json',parent/'data/MANIFEST.json')}
    monkeypatch.setattr(prepare,'PROJECT',tmp_path);monkeypatch.setattr(prepare,'config',lambda:cfg)
    monkeypatch.setattr(prepare,'module',lambda *args:SimpleNamespace(signature=lambda:'old'))
    monkeypatch.setattr(prepare,'verify_external',lambda *args:{'status':'COMPLETE'})
    inputs={};reused,costs=prepare.reusable_visuals(inputs,rows,{'model_file_stats':model})
    assert reused['state0']['reuse_scope']=='visual_features_and_teacher_actions_only_not_physics'
    assert hashes=={p:prepare.sha(p) for p in hashes}

def test_actual_batch_entrypoints_resolve_same_revision_without_gpu():
    import jt_cli as cli
    assert cli.check_launchers()==['cache','train','evaluate','cpu']

def test_old_environment_root_is_rejected_before_submission(monkeypatch,tmp_path):
    import jt_cli as cli
    actual_root=cli.ROOT
    folder=tmp_path/'slurm';folder.mkdir()
    # Reproduce the observed failure: the fixed batch sources an environment
    # whose JT_ROOT still names the old run. Execute the real shell and old CLI.
    environment=(actual_root/'slurm/environment.sh').read_text()
    parent=actual_root.parent/'wrist_spatial_target_joint_v1'
    environment=environment.replace(environment.splitlines()[1],f'JT_ROOT="{parent}"')
    (folder/'environment.sh').write_text(environment)
    script=(actual_root/'slurm/cache.sbatch').read_text().replace(
        str(actual_root/'slurm/environment.sh'),str(folder/'environment.sh'))
    (folder/'cache.sbatch').write_text(script)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'queue',lambda:pytest.fail('failed preflight must not query or submit Slurm'))
    with pytest.raises(RuntimeError,match='Batch launcher check failed for cache'):
        cli.submit()

def test_reset_history_uses_original_workers_and_all_trajectories():
    from jt_history import source_histories
    # The unselected trajectory still contributes a reset in the original worker.
    episodes=[]
    for cohort,states in [('fit',(40,41)),('verify',(42,43))]:
        for task in (0,4):
            for state in states:
                episodes.append({'suite':'spatial','task_id':task,'state_index':state,
                    'episode_id':f'{task}_{state}','rollout_seed':state,'cohort':cohort})
    history=source_histories(episodes)
    assert history['4_40']==[] and history['4_41']==[]
    assert [e['episode_id'] for e in history['4_42']]==['4_40']
    assert [e['episode_id'] for e in history['4_43']]==['4_41']

def test_installed_sampler_bug_and_historical_warmup_restore_rng(monkeypatch):
    import ast
    from jt_history import replay_previous_resets
    # Execute installed pure-Python definitions, without importing the simulator.
    base=Path('/home/caijiasheng25/Desktop/LIBERO/libero/libero/envs')
    scope={'np':np,'copy':copy.copy,'MujocoObject':type('Unused',(),{})}
    nodes=[n for n in ast.parse((base/'regions/object_property_sampler.py').read_text()).body
           if isinstance(n,ast.ClassDef) and n.name in ('ObjectPropertySampler','OpenCloseSampler')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'installed_samplers','exec'),scope)
    cls=next(n for n in ast.parse((base/'bddl_base_domain.py').read_text()).body
             if isinstance(n,ast.ClassDef) and n.name=='BDDLBaseDomain')
    add=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_add_placement_initializer')
    exec(compile(ast.Module(body=[add],type_ignores=[]),'installed_initializer','exec'),scope)
    def environment():
        inner=SimpleNamespace(hard_reset=True,deterministic_reset=False,
            parsed_problem={'fixtures':{},'objects':{},'regions':{},'initial_state':[['open','cabinet']],
                            'problem_name':'minimal'},objects_dict={},fixtures_dict={},
            object_states_dict={'cabinet':SimpleNamespace(set_joint=lambda x:None)},object_property_initializers=[])
        inner.get_object=lambda name:SimpleNamespace(name=name,object_properties={'articulation':{'default_open_ranges':[0.,1.]}})
        for _ in range(2):scope['_add_placement_initializer'](inner)
        return SimpleNamespace(env=inner,seed=np.random.seed)
    def reset(env):
        scope['_add_placement_initializer'](env.env)
        for sampler in env.env.object_property_initializers:sampler.sample()
    runtime=SimpleNamespace(bind_environment_context=lambda e:None,reset_environment=reset,renderer_metadata=lambda e:None)
    # Historical source: first episode then second, with its own seed each time.
    original=environment();original.seed(40);reset(original);original.seed(42);reset(original)
    expected=np.random.random(5)
    # Old fix2: fresh environment, only current reset -> the observed mismatch.
    fresh=environment();fresh.seed(42);reset(fresh)
    assert not np.array_equal(np.random.random(5),expected)
    # fix3: replay earlier resets BEFORE setting current seed.
    repaired=environment()
    entry={'suite':'s','task_id':4,'source_reset_history':[{'suite':'s','task_id':4,'rollout_seed':40,'episode_id':'fit'}]}
    proof=replay_previous_resets(repaired,entry,runtime)
    repaired.seed(42);reset(repaired)
    np.testing.assert_array_equal(np.random.random(5),expected)
    assert len(proof['before'])==2 and len(repaired.env.object_property_initializers)==4

def test_history_rejects_soft_reset_or_other_task():
    from jt_history import replay_previous_resets
    env=SimpleNamespace(env=SimpleNamespace(hard_reset=False,deterministic_reset=False))
    with pytest.raises(RuntimeError,match='hard-reset'):
        replay_previous_resets(env,{},None)
    env.env.hard_reset=True;env.env.object_property_initializers=[]
    with pytest.raises(RuntimeError,match='task boundary'):
        replay_previous_resets(env,{'suite':'a','task_id':4,
            'source_reset_history':[{'suite':'a','task_id':0}]},None)

def test_complete_cache_migration_preserves_bytes_and_rejects_bad_pair(monkeypatch,tmp_path):
    import jt_prepare as prep
    root=tmp_path/'new';(root/'data/cache').mkdir(parents=True)
    parent=tmp_path/'old.pt';fingerprint={'arrays':{'physics':'same'}}
    payload={'all_physics_recomputed_with_fresh_environments':True,
        'capture':{'fingerprint':fingerprint},'row':{'sample_fingerprint':fingerprint,'cohort':'fit'},
        'paired_decision_fingerprint':fingerprint,'outcomes':[{'exact_start_fingerprint':fingerprint,
            'environment_isolation':'fresh_environment_for_each_condition'} for _ in range(5)]}
    torch.save(payload,parent);before=prep.sha(parent)
    monkeypatch.setattr(prep,'ROOT',root)
    import jt_common as common
    monkeypatch.setattr(common,'ROOT',root)
    monkeypatch.setattr(prep,'finish',lambda path,*args,**kwargs:path.write_text(json.dumps(kwargs)))
    info={'path':str(parent),'sha256':before,'receipt':'old.json'}
    assert prep.reuse_complete_caches({'state':info})==['state']
    assert (root/'data/cache/state.pt').is_symlink() and prep.sha(parent)==before
    payload['outcomes'][0]['exact_start_fingerprint']={'arrays':{'physics':'wrong'}}
    torch.save(payload,parent)
    with pytest.raises(RuntimeError,match='paired starts'):
        prep.reuse_complete_caches({'bad':dict(info,sha256=prep.sha(parent))})
    assert not (root/'data/cache/bad.pt').exists()

def test_cache_cannot_bypass_replay_startup_check(monkeypatch):
    import jt_cache as cache
    monkeypatch.setattr(cache.native,'require_gpu',lambda stage:None)
    monkeypatch.setattr(cache,'complete',lambda path:False)
    with pytest.raises(RuntimeError,match='startup check'):
        cache.run(0,SimpleNamespace())

@pytest.mark.parametrize('old_prefix',['stj1_','stjf1_','stjf2_'])
def test_active_previous_revision_blocks_submission(monkeypatch,old_prefix):
    import jt_cli as cli
    monkeypatch.setattr(cli,'check_launchers',lambda:None)
    monkeypatch.setattr(cli,'lock',lambda name:nullcontext())
    monkeypatch.setattr(cli,'queue',lambda:[['123','RUNNING',old_prefix+'old_cache0','node']])
    with pytest.raises(RuntimeError,match='Other revision'):
        cli.submit()
