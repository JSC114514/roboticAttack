"""CPU checks for objective fidelity, shared conditions and resource-safe execution."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import fc_common as common
import fc_math as math
import fc_train as train
import fc_evaluate as evaluate
import fc_report as report
import fc_cli as cli


def test_uada_targets_respect_unmasked_gripper_and_all_slots():
    clean=torch.tensor([[[-.2,0.,.3,-.4,.2,0.,.49],[.2,-.1,0.,0.,-.4,.8,.5]]]*4).reshape(1,8,7)
    target=math.far_targets(clean)
    assert target.shape==(1,8,7)
    assert target[0,0].tolist()==[1,-1,-1,1,-1,-1,1]
    assert target[0,1].tolist()==[-1,1,-1,-1,1,-1,0]
    assert set(target[...,6].flatten().tolist())=={0.,1.}
    lo=clean.new_tensor([-1]*6+[0]);hi=torch.ones(7)
    assert torch.all((target-clean).abs()>=torch.minimum((hi-clean).abs(),(lo-clean).abs()))


@pytest.mark.parametrize('method,dims',[('uada_clean_xyz',3),('uada_clean_7dof',7)])
def test_objective_has_only_declared_prediction_gradients(method,dims):
    action=torch.full((1,8,7),.2,requires_grad=True);clean=torch.full_like(action,.3,requires_grad=True)
    loss=math.action_loss(action,clean,method)
    grad,reference_grad=torch.autograd.grad(loss,(action,clean),allow_unused=True)
    assert reference_grad is None
    assert torch.all(grad[...,:dims]!=0) and torch.all(grad[...,dims:]==0)
    target=math.far_targets(clean)
    assert torch.allclose(loss,torch.nn.functional.mse_loss(action[...,:dims],target[...,:dims]))


@pytest.mark.parametrize('method',['uada_clean_xyz','uada_clean_7dof'])
def test_staged_two_branch_gradient_matches_full_nonlinear_rgb_gradient(monkeypatch,method):
    ref=train.Engine.staged_gradient.__globals__['ref']
    def pixels(_obs,patch,_wrapper):return patch[None].repeat(1,4,1,1),None
    monkeypatch.setattr(ref,'patch_pixels',pixels)
    class TinyEngine:
        wrapper=None
        def spatial(self,p):return p[:,6:9].square().mean((-1,-2))[:,None]
        def siglip(self,p):return p.sin().mean((-1,-2))[:,None]
        def command(self,a):return a[...,:3]
        def action(self,_obs,_lang,p=None,replacements=None):
            caps=replacements or {'dino':self.spatial(p),'siglip':self.siglip(p[:,9:12])}
            d,s=caps['dino'],caps['siglip']
            action=torch.cat([d+s.square(),s+d.sin(),(s+d).mean(-1,keepdim=True)],-1).expand(1,8,7)
            return action,caps
    engine=TinyEngine();patch=torch.full((3,2,2),.3,requires_grad=True)
    row={'observation':{},'language':'','teacher':{'clean':torch.full((1,8,7),.2),'unit':[1.,0.,0.]}}
    staged,metrics=train.Engine.staged_gradient(engine,row,patch,method)
    action,_=engine.action({},'',pixels({},patch,None)[0]);loss=math.action_loss(action,row['teacher']['clean'],method)
    direct=torch.autograd.grad(loss,patch)[0]
    assert torch.allclose(staged,direct,rtol=1e-6,atol=1e-7)
    assert metrics['dino_gradient_l2']>0 and metrics['siglip_gradient_l2']>0


def test_private_trainer_preserves_original_sampler_optimizer_and_output_root():
    module=train.training_module()
    assert module.ROOT==common.ROOT and module.config is common.config
    assert module.records is common.records and module.make_engine is train.make_engine
    assert 'action_loss' in train.Engine.staged_gradient.__code__.co_names
    assert 'objective' not in train.Engine.staged_gradient.__code__.co_names
    assert train.Engine.gate.__globals__['ROOT']==common.ROOT
    assert [m for _,m,_ in common.candidates()]==['uada_clean_xyz','uada_clean_7dof']*2


def test_sampler_matches_original_sealed_relations_for_both_seeds():
    rows=list(common.source.records('fit'))
    assert len(rows)==237 and sum(not r['teacher']['no_target'] for r in rows)==217
    for seed in (42,43):
        expected=math.sample_schedule(rows,seed)
        folder=common.PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1/training'/f'matched_relation_s{seed}'
        saved=common.read(folder/'COMPLETE.json');observed={}
        for p in saved['artifacts']:
            if p.endswith('.jsonl'):
                for line in (folder.parents[1]/p).read_text().splitlines():
                    value=json.loads(line);observed[value['step']]=value['state_id']
        assert [observed[i+1] for i in range(4000)]==expected


def test_clean_and_patch_first_forward_checks(monkeypatch):
    calls=[]
    wrapper=SimpleNamespace(predict_action=lambda *_a,**_k:(calls.append(1) or np.zeros((8,7))))
    clean=evaluate.FirstCleanCheck(wrapper,None);clean.predict_action({},'')
    assert clean.error==0 and len(calls)==2
    patch=evaluate.FirstCleanCheck(wrapper,np.zeros((8,7)));patch.predict_action({},'');assert patch.error is None
    patch.predict_action({},'');assert patch.error==0
    patch=evaluate.FirstCleanCheck(wrapper,np.full((8,7),2e-5));patch.predict_action({},'')
    with pytest.raises(RuntimeError,match='First clean'):patch.predict_action({},'')


@pytest.mark.parametrize('fail',[False,True])
def test_clean_before_patch_fresh_resets_and_cleanup(tmp_path,monkeypatch,fail):
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    names=['clean','patch'];monkeypatch.setattr(evaluate,'condition_names',lambda:names)
    def receipt(c,n):return tmp_path/'episodes'/c['start']/n/'COMPLETE.json'
    monkeypatch.setattr(evaluate,'episode_receipt',receipt)
    monkeypatch.setattr(evaluate,'complete',lambda p:Path(p).exists())
    def finish(p,files):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    monkeypatch.setattr(evaluate,'finish',finish)
    monkeypatch.setattr(evaluate,'episode_files',lambda p:{'TRACE.parquet':tmp_path/'TRACE.parquet'})
    monkeypatch.setattr(evaluate.pd,'read_parquet',lambda *_:pd.DataFrame([{'predicted_actions':np.zeros((8,7))}]))
    monkeypatch.setattr(evaluate.native,'runtime_config',lambda:{})
    initial=tmp_path/'initial.pt';torch.save([np.zeros(2)],initial)
    case=dict(start='test',task_name='task',suite='libero_goal',task_id=0,state_index=0,initial_file=str(initial),reference_start=str(tmp_path/'paired'))
    created=[];closed=[];draws=[]
    def make_env(*_):
        value=object();created.append(value);draws.append((np.random.rand(),torch.rand(1).item()))
        return value,SimpleNamespace(name='task'),[np.zeros(2)]
    runtime=SimpleNamespace(make_env=make_env,close_environment=closed.append)
    def episode(env,task,init,case,name,patch,checked,attempt,monitor,external):
        checked.predict_action({},'')
        if name!='clean':checked.predict_action({},'')
        if fail:raise RuntimeError('injected failure')
        p=Path(case['reference_start']);p.mkdir(exist_ok=True);(p/'START.json').write_text('{}')
    monitor=SimpleNamespace(data={'completed':0},update=lambda **_:None)
    wrapper=SimpleNamespace(predict_action=lambda *_a,**_k:np.zeros((8,7)))
    if fail:
        with pytest.raises(RuntimeError,match='injected'):evaluate.run_case(runtime,wrapper,episode,case,dict.fromkeys(names),dict.fromkeys(names),monitor)
        assert len(created)==1
    else:
        evaluate.run_case(runtime,wrapper,episode,case,dict.fromkeys(names),dict.fromkeys(names),monitor)
        assert len(created)==2 and created[0] is not created[1] and draws[0]==draws[1]
    assert created==closed


def test_completed_worker_does_not_load_model(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    monkeypatch.setattr(evaluate,'verify',lambda:{})
    monkeypatch.setattr(evaluate,'conditions',lambda:dict.fromkeys(common.condition_names()))
    monkeypatch.setattr(evaluate,'cases',lambda _: [dict(start=str(i)) for i in range(40)])
    monkeypatch.setattr(evaluate,'complete',lambda p:p.name=='COMPLETE.json')
    monkeypatch.setattr(evaluate,'load_wrapper',lambda *_:pytest.fail('Unnecessary model load'))
    monkeypatch.setattr(evaluate.native,'runtime_module',lambda:pytest.fail('Unnecessary simulator import'))
    result=[];monkeypatch.setattr(evaluate,'finish',lambda p,files,**kw:result.append(len(files)))
    evaluate.run(0,'original40',SimpleNamespace(update=lambda **_:None));assert result==[260]


def test_statistics_separate_old_new_missing_and_clean_failed_starts(monkeypatch):
    monkeypatch.setattr(report,'config',lambda:{'bootstrap_draws':20})
    records=[]
    for cohort in ('original40','expanded40'):
        for i in range(4):
            for name in ('clean','matched_relation_s42','uada_clean_xyz_s42'):
                if cohort=='expanded40' and name=='uada_clean_xyz_s42':continue
                failed=(name!='clean' and i==0) or i==3
                records.append(dict(cohort=cohort,start=cohort+str(i),suite='libero_goal',task_id=0,condition=name,
                    clean_success=int(i!=3),success=int(not failed),induced_failure=int(name!='clean' and i==0)))
    summary,contrasts=report.statistics(pd.DataFrame(records));look={(r['cohort'],r['condition']):r for r in summary}
    assert look['original40','matched_relation_s42']['robot_task_SR']==.5
    assert look['original40','matched_relation_s42']['CASR']==1/3
    assert look['expanded40','uada_clean_xyz_s42']['CASR'] is None
    pair=next(r for r in contrasts if r['cohort']=='all80' and r['b']=='uada_clean_xyz_s42')
    assert pair['paired']==3 and pair['CASR_difference']==0


def test_audited_episode_bound_to_new_root_only():
    fn=evaluate.rollout_functions()
    assert fn.__globals__['ROOT']==common.ROOT and fn.__globals__['conditions'] is common.conditions


def test_manual_dry_run_dependency_graph_has_two_gpu_lanes(monkeypatch,capsys):
    monkeypatch.setattr(cli,'check_launchers',lambda:None);monkeypatch.setattr(cli,'complete',lambda _:False)
    monkeypatch.setattr(cli,'done',lambda stage:stage=='prepare')
    monkeypatch.setattr(cli,'queue',lambda:pytest.fail('Dry-run contacted Slurm'))
    cli.submit(True);lines=capsys.readouterr().out.splitlines()
    assert len(lines)==11
    assert '--dependency=afterok:DRY1' in lines[2] and '--dependency=afterok:DRY2' in lines[3]
    assert '--dependency=afterok:DRY3:DRY4' in lines[4]
    assert '--dependency=afterok:DRY6:DRY7' in lines[8] and '--dependency=afterok:DRY6:DRY7' in lines[9]
    assert '--dependency=afterany:DRY9:DRY10:DRY8' in lines[10]


def test_active_revision_is_not_duplicated(monkeypatch,capsys):
    monkeypatch.setattr(cli,'check_launchers',lambda:None);monkeypatch.setattr(cli,'complete',lambda _:False)
    monkeypatch.setattr(cli,'queue',lambda:[['1','RUNNING','srfair1_old_train0','node']])
    monkeypatch.setattr(cli.subprocess,'run',lambda *_a,**_k:pytest.fail('Duplicate submitted'))
    cli.submit();assert 'ALREADY_ACTIVE' in capsys.readouterr().out


def test_real_batch_launchers_and_resources():
    assert cli.check_launchers()==['train','evaluate','cpu']
    for kind,memory,cpus in [('train','16G',4),('evaluate','12G',4),('cpu','4G',2)]:
        text=(common.ROOT/'slurm'/(kind+'.sbatch')).read_text()
        assert f'--mem={memory}' in text and f'--cpus-per-task={cpus}' in text
        assert ('--gres=gpu:4090:1' in text)==(kind!='cpu')
