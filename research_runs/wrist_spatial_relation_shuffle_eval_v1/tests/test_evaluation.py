"""CPU regression tests for provenance, strict pairing, resume and manual dispatch."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import se_common as common
import se_prepare as prepare
import se_evaluate as evaluate
import se_report as report
import se_cli as cli


def test_four_final_assets_direct_sources_and_roundtrip():
    original=common.read(common.PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1/data/MANIFEST.json')
    for spec in common.config()['assets']:
        asset,paths=prepare.attest_asset(spec,original)
        parent='wrist_spatial_relation_shuffle_v1' if spec['method']=='matched_relation' else 'wrist_spatial_relation_shuffle_v1_fix1'
        assert Path(asset['path'])==common.PROJECT/'research_runs'/parent/'training'/spec['condition']/'checkpoint_004000.png'
        assert asset['verified_updates']==4000 and paths


def test_wrong_final_budget_rejected(monkeypatch):
    original=common.read(common.PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1/data/MANIFEST.json')
    load=prepare.torch.load
    def altered(*args,**kwargs):
        data=load(*args,**kwargs);data['step']=3500;return data
    monkeypatch.setattr(prepare.torch,'load',altered)
    with pytest.raises(RuntimeError,match='Checkpoint metadata'):
        prepare.attest_asset(common.config()['assets'][0],original)


def test_missing_results_and_clean_failure_do_not_become_attack_success(monkeypatch):
    monkeypatch.setattr(report,'config',lambda:{'methods':common.config()['methods'],'bootstrap_draws':50,'bootstrap_seed':42})
    rows=[]
    for i in range(40):
        rows.append(dict(start=str(i),suite='libero_goal',task_id=i//4,condition='clean',clean_success=int(i!=39),success=int(i!=39),induced_failure=0))
    # matched42: all forty; clean-failed start also fails but adds no CASR failure.
    for i in range(40):
        rows.append(dict(start=str(i),suite='libero_goal',task_id=i//4,condition='matched_relation_s42',clean_success=int(i!=39),success=int(i not in (0,39)),induced_failure=int(i==0)))
    for name in ('shuffled_relation_s42','initial_s43'):
        for i in range(39):
            rows.append(dict(start=str(i),suite='libero_goal',task_id=i//4,condition=name,clean_success=1,success=1,induced_failure=0))
    summary,contrasts=report.statistics(pd.DataFrame(rows));by={r['condition']:r for r in summary}
    assert by['matched_relation_s42']['robot_task_SR']==38/40
    assert by['matched_relation_s42']['CASR']==1/39
    assert by['initial_s43']['robot_task_SR']==1 and by['initial_s43']['episodes_completed']==39
    assert by['matched_relation_s43']['CASR'] is None
    assert contrasts[0]['paired']==39 and contrasts[0]['matched_only_failure']==1
    assert contrasts[1]['paired']==0 and contrasts[1]['CASR_difference'] is None


def test_first_clean_action_checked_on_existing_second_call(monkeypatch):
    monkeypatch.setattr(evaluate,'config',lambda:{'action_tolerance':1e-5})
    wrapper=SimpleNamespace(predict_action=lambda *_args,**_kw:np.zeros((8,7)))
    checked=evaluate.FirstCleanCheck(wrapper,np.zeros((8,7)))
    checked.predict_action(None,'');assert checked.error is None
    checked.predict_action(None,'');assert checked.error==0 and checked.calls==2
    checked=evaluate.FirstCleanCheck(wrapper,np.full((8,7),2e-5))
    checked.predict_action(None,'')
    with pytest.raises(RuntimeError,match='Historical first clean'):checked.predict_action(None,'')


@pytest.mark.parametrize('fail',[False,True])
def test_fresh_environment_per_condition_and_close_on_failure(tmp_path,monkeypatch,fail):
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    monkeypatch.setattr(evaluate,'BASE',tmp_path/'baseline')
    monkeypatch.setattr(evaluate,'verify_external',lambda *_:{'artifacts':{'attempt/TRACE.parquet':'unused'}})
    monkeypatch.setattr(evaluate.pd,'read_parquet',lambda *_:pd.DataFrame([{'predicted_actions':np.zeros((8,7))}]))
    monkeypatch.setattr(evaluate,'complete',lambda _:False)
    completions=[];monkeypatch.setattr(evaluate,'finish',lambda p,files:completions.append(p))
    monkeypatch.setattr(evaluate.native,'runtime_config',lambda:{})
    initial=tmp_path/'initial.pt';torch.save([np.zeros(2)],initial)
    case=dict(start='case',task_name='task',suite='libero_goal',task_id=0,state_index=0,initial_file=str(initial))
    created=[];closed=[];draws=[]
    def make_env(*_args):
        env=object();created.append(env);draws.append((np.random.rand(),torch.rand(1).item()))
        return env,SimpleNamespace(name='task'),[np.zeros(2)]
    runtime=SimpleNamespace(make_env=make_env,close_environment=lambda env:closed.append(env))
    def episode(env,task,init,case,name,patch,checked,attempt,monitor,external):
        checked.predict_action(None,'');checked.predict_action(None,'')
        if fail:raise RuntimeError('simulated rendering failure')
    monitor=SimpleNamespace(data={'completed':0},update=lambda **_kw:None)
    wrapper=SimpleNamespace(predict_action=lambda *_args,**_kw:np.zeros((8,7)))
    if fail:
        with pytest.raises(RuntimeError,match='simulated rendering'):
            evaluate.run_case(runtime,wrapper,episode,case,{'a':None,'b':None},{'a':None,'b':None},monitor,0)
        assert not completions and len(created)==1
    else:
        evaluate.run_case(runtime,wrapper,episode,case,{'a':None,'b':None},{'a':None,'b':None},monitor,0)
        assert len(completions)==2 and created[0] is not created[1] and draws[0]==draws[1]
    assert created==closed


def test_resume_all_finished_does_not_load_model_or_runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    monkeypatch.setattr(evaluate,'verify',lambda:{'cases':[{'start':str(i)} for i in range(40)]})
    monkeypatch.setattr(evaluate,'conditions',lambda:{name:None for name in common.config()['methods']})
    monkeypatch.setattr(evaluate,'complete',lambda p:p.name=='COMPLETE.json')
    monkeypatch.setattr(evaluate,'load_wrapper',lambda *_:pytest.fail('unnecessary model load'))
    monkeypatch.setattr(evaluate.native,'runtime_module',lambda:pytest.fail('unnecessary simulator import'))
    calls=[];monkeypatch.setattr(evaluate,'finish',lambda p,files,**kw:calls.append((files,kw)))
    evaluate.run(0,SimpleNamespace(update=lambda **_kw:None))
    assert len(calls[0][0])==80 and calls[0][1]['new_closedloop_episodes']==80


def test_audited_adapter_uses_new_output_root_only():
    old=common.read(common.PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1_fix1/data/MANIFEST.json')['signature']
    fn=evaluate.rollout_functions()
    assert fn.__globals__['ROOT']==common.ROOT
    assert fn.__globals__['conditions'] is evaluate.conditions
    assert common.read(common.PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1_fix1/data/MANIFEST.json')['signature']==old


def test_signature_and_receipt_detect_changed_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'signature',lambda:'test')
    asset=tmp_path/'asset';asset.write_text('good');receipt=tmp_path/'COMPLETE.json'
    common.finish(receipt,[asset]);assert common.complete(receipt)
    asset.write_text('bad')
    with pytest.raises(RuntimeError,match='Changed artifact'):common.complete(receipt)


def test_manual_dry_run_only_two_gpu_lanes_and_cpu_report(monkeypatch,capsys):
    monkeypatch.setattr(cli,'check_launchers',lambda:['evaluate','cpu'])
    monkeypatch.setattr(cli,'done',lambda stage:stage=='prepare')
    monkeypatch.setattr(cli,'complete',lambda _:False)
    monkeypatch.setattr(cli,'queue',lambda:pytest.fail('dry-run queried Slurm'))
    cli.submit(True);output=capsys.readouterr().out
    assert output.count('sbatch --parsable')==3 and output.count('evaluate.sbatch')==2
    assert 'afterany:DRY1:DRY2' in output and 'train' not in output


def test_related_pipeline_blocks_duplicate_submission(monkeypatch,capsys):
    monkeypatch.setattr(cli,'check_launchers',lambda:None)
    monkeypatch.setattr(cli,'complete',lambda _:False)
    monkeypatch.setattr(cli,'queue',lambda:[['999','RUNNING','srrf1_old_evaluate0','node']])
    monkeypatch.setattr(cli.subprocess,'run',lambda *_args,**_kw:pytest.fail('duplicate submitted'))
    cli.submit();assert 'no duplicate jobs' in capsys.readouterr().out


def test_actual_batch_shells_and_resource_requests():
    assert cli.check_launchers()==['evaluate','cpu']
    gpu=(common.ROOT/'slurm/evaluate.sbatch').read_text();cpu=(common.ROOT/'slurm/cpu.sbatch').read_text()
    assert '--partition=rtx4090' in gpu and '--gres=gpu:4090:1' in gpu and '--cpus-per-task=4' in gpu and '--mem=12G' in gpu
    assert '--cpus-per-task=2' in cpu and '--mem=4G' in cpu and '--gres' not in cpu
