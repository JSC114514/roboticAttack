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
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli,'signature',lambda:'a'*64)
    monkeypatch.setattr(cli,'complete',lambda path:False);monkeypatch.setattr(cli,'done',lambda stage:False)
    monkeypatch.setattr(cli,'lock',lambda name:nullcontext())
    monkeypatch.setattr(cli,'queue',lambda:pytest.fail('dry run queried live queue'))
    cli.submit(dry=True);lines=capsys.readouterr().out.splitlines()
    assert len(lines)==12
    assert 'afterok:DRY5' in next(line for line in lines if line.endswith('train2'))
    assert 'afterok:DRY6' in next(line for line in lines if line.endswith('train3'))
    assert 'afterok:DRY7:DRY8' in next(line for line in lines if line.endswith('select'))
    assert 'afterany:DRY10:DRY11' in lines[-1]

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
