"""CPU/toy only: no real model loading, renderer, GPU or Slurm calls."""
from __future__ import annotations
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common
import new_objectives as feature
import train
import data
import analyze
import evaluate
import pipeline
import engine as engine_module
from engine import OFTEngine
from source_transform import RandomPatchTransform

torch.set_num_threads(2)


def footprint():
    result = torch.zeros(256)
    result[32:48] = 1
    return result


def test_contract_shared_data_budget_initialization_and_resources():
    cfg = common.config()
    old = common.base.config()
    assert common.METHODS == ("dino_attention", "dino_multilayer", "dino_affinity")
    for key in ("training_suite", "seed", "patch_shape", "learning_rate", "warmup_outer_iterations", "schedule_outer_iterations", "inner_loop", "outer_iterations", "eot", "evaluation", "data_seed"):
        assert cfg[key] == old[key]
    assert cfg['outer_iterations']*cfg['inner_loop'] == 12500
    assert len(common.deployment_conditions()) * 4 * 10 * 2 == 240
    assert len(common.analysis_conditions()) * 4 * 10 * 2 == 560
    assert cfg['resources']['training']['max_concurrent'] == 3
    assert cfg['resources']['evaluation']['max_concurrent'] == 4
    patch, initial, opt, schedule = train.initialize_attack(cfg, 'cpu')
    assert torch.equal(initial, torch.rand(3,50,50,generator=torch.Generator().manual_seed(42)))
    assert opt.param_groups[0]['params'][0] is patch and patch.numel()==7500
    for _ in range(250):
        opt.zero_grad(); patch.mean().backward(); opt.step(); schedule.step()
    assert .0018 < opt.param_groups[0]['lr'] < .002


def test_attention_exact_native_qk_norm_scale_and_complete_prefix_denominator():
    qkv = torch.randn(1,261,24,generator=torch.Generator().manual_seed(18),requires_grad=True)
    attn = SimpleNamespace(num_heads=2, q_norm=lambda x:x*1.2, k_norm=lambda x:x*.8, scale=.4)
    loss, detail = feature.attention_objective({8:qkv}, {8:attn}, footprint())
    q,k,_ = qkv.reshape(1,261,3,2,4).permute(2,0,3,1,4).unbind(0)
    logits = ((q*1.2)[:,:,5:][:,:,footprint().eq(0)]*.4) @ (k*.8).transpose(-2,-1)
    mass = logits.softmax(-1)[...,37:53].sum(-1)
    torch.testing.assert_close(loss, -mass.log().mean())
    assert detail['patch_attention_mass'] == pytest.approx(float(mass.detach().mean()),abs=1e-7)
    wrong = logits[...,5:].softmax(-1)[...,32:48].sum(-1)
    assert abs(float(loss.detach()) - float(-wrong.log().mean().detach())) > 1e-4
    loss.backward()
    assert torch.isfinite(qkv.grad).all() and qkv.grad.norm()>0
    updated = (qkv.detach() - .05*qkv.grad).requires_grad_()
    improved, after = feature.attention_objective({8:updated},{8:attn},footprint())
    assert improved < loss and after['patch_attention_mass'] > detail['patch_attention_mass']


def test_attention_support_and_extreme_logits_are_safe():
    with pytest.raises(RuntimeError,match='layers/captures'):
        feature.attention_objective({}, {}, torch.zeros(256))
    with pytest.raises(RuntimeError,match='too few'):
        feature.attention_objective({}, {}, torch.ones(256))
    qkv = (torch.randn(1,261,24,generator=torch.Generator().manual_seed(3))*20).requires_grad_()
    attn = SimpleNamespace(num_heads=2, q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(), scale=.5)
    loss,_ = feature.attention_objective({8:qkv},{8:attn},footprint())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(qkv.grad).all()


def test_empty_visible_patch_does_not_resample_or_disconnect_optimizer_graph():
    qkv=torch.randn(1,261,24,requires_grad=True)
    attn=SimpleNamespace(num_heads=2,q_norm=torch.nn.Identity(),k_norm=torch.nn.Identity(),scale=.5)
    loss,detail=feature.attention_objective({8:qkv},{8:attn},torch.zeros(256))
    loss.backward()
    assert loss==0 and qkv.grad is not None and qkv.grad.abs().sum()==0
    assert detail['attention_objective_active']==0 and detail['patch_key_tokens']==0


def test_failed_chain_invalid_dependents_do_not_prevent_manual_resume(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'preflight',lambda:{'signature':'toy'})
    monkeypatch.setattr(pipeline,'run_signature',lambda:'toy')
    common.atomic_json({'signature':'toy','jobs':[{'job_id':'555'},{'job_id':'556'}]},tmp_path/'reports/SUBMISSION_RECEIPT.json')
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        if command[0]=='squeue':
            return SimpleNamespace(returncode=0,stdout='555|PENDING|N/A|DependencyNeverSatisfied\n556|PENDING|gpu:4090:1|DependencyNeverSatisfied\n',stderr='')
        return SimpleNamespace(returncode=0,stdout=str(2000+len(commands))+'\n',stderr='')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    pipeline.submit()
    assert len([c for c in commands if c[0]=='sbatch'])==5
    receipt=common.load_json(tmp_path/'reports/SUBMISSION_RECEIPT.json')
    assert receipt['waiting_for_existing_gpu_jobs']==[] and receipt['ignored_invalid_pending_gpu_jobs']==['556']


def test_signed_multilayer_reference_detached_sign_correct_and_coverage_excluded():
    clean = torch.randn(1,256,8,generator=torch.Generator().manual_seed(12),requires_grad=True)
    refs = feature.multilayer_reference({8:clean,16:clean*.8,22:clean*1.2})
    assert all(not r['scale'].requires_grad and r['strong'].sum(-1).eq(4).all() for r in refs.values())
    current = (clean.detach()+.15*torch.randn(1,256,8,generator=torch.Generator().manual_seed(17))).requires_grad_()
    loss,detail=feature.multilayer_objective({8:current,16:current*.8,22:current*1.2},refs,footprint())
    loss.backward()
    assert clean.grad is None and current.grad.norm()>0
    assert current.grad[:,32:48].abs().max()==0
    after = current.detach()-.03*current.grad
    newer,details=feature.multilayer_objective({8:after,16:after*.8,22:after*1.2},refs,footprint())
    assert newer<loss
    assert details['layer8_strong_rms'] < detail['layer8_strong_rms']
    assert details['layer8_weak_rms'] > detail['layer8_weak_rms']


def test_multilayer_zero_low_responses_remain_finite():
    clean = torch.zeros(1,256,8)
    clean[...,0]=3
    current = clean.clone().requires_grad_()
    loss,_=feature.multilayer_objective({8:current},feature.multilayer_reference({8:clean}),footprint())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()


def test_affinity_excludes_self_and_support_norm_invariant_and_reverses_relations():
    clean = torch.randn(1,256,8,generator=torch.Generator().manual_seed(19),requires_grad=True)
    current=(clean.detach()+.3*torch.randn(1,256,8,generator=torch.Generator().manual_seed(20))).requires_grad_()
    loss,detail = feature.affinity_objective(clean,current,footprint(),8)
    torch.testing.assert_close(loss,feature.affinity_objective(clean*2,current*3,footprint(),8)[0])
    indices=feature.unpatched_tokens(footprint())
    c=F.normalize(clean.detach().index_select(1,indices),dim=-1)[0]
    z=F.normalize(current.index_select(1,indices),dim=-1)[0]
    matrix=c@c.T; eye=torch.eye(len(indices),dtype=torch.bool)
    positives=matrix.masked_fill(eye,-torch.inf).topk(8,-1).indices
    negatives=matrix.masked_fill(eye,torch.inf).scatter(1,positives,torch.inf).topk(8,-1,largest=False).indices
    assert not (positives==torch.arange(len(indices))[:,None]).any()
    assert not (negatives==torch.arange(len(indices))[:,None]).any()
    assert not (positives[:,:,None]==negatives[:,None,:]).any()
    expected=(z@z.T).gather(1,positives).mean()-(z@z.T).gather(1,negatives).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert clean.grad is None and current.grad.norm()>0 and current.grad[:,32:48].abs().max()==0
    newer,_=feature.affinity_objective(clean,current.detach()-.1*current.grad,footprint(),8)
    assert newer<loss and detail['affinity_query_tokens']==240


class ToyAttention(torch.nn.Module):
    def __init__(self, channels=8):
        super().__init__(); self.qkv=torch.nn.Linear(channels,3*channels)
        self.num_heads,self.scale=2,.5
        self.q_norm=self.k_norm=torch.nn.Identity()
    def forward(self,x):
        return self.qkv(x)[...,:x.shape[-1]].mean(1,keepdim=True).expand_as(x)*.05


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.attn=ToyAttention()
    def forward(self,x): return x+self.attn(x)


class ToyDino(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=torch.nn.ModuleList([ToyBlock() for _ in range(24)])
        self.num_prefix_tokens=5; self.project=torch.nn.Linear(3,8)
        self.register_buffer('position',torch.randn(1,261,8)*.15)
    def forward(self,pixels):
        image=F.adaptive_avg_pool2d(pixels[:,6:9],(16,16)).flatten(2).transpose(1,2)
        x=self.project(image)
        x=torch.cat((x.mean(1,keepdim=True).expand(-1,5,-1),x),1)+self.position
        for block in self.blocks[:-1]: x=block(x)
        return x[:,5:]


@pytest.mark.parametrize('method',common.METHODS)
def test_readonly_hooks_capture_actual_layers_gradients_and_cleanup(method):
    dino=ToyDino().eval().requires_grad_(False)
    engine=SimpleNamespace(dino=dino,direct_dino=dino)
    pixels=torch.rand(1,12,32,32,requires_grad=True)
    expected=dino(pixels)
    dense,captures=feature.features_forward(engine,pixels,method,common.config())
    torch.testing.assert_close(dense,expected)
    refs=None
    if method=='dino_multilayer': refs=feature.multilayer_reference({k:v.detach() for k,v in captures.items()})
    if method=='dino_affinity': refs=(dense.detach()+.1*torch.randn_like(dense))
    loss,_=feature.objective(method,dense,captures,refs,engine,footprint(),common.config())
    loss.backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.norm()>0
    assert all(p.grad is None for p in dino.parameters())
    assert all(not b._forward_hooks and not b.attn.qkv._forward_hooks for b in dino.blocks)
    with pytest.raises(ValueError):
        with feature.capture_layers(dino,[8],True): raise ValueError('interrupt')
    assert not dino.blocks[8].attn.qkv._forward_hooks


def test_runtime_copies_and_episode_ast_match_successful_v1():
    for name in common.COPIED_FILES:
        assert (ROOT/'code'/name).read_bytes()==(common.BASE_ROOT/'code'/name).read_bytes()
    def function(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
    assert function(ROOT/'code/evaluate.py','episode')==function(common.BASE_ROOT/'code/evaluate.py','episode')
    source=(ROOT/'code/evaluate.py').read_text()
    assert 'from train import' not in source and 'from new_objectives import' not in source
    assert common.deployment_conditions()[0]=='dino_attention@250'


def test_original_eot_keeps_front_and_has_differentiable_rgb(monkeypatch):
    engine=object.__new__(OFTEngine)
    engine.transform=RandomPatchTransform('cpu',resize_patch=False)
    engine.wrapper=SimpleNamespace(processor=SimpleNamespace(image_processor=None),_official_openvla_utils=None,gen_cfg=None,dtype=torch.float32)
    monkeypatch.setattr(engine_module,'aligned_pixels',lambda front,wrist,*args:torch.cat((front,front,wrist,wrist))[None])
    record={'observation':{'full_image':np.full((224,224,3),128,np.uint8),'wrist_image':np.full((224,224,3),25,np.uint8)}}
    patch=torch.rand(3,50,50,requires_grad=True)
    pixels,support=engine.pixels(record,patch,15,diverse=True)
    assert pixels[:,:6].eq(128/255).all() and support.shape==(256,)
    pixels[:,6:9].mean().backward()
    assert torch.isfinite(patch.grad).all() and patch.grad.norm()>0


class ToyEngine:
    calls=0
    fail_on=None
    def __init__(self,wrapper):
        self.wrapper=wrapper; self.dino=ToyDino().eval().requires_grad_(False)
    def base(self,record):
        return {'pixel_values':torch.full((1,12,32,32),.3)}, np.zeros(8)
    def pixels(self,record,patch,seed,diverse=False):
        ToyEngine.calls+=1
        if ToyEngine.fail_on==ToyEngine.calls: raise RuntimeError('simulated interruption')
        front=torch.full((1,6,32,32),.3)
        wrist=F.interpolate(patch[None],size=(32,32),mode='bilinear',align_corners=False)
        return torch.cat((front,wrist,wrist),1),footprint()
    def direct_dino(self,pixels): return self.dino(pixels)
    def validate(self,*args): return {'status':'CPU_TOY_ONLY'}


def install_toy_training(monkeypatch,tmp_path):
    cfg=copy.deepcopy(common.config())
    cfg.update(outer_iterations=4,inner_loop=3,warmup_outer_iterations=1,resume_every_outer=1,checkpoint_outer_iterations=[0,2,4],evaluation_checkpoints=[4])
    records={str(i):{'uid':str(i),'episode_id':f'episode{i}','step_index':i,'task_id':i} for i in range(4)}
    bank={'records':records,'candidates':[[str(i)] for i in range(4)]}
    wrapper=SimpleNamespace(vla=torch.nn.Linear(1,1).requires_grad_(False),action_head=None,proprio_projector=None)
    monkeypatch.setattr(train,'config',lambda:cfg)
    monkeypatch.setattr(train,'run_signature',lambda:'CPU_TOY_SIGNATURE')
    monkeypatch.setattr(train,'prepared',lambda:None)
    monkeypatch.setattr(train,'require_gpu',lambda stage:'CPU_TOY_ONLY')
    monkeypatch.setattr(train,'load_bank',lambda:bank)
    monkeypatch.setattr(train,'load_wrapper',lambda *args:wrapper)
    monkeypatch.setattr(train,'OFTEngine',ToyEngine)
    monkeypatch.setattr(train,'patch_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(train.torch.cuda,'manual_seed_all',lambda seed:None)
    monkeypatch.setattr(train.torch.cuda,'get_rng_state_all',lambda:[torch.tensor([1],dtype=torch.uint8)])
    monkeypatch.setattr(train.torch.cuda,'set_rng_state_all',lambda states:None)
    return cfg,bank


@pytest.mark.parametrize('method',common.METHODS)
def test_full_loop_actual_rgb_updates_and_exact_cpu_resume(method,monkeypatch,tmp_path):
    cfg,bank=install_toy_training(monkeypatch,tmp_path)
    full,partial=tmp_path/'full',tmp_path/'partial'
    monkeypatch.setattr(train,'ROOT',full)
    ToyEngine.calls,ToyEngine.fail_on=0,None
    train.train(method)
    completed=torch.load(full/'training'/method/'resume.pt',weights_only=False)
    assert completed['outer']==4 and not torch.equal(completed['patch'],completed['initial'])
    assert [r['uid'] for r in completed['rows']]==[str(i) for i in range(4)]
    assert sum(r['effective_rgb_updates'] for r in completed['rows'])>0
    assert all(r['optimizer_updates']==(i+1)*3 for i,r in enumerate(completed['rows']))
    monkeypatch.setattr(train,'ROOT',partial)
    ToyEngine.calls,ToyEngine.fail_on=0,5
    with pytest.raises(RuntimeError,match='simulated interruption'): train.train(method)
    checkpoint=torch.load(partial/'training'/method/'resume.pt',weights_only=False)
    assert checkpoint['outer']==1
    ToyEngine.fail_on=None
    train.train(method)
    resumed=torch.load(partial/'training'/method/'resume.pt',weights_only=False)
    assert torch.equal(resumed['patch'],completed['patch']) and resumed['scheduler']==completed['scheduler']
    state=list(resumed['optimizer']['state'].values())[0]
    original=list(completed['optimizer']['state'].values())[0]
    assert int(state['step'])==12
    assert torch.equal(state['exp_avg'],original['exp_avg']) and torch.equal(state['exp_avg_sq'],original['exp_avg_sq'])
    train.train(method) # COMPLETE does not load model or perform another update.


def synthetic_raw():
    rows=[]
    for suite in common.SUITES:
        for task in range(10):
            for state in [40,41]:
                for cond in common.analysis_conditions():
                    success=1 if cond=='clean' else int(task%2==0)
                    rows.append({'suite':suite,'task_id':task,'state_index':state,'condition':cond,'rollout_seed':task*100+state,
                                 'success':success,'failure':1-success})
    return pd.DataFrame(rows)


def test_all_six_methods_paired_statistics_and_no_suite_average():
    summary,contrasts,tasks=analyze.summarize(synthetic_raw(),draws=50)
    assert len(summary)==28 and len(tasks)==240 and len(contrasts)==44
    assert set(summary['method'])=={'clean',*common.ALL_METHODS}
    assert summary.loc[summary.method.ne('clean'),'task_equal_casr'].eq(.5).all()
    assert contrasts.casr_difference.eq(0).all()
    bad=synthetic_raw().iloc[:-1]
    with pytest.raises(RuntimeError,match='incomplete'): analyze.summarize(bad,draws=2)
    bad=synthetic_raw(); bad.loc[0,'rollout_seed']+=1
    with pytest.raises(RuntimeError,match='Unpaired'): analyze.summarize(bad,draws=2)


def test_casr_qualifies_clean_success_and_uses_task_equal_denominators():
    raw=pd.DataFrame([{'task_id':t,'state_index':s,'condition':c,'success':v,'failure':1-v}
                      for t,s,clean,attack in [(0,40,1,0),(0,41,1,1),(1,40,0,0),(1,41,1,0)]
                      for c,v in [('clean',clean),('dino_attention@250',attack)]])
    point,pairs=analyze.task_casr(raw,'dino_attention@250')
    assert point==.75 and pairs.clean_success.sum()==3


def test_evaluate_loads_all_three_and_never_adds_clean(monkeypatch,tmp_path):
    patches={}
    from PIL import Image
    for method in common.METHODS:
        folder=tmp_path/'training'/method; folder.mkdir(parents=True)
        path=folder/'outer_0250.png'; Image.fromarray(np.full((50,50,3),128,np.uint8)).save(path)
        patches[method+'@250']={'png':common.sha256_file(path)}
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    monkeypatch.setattr(evaluate,'load_json',lambda p:{'status':'COMPLETE','signature':'toy','patches':patches})
    monkeypatch.setattr(evaluate,'run_signature',lambda:'toy')
    actual,hashes=evaluate.patches()
    assert set(actual)==set(common.deployment_conditions()) and 'clean' not in hashes


def test_submission_afterok_chain_resource_caps_and_no_double_submit(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'preflight',lambda:{'signature':'toy'})
    monkeypatch.setattr(pipeline,'run_signature',lambda:'toy')
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='squeue': return SimpleNamespace(returncode=0,stdout='123|RUNNING|gpu:4090:1|node\n',stderr='')
        return SimpleNamespace(returncode=0,stdout=str(1000+len(calls))+'\n',stderr='')
    monkeypatch.setattr(pipeline.subprocess,'run',run)
    pipeline.submit()
    commands=[c for c in calls if c[0]=='sbatch']
    assert len(commands)==5 and '--dependency=afterany:123' in commands[0]
    for previous,current in zip(commands,commands[1:]):
        prior_id=str(1000+calls.index(previous)+1)
        assert '--dependency=afterok:'+prior_id in current
    receipt=json.loads((tmp_path/'reports/SUBMISSION_RECEIPT.json').read_text())
    assert receipt['training_gpu_cap']==3 and receipt['evaluation_gpu_cap']==4
    old_jobs='\n'.join(j['job_id']+'|PENDING|gpu:4090:1|Dependency' for j in receipt['jobs'])
    monkeypatch.setattr(pipeline.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=old_jobs,stderr=''))
    with pytest.raises(RuntimeError,match='already has active jobs'): pipeline.submit()


def test_gpu_array_caps_and_bash_submission_is_manual():
    train_script=(ROOT/'slurm/02_train_4090.sbatch').read_text()
    eval_script=(ROOT/'slurm/04_eval_4090.sbatch').read_text()
    assert '--array=0-2%3' in train_script and '--gres=gpu:4090:1' in train_script
    assert '--array=0-9%4' in eval_script and '--gres=gpu:4090:1' in eval_script
    assert 'dino_attention dino_multilayer dino_affinity' in train_script
    pipeline_source=(ROOT/'code/pipeline.py').read_text()
    assert 'if args.submit:' in pipeline_source
    assert 'GPU smoke' not in train_script


def test_invalid_pending_dependencies_are_reported_not_waited_or_cancelled():
    queue='123|PENDING|gpu:4090:1|DependencyNeverSatisfied\n124_[0-2%3]|RUNNING|gpu:4090:1|node\n'
    assert pipeline.existing_gpu_jobs(queue)==['124']
    assert pipeline.ignored_invalid_jobs(queue)==['123']
    assert 'scancel' not in (ROOT/'code/pipeline.py').read_text()


def test_final_checkpoint_without_complete_status_recovers_without_model(monkeypatch,tmp_path):
    cfg,_=install_toy_training(monkeypatch,tmp_path)
    monkeypatch.setattr(train,'ROOT',tmp_path)
    ToyEngine.calls,ToyEngine.fail_on=0,None
    train.train('dino_attention')
    folder=tmp_path/'training/dino_attention'
    before=torch.load(folder/'resume.pt',weights_only=False)
    common.atomic_json({'status':'RUNNING','signature':'CPU_TOY_SIGNATURE','outer':3},folder/'status.json')
    def must_not_load(*args): raise AssertionError('Should not load model for committed final resume')
    monkeypatch.setattr(train,'load_wrapper',must_not_load)
    train.train('dino_attention')
    assert common.load_json(folder/'status.json')['status']=='COMPLETE'
    assert torch.equal(torch.load(folder/'resume.pt',weights_only=False)['patch'],before['patch'])


def test_full_loop_assets_validation_uses_actual_png_and_schedule(monkeypatch,tmp_path):
    cfg,bank=install_toy_training(monkeypatch,tmp_path)
    monkeypatch.setattr(train,'ROOT',tmp_path)
    for method in common.METHODS:
        ToyEngine.calls,ToyEngine.fail_on=0,None
        train.train(method)
    monkeypatch.setattr(analyze,'ROOT',tmp_path)
    monkeypatch.setattr(analyze,'config',lambda:cfg)
    monkeypatch.setattr(analyze,'run_signature',lambda:'CPU_TOY_SIGNATURE')
    monkeypatch.setattr(analyze,'prepared',lambda:None)
    monkeypatch.setattr(analyze,'load_bank',lambda:bank)
    analyze.validate_training()
    report=common.load_json(tmp_path/'reports/AUTO_VALIDATION.json')
    assert report['status']=='COMPLETE' and len(report['patches'])==3
    from PIL import Image
    path=tmp_path/'training/dino_attention/outer_0004.png'
    Image.fromarray(np.zeros((50,50,3),np.uint8)).save(path)
    with pytest.raises(RuntimeError,match='PNG does not match'): analyze.validate_training()


def test_automatic_merge_preserves_old_identities_and_writes_560_toy_rows(monkeypatch,tmp_path):
    raw=synthetic_raw()
    raw['signature']='CPU_TOY_SIGNATURE'
    raw['source_training_signature']='CPU_TOY_SIGNATURE'
    raw['decision_blocks']=1
    raw['evaluation_training_hooks']=False
    raw['renderer_vendor']='NVIDIA Corporation'
    raw['renderer']='NVIDIA GeForce RTX 4090/CPU_TOY_METADATA_ONLY'
    from PIL import Image
    patches={}
    for method in common.METHODS:
        folder=tmp_path/'training'/method; folder.mkdir(parents=True)
        png=folder/'outer_0250.png'; Image.fromarray(np.full((50,50,3),128,np.uint8)).save(png)
        patches[method+'@250']={'png':common.sha256_file(png)}
    raw['deployment_png_sha256']=raw.condition.map({key:v['png'] for key,v in patches.items()}).fillna('CPU_TOY_OLD_HASH')
    new=raw.loc[raw.condition.isin(common.deployment_conditions())].copy()
    old=raw.loc[~raw.condition.isin(common.deployment_conditions())].copy()
    old['signature']=old['source_training_signature']=common.BASE_SIGNATURE
    old['source_run']=common.BASE_ROOT.name; old['reused']=True
    old_trace=old.assign(decision_block=0)
    for suite in common.SUITES:
        folder=tmp_path/'evaluation'/suite; folder.mkdir(parents=True)
        for task in range(10):
            part=new.loc[new.suite.eq(suite)&new.task_id.eq(task)]
            rp,tp=folder/f'task_{task:02d}_rollouts.parquet',folder/f'task_{task:02d}_traces.parquet'
            part.to_parquet(rp,index=False); part.assign(decision_block=0).to_parquet(tp,index=False)
            common.atomic_json({'status':'COMPLETE','signature':'CPU_TOY_SIGNATURE','rollouts_sha256':common.sha256_file(rp),
                                'traces_sha256':common.sha256_file(tp)},folder/f'task_{task:02d}_STATUS.json')
    common.atomic_json({'status':'COMPLETE','signature':'CPU_TOY_SIGNATURE','patches':patches},tmp_path/'reports/AUTO_VALIDATION.json')
    original_summary=analyze.summarize
    monkeypatch.setattr(analyze,'ROOT',tmp_path)
    monkeypatch.setattr(analyze,'run_signature',lambda:'CPU_TOY_SIGNATURE')
    monkeypatch.setattr(analyze,'prepared',lambda:None)
    monkeypatch.setattr(analyze,'reference_frames',lambda:(old.copy(),old_trace.copy()))
    monkeypatch.setattr(analyze,'summarize',lambda value:original_summary(value,draws=20))
    analyze.analyze()
    saved=pd.read_parquet(tmp_path/'results/OBJECTIVES250_ROLLOUTS.parquet')
    assert len(saved)==560 and saved.reused.sum()==320
    assert saved.loc[saved.reused,'signature'].eq(common.BASE_SIGNATURE).all()
    assert saved.loc[~saved.reused,'signature'].eq('CPU_TOY_SIGNATURE').all()
    status=common.load_json(tmp_path/'results/STATUS.json')
    assert status['new_episodes']==240 and status['reused_episodes']==320 and status['methods']==6
