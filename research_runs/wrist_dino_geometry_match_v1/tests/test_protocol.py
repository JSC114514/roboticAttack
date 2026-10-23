import ast,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
import common,data,engine,evaluate,train,pipeline,report

@pytest.mark.parametrize('seed',[1,42,43])
def test_full_loss_and_gradient_equal_direct_target_distance(seed):
    torch.manual_seed(seed)
    negative=torch.randn(256,8);positive=negative+.2*torch.randn(256,8)
    value=torch.randn(1,256,8,requires_grad=True)
    result=train.objective(value,negative,positive,'full');loss,_=result
    direction=positive-negative;energy=direction.square().mean(-1);w=energy/energy.sum()
    direct=(w*(value.reshape_as(positive)-positive).square().mean(-1)).sum()/(w*energy).sum()
    assert torch.allclose(loss,direct,atol=1e-4,rtol=1e-5)
    g1=torch.autograd.grad(loss,value,retain_graph=True)[0];g2=torch.autograd.grad(direct,value)[0]
    assert torch.allclose(g1,g2,atol=1e-6,rtol=1e-4)

def test_full_loss_penalizes_orthogonal_change_that_projection_ignores():
    minus=torch.zeros(256,2);plus=minus.clone();plus[:,0]=1
    value=plus.clone();value[:,1]=3
    old=train.objective(value,minus,plus,'projection')[0]
    new=train.objective(value,minus,plus,'full')[0]
    assert float(old)==0 and float(new)>0

def test_degenerate_targets_are_retained_without_fabricated_loss():
    t=torch.zeros(256,4)
    assert train.objective(t,t,t,'full') is None
    assert evaluate.metrics(t,t,t)['q'] is None

def test_fixed_data_cohorts_and_trajectory_separation():
    fit=data.index_rows('fit');verify=data.index_rows('verify')
    assert len(fit)==len(verify)==48
    assert len(set((r['suite'],r['task_id']) for r in fit))==24
    assert not set(r['episode_id'] for r in fit)&set(r['episode_id'] for r in verify)
    assert all(r['frame_rank']==1 for r in fit+verify)
    assert len(data.index_rows('train'))==960 and len(data.index_rows('validation'))==160

def test_seed42_png_is_exact_historical_initialization(tmp_path):
    path=tmp_path/'initial.png';common.save_png(common.initial_patch(seed=42),path)
    assert common.sha(path)==common.sha(common.origin_run()/'training/translation/epoch_00.png')
    assert not torch.equal(common.initial_patch(seed=42),common.initial_patch(seed=43))

def test_slurm_resources_separate_training_and_evaluation():
    for p in (common.ROOT/'slurm').glob('*.sbatch'):
        text=p.read_text()
        if p.stem in ('pilot_report','night_report'):
            assert '--partition=cpu' in text and '--mem=4G' in text and '--gres=' not in text
        else:
            assert '--partition=rtx4090' in text and '--gres=gpu:4090:1' in text and '--cpus-per-task=4' in text
            assert ('--mem=16G' if p.stem in ('pilot_train','night_train') else '--mem=12G') in text

def test_submission_bounds_parallel_arrays_and_preserves_failure_reporting(monkeypatch):
    monkeypatch.setattr(pipeline,'complete',lambda p:False);calls=[]
    def record(stage,indices,deps,kind='afterany'):
        calls.append((stage,indices,list(deps),kind));return stage+'_JOB'
    pipeline.stages('pilot',record)
    assert calls[0][:2]==('pilot_train',[0,1,2]) and calls[1][0]=='oracle'
    assert set(calls[2][2])=={'pilot_train_JOB','oracle_JOB'}
    assert calls[-1][0]=='pilot_report' and calls[-1][3]=='afterany'
    calls.clear();pipeline.stages('night',record,'pilot_report_JOB')
    assert calls[0]==('night_train',[0,1,2],['pilot_report_JOB'],'afterok')
    assert calls[1][0:2]==('night_eval',[0,1,2,3])
    assert calls[-1][0]=='night_report' and calls[-1][3]=='afterany'

def test_resume_only_submits_incomplete_array_elements(monkeypatch):
    monkeypatch.setattr(pipeline,'complete',lambda p:'full_s42' in str(p) or 'libero_object_COMPLETE' in str(p))
    calls=[]
    def record(stage,indices,deps,kind='afterany'):calls.append((stage,indices));return 'JOB'
    pipeline.stages('night',record)
    assert calls[0]==('night_train',[1,2]) and calls[1]==('night_eval',[1,2,3])

def test_casr_is_paired_clean_conditioned_and_task_equal():
    rows=[]
    for task,state,clean,patch in [(0,0,1,0),(0,1,1,1),(1,0,1,1),(1,1,0,0)]:
        for condition,value in [('clean',clean),('full_s42',patch)]:
            rows.append({'split':'heldout','suite':'s','task_id':task,'state_index':state,'condition':condition,'success':value})
    out=report.casr(pd.DataFrame(rows)).iloc[0]
    assert out.eligible_clean_starts==3 and out.failure_switches==1
    assert out['mean']==.25

def test_missing_outputs_write_failure_not_zero_success(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'signature',lambda:'test')
    with pytest.raises(RuntimeError,match='Incomplete'):
        report.check_sources([tmp_path/'missing.json'],'pilot')
    value=json.loads((tmp_path/'results/pilot/STATUS.json').read_text())
    assert value['status']=='FAILED_OR_INCOMPLETE' and not value['scientific_result_available']
    assert 'CASR' not in value

def test_cost_receipt_captures_failed_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'signature',lambda:'test');monkeypatch.setattr(common,'resources',lambda:{'process_rss_hwm_bytes':123})
    p=tmp_path/'cost.json'
    with pytest.raises(ValueError):
        with common.cost_attempt('unit',path=p):raise ValueError('bad')
    value=json.loads(p.read_text());assert value['status']=='FAILED' and value['elapsed_s']>=0
    assert value['process_rss_hwm_bytes']==123

def test_monitor_is_independent_and_closes_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'signature',lambda:'test')
    monkeypatch.setattr(common,'resources',lambda:{})
    with common.Monitor('a') as monitor:monitor.update(completed=1,total=2)
    with pytest.raises(ValueError):
        with common.Monitor('b'):raise ValueError('bad')
    assert common.read_json(tmp_path/'reports/PROGRESS/a.json')['status']=='COMPLETE'
    assert common.read_json(tmp_path/'reports/PROGRESS/b.json')['status']=='FAILED'

@pytest.mark.parametrize('name,allowed',[('NVIDIA GeForce RTX 4090',True),('NVIDIA A800 80GB',False),('NVIDIA GeForce RTX 3090',False)])
def test_gpu_guard_only_allows_allocated_4090(monkeypatch,name,allowed):
    monkeypatch.setenv('SLURM_JOB_ID','test');monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1);monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:name)
    if allowed:assert common.require_gpu('test')==name
    else:
        with pytest.raises(RuntimeError,match='4090'):common.require_gpu('test')

def test_no_automatic_scheduler_calls_in_workers():
    for name in ('worker.py','train.py','evaluate.py','report.py','data.py'):
        text=(common.ROOT/'code'/name).read_text()
        assert 'sbatch' not in text and 'scancel' not in text and 'srun' not in text

def test_egl_environment_uses_single_4090_mapping():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','PYTHONDONTWRITEBYTECODE':'1'}
    r=subprocess.run(['bash','-c','source "$1" gpu; printf "%s|%s" "$OFT_STAGE_GPU" "$MUJOCO_EGL_DEVICE_ID"',
                      'test',str(common.ROOT/'slurm/environment.sh')],env=env,capture_output=True,text=True,check=True)
    assert r.stdout=='4090|2'

def test_parquet_action_chunks_keep_all_slots(tmp_path):
    actions=np.arange(56,dtype=np.float32).reshape(8,7)
    p=tmp_path/'actions.parquet';pd.DataFrame([{'actions':actions.tolist()}]).to_parquet(p)
    assert np.array_equal(common.decode_action_chunk(pd.read_parquet(p).iloc[0].actions),actions)

@pytest.mark.parametrize('kind',['individual','projection','full'])
def test_pilot_runs_real_patch_optimizer_and_seals_pngs(tmp_path,monkeypatch,kind):
    configuration=common.cfg();configuration['pilot_visits']=3
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(train,'ROOT',tmp_path)
    monkeypatch.setattr(common,'cfg',lambda:configuration);monkeypatch.setattr(train,'cfg',lambda:configuration)
    monkeypatch.setattr(common,'signature',lambda:'pilot_test');monkeypatch.setattr(train,'signature',lambda:'pilot_test')
    monkeypatch.setattr(train,'require_gpu',lambda stage:None);monkeypatch.setattr(train,'resources',lambda:{})
    frozen=torch.nn.Linear(1,1).requires_grad_(False)
    class TinyEngine:
        wrapper=SimpleNamespace(device='cpu',vla=frozen)
        def spatial_patch(self,observation,patch):return patch.mean().expand(1,256,2),None
    monkeypatch.setattr(train,'make_engine',TinyEngine)
    rows=[{'state_id':'s'+str(i),'observation':{},'target_features':{
        'translation_minus':torch.zeros(256,2),'translation_plus':torch.ones(256,2)}} for i in range(2)]
    monkeypatch.setattr(train,'compact_pilot',lambda:{'fit':rows,'verify':[]})
    monitor=SimpleNamespace(update=lambda **kw:None)
    train.run_pilot(kind,monitor)
    folder=tmp_path/'training/pilot'/kind
    assert common.complete(folder/'COMPLETE.json')
    receipt=common.read_json(folder/'COMPLETE.json')
    assert receipt['candidate_updates']==(12 if kind=='individual' else 6)
    assert all(parameter.grad is None for parameter in frozen.parameters())
    # A second invocation must resume at the completed boundary, without model work.
    monkeypatch.setattr(train,'make_engine',lambda:pytest.fail('Completed pilot loaded model'))
    train.run_pilot(kind,monitor)

def test_last_night_epoch_is_sealed_without_loading_model(tmp_path,monkeypatch):
    configuration=common.cfg()
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(train,'ROOT',tmp_path)
    monkeypatch.setattr(common,'cfg',lambda:configuration);monkeypatch.setattr(train,'cfg',lambda:configuration)
    monkeypatch.setattr(common,'signature',lambda:'recovery');monkeypatch.setattr(train,'signature',lambda:'recovery')
    monkeypatch.setattr(train,'require_gpu',lambda stage:None);monkeypatch.setattr(train,'resources',lambda:{})
    monkeypatch.setattr(train,'make_engine',lambda:pytest.fail('Final epoch recovery loaded model'))
    folder=tmp_path/'training/night/full_s42';folder.mkdir(parents=True)
    common.atomic_torch({'signature':'recovery','candidate':'full_s42','epoch':50,'optimized':46300,
                         'no_target':1700,'patch':torch.zeros(3,50,50)},folder/'LAST.pt')
    train.run_night('full_s42',SimpleNamespace(update=lambda **kw:None))
    assert common.complete(folder/'COMPLETE.json') and (folder/'final.png').is_file()

def test_oracle_and_png_diagnostics_survive_actual_parquet_schema(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluate,'array',lambda p:np.full((50,50,3),128,dtype=np.uint8))
    class NativeStub:
        def forward(self,observation,language,replacements=None):
            value=float(observation['wrist_image'].mean()/255.)
            if replacements and 'dino' in replacements:value=float(np.asarray(replacements['dino']).mean())
            feature=np.full((1,256,4),value,dtype=np.float32)
            return np.full((8,7),value,dtype=np.float32),{'dino':feature,'b22':feature,'siglip':feature}
    row={'state_id':'s','episode_id':'e','suite':'libero_spatial','task_id':0,'state_index':40,'frame_rank':1,
         'language':'test','geometry':{'no_target':False},
         'observation':{'full_image':np.zeros((224,224,3),dtype=np.uint8),
                        'wrist_image':np.zeros((224,224,3),dtype=np.uint8),'state':np.zeros(8)},
         'target_features':{'translation_plus':torch.ones(256,4),'translation_minus':torch.zeros(256,4)},
         'counterfactual_wrist':{'translation_plus':np.full((224,224,3),255,dtype=np.uint8),
                                  'translation_minus':np.zeros((224,224,3),dtype=np.uint8)}}
    native=NativeStub();reference=evaluate.oracle_record(native,row,'fit')
    path=tmp_path/'oracle.parquet';pd.DataFrame([reference]).to_parquet(path,index=False)
    restored=pd.read_parquet(path).iloc[0]
    assert np.isclose(restored.ideal_direction_l2,np.sqrt(3))
    fake=tmp_path/'patch.png';fake.write_bytes(b'PNG source hash witness')
    result=evaluate.patch_record(native,row,'fit','full',fake,restored)
    assert result['direction_positive']==1. and np.isfinite(result['target_distance'])
    assert result['restore_dino_translation_l2']==0.
