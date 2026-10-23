"""CPU-only provenance, scheduling, native episode IO and playback checks."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(1,str(ROOT.parents[1]/'research_tools'))
import rfix_common as c


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_migration_preserves_next_adam_update_and_all_random_states(tmp_path):
    p=torch.nn.Parameter(torch.rand(3,4,4));opt=torch.optim.Adam([p],lr=.002)
    for _ in range(2):
        opt.zero_grad();p.square().mean().backward();opt.step()
    payload={'signature':'parent','method':'shuffled_relation','seed':42,'step':2,'optimized':2,'no_target':0,
             'patch':p.detach().clone(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
             'cuda_rng':[torch.arange(10,dtype=torch.uint8)],'numpy_rng':np.random.get_state(),'python_rng':__import__('random').getstate()}
    repaired=c.migrate_payload(payload,'parent','repair','shuffled_relation',42,2)
    assert payload['signature']=='parent' and repaired['signature']=='repair'
    assert all(equal(payload[k],repaired[k]) for k in payload if k!='signature')
    path=tmp_path/'migration.pt';torch.save(repaired,path);loaded=torch.load(path,weights_only=False)
    assert all(equal(payload[k],loaded[k]) for k in payload if k!='signature')
    q=torch.nn.Parameter(loaded['patch'].clone());other=torch.optim.Adam([q],lr=.002);other.load_state_dict(loaded['optimizer'])
    for x,o in [(p,opt),(q,other)]:
        o.zero_grad();x.square().mean().backward();o.step()
    assert torch.equal(p,q)
    with pytest.raises(RuntimeError):c.migrate_payload(payload,'wrong','repair','shuffled_relation',42,2)


def module(device='cuda:0',mapping=None,offload=False):
    leaf=NS(hf_device_map=mapping,_hf_hook=NS(offload=offload))
    return NS(parameters=lambda:iter([NS(device=device,numel=lambda:17)]),named_modules=lambda:iter([('',leaf)]))


def test_residency_rejects_cpu_disk_meta_and_offload():
    from rfix_train import residency
    for device,mapping,offload in [('cpu',None,False),('meta',None,False),('cuda:0',{'layer':'disk'},False),('cuda:0',None,True)]:
        wrapper=NS(vla=module(device,mapping,offload))
        with pytest.raises(RuntimeError,match='fully on one GPU'):residency(wrapper)
    assert residency(NS(vla=module(mapping={'all':0})))['single_gpu_resident']


def test_only_one_policy_load_per_process(monkeypatch,tmp_path):
    import rfix_train as t
    calls=[];wrapper=NS(vla=module())
    monkeypatch.setattr(t,'ROOT',tmp_path);monkeypatch.setattr(t,'_loaded',False)
    monkeypatch.setattr(t.native,'load_wrapper',lambda key:calls.append(key) or wrapper)
    monkeypatch.setattr(t.source,'module',lambda *args:NS(Engine=lambda w,cp:NS(wrapper=w)))
    t.make_engine()
    with pytest.raises(RuntimeError,match='one policy load'):t.make_engine()
    assert calls==['libero_goal']


@pytest.mark.parametrize('index,expected',[(0,1),(1,3)])
def test_worker_only_runs_one_shuffled_candidate(monkeypatch,index,expected):
    import rfix_train as t
    calls=[]
    monkeypatch.setattr(t,'training_module',lambda:NS(run=lambda i,m:calls.append(i)))
    t.run(index,None);assert calls==[expected]


def test_dependency_graph_does_not_train_completed_matched_arms(monkeypatch,tmp_path):
    import rfix_cli as cli
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'complete',lambda p:'matched_relation' in str(p))
    monkeypatch.setattr(cli,'read',lambda p:{'cases':[{'start':'a'},{'start':'b'}]})
    calls=[]
    def one(stage,deps=(),after='afterok'):
        calls.append((stage,list(deps),after));return stage+'_JOB'
    cli.stages(one)
    assert [r[0] for r in calls]==['train0','train1','select','evaluate0','evaluate1','report']
    assert calls[2][1]==['train0_JOB','train1_JOB']
    assert calls[-1]==('report',['evaluate0_JOB','evaluate1_JOB'],'afterany')


def test_missing_selection_reports_waiting_without_exception(monkeypatch,tmp_path):
    from contextlib import nullcontext
    import rfix_report as r
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'BASE',tmp_path/'base')
    monkeypatch.setattr(r,'lock',lambda name:nullcontext())
    monkeypatch.setattr(r,'complete',lambda p:False)
    monkeypatch.setattr(r,'external_complete',lambda root,p:p.parent.name in ('clean','initial_s42','initial_s43','uada500'))
    (tmp_path/'results').mkdir();(tmp_path/'data').mkdir()
    cases=[{'start':f's{i}','suite':'libero_goal','task_id':i,'state_index':2,'rollout_seed':i} for i in range(2)]
    (tmp_path/'data/MANIFEST.json').write_text(json.dumps({'cases':cases}))
    def folder(root,p):
        case=cases[int(p.parents[1].name[1:])];target=tmp_path/'mock'/case['start']/p.parent.name;target.mkdir(parents=True,exist_ok=True)
        (target/'EPISODE.json').write_text(json.dumps({'rollout_seed':case['rollout_seed'],'checkpoint':'libero_goal',
            'success':int(case['task_id']==0),'executed_steps':2,'events':{}}))
        return target
    monkeypatch.setattr(r,'episode_path',folder)
    r.run(NS(update=lambda **kw:None))
    result=json.loads((tmp_path/'results/STATUS.json').read_text())
    assert result['status']=='WAITING_TRAINING' and not result['selection_complete']
    assert result['clean_successes']==1 and result['native_new_episodes']==0
    assert next(s for s in result['summary'] if s['condition']=='shuffled_relation_s42')['CASR'] is None


def test_raw_camera_writer_and_both_playback_formats(tmp_path):
    import imageio.v2 as imageio
    from rfix_evaluate import CameraWriters
    from export_singleview_videos import encode,metadata
    obs={'full_image':np.full((224,224,3),65,np.uint8),'wrist_image':np.full((224,224,3),180,np.uint8)}
    with CameraWriters(tmp_path,20) as writer:
        for _ in range(3):writer.append_data(obs)
    for view,value in [('front',65),('wrist',180)]:
        assert metadata(tmp_path/(view+'.mp4'))['size']==(224,224)
        result=encode(tmp_path/(view+'.mp4'),tmp_path/(view+'.webm'))
        assert result['decoded_complete']
        with imageio.get_reader(tmp_path/(view+'.webm')) as reader:
            frames=list(reader.iter_data())
        assert len(frames)==3 and abs(frames[0].mean()-value)<4


def test_legacy_crop_removes_text_bar_and_separates_cameras(tmp_path):
    import imageio.v2 as imageio
    from export_singleview_videos import encode
    image=np.zeros((288,448,3),np.uint8);image[:64]=[0,255,0]
    image[64:,:224]=[255,0,0];image[64:,224:]=[0,0,255]
    source=tmp_path/'legacy.mp4'
    with imageio.get_writer(source,fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as writer:
        for _ in range(3):writer.append_data(image)
    for view,x,channel in [('front',0,0),('wrist',224,2)]:
        target=tmp_path/(view+'.mp4');encode(source,target,f'crop=224:224:{x}:64')
        with imageio.get_reader(target) as reader:pixels=reader.get_data(0)
        assert pixels.shape==(224,224,3) and pixels[...,channel].mean()>245 and pixels[...,1].mean()<5
