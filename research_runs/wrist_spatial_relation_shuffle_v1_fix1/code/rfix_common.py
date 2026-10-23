"""Explicit provenance migration; sealed parent runs and libraries stay read-only."""
from functools import lru_cache
from pathlib import Path
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
OLD=PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1'
BASE=PROJECT/'research_runs/wrist_spatial_relation_expand_v1'
TOOLS=PROJECT/'research_tools/spatial_followup'
os.environ['FOLLOWUP_ROOT']=str(OLD)
sys.path.insert(1,str(TOOLS))
import sf_common as legacy
source,native=legacy.source,legacy.native
read,sha,atomic,require=legacy.read,legacy.sha,legacy.atomic,legacy.require
external_complete=legacy.external_complete


def config():return read(ROOT/'config.json')


@lru_cache(None)
def signature():
    require(legacy.signature()=='8a7e5d51e904249f7a031b7f8f52ffd83187ad56d70907b6eb0cecaafc058820','Parent code changed')
    paths=[ROOT/'config.json',ROOT/'README.md',ROOT/'run.sh',PROJECT/'research_tools/export_singleview_videos.py']
    paths += [p for d in ('code','slurm','tests') for p in (ROOT/d).glob('*') if p.is_file()]
    return native.digest({'parent':legacy.signature(),'files':{str(p.relative_to(PROJECT)):sha(p) for p in sorted(paths)}})


def complete(path):
    path=Path(path)
    if not path.exists():return False
    r=read(path);require(r['status']=='COMPLETE' and r['signature']==signature(),'Stale repaired receipt: '+str(path))
    require(all((ROOT/k).is_file() and sha(ROOT/k)==v for k,v in r['artifacts'].items()),'Changed repaired artifacts')
    return True


def finish(path,files,**fields):
    atomic({'status':'COMPLETE','signature':signature(),'completed_unix':time.time(),
            'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in files},**fields},path)


def candidates():return [(f'{m}_s{s}',m,s) for s in config()['seeds'] for m in config()['methods']]


def records(cohort):yield from legacy.records(cohort)


def verify():
    v=read(ROOT/'data/MANIFEST.json');require(v['signature']==signature(),'Repair preparation revision changed')
    for p,h in v['inputs'].items():require(sha(p)==h,'Changed migration input: '+p)
    for p,expected in v['model_file_stats'].items():
        stat=Path(p).stat();require([stat.st_size,stat.st_mtime_ns]==expected,'Changed checkpoint: '+p)
    return v


def migrate_payload(value,expected_signature,new_signature,method,seed,step):
    """Only rebind execution identity; optimizer/patch/RNG are unmodified objects."""
    require(value['signature']==expected_signature,'Parent checkpoint identity differs')
    require(value['method']==method and value['seed']==seed and value['step']==step,'Wrong checkpoint selected')
    require(value['optimized']+value['no_target']==step,'Invalid update counters')
    state=value['optimizer']['state'];require(len(state)==1,'Expected patch-only Adam')
    adam=next(iter(state.values()))
    require(int(adam['step'])==value['optimized'],'Adam counter differs from actual updates')
    return {**value,'signature':new_signature}


def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return verify()
    import torch
    original=legacy.verify()
    require(external_complete(BASE,BASE/'results/COMPLETE.json'),'Expansion incomplete or changed')
    require(config()['updates']==4000 and config()['methods']==['matched_relation','shuffled_relation'],'Scientific budget changed')
    scientific=legacy.config()
    for key,value in scientific.items():require(config()[key]==value,'Scientific configuration changed: '+key)
    for d in ('data','training','reports/PROGRESS','reports/COSTS','reports/GATES','reports/locks','logs','results','episodes'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    inputs=dict(original['inputs'])
    for p in (OLD/'data/MANIFEST.json',OLD/'data/TARGETS.json',BASE/'results/COMPLETE.json'):inputs[str(p)]=sha(p)
    migrations=[];proofs=[]
    for name,method,seed in candidates():
        step=4000 if method=='matched_relation' else 2000
        old=OLD/'training'/name/f'checkpoint_{step:06d}.json'
        require(legacy.complete(old),'Parent checkpoint invalid: '+str(old))
        if method=='matched_relation':require(legacy.complete(old.parent/'COMPLETE.json'),'Completed arm invalid')
        source_pt=old.with_suffix('.pt');source_png=old.with_suffix('.png')
        inputs.update({str(p):sha(p) for p in (old,source_pt,source_png)})
        if method=='matched_relation':
            parent_complete=old.parent/'COMPLETE.json';inputs[str(parent_complete)]=sha(parent_complete)
            for name in read(parent_complete)['artifacts']:
                p=OLD/name;inputs[str(p)]=sha(p)
        folder=ROOT/'training'/name;folder.mkdir(parents=True,exist_ok=True)
        payload=torch.load(source_pt,map_location='cpu',weights_only=False)
        rebound=migrate_payload(payload,legacy.signature(),signature(),method,seed,step)
        destination=folder/source_pt.name
        native.atomic_torch(rebound,destination);shutil.copy2(source_png,destination.with_suffix('.png'))
        native.atomic_torch(rebound,folder/'LAST.pt')
        # The migrated tensors must serialize/deserialize identically, including all RNGs.
        restored=torch.load(destination,map_location='cpu',weights_only=False)
        require(torch.equal(restored['patch'],payload['patch']) and torch.equal(restored['torch_rng'],payload['torch_rng']), 'Migration changed patch/RNG')
        evidence={'parent_checkpoint':str(source_pt),'parent_sha256':sha(source_pt),'parent_signature':legacy.signature(),
                  'new_checkpoint':str(destination),'new_sha256':sha(destination),'step':step,'method':method,'seed':seed,
                  'identity_only_rebound':True,'parent_run_modified':False,
                  'status':'REUSED_COMPLETED' if step==4000 else 'RESUME_FROM_LAST_SEALED_2000',
                  'optimizer_schedule_sampling_rng_preserved':True}
        atomic(evidence,folder/'MIGRATION.json');migrations.append(evidence)
        new_receipt=destination.with_suffix('.json')
        finish(new_receipt,[destination,destination.with_suffix('.png'),folder/'MIGRATION.json'],
               step=step,optimized=payload['optimized'],no_target=payload['no_target'],method=method,seed=seed,terminal=step==4000)
        if step==4000:
            finish(folder/'COMPLETE.json',[new_receipt,folder/'LAST.pt',folder/'MIGRATION.json'],
                   candidate=name,step=step,optimized=payload['optimized'],no_target=payload['no_target'],
                   budget_complete=True,stop_reason='reused_completed_parent_no_GPU')
        proofs += [new_receipt,folder/'MIGRATION.json']
    for seed in config()['seeds']:
        p=OLD/f'data/initial_s{seed}.png';inputs[str(p)]=sha(p);shutil.copy2(p,ROOT/f'data/initial_s{seed}.png')
    parent_failures=[]
    for p in sorted((OLD/'reports/COSTS').glob('*.json')):
        cost=read(p)
        if cost.get('status')=='FAILED' and cost.get('stage','').startswith('train'):
            inputs[str(p)]=sha(p)
            parent_failures.append({'source':str(p),'sha256':sha(p),'stage':cost['stage'],
                                    'elapsed_s':cost['elapsed_s'],'error':cost.get('error'),
                                    'process_rss_hwm_bytes':cost.get('process_rss_hwm_bytes'),
                                    'cuda_peak_allocated_bytes':cost.get('cuda_peak_allocated_bytes')})
    value={**original,'signature':signature(),'inputs':inputs,'migrations':migrations,'parent_failed_costs':parent_failures,
           'reuse_completed_candidates':['matched_relation_s42','matched_relation_s43'],
           'resume_candidates':['shuffled_relation_s42','shuffled_relation_s43'],'resume_steps':2000,
           'single_model_per_process':True,'raw_camera_videos':True}
    atomic(value,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',*proofs,*[ROOT/f'data/initial_s{s}.png' for s in config()['seeds']]])
    return verify()


@lru_cache(None)
def support_module():
    # A private module object, so progress/lock paths never point to the parent.
    m=source.module('rfix_support_private',TOOLS/'sf_common.py')
    m.ROOT=ROOT;m.signature=signature
    return m


def lock(name):return support_module().lock(name)
def Monitor(name):return support_module().Monitor(name)


def training_module():
    from rfix_train import make_engine
    m=source.module('rfix_training_private',source.ROOT/'code/sr_train.py')
    for key,value in dict(ROOT=ROOT,config=config,candidates=candidates,records=records,complete=complete,
                          finish=finish,signature=signature,make_engine=make_engine).items():setattr(m,key,value)
    return m
