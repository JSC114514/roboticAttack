"""Sealed, independent action-restoration diagnosis. All sources read-only."""
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
def read(path): return json.loads(Path(path).read_text())
def config(): return read(ROOT/'config.json')
SOURCE=PROJECT/config()['source_run']
AUDIT=PROJECT/config()['audit_run']
sys.path.insert(1,str(AUDIT/'code'))
import audit_common as audited
require,sha,atomic=audited.require,audited.sha,audited.atomic

@lru_cache(None)
def adapters():
    require(audited.signature()==config()['audit_signature'],'Historical audit code revision changed')
    s=audited.source_adapter()
    require(s.signature()==config()['source_signature'],'Historical source code changed')
    return s

@lru_cache(None)
def signature():
    files=[ROOT/'config.json',ROOT/'README.md',ROOT/'run.sh']
    files += [p for d in ('code','slurm','tests') for p in (ROOT/d).glob('*') if p.is_file()]
    adapters()
    payload={'audit':audited.signature(),'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def complete(path):
    path=Path(path)
    if not path.exists(): return False
    v=read(path); require(v['signature']==signature(),'Restoration revision changed: '+str(path))
    require(v['status']=='COMPLETE','Invalid completion receipt: '+str(path))
    require(all((ROOT/k).is_file() and sha(ROOT/k)==h for k,h in v['artifacts'].items()),'Changed restoration artifact: '+str(path))
    return True

def finish(path,artifacts,**fields):
    atomic({'signature':signature(),'status':'COMPLETE','completed_unix':time.time(),
        'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in artifacts},**fields},path)

@contextmanager
def lock(name):
    p=ROOT/f'reports/locks/{name}.lock'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB); yield

def resources():
    v={'process_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
       'measurement_scope':'parent Python process; ffmpeg children excluded',
       'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0}
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():
            v.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                     cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),cuda_device_name=torch.cuda.get_device_name(0))
    return v

class Monitor:
    def __init__(self,stage):
        self.stage=stage; self.values={'phase':stage}; self.guard=threading.Lock(); self.stop=threading.Event(); self.start=time.monotonic()
        self.sampled_peak=0
    def update(self,**v):
        with self.guard:self.values.update(v)
    def write(self):
        with self.guard:v=dict(self.values)
        v.update(heartbeat_unix=time.time(),elapsed_s=time.monotonic()-self.start,job_id=os.environ.get('SLURM_JOB_ID'),**resources())
        if self.stage=='evaluate':
            try:
                r=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                    '--query-gpu=name,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                v['whole_device_memory_sample']=r.stdout.strip() or r.stderr.strip()
                if r.returncode==0:self.sampled_peak=max(self.sampled_peak,float(r.stdout.strip().split(',')[-2]))
                v.update(sampled_whole_device_peak_memory_mib=self.sampled_peak,
                         gpu_memory_scope='whole device, includes EGL and other processes; MiB')
            except (KeyError,OSError,ValueError,subprocess.TimeoutExpired) as error:v['gpu_sample_error']=repr(error)
        atomic(v,ROOT/f'reports/PROGRESS/{self.stage}.json')
    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(30):self.write()
        self.thread=threading.Thread(target=heartbeat,daemon=True);self.thread.start();self.write();return self
    def __exit__(self,typ,error,tb):
        self.stop.set();self.thread.join(timeout=6)
        self.update(status='FAILED' if typ else 'COMPLETE',error=repr(error) if typ else None);self.write()
        atomic({'stage':self.stage,'status':'FAILED' if typ else 'COMPLETE','error':repr(error) if typ else None,
            'traceback':''.join(traceback.format_exception(typ,error,tb)) if typ else None,
            'elapsed_s':time.monotonic()-self.start,'job_id':os.environ.get('SLURM_JOB_ID'),
            'signature':signature(),'sampled_whole_device_peak_memory_mib':self.sampled_peak,**resources()},
            ROOT/f'reports/COSTS/{self.stage}_{time.time_ns()}.json')

def artifact_folder(receipt,suffix):
    v=read(receipt); paths=[Path(k) for k in v['artifacts'] if k.endswith(suffix)]
    require(len(paths)==1,'Ambiguous successful attempt: '+str(receipt)); return (ROOT/paths[0]).parent

def prepared_expansion(source_manifest,inputs):
    import torch
    c=config(); states=c['expansion_states']; old=read(SOURCE/'config.json')
    require(not(set(states)&set(old['fit_states']+old['verify_states']+old['eval_states'])),'Expansion overlaps current protocol states')
    starts=[]
    for task in range(10):
        entry=source_manifest['initial_states'][f'libero_goal/{task}']; path=Path(entry['path'])
        require(sha(path)==source_manifest['inputs'][str(path)],'Expansion initial states changed')
        inputs[str(path)]=sha(path); saved=torch.load(path,map_location='cpu',weights_only=False)
        require(len(saved)>max(states),'Expansion initial states unavailable')
        starts += [{'suite':'libero_goal','task_id':task,'state_index':state,'initial_file':str(path),'task_name':entry['task_name'],
                    'rollout_seed':old['rollout_seed_base']+old['suites'].index('libero_goal')*10000+task*100+state} for state in states]
    selected=read(SOURCE/'data/SELECTED.json')['candidates']
    for name,value in selected.items():require(sha(value['path'])==value['png_sha256'],'Expansion PNG differs from original selection: '+name)
    patches={'clean':None,'uada500':source_manifest['uada_png']}
    patches.update({f'initial_s{s}':str(SOURCE/f'data/initial_s{s}.png') for s in (42,43)})
    patches.update({name:v['path'] for name,v in selected.items()})
    require(len(patches)==8 and len(starts)==40,'Unexpected expansion scope')
    for p in patches.values():
        if p:inputs[str(p)]=sha(p)
    runtime=adapters().native.runtime_config()
    return {'status':'PREPARED_NOT_SUBMITTED','checkpoint':'libero_goal','checkpoint_path':source_manifest['checkpoint'],
        'source_signature':config()['source_signature'],'starts':starts,'conditions':patches,
        'condition_sha256':{k:sha(p) if p else None for k,p in patches.items()},
        'condition_order':['clean','initial_s42','initial_s43','uada500','radial_reverse_s42','radial_reverse_s43','translation_far_boundary_s42','translation_far_boundary_s43'],
        'patch_shape':[3,50,50],'patch_xy':config()['patch_xy'],'eot':False,'max_steps':300,
        'num_steps_wait':runtime['num_steps_wait'],'num_open_loop_steps':runtime['num_open_loop_steps'],
        'conditional_attack_score':'Paired clean-success starts only; report all40 clean outcomes and explicit skips, never missing as success',
        'maximum_episodes':320,'requires_discussion_before_submission':True,'action_restoration':False,
        'independence_claim':'New relative to the current method protocol; not claimed unseen in all historical research',
        'reason':'Expand native PNG evidence only after interpreting restoration results; no automatic threshold or submission'}

def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return verify()
    import numpy as np
    s=adapters();require(audited.complete(AUDIT/'results/COMPLETE.json'),'Prior video audit incomplete')
    require(s.complete(SOURCE/'reports/PREPARE_COMPLETE.json') and s.complete(SOURCE/'reports/SELECT_COMPLETE.json'),'Original initial PNG / final selection receipt changed')
    c=config()
    require(c['checkpoint']=='libero_goal' and c['primary']=='radial_reverse_s42' and c['order_seed']==42,'Frozen model/primary/order protocol changed')
    require(c['methods']==['restore_radial','restore_xyz','restore_pose_grip'] and c['max_concurrent_gpus']==1 and c['max_steps']==300,'Restoration scope/resource protocol changed')
    require(c['patch_xy']==s.native.cfg()['patch_xy']==[26,137],'Pinned native patch position changed')
    am=audited.verify(); source=read(SOURCE/'data/MANIFEST.json'); inputs=dict(am['inputs'])
    # Include the audited videos/receipts actually reused in comparisons, not a history-wide scan.
    for p in (AUDIT/'data/MANIFEST.json',AUDIT/'results/COMPLETE.json'):inputs[str(p)]=sha(p)
    for p,expected in source['model_file_stats'].items():
        stat=Path(p).stat();require([stat.st_size,stat.st_mtime_ns]==expected,'Checkpoint metadata changed: '+p)
    grouped={}
    for e in am['entries']:grouped.setdefault(e['start'],{})[e['episode']['condition']]=e
    cases=[]; rng=np.random.default_rng(config()['order_seed'])
    for start,entries in sorted(grouped.items()):
        e=dict(entries[config()['primary']]);require(e['episode']['checkpoint']=='libero_goal' and e['episode']['max_steps']==300,'Source checkpoint/horizon changed')
        require(s.complete(e['source_receipt']),'Source deployment receipt changed')
        videos={}
        for condition in ('clean',config()['primary']):
            receipt=AUDIT/'replays'/start/condition/'COMPLETE.json';require(audited.complete(receipt),'Source video changed')
            inputs[str(receipt)]=sha(receipt)
            for k,h in read(receipt)['artifacts'].items():inputs[str(AUDIT/k)]=h
            videos[condition]=str(next(AUDIT/k for k in read(receipt)['artifacts'] if k.endswith('/video.mp4')))
        e.update(order=rng.permutation(config()['methods']).tolist(),baseline_episodes={c:v['episode'] for c,v in entries.items()},baseline_videos=videos)
        cases.append(e)
    require(len(cases)==18,'Restoration must include all 18 eligible starts')
    require({e['episode']['png_sha256'] for e in cases}=={sha(cases[0]['patch'])},'Primary PNG changed across starts')
    for d in ('data','episodes','results','comparisons','logs','reports/COSTS','reports/PROGRESS','reports/GATES'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    expansion=prepared_expansion(source,inputs);atomic(expansion,ROOT/'data/EXPANSION_PREPARED.json')
    v={'signature':signature(),'cases':cases,'inputs':inputs,'model_file_stats':source['model_file_stats'],
       'reference_replays':18,'new_restoration_episodes':54,'reference_steps':sum(e['episode']['executed_steps'] for e in cases),
       'maximum_new_steps':54*config()['max_steps'],'patch_sha256':sha(cases[0]['patch']),
       'claim':'Privileged action restoration diagnoses failure; never a new PNG attack score'}
    atomic(v,ROOT/'data/MANIFEST.json');finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',ROOT/'data/EXPANSION_PREPARED.json'])
    return v

def verify():
    v=read(ROOT/'data/MANIFEST.json');require(v['signature']==signature(),'Prepared code changed')
    for p,h in v['inputs'].items():require(sha(p)==h,'Pinned input changed: '+p)
    for p,expected in v['model_file_stats'].items():
        stat=Path(p).stat();require([stat.st_size,stat.st_mtime_ns]==expected,'Checkpoint metadata changed: '+p)
    return v
