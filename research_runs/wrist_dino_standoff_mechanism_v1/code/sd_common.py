"""Isolated protocol, frozen input provenance and restartable receipts."""
from __future__ import annotations
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'

def require(condition, message):
    if not condition: raise RuntimeError(message)

def read_json(path): return json.loads(Path(path).read_text())
def config(): return read_json(ROOT/'config.json')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def atomic_json(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(tmp,path)
def atomic_torch(value,path):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');torch.save(value,tmp);os.replace(tmp,path)
def atomic_parquet(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp');frame.to_parquet(tmp,index=False);os.replace(tmp,path)

@lru_cache(maxsize=1)
def runtime_config():
    c=read_json(PROJECT/config()['runtime_run']/'config.json')['evaluation']
    return {**c,'env_image_resolution':256,'policy_image_resolution':224}

@lru_cache(maxsize=1)
def native():
    # Only the read-only loader/environment APIs are used; no historical worker
    # or writer is invoked. Their unqualified imports use their original common.
    code=PROJECT/'research_runs/wrist_dino_geometry_match_v1/code'
    sys.path.insert(1,str(code))
    import common
    require(Path(common.__file__).resolve()==code/'common.py','Ambiguous historical common import')
    return common

def runtime(): return native().runtime_module()
def key(task): return f"{task['suite']}_t{task['task_id']:02d}"
def assigned(index): return config()['tasks'][index::2]
def episode_id(task,state): return f'{key(task)}_s{state}'
def scan_receipt(task,state): return ROOT/f'data/capture/{episode_id(task,state)}/COMPLETE.json'
def scan_path(task,state): return scan_receipt(task,state).parent/'SELECTION.pt'
def case_root(case): return ROOT/'evaluation'/case['case_id']

@lru_cache(maxsize=1)
def signature():
    paths=[ROOT/'config.json',ROOT/'README.md',ROOT/'run.sh']
    for folder in ('code','slurm','tests'): paths.extend(p for p in (ROOT/folder).glob('*') if p.is_file())
    external=[PROJECT/config()['runtime_run']/'config.json']
    external.extend((PROJECT/config()['runtime_run']/'code').glob('*.py'))
    external += [PROJECT/'research_runs/wrist_dino_geometry_match_v1/code/common.py',
                 PROJECT/'research_runs/wrist_dino_geometry_match_v1/config.json',
                 PROJECT/'VLAAttacker/multiview/openvla_oft_wrapper.py',
                 PROJECT/'VLAAttacker/multiview/openvla_oft_experiments.py']
    return digest({str(p):sha(p) for p in sorted(paths+external)})

def complete(path):
    path=Path(path)
    if not path.exists(): return False
    r=read_json(path)
    require(r['signature']==signature(),'Stale revision: '+str(path))
    require(all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in r['artifacts'].items()),'Changed artifact: '+str(path))
    return r['status']=='COMPLETE'

def finish(path,artifacts,**extra):
    atomic_json({'signature':signature(),'status':'COMPLETE','completed_unix':time.time(),
        'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in artifacts},**extra},path)

def resources():
    v={'process_rss_hwm_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
    # Never probe CUDA from login-node preflight or CPU reporting.
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        v.update(cuda_device_name=torch.cuda.get_device_name(0),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    return v

class Monitor:
    def __init__(self,name):
        self.name=name;self.fields={};self.lock=threading.RLock();self.started=time.monotonic();self.event=threading.Event()
    def update(self,**fields):
        with self.lock:self.fields.update(fields);self.flush()
    def flush(self):
        with self.lock:
            f=dict(self.fields);elapsed=time.monotonic()-self.started
            n=f.get('completed',0)-f.get('resume_completed',0);total=f.get('total',0)
            eta=elapsed/n*(total-f.get('completed',0)) if n>0 and total>=f.get('completed',0) else None
            atomic_json({'signature':signature(),'worker':self.name,'job_id':os.environ.get('SLURM_JOB_ID'),
                'heartbeat_unix':time.time(),'elapsed_s':elapsed,'estimated_remaining_s':eta,
                **resources(),**f},ROOT/f'reports/PROGRESS/{self.name}.json')
    def _loop(self):
        while not self.event.wait(30):self.flush()
    def __enter__(self):
        self.update(status='RUNNING',phase='starting');self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def __exit__(self,kind,error,tb):
        self.event.set();self.thread.join();self.update(status='FAILED' if kind else 'COMPLETE',error=repr(error) if error else None)

@contextmanager
def cost(stage):
    p=ROOT/f'reports/COSTS/{stage}_{time.time_ns()}.json';started=time.monotonic()
    base={'signature':signature(),'stage':stage,'job_id':os.environ.get('SLURM_JOB_ID'),'started_unix':time.time()}
    atomic_json({**base,'status':'STARTED'},p)
    try: yield
    except BaseException as error:
        atomic_json({**base,'status':'FAILED','elapsed_s':time.monotonic()-started,'error':repr(error),
            'traceback':traceback.format_exc(),**resources()},p);raise
    else:atomic_json({**base,'status':'COMPLETE','elapsed_s':time.monotonic()-started,**resources()},p)

def require_gpu():
    import torch
    require(bool(os.environ.get('SLURM_JOB_ID')),'GPU work requires user-submitted Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one allocated GPU required')
    require('4090' in torch.cuda.get_device_name(0),'Expected allocated 4090')

def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):
        verify_inputs();return read_json(ROOT/'data/MANIFEST.json')
    import numpy as np
    import pandas as pd
    import torch
    c=config();require(c['block']==22 and c['checkpoint_suite']=='libero_spatial' and c['max_concurrent_gpus']==2,'Protocol changed')
    require(c['distance_m']==.05 and c['max_steps']==128 and c['random_directions']==4,'Study constants changed')
    for folder in ('data','evaluation','results','logs','reports/locks','reports/PROGRESS','reports/COSTS','reports/SUBMISSIONS'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    source=PROJECT/c['source_run']/'results'
    traces=source/'LONG4_TRACES.parquet';rollouts=source/'LONG4_ROLLOUTS.parquet'
    library=Path(runtime_config()['paths']['libero_root'])/'libero/libero'
    mapping=library/'benchmark/libero_suite_task_map.py'
    nodes=ast.parse(mapping.read_text()).body
    names=ast.literal_eval(next(n.value for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets)))
    inputs={str(p):sha(p) for p in (traces,rollouts,mapping)};episodes=[];init_meta={}
    audited_path=PROJECT/'research_runs/wrist_dino_geometry_rgb2_v1_fix1/data/MANIFEST.json'
    audited=read_json(audited_path);inputs[str(audited_path)]=sha(audited_path)
    for task in c['tasks']:
        suite,tid=task['suite'],task['task_id'];name=names[suite][tid]
        path=library/'init_files'/suite/(name+'.pruned_init')
        initial=torch.load(path,map_location='cpu',weights_only=False)
        require(all(s<len(initial) and np.isfinite(np.asarray(initial[s])).all() for s in task['states']),'Initial state missing/nonfinite')
        inputs[str(path)]=sha(path);init_meta[key(task)]={'path':str(path),'task_name':name,'count':len(initial)}
        require(audited['inputs'].get(str(path))==inputs[str(path)],'Initial states changed since audited replay cache')
        bddl=library/'bddl_files'/suite/(name+'.bddl');inputs[str(bddl)]=sha(bddl)
        filters=[('condition','==','clean'),('suite','==',suite),('task_id','==',tid)]
        rs=pd.read_parquet(rollouts,filters=filters);ts=pd.read_parquet(traces,filters=filters)
        for state in task['states']:
            record=rs[rs.state_index==state];require(len(record)==1,'Ambiguous/missing source clean rollout')
            r=record.iloc[0];chunks=ts[(ts.state_index==state)&(ts.rollout_seed==r.rollout_seed)].sort_values('decision_block')
            require(chunks.decision_block.tolist()==list(range(len(chunks))),'Nonconsecutive saved decisions')
            cumulative=0
            for row in chunks.itertuples(index=False):
                a=np.stack([np.asarray(v,dtype=np.float32) for v in row.predicted_environment_action_chunk])
                require(a.shape==(8,7) and np.isfinite(a).all() and 1<=row.executed_slots<=8,'Invalid saved action')
                cumulative+=int(row.executed_slots);require(cumulative==row.step_end,'Saved execution count mismatch')
            require(cumulative==int(r.executed_steps),'Incomplete action trajectory')
            episodes.append({**task,'episode_id':episode_id(task,state),'state_index':state,'rollout_seed':int(r.rollout_seed),
                'cohort':'object_holdout' if suite=='libero_10' else ('discovery' if state in (40,41) else 'validation'),
                'source_success':int(r.success),'decision_count':len(chunks),'executed_steps':cumulative})
        del initial
    require(len(episodes)==14,'Expected fourteen trajectories')
    checkpoint=Path(runtime_config()['checkpoints']['libero_spatial'])
    for filename in ('config.json','modeling_prismatic.py','configuration_prismatic.py','dataset_statistics.json',
        'preprocessor_config.json','processor_config.json','tokenizer_config.json','model.safetensors.index.json'):
        p=checkpoint/filename;inputs[str(p)]=sha(p)
    for filename in ('modeling_prismatic.py','configuration_prismatic.py'):
        p=Path(runtime_config()['paths']['openvla_oft_root'])/'prismatic/extern/hf'/filename
        require(sha(p)==inputs[str(checkpoint/filename)],'Checkpoint code sync would be required; refusing')
        inputs[str(p)]=sha(p)
    require(read_json(checkpoint/'config.json')['auto_map']=={'AutoConfig':'configuration_prismatic.OpenVLAConfig','AutoModelForVision2Seq':'modeling_prismatic.OpenVLAForActionPrediction'},'Checkpoint auto_map changed')
    weights={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        for pattern in ('*.safetensors','*.pt') for p in checkpoint.glob(pattern)}
    require(weights,'Checkpoint weights not found')
    require(not torch.cuda.is_initialized() and 'mujoco' not in sys.modules,'CPU preparation must not initialize CUDA/MuJoCo')
    # Pin geometry definitions and simulator API without importing MuJoCo on login CPU.
    assets=library/'assets'
    for rel in ('articulated_objects/wooden_cabinet.xml','stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml',
                'stable_hope_objects/alphabet_soup/alphabet_soup.xml','turbosquid_objects/black_book/black_book.xml'):
        p=assets/rel;inputs[str(p)]=sha(p)
    manifest={'signature':signature(),'inputs':inputs,'weight_file_stats':weights,'initial_states':init_meta,'episodes':episodes,
        'trajectories':14,'max_states':28,'max_short_rollouts':84,'checkpoint_suite':'libero_spatial',
        'mujoco_imported':False,'model_loaded':False,'jobs_submitted':False}
    atomic_json(manifest,ROOT/'data/MANIFEST.json');finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json'],**resources())
    return manifest

def verify_inputs():
    m=read_json(ROOT/'data/MANIFEST.json');require(m['signature']==signature(),'Manifest revision changed')
    for p,h in m['inputs'].items():require(sha(p)==h,'Read-only input changed: '+p)
    for p,st in m['weight_file_stats'].items():
        s=Path(p).stat();require({'size':s.st_size,'mtime_ns':s.st_mtime_ns}==st,'Checkpoint weight file changed')
    return m

def source_chunks(episode):
    import pandas as pd
    p=PROJECT/config()['source_run']/'results/LONG4_TRACES.parquet'
    return pd.read_parquet(p,filters=[('condition','==','clean'),('suite','==',episode['suite']),
        ('task_id','==',episode['task_id']),('state_index','==',episode['state_index']),
        ('rollout_seed','==',episode['rollout_seed'])]).sort_values('decision_block')

def worker_complete(stage,index): return complete(ROOT/f'reports/{stage.upper()}_{index}_COMPLETE.json')
