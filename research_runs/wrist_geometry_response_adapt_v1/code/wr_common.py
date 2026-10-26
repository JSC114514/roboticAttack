"""Shared contracts for wrist-only robustness adaptation; CPU-safe imports."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'

def require(ok, message):
    if not bool(ok): raise RuntimeError(message)

def read(path): return json.loads(Path(path).read_text())
def config(): return read(ROOT / 'config.json')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
def atomic(value, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name('.'+p.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)
def save_torch(value, path):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp');torch.save(value,tmp);tmp.replace(p)

@lru_cache(None)
def signature():
    c=config(); paths=[ROOT/'config.json']+sorted((ROOT/'code').glob('*.py'))
    paths += [PROJECT/p for p in c['dependency_files']]
    return digest({str(p):sha(p) for p in paths})

def complete(path):
    p=Path(path)
    if not p.is_file(): return False
    v=read(p);require(v.get('status')=='COMPLETE' and v.get('signature')==signature(), 'Stale/incomplete receipt: '+str(p))
    for rel,h in v['artifacts'].items():
        q=(ROOT/rel).resolve()
        require(q.is_relative_to(ROOT.resolve()) and q.is_file() and sha(q)==h, 'Changed artifact: '+str(q))
    return True
def finish(path, files, **extra):
    atomic(dict(status='COMPLETE',signature=signature(),completed_unix=time.time(),
                artifacts={str(Path(p).relative_to(ROOT)):sha(p) for p in files},**extra),path)

@contextmanager
def lock(name):
    p=ROOT/'reports/locks'/(name+'.lock');p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def resources():
    v=dict(process_rss_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
           cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0)
    t=sys.modules.get('torch')
    if t is not None and t.cuda.is_initialized():
        v.update(cuda_peak_allocated_bytes=t.cuda.max_memory_allocated(),
                 cuda_peak_reserved_bytes=t.cuda.max_memory_reserved(),device=t.cuda.get_device_name(0))
    return v

class Monitor:
    def __init__(self,stage):
        self.stage=stage;self.data={};self.started=time.monotonic();self.guard=threading.Lock();self.stop=threading.Event()
    def update(self,**v):
        with self.guard:self.data.update(v)
    def write(self):
        with self.guard:v=dict(self.data)
        elapsed=time.monotonic()-self.started;done=v.get('completed',0)-v.get('resumed',0)
        if done>0 and v.get('total'):
            v.update(throughput_per_s=done/elapsed,estimated_remaining_s=max(0,v['total']-v['completed'])*elapsed/done)
        atomic({**v,'stage':self.stage,'heartbeat_unix':time.time(),'elapsed_s':elapsed,
                'job_id':os.environ.get('SLURM_JOB_ID'),**resources()},ROOT/'reports/PROGRESS'/(self.stage+'.json'))
    def __enter__(self):
        def pulse():
            while not self.stop.wait(30):self.write()
        self.thread=threading.Thread(target=pulse,daemon=True);self.thread.start();self.write();return self
    def __exit__(self,kind,error,tb):
        self.stop.set();self.thread.join(5)
        self.update(status='FAILED' if kind else 'COMPLETE',error=repr(error) if kind else None)
        self.write()
        atomic(dict(stage=self.stage,status='FAILED' if kind else 'COMPLETE',error=repr(error) if kind else None,
                    traceback=''.join(traceback.format_exception(kind,error,tb)) if kind else None,
                    elapsed_s=time.monotonic()-self.started,job_id=os.environ.get('SLURM_JOB_ID'),**resources()),
               ROOT/'reports/COSTS'/(self.stage+'_'+str(time.time_ns())+'.json'))

def candidates(stage):
    c=config()
    if stage=='pilot':return c['pilot_candidates']
    p=ROOT/'results/pilot/SELECTION.json'
    require(p.is_file(),'Run and review the pilot before submitting main training')
    selected=read(p)
    require(selected.get('signature')==signature(),'Pilot selection revision changed')
    require(selected.get('complete') is True,'Pilot comparison is incomplete')
    rows=selected.get('selected',selected.get('methods',{}))
    if isinstance(rows,list):rows={x['method']:x for x in rows}
    out=[]
    for method in ('bc','augment','consistency','constrained'):
        value=rows[method]
        template=next(x for x in c['pilot_candidates'] if x['name']==value) if isinstance(value,str) else value
        for seed in c['main_seeds']:
            out.append(dict(template,name=f'{method}_s{seed}',seed=seed,updates=c['stages']['main']['updates']))
    return out

def candidate(stage,name):
    return next(x for x in candidates(stage) if x['name']==name)

def task_map():
    import ast
    p=Path(config()['libero_root'])/'libero/libero/benchmark/libero_suite_task_map.py'
    tree=ast.parse(p.read_text())
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(getattr(t,'id','')=='libero_task_map' for t in n.targets)))

def cases(stage):
    c=config();names=task_map();out=[]
    for j,s in enumerate(c['suites']):
        tasks=c['task_splits'][s]['development'] if stage=='pilot' else range(10)
        for task in tasks:
            for state in c['stages'][stage]['eval_states']:
                name=names[s][task]
                out.append(dict(suite=s,suite_index=j,task_id=task,task_name=name,state_index=state,
                    start=f'{s}_t{task:02d}_i{state:02d}',rollout_seed=c['seed_base']+j*10000+task*100+state,
                    initial_file=str(Path(c['libero_root'])/'libero/libero/init_files'/s/(name+'.pruned_init'))))
    return out

def profiles(stage):
    d=config()['stages'][stage]['eval_shift_m']
    return [dict(name='native',axis='x',shift_m=0.)]+[
        dict(name=f'wrist_{axis}_{sign}',axis=axis,shift_m=k*d)
        for axis in ('x','z') for sign,k in (('minus',-1),('plus',1))]

def require_gpu():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES'),
            'GPU work requires a user-submitted Slurm allocation')

@lru_cache(None)
def native_runtime():
    require_gpu();c=config();p=PROJECT/c['runtime_run']/'code'
    sys.path.insert(0,str(p))
    spec=importlib.util.spec_from_file_location('wr_native_runtime',p/'model_runtime.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def runtime_config():
    c=config();old=read(PROJECT/c['runtime_run']/'config.json')['evaluation']
    return dict(old,checkpoints={s:c['model_dir'] for s in c['suites']},
                env_image_resolution=256,policy_image_resolution=224,
                num_steps_wait=10,num_open_loop_steps=8,max_steps_by_suite=c['max_steps'])

@lru_cache(None)
def native_helpers():
    p=PROJECT/'research_runs/wrist_joint_checkpoint_patch_retest_v1/code';sys.path.insert(0,str(p))
    import jr_common
    jr_common.ROOT=ROOT;jr_common.config=config;jr_common.runtime_config=runtime_config
    jr_common.runtime=native_runtime;jr_common.atomic=atomic;jr_common.resources=resources
    import jr_rollout
    jr_rollout.ROOT=ROOT
    return jr_rollout

def load_wrapper(monitor):
    require_gpu();r=native_runtime();r.prepare_graphics_backend();r.add_paths(runtime_config())
    return native_helpers().load_wrapper(monitor)

def switch_suite(wrapper,suite):
    wrapper.cfg.task_suite_name=suite;wrapper.gen_cfg.task_suite_name=suite
    wrapper.cfg.unnorm_key=suite+'_no_noops';wrapper.gen_cfg.unnorm_key=suite+'_no_noops'
    require(wrapper.gen_cfg.unnorm_key in wrapper.vla.norm_stats,'Suite normalization missing')
