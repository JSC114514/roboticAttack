"""Independent audit artifacts. Historical experiments are never written to."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]

def read(path): return json.loads(Path(path).read_text())
def config(): return read(ROOT/'config.json')
SOURCE = PROJECT/config()['source_run']

def require(value, message):
    if not value: raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def atomic(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)

@lru_cache(None)
def source_adapter():
    sys.path.insert(1, str(SOURCE/'code'))
    import sr_common
    require(sr_common.signature() == config()['source_signature'], 'Historical source revision changed')
    return sr_common

@lru_cache(None)
def signature():
    files = [ROOT/'config.json', ROOT/'README.md', ROOT/'run.sh']
    files += [p for d in ('code','slurm','tests') for p in (ROOT/d).glob('*') if p.is_file()]
    payload = {'source_signature': source_adapter().signature(),
               'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def complete(path):
    path = Path(path)
    if not path.exists(): return False
    r = read(path)
    require(r['signature'] == signature(), 'Audit code changed; use a new revision directory: '+str(path))
    require(all((ROOT/k).is_file() and sha(ROOT/k) == v for k,v in r['artifacts'].items()),
            'Audit artifact changed: '+str(path))
    return r['status'] == 'COMPLETE'

def finish(path, artifacts, **extra):
    atomic({'status':'COMPLETE', 'signature':signature(), 'completed_unix':time.time(),
            'artifacts':{str(Path(p).relative_to(ROOT)):sha(p) for p in artifacts}, **extra}, path)

@contextmanager
def lock(name):
    p = ROOT/f'reports/locks/{name}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as handle:
        # Never wait on a held report lock, including locks from historical jobs.
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

class Monitor:
    def __init__(self, stage):
        self.stage=stage; self.started=time.monotonic(); self.guard=threading.Lock()
        self.values={'phase':stage, 'completed_steps':0}; self.done=threading.Event()
    def update(self, **values):
        with self.guard: self.values.update(values)
    def write(self):
        with self.guard: values=dict(self.values)
        elapsed=time.monotonic()-self.started; completed=values.get('completed_steps',0)
        total=values.get('total_steps',0)
        new_steps=completed-values.get('resumed_steps',0)
        values.update(heartbeat_unix=time.time(), elapsed_s=elapsed, job_id=os.environ.get('SLURM_JOB_ID'),
                      process_rss_hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        values['steps_per_second']=new_steps/elapsed if new_steps>0 else None
        values['estimated_remaining_s']=(total-completed)*elapsed/new_steps if new_steps>0 and total else None
        values.update(cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0,
                      cuda_note='No PyTorch model/tensors on GPU; EGL graphics allocations measured separately')
        if self.stage=='replay':
            try:
                r=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                    '--query-gpu=name,memory.used,memory.total','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=5)
                values['whole_device_graphics_memory_sample']=r.stdout.strip() or r.stderr.strip()
                values['gpu_memory_scope']='whole device, includes other processes; MiB'
                if r.returncode==0:
                    sample=float(r.stdout.strip().split(',')[-2])
                    self.gpu_peak_mib=max(getattr(self,'gpu_peak_mib',0),sample)
                    values['sampled_whole_device_peak_memory_mib']=self.gpu_peak_mib
            except (OSError, KeyError, subprocess.TimeoutExpired) as e: values['gpu_sample_error']=repr(e)
        atomic(values, ROOT/f'reports/PROGRESS/{self.stage}.json')
    def __enter__(self):
        def heartbeat():
            while not self.done.wait(30): self.write()
        self.thread=threading.Thread(target=heartbeat,daemon=True); self.thread.start(); self.write(); return self
    def __exit__(self, typ, error, tb):
        self.done.set(); self.thread.join(timeout=6)
        self.update(status='FAILED' if typ else 'COMPLETE', error=repr(error) if typ else None); self.write()
        atomic({'stage':self.stage,'status':'FAILED' if typ else 'COMPLETE','error':repr(error) if typ else None,
                'traceback':''.join(traceback.format_exception(typ,error,tb)) if typ else None,
                'elapsed_s':time.monotonic()-self.started,'process_rss_hwm_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'job_id':os.environ.get('SLURM_JOB_ID'),'cuda_peak_allocated_bytes':0,'cuda_peak_reserved_bytes':0,
                'sampled_whole_device_peak_memory_mib':getattr(self,'gpu_peak_mib',None),
                'cuda_note':'No PyTorch CUDA allocations; EGL memory samples in PROGRESS'},
               ROOT/f'reports/COSTS/{self.stage}_{time.time_ns()}.json')

def attempt_from_receipt(receipt):
    candidates=[SOURCE/k for k in receipt['artifacts'] if k.endswith('/STEPS.parquet')]
    require(len(candidates)==1, 'Expected one successful attempt in source receipt')
    return candidates[0].parent

def prepare():
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify()
    import pandas as pd
    import imageio_ffmpeg
    s=source_adapter(); entries=[]; inputs={}
    def pin(path):
        path=Path(path); inputs[str(path)]=sha(path); return str(path)
    for name in ('config.json','data/MANIFEST.json','data/SELECTED.json') : pin(SOURCE/name)
    old=read(SOURCE/'data/MANIFEST.json'); selected=read(SOURCE/'data/SELECTED.json')
    patches={'clean':None,'initial_s42':SOURCE/'data/initial_s42.png','uada500':Path(old['uada_png'])}
    patches.update({k:Path(v['path']) for k,v in selected['candidates'].items()})
    rows=[]
    for p in sorted((SOURCE/'evaluation/episodes').glob('*/*/COMPLETE.json')):
        require(s.complete(p), 'Invalid historical receipt: '+str(p)); pin(p)
        r=read(p); meta=read(p.parent/'EPISODE.json'); rows.append(meta)
        # All successful receipts, including both seeds, are pinned for the CPU summary.
        for name in ('EPISODE.json','TRACE.parquet'): pin(p.parent/name)
        pin(attempt_from_receipt(r)/'STEPS.parquet')
        clean=read(p.parent.parent/'clean/EPISODE.json')
        if not clean['success'] or meta['condition'] not in config()['video_conditions']: continue
        attempt=attempt_from_receipt(r)
        for k in r['artifacts']: pin(SOURCE/k)
        init=Path(old['initial_states'][f"{meta['suite']}/{meta['task_id']}"]['path']); pin(init)
        require(sha(init)==old['inputs'][str(init)], 'Initial state file changed')
        patch=patches[meta['condition']]
        if patch is not None:
            pin(patch); require(sha(patch)==meta['png_sha256'], 'Deployment PNG hash changed')
        trace=pd.read_parquet(p.parent/'TRACE.parquet'); steps=pd.read_parquet(attempt/'STEPS.parquet')
        require(len(steps)==meta['executed_steps']==sum(trace.executed_slots), 'Historical step/trace count mismatch')
        entries.append({'key':p.parent.parent.name+'/'+meta['condition'], 'start':p.parent.parent.name,
            'episode':meta,'trace':str(p.parent/'TRACE.parquet'),'steps':str(attempt/'STEPS.parquet'),
            'attempt':str(attempt),'paired_start':str(p.parent.parent/'paired_start'),
            'initial_file':str(init),'task_name':old['initial_states'][f"{meta['suite']}/{meta['task_id']}"]['task_name'],
            'patch':str(patch) if patch else None,'source_receipt':str(p)})
    clean_success=[r for r in rows if r['condition']=='clean' and r['success']]
    require(len(rows)==206 and len(clean_success)==18 and len(entries)==90, 'Source coverage differs from frozen audit scope')
    require({r['suite'] for r in clean_success}=={'libero_goal'}, 'Revisit checkpoint coverage before extending audit')
    # The first clean/primary pair supplies measured throughput, then all 90 complete.
    order=['clean','radial_reverse_s42','initial_s42','uada500','translation_far_boundary_s42']
    entries.sort(key=lambda e:(e['start'],order.index(e['episode']['condition'])))
    encoder=Path(imageio_ffmpeg.get_ffmpeg_exe()); require(encoder.is_file(), 'No local video encoder')
    pin(encoder)
    manifest={'signature':signature(),'source_signature':s.signature(),'entries':entries,'inputs':inputs,
        'historical_rows':rows,'replays':len(entries),'total_steps':sum(e['episode']['executed_steps'] for e in entries),
        'encoder':str(encoder),'claim':'Saved-action visual reconstruction, not new independent policy episodes',
        'initial_visibility':'Unavailable until first verified native sensor callback; never force sample/reset clocks'}
    for d in ('logs','replays','comparisons','results','reports/PROGRESS','reports/COSTS'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    atomic(manifest,ROOT/'data/MANIFEST.json'); finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json'])
    return manifest

def verify():
    v=read(ROOT/'data/MANIFEST.json'); require(v['signature']==signature(),'Prepared audit revision changed')
    for p,h in v['inputs'].items(): require(sha(p)==h, 'Pinned historical input changed: '+p)
    return v
