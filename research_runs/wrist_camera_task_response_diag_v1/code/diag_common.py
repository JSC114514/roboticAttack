"""Run-local IO and established read-only OFT interfaces; no GPU on import."""
from functools import lru_cache
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
SOURCE=PROJECT/'research_runs/wrist_geometry_response_adapt_v1'
PYTHON='/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'
sys.path.insert(0,str(SOURCE/'code'))
import wr_common as wc
read=wc.read;atomic=wc.atomic;require=wc.require;sha=wc.sha;resources=wc.resources
source_signature=wc.signature()
SOURCE_CONFIG=read(SOURCE/'config.json')

def own_config():return read(ROOT/'config.json')
def config():
    return {**SOURCE_CONFIG,**own_config()}
@lru_cache(None)
def signature():
    paths=[ROOT/'config.json',ROOT/'run.sh']+sorted((ROOT/'code').glob('*.py'))
    return wc.digest({'source_signature':source_signature,'files':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
def complete(path):
    p=Path(path)
    if not p.exists():return False
    v=read(p);require(v.get('signature')==signature(),'Changed source/config for completed output: '+str(p))
    require(v.get('status')=='COMPLETE','Malformed completion receipt: '+str(p))
    for rel,h in v['artifacts'].items():
        q=ROOT/rel;require(q.is_file() and sha(q)==h,'Changed output: '+str(q))
    return True

def finish(path,files,**extra):
    import time
    atomic(dict(status='COMPLETE',signature=signature(),artifacts={str(Path(p).relative_to(ROOT)):sha(p) for p in files},completed_unix=time.time(),**extra),path)

def conditions():
    names={-.02:'minus',0.:'zero',.02:'plus'}
    return [dict(name='target_'+names[t]+'_camera_'+names[c],target_shift_m=t,camera_shift_m=c)
            for t in own_config()['target_shifts_m'] for c in own_config()['camera_shifts_m']]

def preflight():
    require(source_signature==own_config()['source_signature'],'Historical source revision changed')
    manifest=read(SOURCE/'data/pilot/MANIFEST.json')
    require(manifest['signature']==source_signature,'Historical manifest revision differs')
    rows=[r for r in manifest['camera']['validation'] if r['suite'] in config()['tasks'] and
          r['task_id'] in config()['tasks'][r['suite']] and r['state_index'] in config()['state_indices']]
    groups={k:[r for r in rows if r['episode_id']==k] for k in sorted({r['episode_id'] for r in rows})}
    require(len(groups)==12 and all(len(v)==5 for v in groups.values()),'Expected 12 trajectories with five observations')
    require(not ({r['episode_id'] for r in rows}&{r['episode_id'] for r in manifest['camera']['train']}),'Trajectory split overlap')
    for name in config()['models'][1:]:
        d=SOURCE/f'training/pilot/{name}';v=read(d/'COMPLETE.json')
        require(v['signature']==source_signature and v['status']=='COMPLETE','Source adapter incomplete')
        final=d/'final.pt';require(final.is_file(),'Adapter missing')
        require(v['artifacts'].get(str(final.relative_to(SOURCE)))==sha(final),'Source adapter hash differs')
    for episode in groups:
        d=SOURCE/f'data/pilot/camera/{episode}';v=read(d.with_suffix('.json'))
        require(v['signature']==source_signature,'Historical camera receipt changed')
        for rel,h in v['artifacts'].items():require(sha(SOURCE/rel)==h,'Historical camera cache differs')
    for name in ('action_head--300000_checkpoint.pt','proprio_projector--300000_checkpoint.pt','dataset_statistics.json'):
        require((Path(config()['model_dir'])/name).is_file(),'Joint checkpoint component missing: '+name)
    return dict(status='CPU_PREFLIGHT_PASS',candidate_trajectories=len(groups),cached_observations=len(rows),
                max_diagnostic_branches=324,max_diagnostic_control_steps=2592,GPU_tested=False,
                replay_costs_additional=True,signature=signature(),**resources())

def source_rows():
    m=read(SOURCE/'data/pilot/MANIFEST.json')
    return [r for r in m['camera']['validation'] if r['suite'] in config()['tasks'] and
            r['task_id'] in config()['tasks'][r['suite']] and r['state_index'] in config()['state_indices']]

def choose(rows):
    eligible=[r for r in rows if r['visible_pixels']>0 and not r['truth']['contact'] and not r['truth']['grasp']]
    return min(eligible,key=lambda r:(r['truth']['distance_m'],r['row']['decision_block'])) if eligible else None

# All rebinding is process-local. Historical source files, receipts and configs stay read-only.
wc.ROOT=ROOT;wc.config=config;wc.signature=signature;wc.complete=complete;wc.finish=finish
Monitor=wc.Monitor
native_runtime=wc.native_runtime;runtime_config=wc.runtime_config;native_helpers=wc.native_helpers
load_wrapper=wc.load_wrapper;switch_suite=wc.switch_suite;require_gpu=wc.require_gpu
