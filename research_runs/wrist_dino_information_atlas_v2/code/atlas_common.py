"""Frozen contracts and atomic artifacts for passive DINO information readout."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PYTHON = '/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def atomic_text(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(value); os.replace(tmp,path)


def config():
    return load_json(ROOT/'config.json')


def runtime_config():
    c=load_json(config()['runtime_config'])['evaluation']
    return {**c,'env_image_resolution':224,'policy_image_resolution':224}


def libraries():
    names=['torch','numpy','pandas','scipy','scikit-learn','timm','transformers','robosuite','mujoco']
    result={}
    for name in names:
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]='NOT_INSTALLED'
    return result


def source_hashes():
    paths=[ROOT/'config.json',ROOT/'PLAN.md',ROOT/'README.md',ROOT/'run.sh',ROOT/'COPIED_SOURCES.json']
    for folder in ('code','tests','slurm'):
        paths.extend(p for p in sorted((ROOT/folder).glob('*')) if p.is_file())
    c=config()
    paths += [Path(c[k]) for k in ('state_manifest','task_map','runtime_config')]
    paths += [Path(p) for p in c['old_labels'].values()]
    states=load_json(c['state_manifest'])['records']
    paths += sorted({Path(r['state_file']).parent.parent.parent.parent/'episodes.json' for r in states})
    paths += [Path(r['bddl_path']) for r in load_json(c['task_map'])['tasks']]
    rt=runtime_config(); native=Path(rt['paths']['openvla_oft_root'])
    paths += [PROJECT/'VLAAttacker/multiview/openvla_oft_wrapper.py',
              PROJECT/'VLAAttacker/multiview/openvla_oft_experiments.py',
              native/'prismatic/extern/hf/modeling_prismatic.py',
              native/'prismatic/extern/hf/processing_prismatic.py',
              native/'experiments/robot/openvla_utils.py']
    suite=Path(sysconfig.get_path('purelib'))/'robosuite'
    paths += [suite/p for p in ('utils/observables.py','utils/binding_utils.py','environments/base.py',
                               'environments/robot_env.py','robots/single_arm.py')]
    libero=Path(rt['paths']['libero_root'])/'libero/libero/envs'
    paths += [libero/'env_wrapper.py',libero/'bddl_base_domain.py']
    return {str(p):sha256_file(p) for p in paths}


def model_identity():
    checkpoint=Path(runtime_config()['checkpoints']['libero_spatial'])
    small=sorted(checkpoint.glob('*.json'))+[checkpoint/n for n in ('modeling_prismatic.py','configuration_prismatic.py')]
    small+=sorted(checkpoint.glob('action_head*.pt'))+sorted(checkpoint.glob('proprio_projector*.pt'))
    weights=sorted(checkpoint.glob('*.safetensors'))
    require(weights,'Missing checkpoint weights')
    return {'small_files':{str(p):sha256_file(p) for p in small},
            'weight_stats':{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in weights}}


def freeze_id():
    return digest({'sources':source_hashes(),'model':model_identity(),'libraries':libraries()})


def episode_inputs(states):
    episodes={}
    for state in states:
        folder=Path(state['state_file']).parent.parent
        episode=state['episode_id']
        if episode in episodes: continue
        entries=load_json(folder.parent.parent/'episodes.json')
        row=next(r for r in entries if r['episode_id']==episode)
        require(row['clean_success'] and row['state_count']==5,'Invalid source episode')
        episodes[episode]=row
    return episodes


def parts_selection(states):
    """Selection is independent of RGB, visibility, probes and effects."""
    c=config()['parts']; selected=[]
    for task in range(10):
        for split in ('discovery','validation'):
            rows=[r for r in states if r['task_id']==task and r['split']==split]
            names=sorted({r['episode_id'] for r in rows})[:c['episodes_per_task_split']]
            for name in names:
                r=min((r for r in rows if r['episode_id']==name),
                      key=lambda r:(abs(r['normalized_progress']-c['progress_target']),r['step_id']))
                selected += [{'state_id':r['state_id'],'episode_id':name,'task_id':task,'split':split,'view':v}
                             for v in config()['views']]
    require(len(selected)==80,'Parts selection must contain exactly 80 images')
    return selected


def preflight():
    """File/schema/hash validation only; never imports a VLA or scheduler."""
    import numpy as np
    c=config(); states=load_json(c['state_manifest'])['records']
    require(len(states)==1200 and len({r['state_id'] for r in states})==1200,'State grid changed')
    refs={c['state_manifest']:sha256_file(c['state_manifest']),c['task_map']:sha256_file(c['task_map']),
          c['runtime_config']:sha256_file(c['runtime_config'])}
    for split in ('discovery','validation'):
        rows=[r for r in states if r['split']==split]
        require(len(rows)==600 and len({r['episode_id'] for r in rows})==120,'Split grid changed')
        require(all(sum(r['task_id']==t for r in rows)==60 for t in range(10)),'Unbalanced tasks')
    require(not {r['episode_id'] for r in states if r['split']=='discovery'} &
            {r['episode_id'] for r in states if r['split']=='validation'},'Trajectory leakage')
    for r in states:
        require(sha256_file(r['state_file'])==r['state_file_sha256'],'Source state changed: '+r['state_id'])
        refs[r['state_file']]=r['state_file_sha256']
        require(not r['selection_used_attack_or_attribution_outcome'],'Outcome-informed source selection')
        with np.load(r['state_file'],allow_pickle=False) as z:
            for name,shape in [('front_rgb',(224,224,3)),('wrist_rgb',(224,224,3)),('proprio',(8,)),
                               ('sim_state',(92,)),('clean_action_chunk',(8,7))]:
                value=z[name]
                require(value.shape==shape and np.isfinite(value).all(),f'Invalid source array {r["state_id"]}/{name}')
                if name.endswith('_rgb'): require(value.dtype==np.uint8,'RGB source dtype changed')
            require(int(z['step_id'])==r['step_id'] and str(z['language']).strip(),'State metadata mismatch')
    episodes=episode_inputs(states)
    require(len(episodes)==240,'Episode grid changed')
    for row in episodes.values():
        for key in ('initial_state_file','trajectory_file'):
            p=row[key]; require(sha256_file(p)==row[key+'_sha256'],'Replay input changed: '+p)
            refs[p]=row[key+'_sha256']
        with np.load(row['trajectory_file'],allow_pickle=False) as z:
            require(z['executed_actions'].shape==(row['active_steps'],7),'Invalid executed-action trace')
            require(np.isfinite(z['executed_actions']).all(),'Nonfinite saved actions')
        with np.load(row['initial_state_file'],allow_pickle=False) as z:
            require(z['sim_state'].shape==(92,) and np.isfinite(z['sim_state']).all(),'Invalid initial state')
    for r in states:
        p=str(Path(r['state_file']).parent.parent.parent.parent/'episodes.json')
        if p not in refs: refs[p]=sha256_file(p)
    for row in load_json(c['task_map'])['tasks']:
        require(sha256_file(row['bddl_path'])==row['bddl_sha256'],'BDDL changed')
        refs[row['bddl_path']]=row['bddl_sha256']
    for p in c['old_labels'].values(): refs[p]=sha256_file(p)
    for p,h in load_json(ROOT/'COPIED_SOURCES.json').items():
        require(sha256_file(p)==h,'Copied source changed'); refs[p]=h
    sid=freeze_id(); path=ROOT/'reports/PREFLIGHT.json'
    if path.exists() and ((ROOT/'reports/SUBMISSION.json').exists() or any((ROOT/'data').glob('task_*/COMPLETE.json'))):
        previous=load_json(path)
        require(previous['signature']==sid and previous['input_hashes']==refs,
                'Existing run has different sources/inputs; preserve it in its own directory')
    result={'status':'CPU_PREFLIGHT_PASS','signature':sid,'input_hashes':refs,'sources':source_hashes(),
            'model':model_identity(),'libraries':libraries(),'states':states,'episodes':episodes,
            'parts_selection':parts_selection(states),'resources':c['resources'],
            'gpu_runtime_validation':'PENDING_USER_SUBMISSION','scheduler_submitted':False}
    atomic_json(result,path)
    return result


def prepared(verify_inputs=False):
    r=load_json(ROOT/'reports/PREFLIGHT.json')
    require(r['signature']==freeze_id(),'Source/runtime/model changed after preparation')
    if verify_inputs:
        for p,h in r['input_hashes'].items():
            require(sha256_file(p)==h,'Prepared input changed: '+p)
    return r


def receipt_valid(path,signature=None):
    path=Path(path)
    if not path.exists(): return False
    r=load_json(path)
    require(r['status']=='COMPLETE' and r['signature']==(signature or prepared()['signature']),
            'Completion identity changed: '+str(path))
    require(all(sha256_file(ROOT/p)==h for p,h in r['artifacts'].items()),'Committed artifact changed')
    return True


def commit(path,files,signature,**fields):
    atomic_json({'status':'COMPLETE','signature':signature,**fields,
                 'artifacts':{str(Path(p).relative_to(ROOT)):sha256_file(p) for p in files}},path)


def resources():
    u=resource.getrusage(resource.RUSAGE_SELF)
    out={'process_rss_hwm_bytes':int(u.ru_maxrss*1024),'process_cpu_seconds':u.ru_utime+u.ru_stime}
    import sys
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        out.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return out


def progress(stage,**fields):
    atomic_json({'stage':stage,'unix':time.time(),**fields,**resources()},ROOT/'reports/PROGRESS.json')
