"""Immutable inputs, atomic artifacts, signatures and independent-stage receipts."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PYTHON='/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python'


def require(ok, message):
    if not bool(ok): raise RuntimeError(message)


def load_json(path): return json.loads(Path(path).read_text())
def config(): return load_json(ROOT/'config.json')
def atlas_root(): return PROJECT/config()['atlas']


def atlas_path():
    value=str(atlas_root()/'code')
    if value not in sys.path: sys.path.append(value)


def runtime_config():
    c=load_json(atlas_root()/'config.json')
    r=load_json(c['runtime_config'])['evaluation']
    return {**r,'env_image_resolution':224,'policy_image_resolution':224}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def atomic_npz(path,**values):
    import numpy as np
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('wb') as f: np.savez_compressed(f,**values)
    os.replace(temporary,path)


def directories():
    for p in ('data','readouts','training','evaluation','results','logs','reports/locks','reports/COSTS',
              'reports/FAILURES','reports/SUBMISSION_ARCHIVE'):
        (ROOT/p).mkdir(parents=True,exist_ok=True)


def source_identity():
    paths=[ROOT/'config.json',ROOT/'README.md',ROOT/'run.sh',ROOT/'PROVENANCE.json']
    for folder in ('code','slurm','tests'):
        paths += sorted(p for p in (ROOT/folder).glob('*') if p.is_file())
    atlas=atlas_root()
    paths += sorted((atlas/'code').glob('*.py'))
    paths += [atlas/'config.json',Path(load_json(atlas/'config.json')['runtime_config'])]
    rt=runtime_config(); native=Path(rt['paths']['openvla_oft_root'])
    paths += [PROJECT/'VLAAttacker/multiview/openvla_oft_wrapper.py',
              PROJECT/'VLAAttacker/multiview/openvla_oft_experiments.py',
              native/'experiments/robot/openvla_utils.py',
              native/'prismatic/extern/hf/modeling_prismatic.py',
              native/'prismatic/extern/hf/processing_prismatic.py']
    import sysconfig
    suite=Path(sysconfig.get_path('purelib'))/'robosuite'
    paths += [suite/p for p in ('utils/observables.py','utils/binding_utils.py','controllers/osc.py',
                               'environments/base.py','environments/robot_env.py','robots/single_arm.py')]
    libero=Path(rt['paths']['libero_root'])/'libero/libero/envs'
    paths += [libero/'env_wrapper.py',libero/'bddl_base_domain.py']
    checkpoint=Path(rt['checkpoints']['libero_spatial'])
    paths += sorted(checkpoint.glob('*.json'))
    paths += [checkpoint/n for n in ('modeling_prismatic.py','configuration_prismatic.py')]
    paths += sorted(checkpoint.glob('action_head*.pt'))+sorted(checkpoint.glob('proprio_projector*.pt'))
    model={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(checkpoint.glob('*.safetensors'))}
    require(model,'Missing model weights')
    versions={}
    for name in ('numpy','scipy','pandas','torch','torchvision','timm','transformers','mujoco','robosuite'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='NOT_INSTALLED'
    return {'sources':{str(p):sha256(p) for p in paths},'model_weight_stats':model,'versions':versions}


def signature(): return digest(source_identity())


def preflight():
    """Small source/cohort check; no policy, renderer, readout training or scheduler."""
    import numpy as np
    from pilot_geometry import cohort, conditions, candidates
    directories(); atlas=atlas_root(); old=load_json(atlas/'reports/PREFLIGHT.json')
    capture=load_json(atlas/'reports/CAPTURE_COMPLETE.json')
    require(capture['status']=='COMPLETE' and capture['signature']==old['signature'],'Atlas capture is incomplete')
    labels_complete=load_json(atlas/'reports/LABELS_COMPLETE.json')
    require(labels_complete['status']=='COMPLETE' and labels_complete['signature']==old['signature'],
            'Atlas aligned labels are incomplete')
    references={}; label_rows=[]
    def reference(path, expected=None):
        p=Path(path); h=sha256(p)
        require(expected is None or h==expected,'Changed upstream artifact: '+str(p))
        references[str(p)]=h
    reference(atlas/'reports/PREFLIGHT.json'); reference(atlas/'reports/CAPTURE_COMPLETE.json')
    reference(atlas/'reports/LABELS_COMPLETE.json')
    atlas_config=load_json(atlas/'config.json')
    reference(atlas_config['task_map'],old['input_hashes'][atlas_config['task_map']])
    for task in range(10):
        for kind in ('labels/',''):
            folder=atlas/f'data/{kind}task_{task:02d}'
            receipt=load_json(folder/'COMPLETE.json')
            require(receipt['status']=='COMPLETE' and receipt['signature']==old['signature'],'Atlas shard changed')
            chain=labels_complete if kind else capture
            reference(folder/'COMPLETE.json',chain['artifacts'][str((folder/'COMPLETE.json').relative_to(atlas))])
            paths=[folder/'RECORDS.json']
            if not kind: paths += [folder/f'global/wrist__b{layer:02d}_spatial.npy' for layer in config()['layers']]
            for path in paths: reference(path,receipt['artifacts'][str(path.relative_to(atlas))])
        label_rows += load_json(atlas/f'data/labels/task_{task:02d}/RECORDS.json')
    chosen=cohort(label_rows,config()['episodes_per_task_split'])
    sources={r['state_id']:r for r in old['states']}
    rows=[]; episodes={}
    for label in chosen:
        row=dict(sources[label['state_id']]); row.update(truth=label['truth'],visibility=label['visibility'])
        row['conditions']=conditions(row['truth'],config()); rows.append(row)
        reference(row['state_file'],row['state_file_sha256'])
        lab=atlas/f'data/labels/task_{row["task_id"]:02d}'
        receipt=load_json(lab/'COMPLETE.json'); p=lab/'states'/(row['state_id']+'.npz')
        reference(p,receipt['artifacts'][str(p.relative_to(atlas))])
        with np.load(row['state_file'],allow_pickle=False) as z:
            require(z['front_rgb'].shape==z['wrist_rgb'].shape==(224,224,3),'Invalid cached RGB')
            require(z['proprio'].shape==(8,) and z['clean_action_chunk'].shape==(8,7),'Invalid source policy shape')
        ep=old['episodes'][row['episode_id']]; episodes[row['episode_id']]=ep
        for key in ('initial_state_file','trajectory_file'): reference(ep[key],ep[key+'_sha256'])
    require(len(rows)==40 and len(episodes)==40,'Cohort must be 40 independent episodes')
    require(all(sum(r['split']==s for r in rows)==20 for s in ('discovery','validation')),'Split count changed')
    tasks=load_json(atlas_config['task_map'])['tasks']
    for task in tasks: reference(task['bddl_path'],task['bddl_sha256'])
    reference(PROJECT/config()['uada_png'])
    for pair in load_json(ROOT/'PROVENANCE.json')['copies']:
        require(sha256(ROOT/pair['destination'])==pair['destination_sha256'],'Copied/adapted code changed')
        reference(PROJECT/pair['source'],pair['source_sha256'])
    identity=source_identity()
    require(identity['model_weight_stats']==old['model']['weight_stats'],
            'Current model weights differ from the model that produced the atlas')
    for p,h in old['model']['small_files'].items(): reference(p,h)
    # Enforce agreement with the actual capture for overlapping native/simulator
    # sources, without rescanning unrelated historical outputs or databases.
    for p,h in identity['sources'].items():
        if p in old['sources']:
            require(h==old['sources'][p],'Atlas/runtime source changed since original capture: '+p)
    for name,version in identity['versions'].items():
        if name in old['libraries']:
            require(version==old['libraries'][name],'Library differs from the atlas capture: '+name)
    sid=digest(identity); path=ROOT/'reports/PREFLIGHT.json'
    if path.exists():
        previous=load_json(path)
        sealed=(ROOT/'reports/SUBMISSION.json').exists() or (ROOT/'reports/PREPARE_COMPLETE.json').exists()
        require(not sealed or (previous['signature']==sid and previous['input_hashes']==references),
                'Sources/inputs differ from existing work; use an independent versioned run directory')
    result={'status':'CPU_PREFLIGHT_PASS','signature':sid,'identity':identity,'input_hashes':references,
            'states':rows,'episodes':episodes,'tasks':tasks,'candidates':candidates(config()),
            'selection_used_action_response':False,'validation_previously_observed':True,
            'gpu_checks':'PENDING_MANUAL_SLURM_SUBMISSION','scheduler_submitted':False}
    atomic_json(result,path); return result


def prepared(verify_inputs=False):
    prep=load_json(ROOT/'reports/PREFLIGHT.json')
    require(prep['signature']==signature(),'Sources/runtime/model changed after preparation')
    if verify_inputs:
        for p,h in prep['input_hashes'].items(): require(sha256(p)==h,'Input changed: '+p)
    return prep


def complete(path,sid):
    path=Path(path)
    if not path.exists(): return False
    receipt=load_json(path)
    require(receipt['status']=='COMPLETE' and receipt['signature']==sid,'Artifact identity mismatch: '+str(path))
    require(all(sha256(ROOT/p)==h for p,h in receipt['artifacts'].items()),'Committed artifact changed: '+str(path))
    for p in receipt['artifacts']:
        child=ROOT/p
        if child.name=='COMPLETE.json' or child.name.endswith('_COMPLETE.json'):
            require(complete(child,sid),'Missing nested artifact receipt: '+str(child))
    return True


def commit(path,files,sid,**fields):
    atomic_json({'status':'COMPLETE','signature':sid,**fields,
                 'artifacts':{str(Path(p).relative_to(ROOT)):sha256(p) for p in files}},path)


def resources():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result={'process_rss_hwm_bytes':int(usage.ru_maxrss*1024),
            'process_cpu_s':usage.ru_utime+usage.ru_stime}
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def progress(stage,**fields):
    result={'stage':stage,'unix':time.time(),**fields,**resources()}
    atomic_json(result,ROOT/'reports/PROGRESS.json')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


def observation(row,wrist=None):
    import numpy as np
    with np.load(row['state_file'],allow_pickle=False) as z:
        obs={'full_image':z['front_rgb'].copy(),'wrist_image':z['wrist_rgb'].copy() if wrist is None else wrist.copy(),
             'state':z['proprio'].copy()}
        language=str(z['language'])
    return obs,language
