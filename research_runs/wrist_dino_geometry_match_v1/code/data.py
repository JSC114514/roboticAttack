"""Read-only audited source shards; no rendering or source-file rewriting."""
from __future__ import annotations
import ast, importlib.util
from functools import lru_cache
from pathlib import Path
from common import (ROOT,PROJECT,cfg,origin_run,runtime_config,runtime_dependency_preflight,
                    read_json,sha,require,signature,atomic_json,finish,complete,initial_patch,save_png)

def index_rows(cohort):
    rows=read_json(origin_run()/'data/INDEX.json')
    if cohort=='train':return [r for r in rows if r['split']=='train']
    if cohort in ('fit','verify'):
        states=cfg()['pilot_fit_states' if cohort=='fit' else 'pilot_verify_states']
        return [r for r in rows if r['split']=='train' and r['state_index'] in states
                and r['frame_rank']==cfg()['pilot_frame_rank']]
    if cohort=='validation':return [r for r in rows if r['split']=='validation']
    raise ValueError(cohort)

def shard_key(row):return row['split'],row['suite'],row['task_id']

@lru_cache(maxsize=40)
def verify_shard(split,suite,task):
    require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Run CPU preflight first')
    relative=f'data/cache/{split}/{suite}/task_{task:02d}.pt'
    path=origin_run()/relative
    entry=read_json(ROOT/'data/SOURCE_MANIFEST.json')['shards'][relative]
    require(sha(path)==entry['sha256'],'Source cache changed: '+relative)
    return path.stat().st_size,path.stat().st_mtime_ns

def load_shard(split,suite,task):
    import torch
    expected=verify_shard(split,suite,task)
    path=origin_run()/f'data/cache/{split}/{suite}/task_{task:02d}.pt'
    require((path.stat().st_size,path.stat().st_mtime_ns)==expected,'Source shard modified during run')
    value=torch.load(path,map_location='cpu',weights_only=False)
    require(value['signature']==read_json(path.with_suffix('.json'))['signature'],'Shard/receipt origin mismatch')
    return value['records']

def iter_records(cohort,suite=None):
    requested=index_rows(cohort)
    if suite is not None:requested=[r for r in requested if r['suite']==suite]
    keys=sorted(set(shard_key(r) for r in requested))
    selected={r['state_id'] for r in requested}
    for split,name,task in keys:
        rows=load_shard(split,name,task)
        kept=[r for r in rows if r['state_id'] in selected]
        require(len(kept)==sum(shard_key(r)==(split,name,task) for r in requested),'Missing source observations')
        yield from kept

def compact_pilot():
    """96 selected records, never the complete multi-layer source bank."""
    return {cohort:list(iter_records(cohort)) for cohort in ('fit','verify')}

def prepare():
    import torch
    from PIL import Image
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return read_json(ROOT/'reports/PREFLIGHT.json')
    origin=origin_run();pref=read_json(origin/'reports/PREFLIGHT.json')
    spec=importlib.util.spec_from_file_location('geometry_source_common',origin/'code/common.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    require(old.signature()==pref['signature'],'Source experiment no longer matches its sealed preparation')
    require(old.complete(origin/'training/translation/COMPLETE.json'),'Old projection baseline incomplete')
    require(old.complete(origin/'results/COMPLETE.json'),'Source experiment incomplete')
    require(pref['source_action_validation']['chunks_checked']==8062,'Source action audit missing')
    require(cfg()['max_concurrent_gpus']==4 and cfg()['patch_shape']==[3,50,50] and cfg()['gpu_type']=='4090',
            'Resource/patch protocol changed')
    require(cfg()['epochs']*cfg()['updates_per_epoch']==48000,'Night training budget changed')
    dependencies=runtime_dependency_preflight()
    selections={name:index_rows(name) for name in ('fit','verify','train','validation')}
    require([len(selections[x]) for x in selections]==[48,48,960,160],'Cohort sizes changed')
    require(not ({r['episode_id'] for r in selections['fit']} & {r['episode_id'] for r in selections['verify']}),'Pilot trajectory leakage')
    shards={}
    for split,suite,task in sorted(set(shard_key(r) for r in selections['train']+selections['validation'])):
        receipt=origin/f'data/cache/{split}/{suite}/task_{task:02d}.json'
        require(old.complete(receipt),'Original cache is not complete')
        value=read_json(receipt);relative=receipt.with_suffix('.pt').relative_to(origin).as_posix()
        shards[relative]={'sha256':value['artifacts'][relative],'origin_signature':value['signature'],
                          'receipt_sha256':sha(receipt)}
    # Parse the public task map without importing MuJoCo on the login node.
    library=Path(runtime_config()['paths']['libero_root'])/'libero/libero'
    map_path=library/'benchmark/libero_suite_task_map.py'
    nodes=ast.parse(map_path.read_text()).body
    task_map=ast.literal_eval(next(n.value for n in nodes if isinstance(n,ast.Assign)
                                  and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets)))
    init_files={}
    for suite in cfg()['suites']:
        for task in cfg()['task_splits'][suite]['validation']+cfg()['task_splits'][suite]['heldout']:
            path=library/'init_files'/suite/(task_map[suite][task]+'.pruned_init')
            states=torch.load(path,map_location='cpu',weights_only=False)
            require(len(states)>48 and all(s<len(states) for s in cfg()['fresh_heldout_states']),'Initial state coverage missing')
            init_files[f'{suite}/{task}']={'path':str(path),'sha256':sha(path),'count':len(states)}
            del states
    for seed in (42,43):save_png(initial_patch(seed=seed),ROOT/f'data/initial_s{seed}.png')
    require(sha(ROOT/'data/initial_s42.png')==sha(origin/'training/translation/epoch_00.png'),'Seed42 initialization differs from old baseline')
    for path in (ROOT/'data/initial_s42.png',ROOT/'data/initial_s43.png',PROJECT/cfg()['uada_png']):
        with Image.open(path) as image:require(image.mode=='RGB' and image.size==(50,50),'Invalid deployment PNG')
    manifest=ROOT/'data/SOURCE_MANIFEST.json'
    atomic_json({'signature':signature(),'origin_signature':pref['signature'],'shards':shards,
                 'init_files':init_files,'task_map_sha256':sha(map_path),'source_files_read_only':True},manifest)
    selection_path=ROOT/'data/SELECTION.json';atomic_json(selections,selection_path)
    report=ROOT/'reports/PREFLIGHT.json'
    atomic_json({'status':'CPU_PREFLIGHT_PASS','signature':signature(),'cohorts':{k:len(v) for k,v in selections.items()},
                 'shards_audited':len(shards),'fresh_states':cfg()['fresh_heldout_states'],
                 'checkpoint_dependencies':dependencies,'source_signature':pref['signature'],
                 'slurm_submitted':False},report)
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[report,manifest,selection_path,ROOT/'data/initial_s42.png',ROOT/'data/initial_s43.png'])
    return read_json(report)
