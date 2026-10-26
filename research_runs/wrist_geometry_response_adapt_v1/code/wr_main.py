"""Stage entrypoints. GPU stages only execute in user-submitted allocations."""
import argparse
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from wr_common import (ROOT, PROJECT, config, require, read, atomic, sha, signature,
                       resources, Monitor, lock, task_map, cases,complete)


def check():
    c=config();model=Path(c['model_dir']);names=task_map()
    for key in ('config.json','model.safetensors.index.json','dataset_statistics.json',
                'action_head--300000_checkpoint.pt','proprio_projector--300000_checkpoint.pt',
                'modeling_prismatic.py','configuration_prismatic.py'):
        require((model/key).is_file(),'Missing model component '+key)
    index=read(model/'model.safetensors.index.json')
    for name in set(index['weight_map'].values()):require((model/name).is_file(),'Missing model weight shard '+name)
    stats=read(model/'dataset_statistics.json')
    for s in c['suites']:
        require(s+'_no_noops' in stats,'Missing joint normalization for '+s)
        data=Path(c['expert_root'])/(s+'_no_noops')/'1.0.0'
        require((data/'dataset_info.json').is_file(),'Missing expert RLDS metadata '+s)
        require(any(data.glob('*.tfrecord*')),'Missing expert RLDS shards '+s)
        split=c['task_splits'][s]
        sets=[set(v) for v in split.values()]
        require(set.union(*sets)==set(range(10)) and sum(map(len,sets))==10,'Task split overlap')
    for rel in c['dependency_files']+[c['camera_index'],c['source_traces']]:
        require((PROJECT/rel).is_file(),'Missing source '+rel)
    for module in ('torch','numpy','pandas','PIL','tensorflow','tensorflow_datasets'):
        require(importlib.util.find_spec(module) is not None,'Required installed dependency missing: '+module)
    from wr_data import camera_members
    counts={stage:{k:len(v) for k,v in camera_members(stage).items()} for stage in ('pilot','main')}
    for stage in ('pilot','main'):
        for case in cases(stage):require(Path(case['initial_file']).is_file(),'Initial file unavailable '+case['initial_file'])
    require(c['adapter']['blocks']==[18,19,20,21,22] and c['adapter']['rank']==8,'Adapter specification changed')
    require(len(c['pilot_candidates'])==8,'Pilot comparison changed')
    result=dict(status='CPU_PREFLIGHT_PASS',model=str(model),camera_counts=counts,
        pilot_closedloop_episodes=32*5*9,main_closedloop_episodes=120*5*13,
        GPU_tested=False,signature=signature(),**resources())
    result['installed_versions']={}
    for name in ('torch','transformers','timm','tensorflow','tensorflow-datasets','mujoco','robosuite','imageio-ffmpeg'):
        try:result['installed_versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['installed_versions'][name]='not_in_distribution_metadata'
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


def verify_sources(stage):
    """Prepared labels and base model must retain their recorded provenance."""
    require(complete(ROOT/f'reports/EXPERT_{stage}_COMPLETE.json'),'Expert data preparation incomplete')
    manifest=read(ROOT/f'data/{stage}/MANIFEST.json')
    require(manifest['signature']==signature(),'Prepared data revision differs')
    stats=Path(config()['model_dir'])/'dataset_statistics.json'
    require(sha(stats)==manifest['stats_sha256'],'Normalization statistics changed')
    for path,expected in manifest['inputs'].items():
        s=Path(path).stat();require([s.st_size,s.st_mtime_ns]==expected,'Expert input changed: '+path)
    identity=ROOT/'reports/ASSETS.json'
    if identity.is_file():
        pinned=read(identity)
        require(pinned['signature']==signature(),'Asset receipt revision differs')
        for path,expected in pinned['model_file_stats'].items():
            s=Path(path).stat();require([s.st_size,s.st_mtime_ns]==expected,'Base checkpoint changed: '+path)
        for path,h in pinned['input_hashes'].items():require(sha(path)==h,'Replay/model input changed: '+path)
    return manifest


def pin_assets():
    c=config();m=Path(c['model_dir']);stats={}
    for p in sorted(m.iterdir()):
        if p.is_file() and p.suffix in ('.safetensors','.bin','.pt','.json','.py'):
            st=p.stat();stats[str(p)]=[st.st_size,st.st_mtime_ns]
    paths=[m/p for p in ('config.json','dataset_statistics.json','model.safetensors.index.json')]
    paths += [PROJECT/c['camera_index'],PROJECT/c['source_traces']]
    paths += sorted({Path(x['initial_file']) for stage in ('pilot','main') for x in cases(stage)})
    value=dict(signature=signature(),model_file_stats=stats,input_hashes={str(p):sha(p) for p in paths})
    prior=ROOT/'reports/ASSETS.json'
    if prior.is_file():require(read(prior)==value,'Pinned source/model identity changed')
    else:atomic(value,prior)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','prepare-expert','prepare-camera','calibrate','train','evaluate','report'])
    p.add_argument('--stage',choices=['pilot','main'],default='pilot')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--candidate');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='check':check();return
    require(0<=a.shard<a.shards<= (2 if a.stage=='pilot' else 4),'Worker sharding exceeds budget')
    if a.command in ('train','evaluate'):require(a.candidate is not None,'Candidate is required')
    name=f'{a.command}_{a.stage}_{a.candidate or "shared"}_{a.shards}_{a.shard}'
    with lock(name),Monitor(name) as monitor:
        if a.command=='prepare-expert':
            check();pin_assets()
            from wr_data import prepare
            prepare(a.stage,monitor)
        elif a.command=='prepare-camera':
            verify_sources(a.stage)
            from wr_data import prepare_camera
            prepare_camera(a.stage,a.shard,a.shards,monitor)
        elif a.command=='calibrate':
            verify_sources(a.stage)
            from wr_train import calibration
            calibration(a.stage,monitor)
        elif a.command=='train':
            verify_sources(a.stage)
            from wr_train import train
            train(a.stage,a.candidate,monitor)
        elif a.command=='evaluate':
            verify_sources(a.stage)
            from wr_evaluate import evaluate
            evaluate(a.stage,a.candidate,a.shard,a.shards,monitor)
        else:
            from wr_report import run
            run(a.stage,monitor)
            from wr_video import export
            export(a.stage,monitor)


if __name__=='__main__':main()
