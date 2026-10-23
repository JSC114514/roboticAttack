"""Attest shared conditions and materialize read-only episode links, never weights."""
from pathlib import Path
import json
import random
import shutil
import numpy as np
import torch
from fc_common import ROOT,PROJECT,SOURCE,BASE,CURRENT,source,native,config,signature,module,read,sha,atomic,require,complete,finish,verify,verify_external,candidates,condition_names
from fc_math import sample_schedule


def reuse_episode(parent,receipt,case,condition,inputs):
    saved=verify_external(parent,receipt)
    items=[parent/p for p in saved['artifacts'] if p.endswith('/EPISODE.json')]
    require(len(items)==1,'Ambiguous external result');meta=read(items[0])
    for field in ('start','suite','task_id','state_index','rollout_seed','task_name'):
        require(meta[field]==case[field],'External start differs: '+field)
    require(meta['checkpoint']=='libero_goal' and meta['condition']==condition and meta['strict_paired_reset'],'External model/deployment differs')
    require(meta['max_steps']==300 and meta['num_open_loop_steps']==8,'Native loop changed')
    folder=ROOT/'episodes'/case['start']/condition;files=[]
    wanted={'EPISODE.json','TRACE.parquet','STEPS.parquet','EVENTS.parquet','START.npz','front.mp4','wrist.mp4','video.mp4'}
    for relative,expected in saved['artifacts'].items():
        p=parent/relative;inputs[str(p)]=expected
        if p.name not in wanted or (p.name=='START.npz' and '/attempt_' not in relative):continue
        destination=folder/'reused'/p.name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():require(sha(destination)==expected,'Reused output collision')
        else:destination.symlink_to(p.resolve())
        files.append(destination)
    proof=folder/'REUSE.json';atomic({'source_receipt':str(receipt),'source_signature':saved['signature'],
        'source_sha256':sha(receipt),'reuse_without_new_simulation':True,'parent_read_only':True},proof)
    inputs[str(receipt)]=sha(receipt);finish(folder/'COMPLETE.json',[*files,proof],reused=True)
    return meta


def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return verify()
    for d in ('data','training','episodes','paired_starts','results','reports/locks','reports/GATES','reports/PROGRESS','reports/COSTS','logs'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    parent=module('fc_completed_eval_common',CURRENT/'code/se_common.py')
    require(parent.signature()==config()['parent_eval_signature'] and parent.complete(CURRENT/'results/COMPLETE.json'),'Parent evaluation changed')
    inherited=parent.verify();original=source.verify_inputs();old=read(PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1/data/MANIFEST.json')
    require(original['model_file_stats']==inherited['model_file_stats']==old['model_file_stats'],'Training model differs')
    inputs={**original['inputs'],**inherited['inputs']}
    for p in (CURRENT/'data/MANIFEST.json',CURRENT/'results/COMPLETE.json',SOURCE/'data/MANIFEST.json',BASE/'data/MANIFEST.json'):
        inputs[str(p)]=sha(p)
    rows=list(source.records('fit'));eligible=[r for r in rows if not r['teacher']['no_target']]
    require(len(rows)==237 and len(eligible)==217,'Training pool changed')
    require({r['state_id'] for r in eligible}=={r['state_id'] for r in old['target_details'] if r['eligible']},'Relation and UADA eligibility differs')
    schedules={}
    for seed in config()['seeds']:
        schedules[str(seed)]=sample_schedule(rows,seed)
        asset=next(a for a in inherited['assets'] if a['condition']==f'matched_relation_s{seed}')
        completed=Path(asset['path']).parent/'COMPLETE.json';parent_root=PROJECT/asset['root']
        proof=verify_external(parent_root,completed);verify_external(parent_root,Path(asset['path']).with_suffix('.json'))
        logs=[parent_root/p for p in proof['artifacts'] if '/ATTEMPT_' in p and p.endswith('.jsonl')]
        actual={}
        for log in logs:
            for line in log.read_text().splitlines():
                item=json.loads(line);actual[item['step']]=item['state_id']
        require([actual.get(n+1) for n in range(4000)]==schedules[str(seed)],'Original relation sample sequence differs')
        initial=SOURCE/f'data/initial_s{seed}.png'
        require(sha(initial)==sha(PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1'/f'data/initial_s{seed}.png'),'Initialization differs')
        shutil.copy2(initial,ROOT/f'data/initial_s{seed}.png');inputs[str(initial)]=sha(initial)
    atomic(schedules,ROOT/'data/SAMPLE_SCHEDULE.json')
    conditions={'clean':None,**{f'initial_s{s}':str(SOURCE/f'data/initial_s{s}.png') for s in config()['seeds']}}
    for asset in inherited['assets']:conditions[asset['condition']]=asset['path']
    for seed in config()['seeds']:
        folder=SOURCE/'training'/f'translation_far_boundary_s{seed}'
        require(source.complete(folder/'COMPLETE.json') and source.complete(folder/'checkpoint_004000.json'),'Physical XYZ baseline incomplete')
        require(read(folder/'checkpoint_004000.json')['optimized']==4000,'Physical baseline budget differs')
        for p in (folder/'COMPLETE.json',folder/'checkpoint_004000.json',folder/'checkpoint_004000.png'):inputs[str(p)]=sha(p)
        conditions[f'translation_far_boundary_s{seed}']=str(folder/'checkpoint_004000.png')
    for name,_,_ in candidates():conditions[name]=str(ROOT/'training'/name/'checkpoint_004000.png')
    conditions={name:conditions[name] for name in condition_names()}
    all_cases=[{**c,'cohort':'original40','reference_start':str(BASE/'paired_starts'/c['start'])} for c in inherited['cases']]
    for task in range(10):
        template=next(c for c in inherited['cases'] if c['task_id']==task and c['state_index']==2)
        states=torch.load(template['initial_file'],map_location='cpu',weights_only=False)
        require(len(states)>max(config()['expanded_eval_states']),'Missing expanded initial states')
        for state in config()['expanded_eval_states']:
            start=f'libero_goal_t{task:02d}_s{state:02d}'
            all_cases.append({**template,'state_index':state,'start':start,'rollout_seed':template['rollout_seed']+state-2,
                              'cohort':'expanded40','reference_start':str(ROOT/'paired_starts'/start)})
    require(len(all_cases)==80 and not (set(config()['expanded_eval_states'])&{40,41,42,43}),'Evaluation/train trajectory overlap')
    reused=0
    for case in all_cases:
        if case['cohort']!='original40':continue
        for name,path in conditions.items():
            parent_root=CURRENT if name.startswith(('matched_relation_','shuffled_relation_')) else BASE
            receipt=parent_root/'episodes'/case['start']/name/'COMPLETE.json'
            if not receipt.exists():continue
            meta=reuse_episode(parent_root,receipt,case,name,inputs)
            require(meta['png_sha256']==(sha(path) if path else None),'External PNG differs')
            reused+=1
        monitor.update(phase='CPU_shared_condition_attestation',completed=reused,total=356)
    require(reused==356,'Previously verified reuse count changed')
    value={'signature':signature(),'cases':all_cases,'conditions':conditions,'inputs':inputs,'model_file_stats':original['model_file_stats'],
        'checkpoint':original['checkpoint'],'eligible_state_ids':[r['state_id'] for r in eligible],
        'reused_episodes':reused,'planned_episodes':1040,'new_episodes':1040-reused,'training_observations':217,
        'reference':'clean native Goal actions; no expert labels','sample_schedule_matches_both_original_seeds':True,
        'historical_uada_reference':[r for r in inherited['baselines'] if r['condition']=='uada500']}
    atomic(value,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',ROOT/'data/SAMPLE_SCHEDULE.json',*[ROOT/f'data/initial_s{s}.png' for s in config()['seeds']]])
    monitor.update(phase='CPU_preparation_complete',completed=reused,total=reused)
    return verify()
