"""CPU-only provenance and selection; never construct a simulator or policy."""
from pathlib import Path
from va_common import ROOT,FAIR,SOURCE,config,read,sha,atomic,finish,complete,signature,require,adapters
from va_math import all_conditions

def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return
    import numpy as np
    import torch
    from PIL import Image
    source,native=adapters()
    from fc_common import complete as fair_complete
    require(fair_complete(FAIR/'reports/PREPARE_COMPLETE.json'),'Fair source manifest completion invalid')
    old=read(SOURCE/'data/MANIFEST.json');fair=read(FAIR/'data/MANIFEST.json')
    require(fair['signature']==config()['source_signature'],'Fair baseline changed')
    require(fair['checkpoint']==old['checkpoint'] and fair['model_file_stats']==old['model_file_stats'],'Model provenance differs between caches and PNGs')
    require(native.cfg()['patch_xy']==config()['patch_xy'],'Native PNG location differs')
    inputs={str(SOURCE/'data/MANIFEST.json'):sha(SOURCE/'data/MANIFEST.json'),
            str(FAIR/'data/MANIFEST.json'):sha(FAIR/'data/MANIFEST.json'),
            old['trace_path']:sha(old['trace_path'])}
    episodes=[e for e in old['episodes'] if e['suite']=='libero_goal']
    require(len(episodes)==24 and sorted({e['task_id'] for e in episodes})==config()['tasks'],'Goal source task coverage differs')
    all_rows=[];cases=[];counts={'fit':0,'verify':0}
    for e in episodes:
        receipt=source.cache_receipt(e['episode_id']);require(source.complete(receipt),'Source cache invalid')
        p=receipt.with_suffix('.pt');inputs[str(p)]=sha(p);inputs[str(receipt)]=sha(receipt)
        rows=torch.load(p,map_location='cpu',weights_only=False)['records'];require(len(rows)>=2,'Too few cached observations')
        counts[e['cohort']]+=len(rows)
        for r in rows:
            require(r['observation']['full_image'].shape==(224,224,3),'Image shape changed')
            all_rows.append({k:r[k] for k in ('state_id','episode_id','task_id','state_index','cohort','decision_block','frame_rank')})
        if e['cohort']=='verify':
            first=min(rows,key=lambda r:r['decision_block']);initial=old['initial_states'][f"libero_goal/{e['task_id']}"]
            inputs[initial['path']]=sha(initial['path'])
            cases.append(dict(e,cache=str(p),initial_file=initial['path'],task_name=initial['task_name'],
                              first_decision=first['decision_block'],first_state_id=first['state_id']))
        monitor.update(phase='CPU_cache_attestation',completed=len(all_rows),total=118)
    require(counts=={'fit':58,'verify':60},'Cache counts differ; preserve rather than replace records')
    require(len(cases)==12 and len(all_conditions())==24,'Branch budget mismatch')
    patches={}
    for key,name in zip(('relation','uada7'),config()['patch_conditions']):
        p=Path(fair['conditions'][name]);require(p.is_file(),'Missing completed PNG')
        require(Image.open(p).size==(50,50),'Unexpected PNG size');inputs[str(p)]=sha(p);patches[key]=str(p)
        proof=p.with_suffix('.json');require(proof.is_file(),'Missing checkpoint receipt')
        v=read(proof);require(v.get('step')==4000 and v.get('status')=='COMPLETE','PNG is not final 4000 checkpoint')
        require(any(Path(k).name==p.name and h==sha(p) for k,h in v['artifacts'].items()),'PNG hash not sealed')
        inputs[str(proof)]=sha(proof)
    require(not ({e['episode_id'] for e in episodes if e['cohort']=='fit'} & {e['episode_id'] for e in cases}),'Trajectory overlap')
    for d in ('data/references','episodes','results','reports/GATES','reports/PROGRESS','reports/COSTS','reports/locks','logs'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    manifest=dict(signature=signature(),inputs=inputs,model_file_stats=old['model_file_stats'],checkpoint=old['checkpoint'],
                  episodes=episodes,cases=cases,rows=all_rows,patches=patches,conditions=all_conditions(),
                  calibration_counts=counts,planned_branches=288,trace_path=old['trace_path'],
                  source_initial_states=old['initial_states'],claim='Goal pre-contact mechanism development screen; no CASR or new trained patch')
    atomic(manifest,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json'])
