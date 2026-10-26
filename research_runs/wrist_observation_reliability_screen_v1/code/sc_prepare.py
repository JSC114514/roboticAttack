"""CPU-only model provenance, predetermined task selection and initial-state checks."""
import copy
from pathlib import Path
from sc_common import ROOT,BASE,config,read,sha,require,signature,atomic,finish,complete,verify


def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify()
    import torch
    parent=read(BASE/'data/MANIFEST.json');proof=read(BASE/'reports/PREPARE_COMPLETE.json')
    require(proof['status']=='COMPLETE' and proof['signature']==parent['signature'],'Parent model provenance not complete')
    require(proof['artifacts']['data/MANIFEST.json']==sha(BASE/'data/MANIFEST.json'),'Parent manifest changed')
    c=config();require(c['revision']==parent['model']['revision'] and c['model_dir']==parent['checkpoint'],'Joint model identity changed')
    inputs={str(BASE/'data/MANIFEST.json'):sha(BASE/'data/MANIFEST.json')}
    inputs.update(parent['model']['small_hashes'])
    inputs[parent['model']['download_receipt']]=sha(parent['model']['download_receipt'])
    cases=[]
    for template in parent['cases']:
        if template['task_id'] not in c['tasks'][template['suite']]: continue
        p=Path(template['initial_file']);inputs[str(p)]=sha(p)
        states=torch.load(p,map_location='cpu',weights_only=False)
        require(len(states)>max(c['screen_states']),'Requested initial state unavailable')
        for state in c['screen_states']:
            row=copy.deepcopy(template);row.update(state_index=state,
                start=f"{row['suite']}_t{row['task_id']:02d}_s{state:02d}",
                rollout_seed=c['seed_base']+10000*row['suite_index']+100*row['task_id']+state)
            cases.append(row)
    conditions=[dict(name=n,path=None,sha256=None) for n in c['conditions']]
    require(len(cases)==32 and len(conditions)==7,'Screen budget changed')
    value=dict(signature=signature(),cases=cases,conditions=conditions,inputs=inputs,
        model=parent['model'],checkpoint=parent['checkpoint'],model_file_stats=parent['model_file_stats'],
        total_episodes=224,training=False,base_signature=parent['signature'])
    for p,s in value['model_file_stats'].items():
        st=Path(p).stat();require([st.st_size,st.st_mtime_ns]==s,'Joint model component changed: '+p)
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    atomic(value,ROOT/'data/MANIFEST.json');atomic(value,ROOT/'data/ASSETS.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',ROOT/'data/ASSETS.json'])
    monitor.update(phase='READY',completed=32,total=32);return value
