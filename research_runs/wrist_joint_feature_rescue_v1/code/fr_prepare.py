"""CPU provenance, fixed two PNGs, and attested clean reuse; no VLA/GPU work."""
import copy
import shutil
from pathlib import Path
from fr_common import (ROOT,PROJECT,SCREEN,BASE,SCREEN_SIGNATURE,config,read,sha,require,
    signature,atomic,finish,complete,verify,episode_receipt,source_files)
from fr_math import conditions

def compatible(meta,case,c):
    for key in ('suite','task_id','state_index','rollout_seed','start','task_name'):
        require(meta[key]==case[key],'Historical clean identity mismatch: '+key)
    require(meta['checkpoint']==c['model_dir'] and meta['revision']==c['revision'],'Historical clean model differs')
    require(meta['condition']=='clean' and meta['num_open_loop_steps']==8,'Historical clean policy differs')
    require(meta['max_steps']==c['max_steps'][case['suite']],'Historical horizon differs')

def reuse(case,c):
    if complete(episode_receipt(case,'clean')):return True
    files=source_files(case)
    if files is None:return False
    required=('START.npz','START.json','FIRST_ACTION.npy','INTERFACE_CHECK.json','EPISODE.json',
              'TRACE.parquet','STEPS.parquet','front.mp4','wrist.mp4')
    require(all(n in files for n in required),'Historical clean lacks required assets')
    compatible(read(files['EPISODE.json']),case,c)
    proof=read(files['INTERFACE_CHECK.json'])
    require(proof['status']=='PASS' and proof.get('passive_sampler_native_reset_identical'), 'Historical native reset gate missing')
    target=episode_receipt(case,'clean').parent/'reused_clean'
    paired=ROOT/'paired_starts'/case['start'];target.mkdir(parents=True,exist_ok=True);paired.mkdir(parents=True,exist_ok=True)
    for n in required:shutil.copyfile(files[n],target/n)
    for n in ('START.npz','START.json'):shutil.copyfile(files[n],paired/n)
    atomic(dict(source_root=str(SCREEN),source_signature=SCREEN_SIGNATURE,source_receipt=str(files['SOURCE_COMPLETE.json']),
        source_receipt_sha256=sha(files['SOURCE_COMPLETE.json']),
        source_files={str(files[n]):sha(files[n]) for n in required},
        GPU_reset_and_first_action_gate='required before intervention at each current start'),target/'REUSE.json')
    finish(episode_receipt(case,'clean'),[target/n for n in required]+[target/'REUSE.json']+
        [paired/n for n in ('START.npz','START.json')],reused=True)
    return True

def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'):return verify()
    import torch
    c=config();parent=read(BASE/'data/MANIFEST.json');proof=read(BASE/'reports/PREPARE_COMPLETE.json')
    require(proof['status']=='COMPLETE' and proof['signature']==parent['signature'],'Joint provenance incomplete')
    require(proof['artifacts']['data/MANIFEST.json']==sha(BASE/'data/MANIFEST.json'),'Joint manifest changed')
    require(c['revision']==parent['model']['revision'] and c['model_dir']==parent['checkpoint'],'Joint checkpoint differs')
    require(c['screen_states']==[10] and c['confirmation_states']==[11] and c['chunk']==8,'Fixed split changed')
    inputs={str(BASE/'data/MANIFEST.json'):sha(BASE/'data/MANIFEST.json')}
    inputs.update(parent['model']['small_hashes'])
    inputs[parent['model']['download_receipt']]=sha(parent['model']['download_receipt'])
    cases=[]
    for template in parent['cases']:
        if template['task_id'] not in c['tasks'][template['suite']]:continue
        p=Path(template['initial_file']);inputs[str(p)]=sha(p)
        states=torch.load(p,map_location='cpu',weights_only=False);require(len(states)>10,'Initial state10 unavailable')
        row=copy.deepcopy(template);row.update(state_index=10,start=f"{row['suite']}_t{row['task_id']:02d}_s10",
            rollout_seed=c['seed_base']+10000*row['suite_index']+100*row['task_id']+10)
        cases.append(row)
    rows=conditions(c)
    require(len(cases)==16 and len(rows)==9 and [r['name'] for r in rows]==c['conditions'],'Fixed budget differs')
    from PIL import Image
    for patch in c['patches']:
        p=PROJECT/patch['path'];require(p.is_file() and sha(p)==patch['sha256'],'Fixed PNG changed')
        with Image.open(p) as image:require(image.size==(50,50) and image.mode=='RGB','PNG geometry/mode differs')
        inputs[str(p)]=sha(p)
    for p,s in parent['model_file_stats'].items():
        st=Path(p).stat();require([st.st_size,st.st_mtime_ns]==s,'Model component changed')
    value=dict(signature=signature(),cases=cases,conditions=rows,inputs=inputs,model=parent['model'],
        checkpoint=parent['checkpoint'],model_file_stats=parent['model_file_stats'],total_episodes=144,
        training=False,oracle_feature_donors='same current unpatched observation',base_signature=parent['signature'])
    atomic(value,ROOT/'data/MANIFEST.json');atomic(value,ROOT/'data/ASSETS.json')
    reused=sum(reuse(case,c) for case in cases)
    atomic(dict(reused_clean=reused,clean_total=16,new_episode_budget=144-reused,source_signature=SCREEN_SIGNATURE,
        GPU_pairing_gate_required=True),ROOT/'data/REUSE_PLAN.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',ROOT/'data/ASSETS.json',ROOT/'data/REUSE_PLAN.json'])
    monitor.update(phase='READY',completed=16,total=16,reused_clean=reused,new_episode_budget=144-reused)
    return value
