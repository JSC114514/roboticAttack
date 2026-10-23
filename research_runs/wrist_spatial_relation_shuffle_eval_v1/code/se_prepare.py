"""CPU-only attestation; no optimizer migration, model load or simulation."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from se_common import ROOT, PROJECT, BASE, config, signature, read, sha, atomic, require, complete, finish, verify, verify_external


def attest_asset(spec, original):
    parent=PROJECT/spec['root']; folder=parent/'training'/spec['condition']
    saved=read(parent/'data/MANIFEST.json')
    require(saved['signature']==spec['source_signature'], 'Source revision changed')
    require(saved['checkpoint']==original['checkpoint'] and saved['model_file_stats']==original['model_file_stats'], 'Training model differs')
    require(saved['eligible']==217 and saved['target_details']==original['target_details'], 'Training pool/targets differ')
    done=verify_external(parent,folder/'COMPLETE.json')
    terminal=verify_external(parent,folder/'checkpoint_004000.json')
    require(done['candidate']==spec['condition'] and done['budget_complete'], 'Wrong completed candidate')
    for value in (done,terminal):
        require(value['step']==value['optimized']==4000 and value['no_target']==0, 'Final budget not complete')
    require(terminal['method']==spec['method'] and terminal['seed']==spec['seed'] and terminal['terminal'], 'Wrong final checkpoint')
    png=parent/spec['png']; checkpoint=folder/'checkpoint_004000.pt'
    require(png==folder/'checkpoint_004000.png' and sha(png)==spec['png_sha256'], 'Wrong final PNG selected')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for field,expected in dict(signature=spec['source_signature'],method=spec['method'],seed=spec['seed'],step=4000,optimized=4000,no_target=0).items():
        require(payload[field]==expected, 'Checkpoint metadata differs: '+field)
    patch=payload['patch'].detach().cpu()
    require(list(patch.shape)==config()['patch_shape'] and torch.isfinite(patch).all(), 'Invalid serialized patch')
    image=np.asarray(Image.open(png).convert('RGB'))
    require(image.shape==(50,50,3) and np.array_equal(image,patch.clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()), 'PNG and final trained tensor differ')
    paths={parent/'data/MANIFEST.json',parent/'config.json',folder/'COMPLETE.json',folder/'checkpoint_004000.json',checkpoint,png}
    paths.update(parent/p for receipt in (done,terminal) for p in receipt['artifacts'])
    return {**spec,'path':str(png),'checkpoint_path':str(checkpoint),'verified_updates':4000}, paths


def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify()
    for directory in ('data','reports/locks','reports/GATES','reports/PROGRESS','reports/COSTS','results','episodes','logs'):
        (ROOT/directory).mkdir(parents=True,exist_ok=True)
    parent=PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1'
    original=read(parent/'data/MANIFEST.json'); baseline=read(BASE/'data/MANIFEST.json')
    require(original['signature']=='8a7e5d51e904249f7a031b7f8f52ffd83187ad56d70907b6eb0cecaafc058820', 'Original revision changed')
    require(original['model_file_stats']==baseline['model_file_stats'] and original['cases']==baseline['cases'], 'Baseline source differs')
    require(original['eligible']==217 and len(original['target_details'])==original['train_frames']==237, 'Training provenance changed')
    inputs={}
    for path,expected in original['inputs'].items():
        require(sha(path)==expected,'Historical input changed: '+path)
        inputs[path]=expected
    for path in (parent/'data/MANIFEST.json',parent/'data/TARGETS.json',BASE/'data/MANIFEST.json'):
        inputs[str(path)]=sha(path)
    assets=[]
    for spec in config()['assets']:
        asset,paths=attest_asset(spec,original);assets.append(asset)
        inputs.update({str(p):sha(p) for p in paths})
    require([a['condition'] for a in assets]==config()['methods'], 'Four-arm order changed')
    cases=baseline['cases']
    require({(c['task_id'],c['state_index']) for c in cases}=={(t,s) for t in range(10) for s in config()['eval_states']} and len(cases)==40, 'Evaluation grid differs')
    baselines=[]
    for case in cases:
        require(case['suite']=='libero_goal' and sha(case['initial_file'])==original['inputs'][case['initial_file']], 'Initial state source differs')
        reference=BASE/'paired_starts'/case['start']
        require(read(reference/'START.json')['rollout_seed']==case['rollout_seed'], 'Reset seed changed')
        for file in ('START.json','START.npz'):inputs[str(reference/file)]=sha(reference/file)
        for condition in ('clean','initial_s42','initial_s43','uada500'):
            receipt=BASE/'episodes'/case['start']/condition/'COMPLETE.json'
            if not receipt.exists():
                require(condition!='clean','Clean evidence missing');continue
            saved=verify_external(BASE,receipt)
            paths=[BASE/p for p in saved['artifacts'] if p.endswith('/EPISODE.json')]
            traces=[BASE/p for p in saved['artifacts'] if p.endswith('/TRACE.parquet')]
            require(len(paths)==len(traces)==1, 'Ambiguous baseline attempt')
            meta=read(paths[0])
            for field in ('start','task_name','task_id','suite','state_index','rollout_seed'):
                require(meta[field]==case[field], 'Baseline identity differs: '+field)
            require(meta['checkpoint']=='libero_goal' and meta['condition']==condition and meta['strict_paired_reset'] and meta['native_PNG_only'], 'Baseline deployment differs')
            require(meta['max_steps']==300 and meta['num_open_loop_steps']==8, 'Control budget differs')
            png=baseline['conditions'][condition]
            require(meta['png_sha256']==(sha(png) if png else None), 'Baseline PNG differs')
            if condition.startswith('initial_'):
                require(sha(png)==sha(parent/'data'/(condition+'.png')), 'Paired initialization differs')
            if png:inputs[str(png)]=sha(png)
            # Baseline physical reset evidence must also match the common reference.
            starts=[BASE/p for p in saved['artifacts'] if p.endswith('/START.npz') and '/attempt_' in p]
            require(len(starts)==1 and sha(starts[0])==sha(reference/'START.npz'), 'Baseline reset arrays differ')
            for p in (receipt,paths[0],traces[0]):inputs[str(p)]=sha(p)
            baselines.append({'start':case['start'],'condition':condition,'receipt':str(receipt),'episode':str(paths[0]),'trace':str(traces[0])})
        monitor.update(phase='CPU_asset_and_baseline_attestation',completed=len(baselines),total=157)
    require(sum(r['condition']=='clean' for r in baselines)==40, 'Incomplete clean evidence')
    value={'signature':signature(),'assets':assets,'cases':cases,'baselines':baselines,'inputs':inputs,
           'model_file_stats':original['model_file_stats'],'checkpoint':original['checkpoint'],
           'original_train_observations':237,'eligible_train_observations':217,
           'target_source':str(parent/'data/TARGETS.json'),'training_performed':False,'maximum_new_episodes':160,
           'scope':'Goal development evaluation; paired within seed; all forty starts retained'}
    atomic(value,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json'],assets=4,episodes=160,training_performed=False)
    monitor.update(phase='CPU_preparation_complete',completed=4,total=4)
    return verify()
