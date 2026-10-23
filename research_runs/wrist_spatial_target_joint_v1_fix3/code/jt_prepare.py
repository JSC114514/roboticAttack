"""CPU-only preparation, last visible pre-contact sample, no response selection."""
import shutil
import torch
from jt_common import ROOT,PROJECT,BASE,SOURCE,source,config,signature,read,sha,atomic,atomic_torch,require,complete,finish,verify,verify_external,module

def reusable_visuals(inputs,rows,original):
    """Reuse immutable image/features only; all five physics branches are recomputed."""
    parent=PROJECT/config()['repair_parent_run'];expected=config()['repair_parent_signature']
    adapter=module('joint_repair_parent_provenance',parent/'code/jt_common.py')
    require(adapter.signature()==expected,'Failed parent code changed')
    old=read(parent/'data/MANIFEST.json')
    strip_history=lambda xs:[{k:v for k,v in x.items() if k!='source_reset_history'} for x in xs]
    require(old['signature']==expected and strip_history(old['selected'])==strip_history(rows),'Parent selection differs')
    require(old['model_file_stats']==original['model_file_stats'],'Parent checkpoint differs')
    scientific=lambda cfg:{k:v for k,v in cfg.items() if not k.startswith('repair_parent_')}
    require(scientific(config())==scientific(read(parent/'config.json')),'Scientific configuration changed during repair')
    for path in (parent/'data/MANIFEST.json',parent/'config.json'):
        inputs[str(path)]=sha(path)
    result={};costs=[]
    import torch
    for entry in rows:
        receipt=parent/'data/cache'/(entry['state_id']+'.json')
        if not receipt.exists():continue
        saved=verify_external(parent,receipt);path=receipt.with_suffix('.pt')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if not payload['valid']:continue
        require(len(payload['outcomes'])==5 and all('steps' in o for o in payload['outcomes']),'Incomplete parent cache')
        inputs[str(receipt)]=sha(receipt);inputs[str(path)]=sha(path)
        result[entry['state_id']]={'path':str(path),'sha256':sha(path),'receipt':str(receipt),
                                  'parent_signature':expected,'reuse_scope':'visual_features_and_teacher_actions_only_not_physics'}
    for path in sorted((parent/'reports/COSTS').glob('*.json')):
        inputs[str(path)]=sha(path);costs.append({'source':str(path),'sha256':sha(path),**read(path)})
    return result,costs

def reuse_complete_caches(visuals):
    """Attest, do not recompute, prior caches with exact source and paired physics."""
    from jt_common import cache_receipt
    reused=[]
    for state_id,info in visuals.items():
        require(sha(info['path'])==info['sha256'],'Parent cache changed during migration')
        parent=torch.load(info['path'],map_location='cpu',weights_only=False)
        require(parent['all_physics_recomputed_with_fresh_environments'],
                'Parent physics was not independently initialized')
        require(parent['capture']['fingerprint']==parent['row']['sample_fingerprint'],
                'Parent did not reproduce source sample')
        require(all(o['exact_start_fingerprint']==parent['paired_decision_fingerprint']
                    and o['environment_isolation']=='fresh_environment_for_each_condition'
                    for o in parent['outcomes']),'Parent paired starts differ')
        receipt=cache_receipt(state_id);path=receipt.with_suffix('.pt')
        if path.exists():require(sha(path)==info['sha256'],'Existing migrated cache differs')
        else:path.symlink_to(info['path'])
        audit=receipt.with_name(receipt.stem+'_REUSE.json')
        atomic({'state_id':state_id,'source':info,'reuse_scope':'complete_validated_visuals_and_fresh_physics',
                'parent_bytes_unchanged':True,'new_GPU_work':False},audit)
        finish(receipt,[path,audit],valid=True,cohort=parent['row']['cohort'],reused_from=info['receipt'])
        reused.append(state_id)
    return reused

def run(monitor):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json'): return verify()
    for name in ('data/cache','training','reports/locks','reports/PROGRESS','reports/COSTS','reports/GATES','logs','results','episodes','paired_starts'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    from jt_history import source_histories
    original=source.verify_inputs();rows=[];missing=[]
    histories=source_histories(original['episodes'])
    inputs=dict(original['inputs'])
    inputs[str(SOURCE/'data/MANIFEST.json')]=sha(SOURCE/'data/MANIFEST.json')
    for episode in original['episodes']:
        receipt=source.cache_receipt(episode['episode_id'])
        require(source.complete(receipt),'Historical source cache invalid')
        path=receipt.with_suffix('.pt');inputs[str(path)]=sha(path);inputs[str(receipt)]=sha(receipt)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        candidates=[r for r in payload['records'] if r['visible_pixels']>0 and not r['geometry']['contact'] and not r['geometry']['grasp']]
        if not candidates:
            missing.append({**episode,'reason':'no_visible_precontact_native_observation'});continue
        row=max(candidates,key=lambda r:r['decision_block'])
        rows.append({**episode,'state_id':row['state_id'],'decision_block':row['decision_block'],
                     'source_shard':str(path),'source_sha256':sha(path),
                     'source_reset_history':histories[episode['episode_id']]})
    fit=[r for r in rows if r['cohort']=='fit'];validation=[r for r in rows if r['cohort']=='verify']
    require([len(fit),len(validation)]==[47,47],'Previously verified 47/47 source selection changed')
    require(not ({r['episode_id'] for r in fit}&{r['episode_id'] for r in validation}),'Trajectory leakage')
    for cohort in (fit,validation):require(len({(r['suite'],r['task_id']) for r in cohort})==24,'Mixed task coverage changed')
    initial=SOURCE/'data/initial_s42.png';inputs[str(initial)]=sha(initial);shutil.copy2(initial,ROOT/'data/initial_s42.png')
    base_manifest=read(BASE/'data/MANIFEST.json');cases=base_manifest['cases']
    require(base_manifest['model_file_stats']==original['model_file_stats'],'Reused baseline checkpoint differs from training checkpoint')
    inputs[str(BASE/'data/MANIFEST.json')]=sha(BASE/'data/MANIFEST.json')
    require(len(cases)==40 and all(c['suite']=='libero_goal' and c['state_index'] in config()['eval_states'] for c in cases),'Evaluation grid changed')
    baselines=[]
    for case in cases:
        require(sha(case['initial_file'])==original['inputs'][case['initial_file']],'Baseline initial file differs')
        for name in ('START.json','START.npz'):
            path=BASE/'paired_starts'/case['start']/name
            require(path.is_file(),'Missing baseline reset evidence')
            inputs[str(path)]=sha(path)
        require(read(BASE/'paired_starts'/case['start']/'START.json')['rollout_seed']==case['rollout_seed'],'Baseline reset seed differs')
        for condition in ('clean','initial_s42','uada500'):
            receipt=BASE/'episodes'/case['start']/condition/'COMPLETE.json'
            if not receipt.exists():continue
            value=verify_external(BASE,receipt)
            paths=[BASE/p for p in value['artifacts'] if p.endswith('/EPISODE.json')]
            require(len(paths)==1,'Ambiguous baseline episode')
            meta=read(paths[0]);require(meta['checkpoint']=='libero_goal' and meta['rollout_seed']==case['rollout_seed'],'Baseline model/seed mismatch')
            if condition=='initial_s42':require(meta['png_sha256']==sha(initial),'Initial PNG mismatch')
            if condition=='uada500':require(meta['png_sha256']==sha(original['uada_png']),'UADA PNG mismatch')
            inputs[str(receipt)]=sha(receipt);inputs[str(paths[0])]=sha(paths[0])
            baselines.append({'start':case['start'],'condition':condition,'receipt':str(receipt),'episode':str(paths[0])})
    require(sum(r['condition']=='clean' for r in baselines)==40,'Missing clean evidence')
    visual_cache,parent_costs=reusable_visuals(inputs,rows,original)
    reused=reuse_complete_caches(visual_cache)
    value={'signature':signature(),'selected':rows,'missing':missing,'cases':cases,'baselines':baselines,
           'reusable_visual_cache':{},'reused_complete_caches':reused,'parent_costs':parent_costs,
           'repair':'reconstruct historical reset/sampler sequence; reuse exact validated caches; no scientific change',
           'inputs':inputs,'model_file_stats':original['model_file_stats'],'checkpoint':original['checkpoint'],
           'initial_states':original['initial_states'],'trace_path':original['trace_path'],
           'data_source':'four_suites_24_tasks_one_native_observation_per_trajectory',
           'evaluation_scope':'Goal development starts 2-5; all 40 patched starts, including clean failures',
           'maximum_new_episodes':160,'selection_uses_validation':False}
    atomic(value,ROOT/'data/MANIFEST.json')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',ROOT/'data/initial_s42.png'],fit=47,verify=47,missing=missing)
    monitor.update(completed=94,total=94,phase='CPU_preparation_complete')

def finalize(monitor):
    """Compute shared token weights and scale using FIT ONLY."""
    import numpy as np
    from jt_common import selected,cached,cache_receipt
    from jt_math import hierarchy_weights,feature_loss
    if complete(ROOT/'reports/TARGETS_COMPLETE.json'):return
    entries=selected();payloads=[(r,cached(r)) for r in entries]
    for row,payload in payloads:
        if payload['valid']:
            energy=(payload['goals'].double()-payload['clean_feature'].double()[None]).square().mean((1,2))
            if bool((energy<=config()['feature_epsilon']).any()):
                payload['valid']=False;payload['invalid_reasons']=['at_least_one_zero_geometry_feature_change']
    keep={cohort:[r['state_id'] for r,p in payloads if r['cohort']==cohort and p['valid']] for cohort in ('fit','verify')}
    fit=[(r,p) for r,p in payloads if r['cohort']=='fit' and p['valid']]
    for cohort in keep:
        require({r['suite'] for r,p in payloads if r['state_id'] in keep[cohort]}==set(config()['suites']),'Counterfactual validity lost a suite: '+cohort)
    weights=hierarchy_weights([r for r,p in fit]);energy=torch.zeros(256,dtype=torch.float64)
    for weight,(entry,payload) in zip(weights,fit):
        diff=payload['goals'].double()-payload['clean_feature'].double()[None]
        energy+=weight*diff.square().mean((0,2))
    require(float(energy.sum())>config()['feature_epsilon'],'Zero geometry feature energy')
    token_weights=(energy/energy.sum()).float();scale=float((token_weights.double()*energy).sum())
    require(scale>config()['feature_epsilon'],'Degenerate common feature scale')
    mean_harm=(np.stack([p['harm'] for r,p in fit])*weights[:,None]).sum(0)
    valid=bool(mean_harm.max()>0)
    tensor=ROOT/'data/FEATURE_STATS.pt';atomic_torch({'token_weights':token_weights,'scale':scale},tensor)
    value={'signature':signature(),'eligible':keep,'invalid':[{'state_id':r['state_id'],'cohort':r['cohort'],'reasons':p['invalid_reasons']} for r,p in payloads if not p['valid']],
           'mean_fit_harm':mean_harm.tolist(),'fixed_rule':int(np.argmax(mean_harm)),
           'proceed':valid,'status':'READY' if valid else 'STOPPED_NO_APPROACH_OBSTRUCTION',
           'scale':scale,'fit_only_statistics':True,'fit_frames':len(fit),'verify_frames':len(keep['verify']),
           'coverage':{cohort:{suite:sorted({r['task_id'] for r,p in payloads if r['state_id'] in keep[cohort] and r['suite']==suite})
                             for suite in config()['suites']} for cohort in keep}}
    atomic(value,ROOT/'data/TARGET_STATS.json')
    finish(ROOT/'reports/TARGETS_COMPLETE.json',[tensor,ROOT/'data/TARGET_STATS.json',*[cache_receipt(r['state_id']) for r in entries]],proceed=valid)
    monitor.update(phase=value['status'],completed=len(entries),total=len(entries))
