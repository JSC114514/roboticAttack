"""Shared frozen reads, clean references, and two bounded GPU screening workers."""
import time
import numpy as np
from va_common import ROOT,SOURCE,config,read,atomic,save_torch,finish,complete,require,verify,cases,adapters,branch_receipt,reference_path,branch_files
from va_math import nonattack_conditions,patch_conditions

def shared(monitor):
    if complete(ROOT/'reports/SHARED_COMPLETE.json'):return
    import torch
    from va_vision import Vision,pooled
    from va_rollout import run_episode,rng_state
    value=verify();vision=Vision();meta=[];features=[];proprios=[];sums={};count=0
    calibration=ROOT/'data/FEATURES_COMPLETE.json'
    if not complete(calibration):
        for e in value['episodes']:
            rows=torch.load(SOURCE/'data/cache'/(e['episode_id']+'.pt'),map_location='cpu',weights_only=False)['records']
            for row in rows:
                if not features:atomic(vision.check(row),ROOT/'reports/GATES/vision.json')
                captured=vision.features(row['observation'],row['language'])
                features.append(pooled(captured));proprios.append(np.array(row['observation']['state'],copy=True))
                meta.append({k:row[k] for k in ('state_id','episode_id','task_id','state_index','cohort','decision_block')})
                if row['cohort']=='fit':
                    count+=1
                    for k,v in captured.items():
                        if k.startswith('b22_'):continue
                        if k not in sums:sums[k]=np.zeros_like(v,dtype=np.float64)
                        sums[k]+=v
                monitor.update(phase='frozen_features',completed=len(meta),total=118)
        require(count==58 and len(meta)==118,'Feature split/count mismatch')
        np.savez_compressed(ROOT/'data/FEATURES.npz',features=np.asarray(features,dtype=np.float32),proprio=np.asarray(proprios,dtype=np.float32))
        np.savez_compressed(ROOT/'data/MEANS.npz',**{k:(v/count).astype(np.float32) for k,v in sums.items()})
        atomic(meta,ROOT/'data/FEATURE_ROWS.json')
        finish(calibration,[ROOT/'data/FEATURES.npz',ROOT/'data/MEANS.npz',ROOT/'data/FEATURE_ROWS.json',ROOT/'reports/GATES/vision.json'])
    refs=[];completed=sum(complete(branch_receipt(c,'clean')) for c in cases());started=time.monotonic();initial_completed=completed
    monitor.update(phase='clean_references',completed=completed,resume_completed=completed,total=12)
    for case in cases():
        receipt=branch_receipt(case,'clean')
        if not complete(receipt):
            detail=run_episode(case,'clean',vision,None,monitor,rng_state())
            completed+=1;monitor.update(phase='clean_references',completed=completed)
            if completed-initial_completed==4:
                atomic(dict(stage='shared',newly_completed=4,elapsed_s=time.monotonic()-started,
                            estimated_remaining_s=(12-completed)*(time.monotonic()-started)/4,
                            excludes_feature_extraction=True),ROOT/'reports/FIRST_FOUR_TIMING_shared.json')
        refs.append(receipt)
    finish(ROOT/'reports/SHARED_COMPLETE.json',[calibration,*refs])

def screen(route,monitor):
    import torch
    from va_vision import Vision
    from va_rollout import run_episode
    require(complete(ROOT/'reports/CALIBRATION_COMPLETE.json'),'Diagnostic calibration not complete')
    require(complete(ROOT/'data/FEATURES_COMPLETE.json'),'Frozen feature/mean cache changed')
    verify();names=nonattack_conditions() if route=='nonattack' else patch_conditions()
    names=[n for n in names if n!='clean'];receipts=[branch_receipt(c,n) for c in cases() for n in names]
    already=sum(complete(p) for p in receipts);monitor.update(completed=already,resume_completed=already,total=len(receipts),phase=route)
    if already==len(receipts):
        finish(ROOT/f'reports/{route.upper()}_COMPLETE.json',receipts);return
    vision=Vision();means=dict(np.load(ROOT/'data/MEANS.npz',allow_pickle=False))
    started=time.monotonic();done=already;new_starts=0
    for case in cases():
        needed=[n for n in names if not complete(branch_receipt(case,n))]
        if not needed:continue
        reference=torch.load(reference_path(case),map_location='cpu',weights_only=False)
        for name in needed:
            run_episode(case,name,vision,means,monitor,reference['constructor_rng'],reference)
            done+=1;monitor.update(completed=done,phase=route)
        new_starts+=1
        if new_starts==4:
            elapsed=time.monotonic()-started
            atomic(dict(stage=route,completed_new_branches=done-already,elapsed_s=elapsed,
                        estimated_remaining_s=(len(receipts)-done)*elapsed/max(1,done-already)),ROOT/f'reports/FIRST_FOUR_TIMING_{route}.json')
    finish(ROOT/f'reports/{route.upper()}_COMPLETE.json',receipts)
