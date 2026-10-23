"""Only compute clean/CF action teachers; never re-render or regenerate trajectories."""
import torch
import time
from at_common import (ROOT,config,cutoff,source_shards,teacher_receipt,complete,finish,signature,
    make_engine,require,atomic_torch,read_json)

def run(index,monitor):
    from at_common import native
    native.require_gpu('action_teacher_cache')
    entries=source_shards()[index::2]
    missing=[e for e in entries if not complete(teacher_receipt(e['episode_id']))]
    if not missing: return
    if time.time()>=cutoff():
        monitor.update(phase='cutoff_before_cache_model_load',completed=0,total=len(entries)*5)
        return
    engine=make_engine()
    for n,e in enumerate(entries):
        receipt=teacher_receipt(e['episode_id'])
        if complete(receipt): continue
        payload=torch.load(e['path'],map_location='cpu',weights_only=False)
        require(payload['signature']==config()['source_signature'],'Source payload changed')
        rows=[]
        for row in payload['records']:
            f=row['target_features']
            energy=float((f['translation_plus'].float()-f['translation_minus'].float()).square().mean(-1).sum())
            invalid=bool(row['geometry']['no_target']) or energy<=1e-12
            engine.verify_patch_path(row)
            with torch.no_grad():
                clean,clean_caps=engine.action(row['observation'],row['language'])
                target,changed=engine.action(row['observation'],row['language'],replacements={
                    'dino':row['target_features']['translation_plus']})
            require(torch.equal(clean_caps['siglip'],changed['siglip']),'Teacher altered SigLIP')
            rows.append({'state_id':row['state_id'],'clean':clean.cpu(),'target':target.cpu(),
                'no_target':invalid,'renderer_no_target':bool(row['geometry']['no_target']),
                'geometry_direction_energy':energy,
                'no_target_reason':'renderer_no_target' if row['geometry']['no_target'] else 'zero_feature_direction' if invalid else None,
                'teacher_translation_delta_l2':float((target[...,:3]-clean[...,:3]).norm(dim=-1).mean())})
            monitor.update(phase='action_teacher',completed=n*5+len(rows),total=len(entries)*5,
                state_id=row['state_id'],cohort=row['cohort'])
        path=receipt.with_suffix('.pt')
        atomic_torch({'signature':signature(),'episode_id':e['episode_id'],'records':rows},path)
        finish(receipt,[path],cohort=e['cohort'],frames=5)
    monitor.update(phase='teachers_complete',completed=len(entries)*5,total=len(entries)*5)
