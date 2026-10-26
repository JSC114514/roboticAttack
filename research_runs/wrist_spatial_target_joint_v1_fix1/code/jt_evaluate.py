"""All 40 starts, native PNG, audited independent camera rollout functions."""
import ast
import copy
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from jt_common import ROOT,PROJECT,BASE,config,source,native,read,sha,atomic,require,complete,finish,verify,eligible,cached,make_engine,verify_external
from jt_math import feature_loss,realization

def conditions():
    chosen=read(ROOT/'data/SELECTED.json')['candidates'];paths={}
    for method,value in chosen.items():
        require(sha(value['path'])==value['png_sha256'],'Final PNG changed');paths[method]=Path(value['path'])
    return paths

def rollout_functions():
    # Reuse only these audited definitions, without importing the old run's CLI/config or writer root.
    from audit_replay import SensorCapture,fingerprint
    from audit_math import visibility,event_summary
    from rr_logger import PhysicsLogger
    from sr_geometry import anchor,truth,controller
    path=PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1_fix1/code/rfix_evaluate.py'
    tree=ast.parse(path.read_text());names={'CameraWriters','paired_start','episode'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==names,'Audited rollout adapter shape changed')
    space=dict(globals(),SensorCapture=SensorCapture,fingerprint=fingerprint,visibility=visibility,
               event_summary=event_summary,PhysicsLogger=PhysicsLogger,anchor=anchor,truth=truth,controller=controller)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
    return space['episode']

def offline(index,engine,paths,monitor):
    from audit_math import visibility
    shared=torch.load(ROOT/'data/FEATURE_STATS.pt',map_location='cpu',weights_only=False)
    chosen=read(ROOT/'data/SELECTED.json')['candidates'];rows=eligible('verify')[index::2];receipts=[]
    for n,entry in enumerate(rows):
        receipt=ROOT/'evaluation/offline'/(entry['state_id']+'.json')
        if complete(receipt):receipts.append(receipt);continue
        payload=cached(entry);row=payload['row'];results=[]
        with torch.no_grad():
            for name,path in paths.items():
                patch=native.load_png(path,engine.wrapper.device)
                data=patch.mul(255).round().byte().permute(1,2,0).cpu().numpy()
                deployed=native.paste_numpy(row['observation'],data)
                action,caps=engine.forward(deployed,row['language'])
                normalized,_=engine.action(deployed,row['language'])
                require(np.max(abs(engine.physical(normalized)-action))<=1e-5,'Native PNG action paths disagree')
                values,_=engine.spatial_patch(row['observation'],patch)
                require(np.array_equal(values.float().cpu().numpy(),caps['dino']),'Final PNG feature path differs')
                losses=[];base=[]
                for k in range(4):
                    goal=payload['goals'][k].float()
                    losses.append(float(feature_loss(torch.from_numpy(caps['dino'][0]),goal,shared['token_weights'],shared['scale'])))
                    base.append(float(feature_loss(payload['clean_feature'],goal,shared['token_weights'],shared['scale'])))
                goal_command=engine.command(normalized,clipped=True).cpu().numpy()[0]
                clean_action=payload['native_clean_actions'];delta=action-clean_action
                results.append({'state_id':entry['state_id'],'episode_id':entry['episode_id'],'suite':entry['suite'],'task_id':entry['task_id'],
                    'condition':name,'native_actions':action.tolist(),'controller_goal_delta_m':goal_command.tolist(),
                    'clean_native_actions':clean_action.tolist(),'teacher_native_actions':payload['teacher_actions'].tolist(),
                    'feature_losses':losses,'realization':realization(losses,base).tolist(),'fit_selected_rule':chosen[name]['chosen_rule'],
                    'next_action_translation_delta':delta[0,:3].tolist(),'all_slot_translation_delta':delta[:,:3].tolist(),
                    **visibility(payload['capture']['clean_mask'],True,config()['patch_xy'])})
        path=receipt.with_suffix('.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        import json
        path.write_text(''.join(json.dumps(r)+'\n' for r in results))
        finish(receipt,[path],conditions=list(paths));receipts.append(receipt)
        monitor.update(phase='offline_full_native_PNG',completed=n+1,total=len(rows))
    return receipts

def run(index,monitor):
    from jt_cache import constructor_rng,restore_constructor_rng
    receipt=ROOT/f'reports/EVALUATE_{index}_COMPLETE.json'
    if complete(receipt):return
    if not read(ROOT/'data/TARGET_STATS.json')['proceed']:return
    require(complete(ROOT/'reports/SELECT_COMPLETE.json'),'Four final PNGs unavailable')
    native.require_gpu('joint_target_evaluate');value=verify();runtime=native.runtime_module();runtime.prepare_graphics_backend()
    engine=make_engine();paths=conditions();episode=rollout_functions();receipts=[]
    from PIL import Image
    patches={name:np.array(Image.open(path).convert('RGB'),copy=True) for name,path in paths.items()}
    assigned=value['cases'][index::2]
    for n,case in enumerate(assigned):
        todo=[name for name in paths if not complete(ROOT/'episodes'/case['start']/name/'COMPLETE.json')]
        if not todo:
            receipts.extend(ROOT/'episodes'/case['start']/name/'COMPLETE.json' for name in paths);continue
        # Recheck the external evidence on which reset reuse depends.
        base_receipt=BASE/'episodes'/case['start']/'clean/COMPLETE.json';verify_external(BASE,base_receipt)
        construction_state=constructor_rng()
        env,task,states=runtime.make_env(native.runtime_config(),case['suite'],case['task_id'])
        try:
            require(task.name==case['task_name'],'Task mapping changed')
            initial=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
            require(np.array_equal(initial,states[case['state_index']]),'Initial file differs')
            for name in paths:
                r=ROOT/'episodes'/case['start']/name/'COMPLETE.json'
                if complete(r):receipts.append(r);continue
                if env is None:
                    restore_constructor_rng(construction_state)
                    env,task,states=runtime.make_env(native.runtime_config(),case['suite'],case['task_id'])
                    require(task.name==case['task_name'] and np.array_equal(initial,states[case['state_index']]),'Fresh evaluation environment differs')
                attempt=r.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
                episode(env,task,initial,case,name,patches[name],engine.wrapper,attempt,monitor,BASE/'paired_starts'/case['start'])
                finish(r,[attempt/p for p in ('START.npz','EPISODE.json','TRACE.parquet','STEPS.parquet','EVENTS.parquet','front.mp4','wrist.mp4')])
                receipts.append(r)
                runtime.close_environment(env);env=None
            monitor.update(phase='native_PNG_closedloop',completed=n+1,total=len(assigned))
        finally:
            if env is not None:runtime.close_environment(env)
    receipts+=offline(index,engine,paths,monitor)
    finish(receipt,receipts,new_closedloop_episodes=len(assigned)*4)
