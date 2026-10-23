"""Saved PNGs use both native visual branches; internal restores are diagnostics only."""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rgb_common import (ROOT, SCALE, MATCH, config, native, module, require, read_json, sha, atomic_json,
                        atomic_parquet, complete, finish, episode_path, make_engine, verify_inputs)
from rgb_cache import records
from rgb_train import objective

def arrays(path):
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (50,50), 'Invalid PNG')
        return np.array(im, copy=True)

def score(feature, row):
    t = row['target_features']
    result = objective(torch.as_tensor(feature), t['translation_minus'], t['translation_plus'], 'full')
    return {'q': None, 'orthogonal': None, 'target_distance': None} if result is None else {
        'q': result[1]['q'], 'orthogonal': result[1]['orthogonal'], 'target_distance': result[1]['loss']}

def offline_record(engine, row, paths, individual=False):
    clean, features = engine.forward(row['observation'], row['language'])
    minus, _ = engine.forward(row['observation'], row['language'], {'dino': row['target_features']['translation_minus']})
    plus, _ = engine.forward(row['observation'], row['language'], {'dino': row['target_features']['translation_plus']})
    direction = plus-minus
    reference = score(features['dino'], row)
    values = []
    for condition, path in {'clean': None, **paths}.items():
        obs = row['observation'] if path is None else native.paste_numpy(row['observation'], arrays(path))
        action, f = (clean, features) if path is None else engine.forward(obs, row['language'])
        delta = action-clean
        norm = float(np.linalg.norm(delta[0,:3])); bn = float(np.linalg.norm(direction[0,:3]))
        dot = float(delta[0,:3]@direction[0,:3])
        m = score(f['dino'], row)
        item = {k: row[k] for k in ('state_id','episode_id','suite','task_id','state_index','frame_rank','cohort')}
        item.update(condition=condition, individual_fit_only=condition.startswith('individual_'), **m,
            q_delta=None if m['q'] is None else m['q']-reference['q'],
            first_translation_delta_l2=norm, direction_cos=dot/(norm*bn) if norm*bn>1e-12 else None,
            direction_positive=float(dot>0) if bn>1e-12 else None,
            actions=action.tolist(), clean_actions=clean.tolist(), ideal_plus_actions=plus.tolist(),
            ideal_minus_actions=minus.tolist(), png_sha256=sha(path) if path else None,
            no_target=bool(row['geometry']['no_target']))
        if condition in ('projection','full') and not individual:
            restored_dino, _ = engine.forward(obs,row['language'],{'dino':features['dino']})
            restored_siglip, _ = engine.forward(obs,row['language'],{'siglip':features['siglip']})
            item.update(restore_clean_dino_delta_l2=float(np.linalg.norm((restored_dino-clean)[0,:3])),
                        restore_clean_siglip_delta_l2=float(np.linalg.norm((restored_siglip-clean)[0,:3])))
        values.append(item)
    return values

def pairing_gate(env, obs, task_id, state, attempt):
    clean = read_json(episode_path(task_id,state,'clean').parent/'EPISODE.json')
    current = {'sim_state': np.asarray(env.get_sim_state()), 'front':obs['full_image'],
               'wrist':obs['wrist_image'], 'proprio':obs['state']}
    np.savez_compressed(attempt/'START.npz', **current)
    from oracle_rollout import compare_start
    with np.load(ROOT/clean['start_npz'],allow_pickle=False) as before:
        details = compare_start(current,before)
    if not all(v['equal'] for v in details.values()):
        atomic_json({'task_id':task_id,'state_index':state,'differences':details}, ROOT/f'reports/PAIRING/{attempt.name}.json')
        raise RuntimeError('Reset/camera mismatch BEFORE executing PNG actions; see reports/PAIRING')

def episode(env, task, initial, seed, patch, wrapper, runtime, monitor, attempt):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw=runtime.reset_wait(env, initial, seed, config()['num_steps_wait'])
    obs=runtime.observation(raw,native.runtime_config())
    # task and state are attached by the caller only for pairing metadata.
    pairing_gate(env,obs,monitor.task_id,monitor.state_index,attempt)
    step=decision=0; success=False; traces=[]; images=[];started=time.monotonic()
    while step<config()['max_steps'] and not success:
        obs=runtime.observation(raw,native.runtime_config())
        deployed=native.paste_numpy(obs,patch)
        with torch.no_grad():
            action=np.asarray(wrapper.predict_action(copy.deepcopy(deployed),task.language),np.float32).reshape(8,7)
            clean=np.asarray(wrapper.predict_action(copy.deepcopy(obs),task.language),np.float32).reshape(8,7)
        require(np.isfinite(action).all() and np.isfinite(clean).all(),'Nonfinite native PNG action')
        if decision in config()['keyframe_decisions']:
            path=attempt/f'frame_{decision:03d}.png'
            Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],axis=1)).save(path);images.append(path)
        before=step; eef=np.asarray(raw['robot0_eef_pos']).copy();used=0
        for slot in range(min(8,config()['max_steps']-step)):
            raw,_,ended,_=env.step(process_libero_action(action[slot].copy()).tolist())
            used+=1;step+=1
            if ended:success=True;break
        delta=action-clean
        traces.append({'decision_block':decision,'step_start':before,'step_end':step,'executed_slots':used,
            'predicted_actions':action.tolist(),'same_observation_clean_actions':clean.tolist(),
            'executed_translation_delta_l2_mean':float(np.linalg.norm(delta[:used,:3],axis=1).mean()),
            'gripper_disagreement':float(((action[:used,6]>.5)!=(clean[:used,6]>.5)).mean()),
            'eef_before':eef.tolist(),'eef_after':np.asarray(raw['robot0_eef_pos']).tolist(),
            'gripper_qpos_after':np.asarray(raw['robot0_gripper_qpos']).tolist(),'success_after_chunk':success})
        decision+=1
        monitor.update(phase='native_png_closedloop',executed_steps=step,decision_blocks=decision)
    obs=runtime.observation(raw,native.runtime_config());deployed=native.paste_numpy(obs,patch)
    path=attempt/'frame_final.png';Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],axis=1)).save(path);images.append(path)
    return {'success':int(success),'failure':int(not success),'executed_steps':step,'decision_blocks':decision,
            'elapsed_s':time.monotonic()-started,'start_npz':str((attempt/'START.npz').relative_to(ROOT))},traces,images

def run(monitor):
    native.require_gpu('rgb2_evaluate')
    verify_inputs()
    paths={'initial':ROOT/'data/initial.png','uada':native.PROJECT/native.cfg()['uada_png'],
           **{k:ROOT/f'training/{k}/final.png' for k in config()['candidates']}}
    engine=make_engine()
    ids=read_json(ROOT/'data/MANIFEST.json')['individual_ids']
    for cohort in ('fit','verify'):
        for row in records(cohort):
            if cohort=='fit' and row['state_id'] not in ids:continue
            part=ROOT/f'evaluation/offline/{row["state_id"]}.json'
            if complete(part):continue
            engine.verify_patch_path(row)
            current=paths if cohort=='verify' else {
                'initial':paths['initial'], **{k:paths[k] for k in config()['candidates']},
                **{'individual_'+k:ROOT/f'training/{k}/individual/{row["state_id"]}/final.png' for k in config()['candidates']}}
            output=offline_record(engine,row,current,individual=cohort=='fit')
            parquet=part.with_suffix('.parquet');atomic_parquet(pd.DataFrame(output),parquet);finish(part,[parquet])
            monitor.update(phase='offline_'+cohort,state_id=row['state_id'])
    runtime=native.runtime_module()
    done=sum(complete(episode_path(t,s,k)) for t in config()['eval_tasks'] for s in config()['eval_states'] for k in config()['candidates'])
    monitor.update(completed=done,resume_completed=done,total=20)
    for task_id in config()['eval_tasks']:
        env,task,initial=runtime.make_env(native.runtime_config(),'libero_spatial',task_id)
        try:
            for state in config()['eval_states']:
                conditions=list(config()['candidates']);random.Random(config()['order_seed']+task_id*100+state).shuffle(conditions)
                for kind in conditions:
                    receipt=episode_path(task_id,state,kind)
                    if complete(receipt):continue
                    attempt=receipt.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=False)
                    clean=read_json(episode_path(task_id,state,'clean').parent/'EPISODE.json')
                    monitor.task_id,monitor.state_index=task_id,state
                    monitor.update(task_id=task_id,state_index=state,condition=kind,completed=done,total=20)
                    result,trace,images=episode(env,task,initial[state],clean['rollout_seed'],arrays(paths[kind]),engine.wrapper,runtime,monitor,attempt)
                    meta={'task_id':task_id,'state_index':state,'suite':'libero_spatial','checkpoint':'libero_spatial',
                          'condition':kind,'rollout_seed':clean['rollout_seed'],'reused_baseline':False,
                          'png_sha256':sha(paths[kind]),'internal_interventions':False,**result}
                    atomic_json(meta,receipt.parent/'EPISODE.json');atomic_parquet(pd.DataFrame(trace),receipt.parent/'TRACE.parquet')
                    finish(receipt,[receipt.parent/'EPISODE.json',receipt.parent/'TRACE.parquet',attempt/'START.npz',*images]);done+=1
                    monitor.update(completed=done,total=20)
        finally:runtime.close_environment(env)
    finish(ROOT/'evaluation/COMPLETE.json',[episode_path(t,s,k) for t in config()['eval_tasks'] for s in config()['eval_states'] for k in config()['conditions']],**native.resources())
