"""Offline first, then paired native PNG episodes. No feature replacement in rollout."""
import copy
import hashlib
import time
import numpy as np
import torch
from PIL import Image
from rr_common import (ROOT,PILOT,Pool,target,config,manifest,read_json,native,require,sha,
    atomic_json,atomic_parquet,complete,finish,make_engine,episode_path)
from rr_math import Readout


def png(path):
    with Image.open(path) as image:
        require(image.mode=='RGB' and image.size==(50,50),'Invalid deployed PNG')
        return np.array(image,copy=True)


def offline_record(engine,decoder,row,paths):
    clean,caps=engine.forward(row['observation'],row['language'])
    clean_relation=decoder(torch.from_numpy(caps['dino'])).numpy()[0]
    desired=target(row)
    ideal_plus,_=engine.forward(row['observation'],row['language'],{'dino':row['target_features']['translation_plus']})
    ideal_minus,_=engine.forward(row['observation'],row['language'],{'dino':row['target_features']['translation_minus']})
    reference=ideal_plus-ideal_minus;length=float(np.linalg.norm(reference[0,:3]))
    records=[]
    for name,path in paths.items():
        obs=row['observation'] if path is None else native.paste_numpy(row['observation'],png(path))
        action,feature=(clean,caps) if path is None else engine.forward(obs,row['language'])
        relation=decoder(torch.from_numpy(feature['dino'])).numpy()[0]
        movement=relation-clean_relation;difference=action-clean
        item={k:row[k] for k in ('state_id','episode_id','task_id','state_index','frame_rank','cohort')}
        item.update(condition=name,no_target=desired is None,actual_delta_m=desired.tolist() if desired is not None else None,
            relation_delta_cm=(movement*100).tolist(),
            relation_error_cm=float(np.linalg.norm(movement-desired)*100) if desired is not None else None,
            first_action_projection=float(difference[0,:3]@reference[0,:3]/length) if length>1e-12 else None,
            first_action_delta_l2=float(np.linalg.norm(difference[0,:3])),
            chunk_translation_delta_l2=float(np.linalg.norm(difference[:,:3],axis=-1).mean()),
            actions=action.tolist(),clean_actions=clean.tolist(),ideal_plus_actions=ideal_plus.tolist(),
            ideal_minus_actions=ideal_minus.tolist(),png_sha256=sha(path) if path is not None else None)
        if name=='relation':
            restored,unchanged=engine.forward(obs,row['language'],{'dino':caps['dino']})
            require(np.array_equal(unchanged['siglip'],feature['siglip']),'DINO restoration altered actual patched SigLIP')
            item.update(restore_clean_dino_first_delta_l2=float(np.linalg.norm((restored-clean)[0,:3])),
                restore_clean_dino_chunk_delta_l2=float(np.linalg.norm((restored-clean)[:,:3],axis=-1).mean()),
                restore_clean_dino_actions=restored.tolist(),restore_is_internal_diagnostic=True)
        records.append(item)
    return records


def start_arrays(env,obs):
    return {'sim_state':np.array(env.get_sim_state(),copy=True),'front':np.array(obs['full_image'],copy=True),
            'wrist':np.array(obs['wrist_image'],copy=True),'proprio':np.array(obs['state'],copy=True)}


def comparisons(current,reference):
    values={}
    for k,value in current.items():
        before=reference[k];same=value.shape==before.shape
        values[k]={'equal':bool(same and np.array_equal(value,before)),
            'max_abs_difference':float(np.max(np.abs(value.astype(float)-before.astype(float)))) if same else None}
    return values


def fingerprints(arrays):
    output={}
    for k,x in arrays.items():
        x=np.ascontiguousarray(x)
        output[k]=hashlib.sha256(str((x.shape,x.dtype)).encode()+x.tobytes()).hexdigest()
    return output


def baseline_arrays_valid(entry,current):
    if not entry['eligible']:return False,{'reason':entry['reason']}
    proof={};hashes=fingerprints(current)
    aliases={'sim_state':'start_sim_sha256','front':'start_front_sha256','wrist':'start_wrist_sha256','proprio':'start_proprio_sha256'}
    for name,paths in entry['conditions'].items():
        meta=read_json(paths['meta'])
        with np.load(paths['start'],allow_pickle=False) as before:
            values=comparisons(current,before)
        # Some copied legacy EPISODEs point at clean START arrays. Check the
        # condition's independently saved fingerprints as well, rather than trust that pointer.
        for k,field in aliases.items():
            # Clean owns a receipt-sealed START array. Patched legacy conditions
            # may point at that clean array, so they additionally need their own fingerprints.
            if (name!='clean' and field not in meta) or (field in meta and meta[field]!=hashes[k]):
                values[k]['equal']=False
        proof[name]=values
    return all(v['equal'] for row in proof.values() for v in row.values()),proof


def episode(env,task,initial,seed,patch,engine,runtime,monitor,folder,canonical):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw=runtime.reset_wait(env,initial,seed,config()['num_steps_wait'])
    obs=runtime.observation(raw,native.runtime_config());current=start_arrays(env,obs)
    compare=comparisons(current,canonical)
    atomic_json(compare,folder/'PAIRING.json')
    require(all(v['equal'] for v in compare.values()),'New native episode reset mismatch; no actions executed')
    np.savez_compressed(folder/'START.npz',**current)
    steps=decision=0;success=False;traces=[];images=[];started=time.monotonic()
    while steps<config()['max_steps'] and not success:
        obs=runtime.observation(raw,native.runtime_config())
        deployed=obs if patch is None else native.paste_numpy(obs,patch)
        with torch.no_grad():
            actions=np.asarray(engine.wrapper.predict_action(copy.deepcopy(deployed),task.language),np.float32).reshape(8,7)
            clean=actions.copy() if patch is None else np.asarray(engine.wrapper.predict_action(copy.deepcopy(obs),task.language),np.float32).reshape(8,7)
        require(np.isfinite(actions).all() and np.isfinite(clean).all(),'Nonfinite deployed actions')
        if decision in config()['keyframe_decisions']:
            path=folder/f'frame_{decision:03d}.png'
            Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],1)).save(path);images.append(path)
        start=steps;used=0;before=np.array(raw['robot0_eef_pos'],copy=True)
        for slot in range(min(8,config()['max_steps']-steps)):
            raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());steps+=1;used+=1
            if done:success=True;break
        difference=actions-clean
        traces.append({'decision_block':decision,'step_start':start,'step_end':steps,'executed_slots':used,
            'predicted_actions':actions.tolist(),'same_observation_clean_actions':clean.tolist(),
            'executed_translation_delta_l2_mean':float(np.linalg.norm(difference[:used,:3],axis=1).mean()),
            'executed_gripper_disagreement':float(((actions[:used,6]>.5)!=(clean[:used,6]>.5)).mean()),
            'eef_before':before.tolist(),'eef_after':np.asarray(raw['robot0_eef_pos']).tolist(),
            'gripper_qpos_after':np.asarray(raw['robot0_gripper_qpos']).tolist()})
        decision+=1;monitor.update(executed_steps=steps,decision_blocks=decision)
    obs=runtime.observation(raw,native.runtime_config());deployed=obs if patch is None else native.paste_numpy(obs,patch)
    path=folder/'frame_final.png';Image.fromarray(np.concatenate([deployed['full_image'],deployed['wrist_image']],1)).save(path);images.append(path)
    return {'success':int(success),'failure':int(not success),'executed_steps':steps,'decision_blocks':decision,
        'elapsed_s':time.monotonic()-started,**{'start_'+k+'_sha256':v for k,v in fingerprints(current).items()}},traces,images


def run(index,monitor):
    native.require_gpu('relation_evaluate')
    engine=make_engine();pool=Pool('verify');engine.verify_patch_path(pool[0])
    decoder=Readout(PILOT/'readouts/b22.npz');c=config();v=manifest()
    paths={'clean':None,'initial':ROOT/'data/initial.png','uada':native.PROJECT/native.cfg()['uada_png'],
           'relation':ROOT/'training/final.png'}
    artifacts=[];errors=[]
    # Offline validation comes first; rollout errors cannot erase it.
    for i in range(index,len(pool),2):
        row=pool[i];path=ROOT/f'evaluation/offline/{row["state_id"]}.json'
        receipt=path.with_name(path.stem+'_COMPLETE.json')
        if not complete(receipt):
            try:
                values=offline_record(engine,decoder,row,paths)
                atomic_json(values,path);finish(receipt,[path],cohort='verify')
            except Exception as error:
                errors.append({'stage':'offline','state_id':row['state_id'],'error':repr(error)})
                atomic_json(errors,ROOT/f'reports/EVALUATION_ERRORS_{index}.json');continue
        artifacts.append(receipt)
        monitor.update(phase='offline_verify',completed=(i-index)//2+1,total=30,state_id=row['state_id'])
    runtime=native.runtime_module();grid=[(t,s) for t in c['eval_tasks'] for s in c['eval_states']][index::2]
    for n,(task_id,state) in enumerate(grid):
        if all(complete(episode_path(task_id,state,k)) for k in c['conditions']):
            artifacts.extend(episode_path(task_id,state,k) for k in c['conditions']);continue
        env=None
        try:
            env,task,initial=runtime.make_env(native.runtime_config(),'libero_spatial',task_id)
            entry=v['baselines'][f'{task_id}/{state}'];seed=entry['seed']
            raw=runtime.reset_wait(env,initial[state],seed,c['num_steps_wait'])
            obs=runtime.observation(raw,native.runtime_config());current=start_arrays(env,obs)
            canonical_path=ROOT/f'evaluation/starts/t{task_id:02d}_s{state}.npz'
            canonical_path.parent.mkdir(parents=True,exist_ok=True)
            # A resumed start must still match the new run's original canonical arrays.
            if canonical_path.exists():
                with np.load(canonical_path,allow_pickle=False) as before:compare=comparisons(current,before)
                require(all(x['equal'] for x in compare.values()),'Resume reset differs from canonical start')
            else:np.savez_compressed(canonical_path,**current)
            reuse,proof=baseline_arrays_valid(entry,current)
            proof_path=ROOT/f'evaluation/starts/t{task_id:02d}_s{state}_BASELINE.json'
            atomic_json({'reuse':reuse,'proof':proof,'seed':seed},proof_path)
            if reuse:
                for condition in ('clean','initial','uada'):
                    receipt=episode_path(task_id,state,condition)
                    if complete(receipt):continue
                    old=entry['conditions'][condition];meta=read_json(old['meta'])
                    meta.update(reused_baseline=True,source_receipt=old['receipt'],
                        start_npz=str(canonical_path.relative_to(ROOT)),baseline_pairing=str(proof_path.relative_to(ROOT)))
                    path=receipt.parent/'EPISODE.json';atomic_json(meta,path)
                    finish(receipt,[path,canonical_path,proof_path],reused_baseline=True)
            for condition in c['conditions']:
                receipt=episode_path(task_id,state,condition)
                if complete(receipt):artifacts.append(receipt);continue
                attempt=receipt.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=False)
                monitor.update(phase='native_png_closedloop',task_id=task_id,state_index=state,condition=condition,
                               completed=n,total=len(grid))
                result,traces,images=episode(env,task,initial[state],seed,
                    png(paths[condition]) if paths[condition] is not None else None,engine,runtime,monitor,attempt,current)
                # Keep native fingerprint field names consistent with legacy metadata.
                result['start_sim_sha256']=result.pop('start_sim_state_sha256')
                meta={'suite':'libero_spatial','checkpoint':'libero_spatial','task_id':task_id,'state_index':state,
                    'condition':condition,'rollout_seed':seed,'png_sha256':sha(paths[condition]) if paths[condition] else None,
                    'reused_baseline':False,'internal_interventions':False,'start_npz':str((attempt/'START.npz').relative_to(ROOT)),**result}
                meta_path=receipt.parent/'EPISODE.json';trace_path=receipt.parent/'TRACE.parquet'
                import pandas as pd
                atomic_json(meta,meta_path);atomic_parquet(pd.DataFrame(traces),trace_path)
                finish(receipt,[meta_path,trace_path,attempt/'START.npz',attempt/'PAIRING.json',*images]);artifacts.append(receipt)
        except Exception as error:
            errors.append({'stage':'rollout','task_id':task_id,'state_index':state,'error':repr(error)})
            atomic_json(errors,ROOT/f'reports/EVALUATION_ERRORS_{index}.json')
        finally:
            if env is not None:runtime.close_environment(env)
    if errors:raise RuntimeError(f'{len(errors)} evaluation errors retained; completed pairs/offline results are reusable')
    finish(ROOT/f'evaluation/WORKER_{index}_COMPLETE.json',artifacts,**native.resources())
