"""Full native LIBERO episodes with a persistent wrist camera geometry change."""
import gc
from pathlib import Path
import time
import numpy as np
import torch
from wr_common import (ROOT, PROJECT, config, read, require, atomic, complete, finish, signature,
    cases, profiles, candidate, resources, load_wrapper, switch_suite, native_runtime,
    runtime_config, native_helpers)
from wr_engine import Engine
from wr_camera import CameraSamples
from wr_data import geometry_module


def folder(stage,name,case,profile):
    return ROOT/f'evaluation/{stage}/{name}'/case['start']/profile['name']

def target(env,case):
    geo=geometry_module()
    try:return geo,geo.anchor(env,case['suite'],case['task_id']),None
    except (RuntimeError,KeyError,ValueError,AttributeError) as error:
        return geo,None,repr(error)

def truth(env,geo,anchor):
    if anchor is not None:return geo.truth(env,anchor)
    return dict(eef_xyz=np.asarray(env.sim.data.site_xpos[env.env.robots[0].eef_site_id]).tolist(),
                target_xyz=None,distance_m=None,contact=None,grasp=None)

def episode(stage,name,engine,case,profile,monitor,check):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action,get_libero_dummy_action
    import pandas as pd
    r=native_runtime();rc=runtime_config();utils=native_helpers();c=config()
    out=folder(stage,name,case,profile);out.mkdir(parents=True,exist_ok=True)
    receipt=out/'COMPLETE.json'
    if complete(receipt):return
    attempt=out/('attempt_'+str(time.time_ns()));attempt.mkdir()
    started=time.monotonic();env=None;switch_suite(engine.wrapper,case['suite'])
    utils.seed_all(case['rollout_seed'])
    try:
        env,task,initial=r.make_env(rc,case['suite'],case['task_id'])
        require(task.name==case['task_name'],'Task name changed')
        pinned=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
        require(np.array_equal(pinned,initial[case['state_index']]),'Initial-state file mismatch')
        env.seed(case['rollout_seed']);r.bind_environment_context(env);r.reset_environment(env)
        gc.collect();r.renderer_metadata(env)
        raw=env.set_init_state(pinned)
        require(np.array_equal(env.get_sim_state(),pinned),'Reset physics differs from initial-state file')
        geo,anchor,target_error=target(env,case)
        ids=anchor['render_ids'] if anchor is not None else None
        with CameraSamples(env,c['rgb_tolerance'],ids) as camera:
            for i in range(rc['num_steps_wait']):
                camera.arm('warmup_'+str(i))
                raw,_,done,_=env.step(get_libero_dummy_action());require(not done,'Task ended during fixed warmup')
            obs=r.observation(raw,rc);fingerprint=utils.fingerprint(env)
            arrays=dict(sim_state=np.asarray(env.get_sim_state()).copy(),proprio=obs['state'],
                        front=obs['full_image'],wrist=obs['wrist_image'])
            pair=ROOT/f'evaluation/{stage}/paired_starts'/case['start']
            meta=dict(fingerprint=fingerprint,rollout_seed=case['rollout_seed'])
            if name=='base' and profile['name']=='native':
                if (pair/'START.json').exists():utils.assert_start(arrays,meta,pair)
                else:
                    pair.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(pair/'START.npz',**arrays);atomic(meta,pair/'START.json')
            else:
                require((pair/'START.json').is_file(),'Original native start missing')
                utils.assert_start(arrays,meta,pair)
            check_values=engine.gate(obs,task.language,gradient=False) if check else dict(status='PASS')
            first=engine.native(obs,task.language)
            if name=='base':
                first_path=pair/'BASE_FIRST_ACTION.npy'
                if profile['name']=='native':np.save(first_path,first)
                else:require(np.max(abs(first-np.load(first_path)))<=c['action_tolerance'],'Paired base first action differs')
            require(utils.fingerprint(env)==fingerprint,'Action/interface forward changed simulator state')
            atomic(check_values,attempt/'INTERFACE_CHECK.json')
            fps=1/float(env.env.control_timestep);limit=c['max_steps'][case['suite']]
            step=0;decision=0;success=False;steps=[];traces=[];audits=[]
            first_contact=None;first_grasp=None;lost=visible=0
            initial_visibility=None

            def deployed(current,at_decision):
                native=r.observation(current,rc)
                # Zero shift is deliberately rendered at startup as an interface
                # check, while unmodified policy pixels remain the native pixels.
                if profile['shift_m']==0:
                    if step==0:
                        _,audit=camera.shifted_raw(current,'x',0.,audit_visibility=True)
                        audits.append(dict(control_step=step,**audit))
                    return native
                changed,audit=camera.shifted_raw(current,profile['axis'],profile['shift_m'],
                    audit_visibility=at_decision,check_zero=at_decision)
                altered=r.observation(changed,rc)
                require(np.array_equal(native['full_image'],altered['full_image']) and
                        np.array_equal(native['state'],altered['state']),'Wrist change altered other policy inputs')
                if at_decision:audits.append(dict(control_step=step,**audit))
                return altered

            image=deployed(raw,True)
            with utils.CameraWriters(attempt,fps) as writer:
                writer.append(image)
                while step<limit and not success:
                    before_forward=utils.fingerprint(env)
                    actions=engine.native(image,task.language)
                    require(utils.fingerprint(env)==before_forward,'Policy advanced physical state')
                    begin=step;used=0
                    if audits and audits[-1]['control_step']==step:
                        audit=audits[-1]
                        if 'target_lost' in audit:
                            visible+=int(audit['native_target_pixels']>0);lost+=int(audit['target_lost'])
                            if initial_visibility is None:initial_visibility={k:audit[k] for k in ('native_target_pixels','shifted_target_pixels','target_lost')}
                    for slot in range(min(8,limit-step)):
                        prior=truth(env,geo,anchor)
                        command=np.asarray(process_libero_action(actions[slot].copy()),float)
                        require(command.shape==(7,) and np.isfinite(command).all(),'Invalid controller command')
                        camera.arm('step_'+str(step+1))
                        raw,_,done,_=env.step(command.tolist());step+=1;used+=1
                        after=truth(env,geo,anchor)
                        if after['contact'] and first_contact is None:first_contact=step
                        if after['grasp'] and first_grasp is None:first_grasp=step
                        steps.append(dict(step=step,decision=decision,slot=slot,
                            action=actions[slot].tolist(),controller_command=command.tolist(),
                            eef_xyz=after['eef_xyz'],target_xyz=after['target_xyz'],distance_m=after['distance_m'],
                            actual_eef_displacement_m=(np.asarray(after['eef_xyz'])-prior['eef_xyz']).tolist(),
                            contact=after['contact'],grasp=after['grasp'],task_success=bool(done)))
                        image=deployed(raw,step%8==0)
                        writer.append(image)
                        if done:success=True;break
                    traces.append(dict(decision=decision,start_step=begin,end_step=step,
                                       executed_slots=used,actions=actions.tolist()))
                    decision+=1;monitor.update(phase='closedloop',candidate=name,start=case['start'],
                                               profile=profile['name'],executed_steps=step)
            pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet',index=False)
            pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet',index=False)
            atomic(dict(records=audits,fixed_target_annotation_error=target_error,
                        visibility_does_not_select_outcomes=True),attempt/'CAMERA.json')
            result=dict(signature=signature(),stage=stage,candidate=name,
                seed=0 if name=='base' else candidate(stage,name)['seed'],**case,
                profile=profile['name'],camera_axis=profile['axis'],camera_shift_m=profile['shift_m'],
                success=bool(success),executed_steps=step,max_steps=limit,decisions=decision,
                first_contact_step=first_contact,first_grasp_step=first_grasp,
                native_target_visible_decisions=visible,target_lost_decisions=lost,
                initial_visibility=initial_visibility,fixed_target_annotation_error=target_error,
                elapsed_s=time.monotonic()-started,fps=fps,checkpoint=c['model_dir'],
                adapter=None if name=='base' else str(ROOT/f'training/{stage}/{name}/final.pt'),
                **resources())
            atomic(result,out/'RESULT.json')
            finish(receipt,[out/'RESULT.json',*[attempt/n for n in ('INTERFACE_CHECK.json','CAMERA.json',
                   'STEPS.parquet','TRACE.parquet','front.mp4','wrist.mp4')]])
        # Central native single-view MP4s are linked without duplication. WebM
        # encoding is deferred to a CPU worker, never holding a GPU for conversion.
        import os
        for view in ('front','wrist'):
            dest=PROJECT/'research_videos'/f'{ROOT.name}_singleview'/stage/name/profile['name']/view
            dest.mkdir(parents=True,exist_ok=True)
            target_path=dest/(case['start']+f'_success_{int(success)}.mp4')
            if not target_path.exists():os.link(attempt/(view+'.mp4'),target_path)
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started,**resources()),attempt/'FAILURE.json')
        raise
    finally:
        if env is not None:r.close_environment(env)


def evaluate(stage,name,shard,shards,monitor):
    allcases=cases(stage)[shard::shards];p=profiles(stage)
    expected=[folder(stage,name,case,profile)/'COMPLETE.json' for case in allcases for profile in p]
    done=sum(complete(r) for r in expected)
    monitor.update(completed=done,resumed=done,total=len(expected))
    offline_needed=shard==0 and not complete(ROOT/f'evaluation/{stage}/{name}/OFFLINE_COMPLETE.json')
    if done==len(expected) and not offline_needed:return
    wrapper=load_wrapper(monitor);engine=Engine(wrapper,gradient_checkpointing=False)
    if name!='base':
        require(complete(ROOT/f'training/{stage}/{name}/COMPLETE.json'),'Final adapter incomplete')
        saved=torch.load(ROOT/f'training/{stage}/{name}/final.pt',map_location='cpu',weights_only=False)
        require(saved['signature']==signature() and saved['step']==candidate(stage,name)['updates'],'Final adapter/checkpoint mismatch')
        engine.load_state_dict(saved['adapter'])
    checked=set();started=time.monotonic();timed=0
    for case in allcases:
        # Native must precede changed profiles so paired reset evidence exists.
        for profile in p:
            receipt=folder(stage,name,case,profile)/'COMPLETE.json'
            if complete(receipt):continue
            episode(stage,name,engine,case,profile,monitor,case['suite'] not in checked)
            checked.add(case['suite']);done+=1;timed+=1;monitor.update(completed=done)
            if timed==4:atomic(dict(episodes=timed,elapsed_s=time.monotonic()-started,
                mean_episode_s=(time.monotonic()-started)/timed,**resources()),
                ROOT/f'reports/THROUGHPUT/eval_{stage}_{name}_{shard}.json')
    if offline_needed:
        from wr_train import offline
        offline(stage,name,engine,monitor)
    require(all(complete(r) for r in expected),'Evaluation shard incomplete')
    finish(ROOT/f'reports/EVALUATE_{stage}_{name}_{shards}_{shard}_COMPLETE.json',expected)
