"""Full-task oracle restoration; donor is rebuilt at the current branch state."""
import time
import numpy as np
from fr_common import (ROOT,PROJECT,config,read,atomic,require,finish,complete,verify,runtime,runtime_config,
    native,geometry,episode_receipt,episode_files,resources)
from fr_vision import Vision
from sc_rollout import warm_history,start_arrays,create,snapshot,native_start_reference
from sc_math import action_difference

def episode(wrapper,vision,case,condition,patch,monitor,gate_needed):
    import pandas as pd
    from jr_math import paste
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    utils=native();r=runtime();c=config();name=condition['name'];mode=condition['mode'];receipt=episode_receipt(case,name)
    attempt=receipt.parent/('attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
    env=None;started=time.monotonic();utils.switch_suite(wrapper,case['suite'])
    try:
        expected_start=native_start_reference(case) if name=='clean' else None
        env,task,initial=create(case)
        with warm_history(env,initial,case['rollout_seed']) as (raw,history,samples):
            geo=geometry();target=geo['anchor'](env,case['suite'],case['task_id'])
            obs=history[-1]['observation'];arrays=start_arrays(env,obs)
            start=dict(fingerprint=utils.fingerprint(env),rollout_seed=case['rollout_seed'])
            np.savez_compressed(attempt/'START.npz',**arrays);atomic(start,attempt/'START.json')
            paired=ROOT/'paired_starts'/case['start']
            if name=='clean':
                baseline,meta=expected_start;require(start==meta,'Sampler changed native reset fingerprint')
                for k in arrays:require(np.array_equal(arrays[k],baseline[k]),'Sampler changed native reset: '+k)
                if paired.exists():utils.assert_start(arrays,start,paired)
                else:
                    paired.mkdir(parents=True);np.savez_compressed(paired/'START.npz',**arrays);atomic(start,paired/'START.json')
                old=None
            else:
                require(complete(episode_receipt(case,'clean')),'Clean source missing')
                utils.assert_start(arrays,start,paired)
                old=np.load(episode_files(episode_receipt(case,'clean'))['FIRST_ACTION.npy'],allow_pickle=False)
            before=utils.fingerprint(env)
            first,_=vision.action(obs,task.language)
            if old is not None:
                error=float(np.max(abs(first-old)));require(error<=c['action_tolerance'],'Reused clean first action differs')
                atomic(dict(status='PASS',start=case['start'],same_reset=True,first_action_max_abs_error=error,
                    source=read(ROOT/'data/REUSE_PLAN.json')['source_signature']),ROOT/'reports/REUSE_GATES'/f"{case['start']}.json")
            np.save(attempt/'FIRST_ACTION.npy',first)
            deployed=paste(obs,patch,c['patch_xy'])
            gate=utils.interface_gate(wrapper,obs,task.language,case['suite']) if gate_needed else dict(status='PASS')
            if gate_needed:gate.update(vision.gate(obs,deployed,task.language,old))
            require(utils.fingerprint(env)==before,'Forward or restoration changed physics/camera/sampling')
            atomic(gate,attempt/'INTERFACE_CHECK.json')
            fps=1/float(env.env.control_timestep);limit=c['max_steps'][case['suite']]
            step=decision=0;success=False;traces=[];steps=[];max_identity_error=0.
            truth=geo['truth'](env,target);first_contact=0 if truth['contact'] else None;first_grasp=0 if truth['grasp'] else None
            with utils.CameraWriters(attempt,fps) as writer:
                writer.append(deployed)
                while step<limit and not success:
                    obs=history[-1]['observation'];deployed=paste(obs,patch,c['patch_xy']);before_forward=utils.fingerprint(env)
                    donor_sample=dict(history[-1]['sensors']['wrist'])
                    # Always build donor from THIS unpatched live observation; no other episode is read.
                    clean,donor=vision.action(obs,task.language)
                    if name=='clean':actions=clean.copy();metrics=dict(dino_native_delta_rms=0.,dino_used_delta_rms=0.,siglip_native_delta_rms=0.,siglip_used_delta_rms=0.)
                    else:
                        actions,caps=vision.action(deployed,task.language,mode,donor)
                        metrics=vision.metrics(caps,donor,mode);del caps
                    del donor
                    require(utils.fingerprint(env)==before_forward,'Donor/restore forward changed native state or sensor clocks')
                    both_error=float(np.max(abs(actions-clean))) if mode=='both' else None
                    if both_error is not None:
                        max_identity_error=max(max_identity_error,both_error)
                        require(both_error<=c['action_tolerance'],'Both restored branches no longer match same-state clean action')
                    used=0;begin=step
                    for slot in range(min(c['chunk'],limit-step)):
                        before_truth=geo['truth'](env,target)
                        command=np.asarray(process_libero_action(actions[slot].copy()),float)
                        require(command.shape==(7,) and np.isfinite(command).all(),'Invalid native control command')
                        raw,_,done,_=env.step(command.tolist());step+=1;used+=1
                        after=geo['truth'](env,target);history.append(snapshot(raw,samples,env,step))
                        if after['contact'] and first_contact is None:first_contact=step
                        if after['grasp'] and first_grasp is None:first_grasp=step
                        steps.append(dict(step=step,time_s=step/fps,decision_block=decision,executed_slot=slot,
                            raw_action=actions[slot].tolist(),same_state_clean_action=clean[slot].tolist(),controller_command=command.tolist(),
                            eef_xyz=after['eef_xyz'],target_xyz=after['target_xyz'],target_center_distance_m=after['distance_m'],
                            actual_eef_displacement_m=(np.asarray(after['eef_xyz'])-before_truth['eef_xyz']).tolist(),
                            contact=after['contact'],grasp=after['grasp'],success=bool(done)))
                        writer.append(paste(history[-1]['observation'],patch,c['patch_xy']))
                        if done:success=True;break
                    stats=wrapper.vla.norm_stats[wrapper.gen_cfg.unnorm_key]['action']
                    traces.append(dict(decision_block=decision,step_start=begin,step_end=step,executed_slots=used,
                        donor_control_step=begin,donor_sim_time_s=before_forward['time'],
                        donor_wrist_frame_id=donor_sample['frame_id'],donor_wrist_sensor_timestamp_s=donor_sample['sensor_timestamp_s'],
                        restoration=mode,oracle_same_state=True,physics_and_sensor_clocks_unchanged=True,
                        predicted_actions=actions.tolist(),same_state_clean_actions=clean.tolist(),
                        both_restore_max_abs_error=both_error,**metrics,**action_difference(actions,clean,stats,used)))
                    decision+=1;monitor.update(current_start=case['start'],condition=name,executed_steps=step,completed_decisions=decision)
            pd.DataFrame(traces).to_parquet(attempt/'TRACE.parquet',index=False)
            pd.DataFrame(steps).to_parquet(attempt/'STEPS.parquet',index=False)
            atomic(dict(**case,condition=name,patch=condition['patch'],restoration=mode,checkpoint=c['model_dir'],revision=c['revision'],
                png_sha256=condition['sha256'],success=int(success),executed_steps=step,max_steps=limit,fps=fps,num_open_loop_steps=8,
                first_target_contact_step=first_contact,first_target_grasp_step=first_grasp,contact_sampling='native control-step endpoint',
                native_PNG_only=(mode=='native'),oracle_feature_restoration=(mode!='native'),
                donor_source='same current branch unpatched observation',model_frozen=True,decisions=decision,
                both_restore_max_abs_error=max_identity_error if mode=='both' else None,elapsed_s=time.monotonic()-started,**resources()),attempt/'EPISODE.json')
            finish(receipt,[attempt/n for n in ('START.npz','START.json','FIRST_ACTION.npy','INTERFACE_CHECK.json','EPISODE.json',
                'TRACE.parquet','STEPS.parquet','front.mp4','wrist.mp4')]+[paired/n for n in ('START.npz','START.json')])
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),case=case,condition=name,elapsed_s=time.monotonic()-started,**resources()),attempt/'FAILURE.json');raise
    finally:
        if env is not None:r.close_environment(env)

def run(stage,index,workers,monitor):
    from jr_math import weighted_shards
    from PIL import Image
    require(1<=workers<=2,'At most two GPU workers');m=verify();utils=native()
    if stage=='clean':cases=m['cases'];rows=[r for r in m['conditions'] if r['name']=='clean']
    else:
        require(stage=='evaluate' and 0<=index<workers,'Invalid worker shard')
        require(all(complete(episode_receipt(case,'clean')) for case in m['cases']),'All clean sources required')
        cases=weighted_shards(m['cases'],workers,config()['max_steps'])[index]
        rows=[r for r in m['conditions'] if r['name']!='clean']
    receipts=[episode_receipt(case,row['name']) for case in cases for row in rows]
    already=sum(complete(p) for p in receipts);monitor.update(completed=already,resumed=already,total=len(receipts))
    if already<len(receipts):
        wrapper=utils.load_wrapper(monitor);vision=Vision(wrapper);checked=set()
        patches={r['name']:np.array(Image.open(PROJECT/r['path']).convert('RGB'),copy=True) if r['path'] else None for r in rows}
        cases_timed=0;timed_seconds=0.
        for case in cases:
            case_started=time.monotonic();new_episodes=0
            order=list(rows);np.random.default_rng(case['rollout_seed']).shuffle(order)
            for row in order:
                if complete(episode_receipt(case,row['name'])):continue
                key=(case['suite'],row['patch'])
                episode(wrapper,vision,case,row,patches[row['name']],monitor,key not in checked);checked.add(key)
                new_episodes+=1
                monitor.update(phase='feature_rescue',completed=monitor.data['completed']+1)
            if new_episodes:
                cases_timed+=1;timed_seconds+=time.monotonic()-case_started
                monitor.update(starts_timed=cases_timed,mean_complete_start_s=timed_seconds/cases_timed)
                if cases_timed==4:
                    atomic(dict(starts=cases_timed,elapsed_s=timed_seconds,mean_start_s=timed_seconds/cases_timed,
                        additional_policy_forwards=True,**resources()),ROOT/'reports/THROUGHPUT'/f'{stage}_{index}_first4.json')
    require(all(complete(p) for p in receipts),'Shard incomplete')
    finish(ROOT/f'reports/{stage}_{workers}_{index}_COMPLETE.json',receipts);monitor.update(phase='COMPLETE')
