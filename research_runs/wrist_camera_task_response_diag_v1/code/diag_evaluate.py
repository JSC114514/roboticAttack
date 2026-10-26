# 发布注释：每个条件从相同重放状态开始，执行完整8个控制步；短窗口任务标志不是完整回合成功率。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""One genuine native action chunk per predeclared physical/view condition."""
import time
import numpy as np
from diag_common import (ROOT,SOURCE,config,read,atomic,require,complete,finish,signature,resources,
    load_wrapper,switch_suite,native_runtime,runtime_config,native_helpers,require_gpu)
from diag_geometry import vector,set_translation
from diag_prepare import replay,rt_observation
from wr_data import geometry_module


def frozen(engine):
    for m in native_helpers().frozen_modules(engine.wrapper):
        m.eval()
        for p in m.parameters():p.requires_grad_(False)
    require(all(not p.requires_grad and p.grad is None for m in native_helpers().frozen_modules(engine.wrapper)
                for p in m.parameters()),'Inference model is not frozen')


def interface_gate(engine,observation,language):
    import torch
    before=engine.active
    engine.active=False
    try:
        routed=engine.native(observation,language)
        engine.backbone.forward=engine.original_forward
        original=engine.native(observation,language)
    finally:
        engine.backbone.forward=engine.visual_forward;engine.active=before
    error=float(np.max(abs(routed-original)))
    require(error<=config()['action_tolerance'],'Explicit wrist routing changed native base action')
    with torch.no_grad():
        normalized=engine.action(observation,language).cpu().numpy()
        restored=engine.wrapper.vla._unnormalize_actions(normalized,engine.wrapper.gen_cfg.unnorm_key)
        native=engine.native(observation,language)
    action_error=float(np.max(abs(restored-native)))
    require(action_error<=config()['action_tolerance'],'Normalized/native action mismatch')
    frozen(engine)
    return dict(status='PASS',routing_max_abs=error,action_max_abs=action_error,all_weights_frozen=True)


def branch(model,engine,a,c,index,monitor,check):
    import torch,pandas as pd
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    episode=a['episode_id'];folder=ROOT/'evaluation'/model/episode/c['name'];folder.mkdir(parents=True,exist_ok=True)
    if complete(folder/'COMPLETE.json'):return
    attempt=folder/('attempt_'+str(time.time_ns()));attempt.mkdir();started=time.monotonic();replay_steps=0
    try:
        with replay(episode,a['selected']['decision_block']) as (env,task,raw,samples,renderer,anchor,count):
            replay_steps=count
            require(samples.latest['fingerprint']==a['base_replay_sample_fingerprint'],'Replayed sample differs from prepared start')
            delta=vector(env,c['target_shift_m']);set_translation(env,anchor,delta)
            require(native_helpers().fingerprint(env)==c['physical_fingerprint'],'Prepared physical variant/start differs')
            truth=geometry_module().truth(env,anchor)
            require(np.allclose(truth['target_xyz'],c['initial_truth']['target_xyz'],atol=1e-10,rtol=0),'Physical target start differs')
            with np.load(ROOT/f'data/anchors/{episode}/CONDITIONS.npz',allow_pickle=False) as f:
                image=dict(full_image=f['front'][index].copy(),wrist_image=f['wrist'][index].copy(),state=f['proprio'][index].copy())
            # The startup gate checks official/native action equivalence on the exact deployed pixels.
            before=native_helpers().fingerprint(env)
            gate=interface_gate(engine,image,task.language) if check else dict(status='REUSED_PER_SUITE')
            with torch.no_grad():
                normalized=engine.action(image,task.language).cpu().numpy()
                actions=engine.native(image,task.language)
            require(native_helpers().fingerprint(env)==before,'Model inference changed simulator state')
            initial=truth;first_contact=0 if truth['contact'] else None;first_grasp=0 if truth['grasp'] else None
            step_rows=[];commands=[];visibility=[];done=False
            fps=1/float(env.env.control_timestep)
            with native_helpers().CameraWriters(attempt,fps) as writer:
                writer.append(image)
                for slot in range(8):
                    command=np.asarray(process_libero_action(actions[slot].copy()),float)
                    require(command.shape==(7,) and np.isfinite(command).all(),'Invalid native controller command')
                    prior=geometry_module().truth(env,anchor);samples.arm('diagnostic_'+str(slot+1))
                    raw,_,done,_=env.step(command.tolist())
                    # mj_step leaves derived poses at the pre-integration substep.
                    # Refresh metric geometry only; native camera snapshots/clocks stay intact.
                    env.sim.forward()
                    after=geometry_module().truth(env,anchor);commands.append(command.tolist())
                    if after['contact'] and first_contact is None:first_contact=slot+1
                    if after['grasp'] and first_grasp is None:first_grasp=slot+1
                    step_rows.append(dict(step=slot+1,slot=slot,action=actions[slot].tolist(),controller_command=command.tolist(),
                        eef_xyz=after['eef_xyz'],target_xyz=after['target_xyz'],distance_m=after['distance_m'],
                        actual_eef_displacement_m=(np.asarray(after['eef_xyz'])-prior['eef_xyz']).tolist(),
                        contact=after['contact'],grasp=after['grasp'],short_task_done=bool(done)))
                    # Target motion is now simulated physically. No synthetic target shift after start.
                    changed,audit=renderer.paired_raw(raw,samples,anchor,np.zeros(3),c['camera_shift_m'])
                    visibility.append(dict(step=slot+1,**audit));writer.append(rt_observation(changed))
            require(len(step_rows)==8,'Diagnostic must execute exactly eight control steps')
            final=geometry_module().truth(env,anchor);frozen(engine)
            result=dict(signature=signature(),model=model,episode_id=episode,suite=a['suite'],task_id=a['task_id'],
                state_index=a['state_index'],selected=a['selected'],condition=c['name'],target_shift_m=c['target_shift_m'],
                camera_shift_m=c['camera_shift_m'],valid=True,executed_steps=len(step_rows),success_flag=any(r['short_task_done'] for r in step_rows),
                final_task_success_flag=bool(done),initial_contact=bool(initial['contact']),initial_grasp=bool(initial['grasp']),
                not_a_full_task_evaluation=True,initial_distance_m=initial['distance_m'],final_distance_m=final['distance_m'],
                approach_m=initial['distance_m']-final['distance_m'],initial_eef_xyz=initial['eef_xyz'],final_eef_xyz=final['eef_xyz'],
                eef_displacement_m=(np.asarray(final['eef_xyz'])-initial['eef_xyz']).tolist(),
                initial_target_xyz=initial['target_xyz'],final_target_xyz=final['target_xyz'],
                actions=actions.tolist(),normalized_actions=normalized.tolist(),controller_commands=commands,
                first_contact_step=first_contact,first_grasp_step=first_grasp,fps=fps,elapsed_s=time.monotonic()-started,
                replay_steps=replay_steps,warmup_steps=config()['num_steps_wait'],adapter=None if model=='base' else str(SOURCE/f'training/pilot/{model}/final.pt'),
                model_checkpoint=config()['model_dir'],physical_freejoint_intervention=True,
                geometry_translation_in_initial_native_camera_snapshots=True,**resources())
            atomic(gate,attempt/'INTERFACE_CHECK.json');atomic(dict(records=visibility),attempt/'CAMERA.json')
            pd.DataFrame(step_rows).to_parquet(attempt/'STEPS.parquet',index=False)
            atomic(result,folder/'RESULT.json')
            # Stable per-branch names consumed by the independent CPU video exporter.
            import os
            for view in ('front','wrist'):
                dest=folder/(view+'.mp4')
                temporary=folder/('.'+view+'_'+str(time.time_ns())+'.mp4')
                os.link(attempt/(view+'.mp4'),temporary)
                os.replace(temporary,dest)
            finish(folder/'COMPLETE.json',[folder/'RESULT.json',attempt/'STEPS.parquet',attempt/'INTERFACE_CHECK.json',
                attempt/'CAMERA.json',folder/'front.mp4',folder/'wrist.mp4'])
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started,replay_steps=replay_steps,**resources()),
               attempt/'FAILURE.json');raise


def evaluate(model,monitor):
    require_gpu();require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'Prepared data incomplete')
    require(model in config()['models'],'Unknown fixed model')
    m=read(ROOT/'data/MANIFEST.json');todo=[];markers=[]
    for a in m['anchors']:
        require(complete(ROOT/f'data/anchors/{a["episode_id"]}/COMPLETE.json'),'Prepared anchor data incomplete')
        for i,c in enumerate(a['conditions']):
            if c['valid']:
                receipt=ROOT/f'evaluation/{model}/{a["episode_id"]}/{c["name"]}/COMPLETE.json';markers.append(receipt)
                if not complete(receipt):todo.append((a,c,i))
    monitor.update(completed=len(markers)-len(todo),resumed=len(markers)-len(todo),total=len(markers),model=model)
    if not todo:
        finish(ROOT/f'reports/EVALUATE_{model}_COMPLETE.json',markers);return
    import torch
    from wr_engine import Engine
    wrapper=load_wrapper(monitor);engine=Engine(wrapper,gradient_checkpointing=False)
    if model!='base':
        saved=torch.load(SOURCE/f'training/pilot/{model}/final.pt',map_location='cpu',weights_only=False)
        require(saved['signature']==config()['source_signature'] and saved['step']==1000,'Wrong final adapter')
        engine.load_state_dict(saved['adapter'])
    engine.active=model!='base';frozen(engine);checked=set();started=time.monotonic();done=len(markers)-len(todo)
    for j,(a,c,i) in enumerate(todo):
        switch_suite(wrapper,a['suite']);branch(model,engine,a,c,i,monitor,a['suite'] not in checked)
        checked.add(a['suite']);done+=1;monitor.update(phase='short_control',completed=done,episode_id=a['episode_id'],condition=c['name'])
        if j==3:atomic(dict(branches=4,mean_branch_s=(time.monotonic()-started)/4,
                           remaining_estimate_s=(len(todo)-4)*(time.monotonic()-started)/4,**resources()),
                           ROOT/f'reports/THROUGHPUT_{model}.json')
    finish(ROOT/f'reports/EVALUATE_{model}_COMPLETE.json',markers,model_loaded_once=True)
