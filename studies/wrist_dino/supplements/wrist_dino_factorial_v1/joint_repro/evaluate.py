# 发布阅读说明：运行原生闭环：每决策8×7命令，实际控制用float32。相同种子/初态配对；clean成功才进入CASR分母。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Unified native PNG closed-loop evaluation, no video frames or encoders.

All conditions reset the same physical init/seed. The clean-success denominator
is fixed separately. Per-step commands/EEF/contact and full8x7decisions remain
available for paired analysis and mechanism diagnostics.
"""
import copy
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import ROOT,config,read,atomic,require,finish,complete,sha,save_npz,Monitor,resources
from .runtime import load_wrapper,Engine,switch_suite,runtime_config


def cases(scope='development'):
    c=config();out=[]
    for si,suite in enumerate(c['suites']):
        tasks=c['splits'][suite]['development'] if scope=='development' else list(range(10))
        states=c['dev_states'] if scope=='development' else c['final_states']
        if scope=='confirmation_extra':tasks=c['splits'][suite]['confirmation'];states=c['confirmation_extra_states']
        for task in tasks:
            for state in states:out.append(dict(suite=suite,task_id=task,state=state,case_id=f'{suite}-task{task:02d}-init{state:02d}',seed=c['seed_base']+10000*si+100*task+state))
    return out


def fingerprint(env):
    arrays=[np.asarray(env.get_sim_state())]
    for name in ['geom_xpos','geom_xmat','site_xpos','cam_xpos','cam_xmat']:arrays.append(np.asarray(getattr(env.sim.data,name)))
    h=hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return dict(physical_camera_sha256=h.hexdigest(),time_s=float(env.sim.data.time))


def paste(obs,patch,view='wrist'):
    """Fixed224px rawRGB paste of final8-bitPNG, before full native preprocessing."""
    out=copy.deepcopy(obs)
    if patch is not None:
        x,y=config()['patch']['eval_xy'];key='full_image' if view=='front' else 'wrist_image'
        require(out[key].shape==(224,224,3),'Deployment image shape changed');out[key][y:y+50,x:x+50]=patch
    return out


def target_geometry(env,suite,task):
    """Reuse fixed interaction entity only for diagnostics; no impact on actions."""
    try:
        from .geometry import reader
        source=reader();a=source.anchor(env,suite,task)
        return lambda:source.truth(env,a)
    except Exception as error:
        env._jr04_geometry_error=repr(error)
        return None


def episode(w,engine,r,case,identity,seed,patch,view,monitor,scope,intervention=None):
    c=config();condition='clean' if identity=='clean' else f'{identity}-seed{seed:02d}'
    folder=ROOT/'outputs/evaluation'/scope/condition/case['case_id'];receipt=folder/'COMPLETE.json'
    # Same current-run model/PNG/seed/start may be reused between development
    # and full40-task evaluation. Preserve original receipts via a local link.
    if scope=='full' and not folder.exists():
        source=ROOT/'outputs/evaluation/development'/condition/case['case_id']
        if complete(source/'COMPLETE.json'):
            row=read(source/'RESULT.json')
            require(row['seed']==case['seed'] and row['checkpoint']==c['model_dir'],'Reuse identity differs')
            if patch is not None:require(row['png_sha256']==hashlib.sha256(patch.tobytes()).hexdigest(),'Reused PNG differs')
            # Carry the clean reset/action reference along with reused results;
            # otherwise a later full-scope patch could create its own baseline.
            source_reference=ROOT/'data/paired_starts/development'/case['case_id']
            target_reference=ROOT/'data/paired_starts/full'/case['case_id']
            require((source_reference/'START.json').is_file() and (source_reference/'FIRST_ACTION.npy').is_file(),'Reused start reference missing')
            if not target_reference.exists():
                target_reference.parent.mkdir(parents=True,exist_ok=True);target_reference.symlink_to(source_reference,target_is_directory=True)
            else:
                require(read(target_reference/'START.json')==read(source_reference/'START.json'),'Reused physical start differs')
                require(np.array_equal(np.load(target_reference/'FIRST_ACTION.npy'),np.load(source_reference/'FIRST_ACTION.npy')),'Reused action reference differs')
            folder.parent.mkdir(parents=True,exist_ok=True);folder.symlink_to(source,target_is_directory=True)
    if complete(receipt):return read(folder/'RESULT.json')
    folder.mkdir(parents=True,exist_ok=True);env=None;started=time.monotonic()
    try:
        from experiments.robot.robot_utils import normalize_gripper_action,invert_gripper_action
        def controller(a):
            a=normalize_gripper_action(a,binarize=True);return invert_gripper_action(a)
        switch_suite(w,case['suite']);np.random.seed(case['seed']);__import__('random').seed(case['seed']);torch.manual_seed(case['seed'])
        env,task,initial=r.make_env(runtime_config(),case['suite'],case['task_id'])
        require(case['state']<len(initial),'Insufficient simulator initial states')
        raw=r.reset_wait(env,initial[case['state']],case['seed'],10);obs=r.observation(raw,runtime_config());start=fingerprint(env)
        reference=ROOT/'data/paired_starts'/('development' if scope=='mechanism_development' else scope)/case['case_id'];reference.mkdir(parents=True,exist_ok=True)
        path=reference/'START.json'
        if path.exists():require(start==read(path),'Paired physical/camera reset differs')
        else:atomic(start,path)
        first=engine.native(obs,task.language)
        first_path=reference/'FIRST_ACTION.npy'
        if first_path.exists():require(float(np.max(abs(first-np.load(first_path))))<=1e-5,'Paired first clean action mismatch')
        else:np.save(first_path,first)
        trace=[];steps=[];success=False;step=0;decision=0;limit=c['max_steps'][case['suite']]
        get_truth=target_geometry(env,case['suite'],case['task_id'])
        while step<limit and not success:
            deployed=paste(obs,patch,view)
            restore_error=None
            if intervention is None:actions=engine.native(deployed,task.language)
            else:
                with torch.no_grad():
                    clean_head,donor=engine.action(obs,task.language)
                    keys=['dino','siglip'] if intervention=='both' else [intervention]
                    changed,_=engine.action(deployed,task.language,replacements={(key,1):donor[key][1] for key in keys})
                    # Match Engine.native's controller interface exactly. The
                    # normalization statistics promote numpy outputs tofloat64;
                    # native deployment explicitly converts back tofloat32.
                    actions=np.asarray(w.vla._unnormalize_actions(changed[0].cpu().numpy(),w.gen_cfg.unnorm_key),np.float32)
                    if intervention=='both':
                        expected=engine.native(obs,task.language)
                        restore_error=float(np.max(abs(actions-expected)))
                        require(restore_error<=1e-5,'Both-branch restore differs from same-state clean')
            clean=actions.copy() if patch is None else engine.native(obs,task.language)
            entry=dict(decision=decision,step_start=step,actions=actions.tolist(),same_state_clean=clean.tolist(),phase_proxy=('early','middle','late')[min(2,step*3//limit)])
            entry.update(controller_action_dtype=str(actions.dtype),both_restore_native_action_error=restore_error)
            if get_truth is not None:entry['truth_before']=get_truth()
            used=0
            for slot in range(min(8,limit-step)):
                before=np.array(raw['robot0_eef_pos'],copy=True);command=controller(actions[slot].copy())
                raw,reward,done,info=env.step(command.tolist());env.sim.forward();step+=1;used+=1
                truth=get_truth() if get_truth is not None else None
                steps.append(dict(step=step,decision=decision,slot=slot,command=command.tolist(),eef_before=before.tolist(),eef_after=np.asarray(raw['robot0_eef_pos']).tolist(),truth=truth,task_success=bool(done)))
                obs=r.observation(raw,runtime_config())
                if done:success=True;break
            entry.update(step_end=step,executed_slots=used);trace.append(entry);decision+=1
            monitor.update(case_id=case['case_id'],condition=condition,executed_steps=step)
        atomic(dict(**case,condition=condition,success=bool(success),executed_steps=step,max_steps=limit,model_revision=c['checkpoint_hf_revision'],checkpoint=c['model_dir'],head_step=300000,unnorm_key=w.gen_cfg.unnorm_key,png_sha256=None if patch is None else hashlib.sha256(patch.tobytes()).hexdigest(),native_png=intervention is None,intervention=intervention,video=False,geometry_available=get_truth is not None,geometry_error=getattr(env,'_jr04_geometry_error',None),model_calls=decision*(4 if intervention=='both' else 3 if intervention is not None else 1 if patch is None else 2)+1,elapsed_s=time.monotonic()-started,**resources()),folder/'RESULT.json')
        atomic(trace,folder/'DECISIONS.json');atomic(steps,folder/'STEPS.json')
        finish(receipt,[folder/'RESULT.json',folder/'DECISIONS.json',folder/'STEPS.json'],steps=step)
        return read(folder/'RESULT.json')
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),case=case,identity=identity,elapsed_s=time.monotonic()-started,**resources()),folder/f'FAILURE-{time.time_ns()}.json');raise
    finally:
        if env is not None:r.close_environment(env)


def run(args):
    identity='clean' if args.command=='clean-eval' else args.id
    scope=args.scope;selected=cases(scope)
    if args.smoke:selected=[x for x in selected if x['state'] in config()['dev_states'][:2]]
    name=f'P02-{scope}' if identity=='clean' else f'{identity}-seed{args.seed:02d}-{scope}'
    if args.smoke:name+='-smoke'
    with Monitor(name) as monitor:
        require(complete(ROOT/'reports/P03_COMPLETE.json'),'P03 GPU interface/gradient gate not complete')
        w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False);patch=None;view='wrist'
        if identity!='clean':
            if identity=='INITIAL':p=ROOT/'outputs/preflight/initial.png'
            else:
                from .methods import spec
                view=spec(identity)['view'];p=ROOT/'outputs/training'/f'{identity}-seed{args.seed:02d}'/'final.png'
                require(complete(p.parent/'COMPLETE.json'),'Final trained patch not complete')
            patch=np.asarray(Image.open(p).convert('RGB'),np.uint8);require(patch.shape==(50,50,3),'PNG dimensions differ')
        # Model loading has finished. Replace its heartbeat label so live
        # reports describe evaluation rather than stale loading progress.
        results=[];monitor.update(phase='evaluating_'+scope,completed=0,total=len(selected))
        for case in selected:
            result=episode(w,engine,r,case,identity,args.seed,patch,view,monitor,scope);results.append(result);monitor.update(completed=len(results))
        summary=ROOT/'reports'/f'{name}_RESULTS.json';atomic(dict(cases=results,successes=sum(x['success'] for x in results),count=len(results),scope=scope,smoke=args.smoke),summary)
        destination=ROOT/'reports'/f'P02_{scope}_COMPLETE.json' if identity=='clean' and not args.smoke else ROOT/'reports'/f'{name}_COMPLETE.json'
        finish(destination,[summary],episodes=len(results))
        if identity=='clean' and scope=='development' and not args.smoke:
            # Reuse the same loaded model for the16fixed initialPNG interface
            # smoke episodes; outcome never filters starting states.
            initial=ROOT/'outputs/preflight/initial.png'
            patch=np.asarray(Image.open(initial).convert('RGB'),np.uint8)
            smoke_cases=[x for x in selected if x['state'] in c['dev_states'][:2]] if (c:=config()) else []
            smoke_results=[]
            monitor.update(phase='initialPNG_16_smoke',completed=0,total=len(smoke_cases))
            for case in smoke_cases:
                smoke_results.append(episode(w,engine,r,case,'INITIAL',42,patch,'wrist',monitor,'development'))
                monitor.update(completed=len(smoke_results))
            require(len(smoke_results)==16,'Fixed smoke set coverage changed')
            out=ROOT/'reports/SMOKE_16_RESULTS.json';atomic(dict(cases=smoke_results,engineering_complete=True),out)
            finish(ROOT/'reports/SMOKE_16_COMPLETE.json',[out],episodes=16)

