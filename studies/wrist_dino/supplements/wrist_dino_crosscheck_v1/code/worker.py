# 发布阅读说明：固定PNG的原生闭环与同状态分支恢复，按完整回合续跑。内部恢复单列；Long09采用已核验的新环境重置修复。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Native cross-model closed loops and same-state Adapter branch restoration.

One process loads one target-family checkpoint once for one suite. All PNGs
share reset/noise addresses. No model weights, patch pixels or hyperparameters
are optimized. Hooks exist only inside the explicitly labelled diagnostic.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import os
import random
import time
import numpy as np
from PIL import Image
from common import ROOT,frozen,read,atomic,sha,done,complete,require,Monitor
from contracts import paste_wrist
from video import Videos

def fingerprint(env):
    h=hashlib.sha256()
    for a in [env.get_sim_state(),env.sim.data.geom_xpos,env.sim.data.geom_xmat,env.sim.data.site_xpos,env.sim.data.cam_xpos,env.sim.data.cam_xmat]:h.update(np.ascontiguousarray(a).tobytes())
    return dict(physical_camera_sha256=h.hexdigest(),time_s=float(env.sim.data.time))

def predicted(policy,obs,language,case,decision):
    return policy.predict(obs,language,suite=case['suite'],task_id=case['task_id'],state_index=case['state_index'],decision=decision)

def adapter_restore(policy,clean_obs,patched_obs,language,case,decision,branch):
    """Restore only second-view outputs, preserving front/CLS/model operators."""
    backbone=policy.model.vision_backbone
    modules={'dino':backbone.featurizer,'siglip':backbone.fused_featurizer}
    donor={k:[] for k in modules};handles=[]
    try:
        for k,m in modules.items():handles.append(m.register_forward_hook(lambda _m,_i,o,k=k:donor[k].append(o)))
        clean=predicted(policy,clean_obs,language,case,decision)
    finally:
        for h in handles:h.remove()
    require(all(len(v)==2 for v in donor.values()),'Adapter native visual routing differs')
    counts={k:0 for k in modules};handles=[];keys=list(modules) if branch=='both' else [branch]
    try:
        for k in keys:
            def hook(_m,_i,out,k=k):
                index=counts[k];counts[k]+=1
                return donor[k][1] if index==1 else None
            handles.append(modules[k].register_forward_hook(hook))
        changed=predicted(policy,patched_obs,language,case,decision)
    finally:
        for h in handles:h.remove()
    require(all(counts[k]==2 for k in keys),'Replacement did not hit both native views')
    error=float(np.max(abs(changed-clean)))
    if branch=='both':require(error<=1e-5,'Dual restore != same-state native clean')
    return changed,error

def load_image(entry):
    if entry is None:return None
    require(sha(entry['png'])==entry['sha256'],'Frozen PNG changed')
    return np.asarray(Image.open(entry['png']).convert('RGB'),np.uint8)

def run_episode(env,task,initial,policy,model,case,entry,monitor,branch=None):
    from graphics_runtime import reset_wait
    cfg=frozen()['config'];condition='clean' if entry is None else entry['id'];name=condition if branch is None else condition+'-restore-'+branch
    folder=ROOT/f'outputs/{model}/{name}/{case["case_id"]}'
    if complete(folder/'COMPLETE.json'):return read(folder/'RESULT.json')
    folder.mkdir(parents=True,exist_ok=True);seed=case['seed']
    np.random.seed(seed);random.seed(seed)
    if model=='vla_adapter_pro':__import__('torch').manual_seed(seed)
    raw=reset_wait(env,initial[case['state_index']],seed,10);start=fingerprint(env)
    ref=ROOT/f'outputs/paired_starts/{model}/{case["case_id"]}.json'
    if ref.exists():require(start==read(ref),'Paired target start differs')
    else:atomic(start,ref)
    patch=load_image(entry);obs=policy.prepare(raw);kwargs=dict(suite=case['suite'],task_id=case['task_id'],state_index=case['state_index'],decision=0)
    # Determinism/native identity check on each start. pi05 repeats exactnoise.
    first=predicted(policy,obs,task.language,case,0)
    first_path=ref.with_suffix('.npy')
    if first_path.exists():
        error=float(np.max(abs(first-np.load(first_path))))
        if error>1e-5:
            np.savez_compressed(folder/'FIRST_ACTION_MISMATCH.npz',expected=np.load(first_path),actual=first,**obs)
            atomic(dict(max_abs=error,start=start,first_reference=str(first_path),case=case),folder/'FIRST_ACTION_MISMATCH.json')
        require(error<=1e-5,'Target first clean action differs')
    else:np.save(first_path,first)
    if not (ROOT/f'reports/INTERFACE_{model}_{case["suite"]}.json').exists():
        zero=predicted(policy,paste_wrist(obs,None),task.language,case,0);require(np.max(abs(first-zero))<=1e-5,'Zero paste != native action')
        checks=dict(native_zero_error=float(np.max(abs(first-zero))),physical_start=start)
        if model=='vla_adapter_pro':
            witness=next(x for x in cfg['patches'] if x['id']=='LR004-O03-seed42');p=load_image(witness)
            same,e=adapter_restore(policy,obs,obs,task.language,case,0,'dino');require(e<=1e-5,'Clean DINO identity failed')
            both,e=adapter_restore(policy,obs,paste_wrist(obs,p),task.language,case,0,'both');checks['dual_restore_error']=e
            checks['weights_frozen']=all(not t.requires_grad for m in [policy.model,policy.head,policy.proprio] if m is not None for t in m.parameters());require(checks['weights_frozen'],'Target weights not frozen')
        atomic(checks,ROOT/f'reports/INTERFACE_{model}_{case["suite"]}.json')
    videos=Videos('cross_model/'+model,name,case['case_id'],getattr(env,'control_freq',getattr(env.env,'control_freq',20)))
    commands=[];decisions=[];step=decision=0;success=False;tick=time.monotonic()
    try:
        while step<cfg['max_steps'][case['suite']] and not success:
            clean_obs=policy.prepare(raw);deployed=paste_wrist(clean_obs,patch)
            if branch is None:actions=predicted(policy,deployed,task.language,case,decision);restore_error=None
            else:actions,restore_error=adapter_restore(policy,clean_obs,deployed,task.language,case,decision,branch)
            environment=policy.environment_actions(actions)
            require(environment.shape==(policy.action_horizon,7) and np.isfinite(environment).all(),'Bad native controller actions')
            used=0;before_step=step
            for command in environment[:min(policy.replan_steps,cfg['max_steps'][case['suite']]-step)]:
                # Record exactly the image input at this realized controlstep.
                videos.append(paste_wrist(policy.prepare(raw),patch))
                before=np.asarray(raw['robot0_eef_pos']).copy();raw,reward,ended,info=env.step(command.tolist());step+=1;used+=1
                commands.append(dict(step=step,decision=decision,command=command.tolist(),eef_before=before.tolist(),eef_after=np.asarray(raw['robot0_eef_pos']).tolist(),task_success=bool(ended)))
                if ended:success=True;break
            decisions.append(dict(decision=decision,step_start=before_step,executed_slots=used,predicted=actions.tolist(),environment=environment.tolist(),restore_error=restore_error));decision+=1
            monitor.update(case=case['case_id'],condition=name,control_steps=step)
        video=videos.close(success)
        result=dict(**case,model=model,condition=name,success=success,executed_steps=step,native_png=branch is None,intervention=branch,png_sha256=entry['sha256'] if entry else None,elapsed_s=time.monotonic()-tick,video=video,checkpoint=cfg['models'][model].get('checkpoint',cfg['models'][model].get('checkpoints',{}).get(case['suite'])))
        atomic(result,folder/'RESULT.json');atomic(commands,folder/'STEPS.json');atomic(decisions,folder/'DECISIONS.json')
        files=[folder/x for x in ['RESULT.json','STEPS.json','DECISIONS.json']]+[ROOT/p for p in video['files']]
        done(folder/'COMPLETE.json',files);return result
    except BaseException as e:
        videos.abort();atomic(dict(error=repr(e),step=step,elapsed_s=time.monotonic()-tick),folder/f'FAILURE-{time.time_ns()}.json');raise

def run(model,suite,mode):
    require(os.environ.get('SLURM_JOB_ID'),'Native GPU work requires Slurm')
    a=frozen();cfg=a['config']
    for p,v in a['assets'].items():
        st=Path(p).stat();require(st.st_size==v['size'] and str(st.st_mtime_ns)==v['mtime_ns'],'Frozen target asset changed: '+p)
    from graphics_runtime import prepare_graphics_backend,make_env,close_environment
    from policies import load_policy
    cases=[x for x in a['cases'] if x['suite']==suite]
    conditions=[None,*cfg['patches']] if mode=='transfer' else [x for x in cfg['patches'] if x['id'] in ['LR004-B01-seed42','LR004-O03-seed42']]
    branches=[None] if mode=='transfer' else ['dino','siglip','both']
    name=f'{model}-{suite}-{mode}'
    if complete(ROOT/f'reports/units/{name}.json'):return
    resumed=0
    for case in cases:
        for entry in conditions:
            for branch in branches:
                label='clean' if entry is None else entry['id'];condition=label if branch is None else label+'-restore-'+branch
                resumed+=int((ROOT/f'outputs/{model}/{condition}/{case["case_id"]}/COMPLETE.json').is_file())
    with Monitor(name) as monitor:
        monitor.update(phase='loading_native_policy',total=len(cases)*len(conditions)*len(branches),completed=resumed,resumed=resumed)
        prepare_graphics_backend();policy=load_policy(model,suite,cfg);completed=0;env=None;task_id=None
        if model=='pi05':
            import common,jax
            common.JAX_DEVICE=jax.devices()[0]
        try:
            for case in cases:
                # Long09 reconstructs static fixture body_pos during reset;
                # set_init_state only restores qpos/qvel, so reusing the same
                # simulator drifts fixtures by millimetres despite identical
                # flattened sim state. Use fresh native envs for this concrete
                # task, retaining the original seed/state/hash/action checks.
                fresh=(suite=='libero_10' and case['task_id']==9)
                if not fresh and task_id!=case['task_id']:
                    if env is not None:close_environment(env)
                    env,task,initial=make_env({'paths':{'libero_root':cfg['libero_root']},'env_image_resolution':256},suite,case['task_id']);task_id=case['task_id']
                # Always collect own clean first. Remainingcondition order is
                # fixed and independent of outcomes; RNG reset pairs eachstart.
                for entry in conditions:
                    for branch in branches:
                        label='clean' if entry is None else entry['id'];condition_name=label if branch is None else label+'-restore-'+branch
                        pending=not complete(ROOT/f'outputs/{model}/{condition_name}/{case["case_id"]}/COMPLETE.json')
                        if fresh and pending:
                            if env is not None:close_environment(env)
                            np.random.seed(case['seed']);random.seed(case['seed'])
                            if model=='vla_adapter_pro':__import__('torch').manual_seed(case['seed'])
                            env,task,initial=make_env({'paths':{'libero_root':cfg['libero_root']},'env_image_resolution':256},suite,case['task_id']);task_id=case['task_id']
                        run_episode(env,task,initial,policy,model,case,entry,monitor,branch);completed+=1;monitor.update(phase='evaluating_'+mode,completed=completed)
            done(ROOT/f'reports/units/{name}.json',episodes=completed)
        finally:
            if env is not None:close_environment(env)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--suite',required=True);p.add_argument('--mode',default='transfer',choices=['transfer','mechanism']);args=p.parse_args();run(args.model,args.suite,args.mode)
