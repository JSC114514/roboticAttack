# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Repair missing interaction labels by replaying saved controller commands.

No model is loaded and no policy is queried. Original task results, action
traces and completion receipts remain unchanged. Geometry goes into separate
sidecars after reset, EEF trajectory and task-done checks match the original.
One allocated compatible GPU supplies the native camera renderer only.
"""
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,config,read,atomic,complete,finish,require,Monitor,sha
from .runtime import native_runtime,runtime_config
from .evaluate import fingerprint,target_geometry


def missing_episodes():
    output=[]
    for scope in ['development','mechanism_development']:
        for p in sorted((ROOT/'outputs/evaluation'/scope).glob('*/*/RESULT.json')):
            # Earlier M02 float64 controller outputs violated the native
            # interface. Keep those files as error evidence, outside analysis.
            if read(p).get('condition','').startswith('M02-'):continue
            if complete(p.parent/'COMPLETE.json') and not read(p)['geometry_available']:
                output.append((scope,p.parent))
    return output


def repair_one(runtime,scope,folder,monitor):
    destination=folder/'INTERACTION_REPLAY_COMPLETE.json'
    if complete(destination):return read(folder/'INTERACTION_REPLAY.json')
    original=read(folder/'RESULT.json');commands=read(folder/'STEPS.json');decisions=read(folder/'DECISIONS.json')
    env=None;started=time.monotonic()
    try:
        seed=original['seed'];np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
        env,task,initial=runtime.make_env(runtime_config(),original['suite'],original['task_id'])
        raw=runtime.reset_wait(env,initial[original['state']],seed,10)
        reference=ROOT/'data/paired_starts'/('development' if scope=='mechanism_development' else scope)/original['case_id']/'START.json'
        require(fingerprint(env)==read(reference),'Interaction replay reset mismatch')
        truth=target_geometry(env,original['suite'],original['task_id'])
        if truth is None:
            result=dict(status='UNAVAILABLE_TARGET',reason=getattr(env,'_jr04_geometry_error',None),original_result_sha256=sha(folder/'RESULT.json'),model_calls=0)
        else:
            starts={r['step_start']:r['decision'] for r in decisions};before=[];after=[];max_eef_error=0.
            for i,record in enumerate(commands):
                if i in starts:before.append(dict(decision=starts[i],step_start=i,truth=truth()))
                error=float(np.max(np.abs(np.asarray(raw['robot0_eef_pos'])-record['eef_before'])))
                require(error<=1e-8,'Interaction replay EEF-before mismatch')
                raw,reward,done,info=env.step(record['command']);env.sim.forward()
                error=max(error,float(np.max(np.abs(np.asarray(raw['robot0_eef_pos'])-record['eef_after']))));max_eef_error=max(max_eef_error,error)
                require(error<=1e-8,'Interaction replay EEF-after mismatch')
                require(bool(done)==bool(record['task_success']),'Interaction replay task result mismatch')
                after.append(dict(step=record['step'],decision=record['decision'],slot=record['slot'],truth=truth()))
                monitor.update(case_id=original['case_id'],condition=original['condition'],replayed_steps=i+1)
            result=dict(status='MATCHED_REPLAY',original_result_sha256=sha(folder/'RESULT.json'),original_steps_sha256=sha(folder/'STEPS.json'),reset_matches=True,max_eef_error_m=max_eef_error,steps=after,decisions=before,model_calls=0,elapsed_s=time.monotonic()-started)
        output=folder/'INTERACTION_REPLAY.json';atomic(result,output);finish(destination,[output],model_calls=0,metadata_only=True)
        return result
    except BaseException as error:
        atomic(dict(status='FAILED',error=repr(error),original_result_sha256=sha(folder/'RESULT.json'),elapsed_s=time.monotonic()-started,model_calls=0),folder/f'INTERACTION_REPLAY_FAILURE-{time.time_ns()}.json')
        raise
    finally:
        if env is not None:runtime.close_environment(env)


def run(args=None):
    if complete(ROOT/'reports/M03_INTERACTION_REPLAY_COMPLETE.json'):return
    require(os.environ.get('SLURM_JOB_ID'),'Renderer requires allocated Slurm GPU')
    name=torch.cuda.get_device_name(0);kind=next((x for x in ['4090','3090','A800'] if x.lower() in name.lower()),None)
    require(kind is not None,'Unsupported/forbidden renderer GPU: '+name);os.environ['OFT_STAGE_GPU']=kind
    with Monitor('M03-interaction-replay') as monitor:
        runtime=native_runtime();runtime.prepare_graphics_backend();episodes=missing_episodes();results=[]
        resumed=sum(complete(folder/'INTERACTION_REPLAY_COMPLETE.json') for _,folder in episodes)
        # Exclude reused receipts from measured throughput/ETA on continuation.
        monitor.update(completed=resumed,total=len(episodes),resumed=resumed,model_calls=0)
        for scope,folder in episodes:
            value=repair_one(runtime,scope,folder,monitor);results.append(dict(scope=scope,path=str(folder.relative_to(ROOT)),status=value['status'],reason=value.get('reason')));monitor.update(completed=max(resumed,len(results)))
        matched=sum(r['status']=='MATCHED_REPLAY' for r in results)
        require(matched>0 or not episodes,'No interaction replay produced usable labels')
        output=ROOT/'reports/M03_INTERACTION_REPLAY.json';atomic(dict(cases=results,matched=matched,unavailable=len(results)-matched,model_calls=0,original_results_unchanged=True),output)
        finish(ROOT/'reports/M03_INTERACTION_REPLAY_COMPLETE.json',[output],episodes=len(results),model_calls=0)
