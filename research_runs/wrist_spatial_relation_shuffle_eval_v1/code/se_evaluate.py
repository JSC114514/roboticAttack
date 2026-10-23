"""Four native PNGs, all starts; one frozen policy per worker, fresh paired environments."""
import ast
import copy
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from se_common import ROOT,PROJECT,BASE,config,source,native,read,sha,atomic,require,complete,finish,verify,verify_external


def conditions():
    assets=read(ROOT/'data/MANIFEST.json')['assets']
    for a in assets:require(sha(a['path'])==a['png_sha256'], 'Final PNG changed')
    return {a['condition']:Path(a['path']) for a in assets}


def rollout_functions():
    from audit_replay import SensorCapture,fingerprint
    from audit_math import visibility,event_summary
    from rr_logger import PhysicsLogger
    from sr_geometry import anchor,truth,controller
    path=PROJECT/'research_runs/wrist_spatial_relation_shuffle_v1_fix1/code/rfix_evaluate.py'
    names={'CameraWriters','paired_start','episode'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==names, 'Audited rollout adapter changed')
    space=dict(globals(),SensorCapture=SensorCapture,fingerprint=fingerprint,visibility=visibility,
               event_summary=event_summary,PhysicsLogger=PhysicsLogger,anchor=anchor,truth=truth,controller=controller)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),space)
    return space['episode']


def constructor_rng():
    return random.getstate(),np.random.get_state(),torch.get_rng_state().clone()


def restore_constructor_rng(value):
    random.setstate(value[0]);np.random.set_state(value[1]);torch.set_rng_state(value[2])


def load_wrapper(index):
    wrapper=native.load_wrapper('libero_goal');devices={};offload=[]
    for name in ('vla','action_head','proprio_projector','noisy_action_projector'):
        model=getattr(wrapper,name,None)
        if model is None:continue
        require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model not frozen/eval')
        for p in model.parameters():devices[str(p.device)]=devices.get(str(p.device),0)+p.numel()
        for path,child in model.named_modules():
            for device in (getattr(child,'hf_device_map',{}) or {}).values():
                if str(device) in ('cpu','disk','meta'):offload.append(name+'.'+path)
            if getattr(getattr(child,'_hf_hook',None),'offload',False):offload.append(name+'.'+path)
    require(len(devices)==1 and next(iter(devices)).startswith('cuda:') and not offload, 'CPU/disk/meta offload prohibited')
    atomic({'status':'PASS','devices':devices,'offload':offload,'one_model_load':True},ROOT/f'reports/GATES/residency_{index}.json')
    return wrapper


class FirstCleanCheck:
    """Use the evaluator's first clean call; no extra camera sampling or simulation steps."""
    def __init__(self,wrapper,expected):self.wrapper=wrapper;self.expected=expected;self.calls=0;self.error=None
    def predict_action(self,*args,**kwargs):
        value=self.wrapper.predict_action(*args,**kwargs);self.calls+=1
        if self.calls==2:
            self.error=float(np.max(np.abs(np.asarray(value,dtype=np.float32).reshape(8,7)-self.expected)))
            require(self.error<=config()['action_tolerance'], 'Historical first clean action differs: '+str(self.error))
        return value


def run_case(runtime,wrapper,episode,case,paths,patches,monitor,index):
    reference=verify_external(BASE,BASE/'episodes'/case['start']/'clean/COMPLETE.json')
    trace=next(BASE/p for p in reference['artifacts'] if p.endswith('/TRACE.parquet'))
    expected=np.stack(pd.read_parquet(trace).iloc[0]['predicted_actions']).astype(np.float32).reshape(8,7)
    construction_state=constructor_rng();receipts=[]
    for name in paths:
        receipt=ROOT/'episodes'/case['start']/name/'COMPLETE.json'
        if complete(receipt):receipts.append(receipt);continue
        restore_constructor_rng(construction_state)
        env=None
        try:
            env,task,states=runtime.make_env(native.runtime_config(),case['suite'],case['task_id'])
            require(task.name==case['task_name'], 'Task mapping changed')
            initial=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
            require(np.array_equal(initial,states[case['state_index']]), 'Initial file differs')
            attempt=receipt.parent/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
            checked=FirstCleanCheck(wrapper,expected)
            episode(env,task,initial,case,name,patches[name],checked,attempt,monitor,BASE/'paired_starts'/case['start'])
            require(checked.error is not None, 'First clean action check was not executed')
            gate=attempt/'INTERFACE_CHECK.json'
            atomic({'status':'PASS','first_clean_action_max_abs_error':checked.error,'tolerance':config()['action_tolerance'],
                    'environment_isolation':'fresh_environment_for_each_condition','constructor_rng_restored':True},gate)
            finish(receipt,[attempt/p for p in ('START.npz','EPISODE.json','TRACE.parquet','STEPS.parquet','EVENTS.parquet','front.mp4','wrist.mp4','INTERFACE_CHECK.json')])
            receipts.append(receipt)
            monitor.update(phase='native_PNG_closedloop',completed=monitor.data.get('completed',0)+1)
        finally:
            if env is not None:runtime.close_environment(env)
    return receipts


def run(index,monitor):
    receipt=ROOT/f'reports/EVALUATE_{index}_COMPLETE.json'
    if complete(receipt):return
    value=verify();paths=conditions();assigned=value['cases'][index::2]
    receipts=[ROOT/'episodes'/c['start']/name/'COMPLETE.json' for c in assigned for name in paths]
    already=sum(complete(p) for p in receipts)
    monitor.update(completed=already,resume_completed=already,total=len(receipts),phase='loading_frozen_policy')
    if already!=len(receipts):
        native.require_gpu('relation_shuffle_evaluate');runtime=native.runtime_module();runtime.prepare_graphics_backend()
        wrapper=load_wrapper(index);episode=rollout_functions()
        from PIL import Image
        patches={name:np.array(Image.open(path).convert('RGB'),copy=True) for name,path in paths.items()}
        for case in assigned:
            if all(complete(ROOT/'episodes'/case['start']/name/'COMPLETE.json') for name in paths):continue
            run_case(runtime,wrapper,episode,case,paths,patches,monitor,index)
    require(all(complete(p) for p in receipts), 'Worker episodes incomplete')
    finish(receipt,receipts,new_closedloop_episodes=len(receipts))
    monitor.update(phase='COMPLETE',completed=len(receipts),total=len(receipts))
