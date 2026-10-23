"""Reuse the exact sampler/Adam/PNG loop; one candidate/model per process."""
import ast
import copy
import numpy as np
import torch
from fc_common import ROOT,PROJECT,SOURCE,source,native,require,atomic,config,module,signature,candidates,records,complete,finish,verify
from fc_math import action_loss
from sr_engine import Engine as SourceEngine


class Engine(SourceEngine):
    pass


def install_audited_methods():
    # Keep the audited two-branch VJP unchanged except for the scalar objective.
    path=SOURCE/'code/sr_engine.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='staged_gradient')
    changed=0
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in node.targets):
            require(ast.unparse(node.value)=="objective(command, row['teacher'], method)",'Audited VJP objective changed')
            node.value=ast.parse("action_loss(action, row['teacher']['clean'], method)",mode='eval').body;changed+=1
    require(changed==1,'Expected one scalar loss substitution')
    space=dict(globals(),ref=source.reference_engine_module())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),space)
    Engine.staged_gradient=space['staged_gradient']
    path=PROJECT/'research_tools/spatial_followup/sf_train.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
    gate=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='gate')
    exec(compile(ast.Module(body=[gate],type_ignores=[]),str(path),'exec'),space)
    Engine.gate=space['gate']


install_audited_methods()
_loaded=False

def make_engine(checkpointing=False):
    global _loaded
    require(not _loaded,'One model load per process');_loaded=True
    wrapper=native.load_wrapper('libero_goal');devices={};offload=[]
    for name in ('vla','action_head','proprio_projector','noisy_action_projector'):
        model=getattr(wrapper,name,None)
        if model is None:continue
        require(not model.training and all(not p.requires_grad for p in model.parameters()),'Model not frozen/eval')
        for p in model.parameters():devices[str(p.device)]=devices.get(str(p.device),0)+p.numel()
        for key,child in model.named_modules():
            if any(str(v) in ('cpu','disk','meta') for v in (getattr(child,'hf_device_map',{}) or {}).values()):offload.append(name+'.'+key)
            if getattr(getattr(child,'_hf_hook',None),'offload',False):offload.append(name+'.'+key)
    require(len(devices)==1 and next(iter(devices)).startswith('cuda:') and not offload,'CPU/disk/meta offload prohibited')
    import os
    atomic({'status':'PASS','devices':devices,'offload':offload},ROOT/f'reports/GATES/residency_{os.environ.get("SLURM_JOB_ID","local")}.json')
    return Engine(wrapper,checkpointing)

def training_module():
    value=module('fc_private_training_loop',SOURCE/'code/sr_train.py')
    for key,item in dict(ROOT=ROOT,config=config,candidates=candidates,records=records,complete=complete,
                         finish=finish,signature=signature,make_engine=make_engine).items():setattr(value,key,item)
    return value

def run(index,monitor):
    verify();training_module().run(index,monitor)
