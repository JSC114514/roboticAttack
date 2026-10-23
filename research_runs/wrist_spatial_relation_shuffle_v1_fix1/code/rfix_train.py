"""One candidate and one fully GPU-resident frozen policy per process."""
from rfix_common import ROOT,TOOLS,config,source,native,require,atomic,candidates,training_module


def residency(wrapper):
    components={name:getattr(wrapper,name,None) for name in ('vla','action_head','proprio_projector','noisy_action_projector')}
    devices={};maps={};offload=[]
    for name,module in components.items():
        if module is None:continue
        for p in module.parameters():
            key=str(p.device);devices[key]=devices.get(key,0)+p.numel()
        for path,child in module.named_modules():
            mapping=getattr(child,'hf_device_map',None)
            if mapping:
                maps[name+'.'+path]={k:str(v) for k,v in mapping.items()}
                for value in mapping.values():
                    text=str(value)
                    if text in ('cpu','disk','meta'):offload.append(name+'.'+path+':'+text)
            hook=getattr(child,'_hf_hook',None)
            if hook is not None and getattr(hook,'offload',False):offload.append(name+'.'+path+':offload_hook')
    cuda=[d for d in devices if d.startswith('cuda:')]
    require(devices and len(cuda)==1 and len(devices)==1 and not offload,
            'Policy must be fully on one GPU; refusing CPU/disk/meta offload: '+str({'devices':devices,'offload':offload}))
    return {'parameter_elements_by_device':devices,'hf_device_maps':maps,'offload':offload,'single_gpu_resident':True}


_loaded=False


def make_engine(checkpointing=False):
    global _loaded
    require(not _loaded,'Only one policy load is permitted per worker process')
    _loaded=True
    wrapper=native.load_wrapper('libero_goal')
    try:proof=residency(wrapper)
    except BaseException as error:
        atomic({'status':'FAILED','error':repr(error)},ROOT/'reports/GATES'/f'residency_{__import__("os").environ.get("SLURM_JOB_ID","local")}.json');raise
    atomic({'status':'PASS',**proof},ROOT/'reports/GATES'/f'residency_{__import__("os").environ.get("SLURM_JOB_ID","local")}.json')
    # Private copy of the original PNG/action interface gate, with new output paths.
    module=source.module('rfix_action_engine_private',TOOLS/'sf_train.py')
    module.ROOT=ROOT;module.config=config
    return module.Engine(wrapper,checkpointing)


def run(index,monitor):
    seed=config()['seeds'][index]
    selected=[i for i,(name,method,s) in enumerate(candidates()) if method=='shuffled_relation' and s==seed]
    require(len(selected)==1,'Expected one resumed candidate per worker')
    training_module().run(selected[0],monitor)


def select(monitor):training_module().select(monitor)
