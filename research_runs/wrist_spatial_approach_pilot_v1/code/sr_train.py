"""Four independently resumable candidates, fixed actual-update budget."""
import json
import random
import time
import numpy as np
import torch
from sr_common import (ROOT,native,config,candidates,records,complete,finish,signature,
    make_engine,require,atomic_torch,atomic_json,read_json)
from sr_math import learning_rate

def checkpoints(name):
    out={}
    for p in sorted((ROOT/f'training/{name}').glob('checkpoint_*.json')):
        if complete(p): out[int(read_json(p)['step'])]=p
    return out

def save(folder,patch,optimizer,step,optimized,no_target,method,seed,terminal=False):
    path=folder/f'checkpoint_{step:06d}.pt'
    value={'signature':signature(),'method':method,'seed':seed,'step':step,
        'optimized':optimized,'no_target':no_target,'patch':patch.detach().cpu(),
        'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    atomic_torch(value,path);native.save_png(patch,path.with_suffix('.png'))
    finish(path.with_suffix('.json'),[path,path.with_suffix('.png')],step=step,optimized=optimized,
        no_target=no_target,method=method,seed=seed,terminal=terminal)
    atomic_torch(value,folder/'LAST.pt')
    return path.with_suffix('.json')

def run(index,monitor):
    name,method,seed=candidates()[index];folder=ROOT/f'training/{name}'
    folder.mkdir(parents=True,exist_ok=True)
    if complete(folder/'COMPLETE.json'): return
    native.require_gpu('action_teacher_train')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    pool=[r for r in records('fit') if not r['teacher']['no_target']]
    require(pool,'No eligible fit frames')
    grouped={}
    for r in pool:grouped.setdefault(r['suite'],{}).setdefault(r['task_id'],[]).append(r)
    require(set(grouped)==set(config()['suites']),'Eligible fit pool lost a suite; refuse to disguise single-suite training')
    engine=make_engine(checkpointing=True)
    initial=native.load_png(ROOT/f'data/initial_s{seed}.png',engine.wrapper.device)
    patch=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([patch],lr=config()['learning_rate'])
    start=optimized=no_target=0;last=folder/'LAST.pt'
    if last.exists():
        v=torch.load(last,map_location='cpu',weights_only=False)
        require(v['signature']==signature() and v['method']==method and v['seed']==seed,'Resume identity changed')
        require(v['optimized']+v['no_target']==v['step'],'Resume counters changed')
        with torch.no_grad():patch.copy_(v['patch'].to(patch.device))
        optimizer.load_state_dict(v['optimizer']);start=v['step'];optimized=v['optimized'];no_target=v['no_target']
        state=optimizer.state.get(patch,{})
        require(int(state.get('step',0))==optimized,'Adam moments/step changed')
    # Non-scientific interface gate at worker start, not a separate experiment.
    gate_row=next(r for r in pool if not r['teacher']['no_target'])
    engine.gate(gate_row,patch,method,name)
    if last.exists():
        torch.set_rng_state(v['torch_rng']);torch.cuda.set_rng_state_all(v['cuda_rng'])
        np.random.set_state(v['numpy_rng']);random.setstate(v['python_rng'])
    monitor.update(phase='training',candidate=name,method=method,seed=seed,completed=start,
        resume_completed=start,total=config()['updates'],excluded_frames=sum(r['teacher']['no_target'] for r in records('fit')))
    step=start;current_visit=None;order=None
    log=folder/f'ATTEMPT_{time.time_ns()}.jsonl'
    try:
        with log.open('x') as stream:
            while step<config()['updates']:
                sampler=random.Random(seed*1000000+step)
                suite=config()['suites'][step%len(config()['suites'])]
                group=grouped[suite];task=sampler.choice(sorted(group));row=sampler.choice(group[task]);step+=1
                lr=learning_rate(step,config()['updates'],config()['warmup'],config()['learning_rate'])
                for group in optimizer.param_groups:group['lr']=lr
                if row['teacher']['no_target']:
                    no_target+=1;metrics={'optimized':False,'no_target':True,'loss':None,'gradient_l2':0.,'pixel_step_l2':0.}
                else:
                    before=patch.detach().clone();optimizer.zero_grad(set_to_none=True)
                    gradient,metrics=engine.staged_gradient(row,patch,method)
                    patch.grad=gradient;optimizer.step()
                    with torch.no_grad():patch.clamp_(0,1)
                    optimizer.zero_grad(set_to_none=True);optimized+=1
                    metrics.update(optimized=True,no_target=False,pixel_step_l2=float((patch.detach()-before).norm()),
                        pixel_from_initial_l2=float((patch.detach()-initial).norm()))
                stream.write(json.dumps({'step':step,'state_id':row['state_id'],'learning_rate':lr,**metrics})+'\n')
                if step%25==0:
                    stream.flush();monitor.update(completed=step,optimized_updates=optimized,no_target_updates=no_target,**metrics)
                if step%config()['checkpoint_interval']==0:
                    save(folder,patch,optimizer,step,optimized,no_target,method,seed)
    except BaseException:
        # Completed update boundary only; failed update does not become a resumable step.
        # Last sealed checkpoint stays authoritative after an exception.
        raise
    receipt=save(folder,patch,optimizer,step,optimized,no_target,method,seed,terminal=True)
    require(optimized+no_target==step,'Terminal counters invalid')
    finish(folder/'COMPLETE.json',[receipt,folder/'LAST.pt',*folder.glob('ATTEMPT_*.jsonl')],
        candidate=name,step=step,optimized=optimized,no_target=no_target,
        budget_complete=step==config()['updates'],stop_reason='budget')
    monitor.update(phase='training_finished',completed=step,total=config()['updates'],optimized_updates=optimized,
        no_target_updates=no_target,budget_complete=step==config()['updates'])

def select(monitor):
    selected={};pairs=[]
    for seed in config()['seeds']:
        names=[f'{m}_s{seed}' for m in config()['methods']]
        available={name:checkpoints(name) for name in names}
        common=set.intersection(*(set(available[n]) for n in names))
        # Only scheduled boundaries count as matched-budget checkpoints.
        common={s for s in common if s>0 and s%config()['checkpoint_interval']==0}
        shared=max(common) if common else None
        require(shared==config()['updates'],'Final predeclared update unavailable; refuse partial-budget selection')
        pair={'seed':seed,'common_step':shared,'matched_budget':shared is not None,'candidates':{}}
        for name in names:
            steps=available[name]
            step=shared if shared is not None else max(steps,default=0)
            if step<=0: pair['candidates'][name]={'status':'NO_TRAINED_CHECKPOINT'};continue
            receipt=steps[step];v=read_json(receipt)
            if v['optimized']==0:continue
            path=receipt.with_suffix('.png')
            selected[name]={'path':str(path),'png_sha256':__import__('sr_common').sha(path),
                'step':step,'optimized':v['optimized'],'no_target':v['no_target'],
                'receipt':str(receipt),'matched_budget':shared is not None,'seed':seed}
            pair['candidates'][name]=selected[name]
        pairs.append(pair)
    require(len(selected)==len(candidates()),'All four final PNGs are required before evaluation')
    p=ROOT/'data/SELECTED.json';atomic_json({'signature':signature(),'primary':config()['primary'],
        'selection_rule':'predeclared_final_update_never_validation_score','candidates':selected,'pairs':pairs},p)
    finish(ROOT/'reports/SELECT_COMPLETE.json',[p,*[Path(v['receipt']) for v in selected.values()]])
    monitor.update(phase='selected_by_budget_only',completed=len(selected),total=len(candidates()))

from pathlib import Path
