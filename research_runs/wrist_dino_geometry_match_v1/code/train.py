"""Paired full-feature versus projection objectives, no action loss or EOT."""
from __future__ import annotations
import math,random,time
import numpy as np
import torch
from common import (ROOT,cfg,signature,require,complete,finish,initial_patch,save_png,
                    atomic_torch,atomic_json,read_json,load_wrapper,resources,require_gpu)
from data import compact_pilot,load_shard,index_rows,iter_records
from engine import Engine,signed_projection

def objective(patched,negative,positive,kind):
    result=signed_projection(patched,negative,positive)
    if result is None:return None
    projection,q,orthogonal,energy=result
    require(kind in ('projection','full'),'Unknown feature objective')
    loss=projection if kind=='projection' else projection+orthogonal
    return loss,{'loss':float(loss.detach()),'projection_loss':float(projection.detach()),
                 'q':float(q.detach()),'orthogonal':float(orthogonal.detach()),'target_energy':energy}

def lr(step,total):
    c=cfg();w=c['warmup_updates']
    return c['learning_rate']*step/w if step<=w else c['learning_rate']*.5*(1+math.cos(math.pi*(step-w)/(total-w)))

def update(engine,patch,optimizer,row,kind,rate,initial):
    for group in optimizer.param_groups:group['lr']=rate
    optimizer.zero_grad(set_to_none=True);before=patch.detach().clone()
    spatial,_=engine.spatial_patch(row['observation'],patch)
    targets=row['target_features']
    result=objective(spatial,targets['translation_minus'],targets['translation_plus'],kind)
    if result is None:
        return {'optimized':False,'no_target':True,'gradient_l2':0.,'loss':None,'q':None,'orthogonal':None,
                'projection_loss':None,'target_energy':0.,'pixel_step_l2':0.,
                'pixel_from_initial_l2':float((patch.detach()-initial).norm())}
    loss,values=result
    require(torch.isfinite(loss),'Nonfinite feature loss')
    loss.backward()
    require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Nonfinite patch gradient')
    grad=float(patch.grad.norm());optimizer.step()
    with torch.no_grad():patch.clamp_(0,1)
    require(all(p.grad is None for p in engine.wrapper.vla.parameters()),'Frozen VLA accumulated gradients')
    return {'optimized':True,'no_target':False,'gradient_l2':grad,
            'pixel_step_l2':float((patch.detach()-before).norm()),
            'pixel_from_initial_l2':float((patch.detach()-initial).norm()),**values}

def make_engine():
    wrapper=load_wrapper(cfg()['training_checkpoint_suite']);engine=Engine(wrapper)
    row=next(iter_records('fit'));engine.verify_patch_path(row)
    return engine

def run_pilot(kind,monitor):
    require_gpu('pilot_'+kind);folder=ROOT/'training/pilot'/kind
    if complete(folder/'COMPLETE.json'):return
    engine=make_engine();cohorts=compact_pilot();rows=cohorts['fit'];visits=cfg()['pilot_visits']
    folder.mkdir(parents=True,exist_ok=True)
    if kind=='individual':
        units=[(loss,row) for loss in ('projection','full') for row in rows]
        total=len(units)*visits;completed=0;artifacts=[]
        for loss,row in units:
            target=folder/loss/row['state_id'];receipt=target/'COMPLETE.json'
            if complete(receipt):completed+=visits;artifacts.append(receipt);continue
            initial=initial_patch(engine.wrapper.device,42);patch=torch.nn.Parameter(initial.clone())
            opt=torch.optim.Adam([patch],lr=cfg()['pilot_lr']);optimized=0;no_target=0
            for step in range(visits):
                values=update(engine,patch,opt,row,loss,cfg()['pilot_lr'],initial)
                optimized+=int(values['optimized']);no_target+=int(values['no_target']);completed+=1
                if step%20==0:monitor.update(phase=loss,state_id=row['state_id'],completed=completed,total=total,**values)
            png=target/'final.png';save_png(patch,png)
            finish(receipt,[png],loss=loss,state_id=row['state_id'],candidate_updates=visits,
                   optimized=optimized,no_target=no_target)
            artifacts.append(receipt)
        finish(folder/'COMPLETE.json',artifacts,kind=kind,seed=42,candidate_updates=total,**resources())
    else:
        loss=kind;initial=initial_patch(engine.wrapper.device,42);patch=torch.nn.Parameter(initial.clone())
        opt=torch.optim.Adam([patch],lr=cfg()['pilot_lr']);start=0;optimized=0;no_target=0
        last=folder/'LAST.pt'
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False)
            require(saved['signature']==signature(),'Pilot resume signature mismatch')
            with torch.no_grad():patch.copy_(saved['patch'].to(patch.device))
            opt.load_state_dict(saved['optimizer']);start=saved['visit'];optimized=saved['optimized'];no_target=saved['no_target']
        monitor.update(resume_completed=start*len(rows))
        log=folder/f'ATTEMPT_{time.time_ns()}.jsonl';total=len(rows)*visits
        with log.open('x') as stream:
            for visit in range(start,visits):
                ordered=list(rows);random.Random(42+visit).shuffle(ordered)
                for n,row in enumerate(ordered):
                    values=update(engine,patch,opt,row,loss,cfg()['pilot_lr'],initial)
                    optimized+=int(values['optimized']);no_target+=int(values['no_target'])
                    step=visit*len(rows)+n+1
                    stream.write(__import__('json').dumps({'step':step,'state_id':row['state_id'],**values})+'\n')
                    if step%25==0:monitor.update(phase='shared_'+loss,completed=step,total=total,**values)
                atomic_torch({'signature':signature(),'visit':visit+1,'patch':patch.detach().cpu(),
                              'optimizer':opt.state_dict(),'optimized':optimized,'no_target':no_target},last)
        png=folder/'final.png';save_png(patch,png)
        finish(folder/'COMPLETE.json',[png,last,*folder.glob('ATTEMPT_*.jsonl')],kind=kind,seed=42,
               candidate_updates=total,optimized=optimized,no_target=no_target,**resources())

def run_night(candidate,monitor):
    require_gpu(candidate);c=cfg();spec=c['night_candidates'][candidate];seed=spec['seed'];kind=spec['loss']
    folder=ROOT/'training/night'/candidate;receipt=folder/'COMPLETE.json'
    if complete(receipt):return
    last=folder/'LAST.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        require(saved['signature']==signature() and saved['candidate']==candidate,'Night resume identity mismatch')
        if saved['epoch']==c['epochs']:
            require(saved['optimized']+saved['no_target']==48000,'Final update counters changed')
            require(tuple(saved['patch'].shape)==(3,50,50) and torch.isfinite(saved['patch']).all(),'Invalid final patch')
            save_png(saved['patch'],folder/'epoch_50.png');save_png(saved['patch'],folder/'final.png')
            finish(receipt,[folder/'final.png',last,*folder.glob('epoch_*.png'),*folder.glob('ATTEMPT_*.jsonl')],
                   candidate=candidate,loss=kind,seed=seed,candidate_updates=48000,
                   optimized=saved['optimized'],no_target=saved['no_target'],final_epoch_recovered=True,**resources())
            monitor.update(phase='final_epoch_recovered',completed=48000,total=48000);return
    engine=make_engine();folder.mkdir(parents=True,exist_ok=True)
    initial=initial_patch(engine.wrapper.device,seed);patch=torch.nn.Parameter(initial.clone())
    opt=torch.optim.Adam([patch],lr=c['learning_rate']);start=0;optimized=0;no_target=0
    last=folder/'LAST.pt';total=c['epochs']*c['updates_per_epoch']
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        require(saved['signature']==signature() and saved['candidate']==candidate,'Night resume identity mismatch')
        start=saved['epoch'];optimized=saved['optimized'];no_target=saved['no_target']
        require(optimized+no_target==start*c['updates_per_epoch'],'Night resume counters mismatch')
        with torch.no_grad():patch.copy_(saved['patch'].to(patch.device))
        opt.load_state_dict(saved['optimizer'])
    monitor.update(resume_completed=start*c['updates_per_epoch'])
    save_png(initial,folder/'epoch_00.png');log=folder/f'ATTEMPT_{time.time_ns()}.jsonl'
    started=time.monotonic();tasks=[(s,t) for s in c['suites'] for t in c['task_splits'][s]['train']]
    with log.open('x') as stream:
        for epoch in range(start,c['epochs']):
            order=list(tasks);random.Random(seed*1000003+epoch).shuffle(order);within=0
            for suite,task in order:
                rows=load_shard('train',suite,task)
                random.Random(seed*1000003+epoch*1009+c['suites'].index(suite)*100+task).shuffle(rows)
                for row in rows:
                    within+=1;step=epoch*c['updates_per_epoch']+within
                    values=update(engine,patch,opt,row,kind,lr(step,total),initial)
                    optimized+=int(values['optimized']);no_target+=int(values['no_target'])
                    stream.write(__import__('json').dumps({'epoch':epoch+1,'step':step,'state_id':row['state_id'],
                                                          'lr':lr(step,total),'elapsed_s':time.monotonic()-started,**values})+'\n')
                    if step%100==0:
                        stream.flush();monitor.update(phase=candidate,completed=step,total=total,epoch=epoch+1,
                                                       optimized=optimized,no_target_updates=no_target,**values)
                del rows
            require(within==960,'Night epoch membership changed')
            atomic_torch({'signature':signature(),'candidate':candidate,'epoch':epoch+1,'patch':patch.detach().cpu(),
                          'optimizer':opt.state_dict(),'optimized':optimized,'no_target':no_target},last)
            if epoch+1 in c['checkpoint_epochs']:save_png(patch,folder/f'epoch_{epoch+1:02d}.png')
    save_png(patch,folder/'final.png')
    finish(receipt,[folder/'final.png',last,*folder.glob('epoch_*.png'),*folder.glob('ATTEMPT_*.jsonl')],
           candidate=candidate,loss=kind,seed=seed,candidate_updates=total,optimized=optimized,no_target=no_target,
           elapsed_s=time.monotonic()-started,**resources())
