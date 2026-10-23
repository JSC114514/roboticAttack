"""Sixteen serial shared patches; only frozen DINO feature matching drives Adam."""
from __future__ import annotations
import math
import os
import time
import numpy as np
import torch
from pilot_common import (ROOT,atomic_json,commit,complete,config,observation,progress,require,resources)
from pilot_engine import target_loss
from pilot_mechanism import features_file
from pilot_pixels import initial_patch,pixels,save_png,verify_initial_png


def learning_rate(step,cfg):
    if step<=cfg['warmup_updates']: return cfg['learning_rate']*step/cfg['warmup_updates']
    fraction=(step-cfg['warmup_updates'])/(cfg['updates']-cfg['warmup_updates'])
    return cfg['learning_rate']*.5*(1+math.cos(math.pi*fraction))


def atomic_torch(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    torch.save(value,temporary); os.replace(temporary,path)


def load_resume(path,sid,candidate):
    # This file is our own local checkpoint, never an external pickle.
    saved=torch.load(path,map_location='cpu',weights_only=False)
    require(saved['signature']==sid and saved['candidate']==candidate,'Optimizer resume identity mismatch')
    return saved


def train_candidate(candidate,discovery,engine,sid):
    cfg=config(); name=candidate['candidate_id']; folder=ROOT/'training'/name
    if complete(folder/'COMPLETE.json',sid): return
    require(len(discovery)==20 and all(r['split']=='discovery' for r in discovery),'Patch fitting requires exactly Discovery20')
    folder.mkdir(parents=True,exist_ok=True)
    initial=initial_patch(candidate['seed'],engine.wrapper.device)
    patch=torch.nn.Parameter(initial.clone()); optimizer=torch.optim.Adam([patch],lr=cfg['learning_rate'])
    latest=folder/'LAST.pt'; start=0; optimized=0; no_target=0
    if latest.exists():
        saved=load_resume(latest,sid,candidate); start=saved['step']; optimized=saved['optimized']; no_target=saved['no_target']
        with torch.no_grad(): patch.copy_(saved['patch'].to(patch.device))
        optimizer.load_state_dict(saved['optimizer'])
    else: save_png(initial,folder/'step_0000.png')
    verify_initial_png(folder/'step_0000.png',candidate['seed'])
    cache=[]; layer=candidate['layer']; target_name=candidate['target']+'_large'
    for row in discovery:
        obs,language=observation(row)
        with np.load(features_file(row),allow_pickle=False) as z:
            clean=torch.as_tensor(z[f'clean__dino_b{layer}'].copy(),device=patch.device)
            target=torch.as_tensor(z[f'{target_name}__dino_b{layer}'].copy(),device=patch.device)
        cache.append((row,obs,clean,target))
    started=time.monotonic(); log=folder/f'TRAIN_ATTEMPT_{time.time_ns()}.jsonl'
    for step in range(start+1,cfg['updates']+1):
        row,obs,clean,target=cache[(step-1)%len(cache)]
        rate=learning_rate(step,cfg)
        for group in optimizer.param_groups: group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        values,_,eot=pixels(engine.wrapper,obs,patch,candidate['seed']*1000003+step,True)
        spatial=engine.spatial(values,layer)
        loss=target_loss(spatial,clean,target)
        before=patch.detach().clone(); grad_l2=0.; grad_max=0.
        if loss is None:
            no_target+=1; loss_value=None
        else:
            require(torch.isfinite(loss),'Nonfinite DINO feature loss')
            loss.backward()
            require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Nonfinite/disconnected patch gradient')
            grad_l2=float(patch.grad.norm()); grad_max=float(patch.grad.abs().max())
            optimizer.step()
            with torch.no_grad(): patch.clamp_(0,1)
            loss_value=float(loss.detach()); optimized+=1
        require(all(p.grad is None for p in engine.wrapper.vla.parameters()),'VLA received parameter gradients')
        record={'step':step,'state_id':row['state_id'],'split':'discovery','loss':loss_value,'lr':rate,
                'gradient_l2':grad_l2,'gradient_max_abs':grad_max,'pixel_step_l2':float((patch.detach()-before).norm()),
                'pixel_from_initial_l2':float((patch.detach()-initial).norm()),'eot':eot,
                'optimized':optimized,'no_target':no_target,'elapsed_s':time.monotonic()-started}
        with log.open('a') as output:
            import json
            output.write(json.dumps(record,allow_nan=False)+'\n')
        if step in cfg['checkpoints']: save_png(patch,folder/f'step_{step:04d}.png')
        if step%cfg['resume_interval']==0 or step==cfg['updates']:
            atomic_torch({'signature':sid,'candidate':candidate,'step':step,'patch':patch.detach().cpu(),
                          'optimizer':optimizer.state_dict(),'optimized':optimized,'no_target':no_target},latest)
        if step%20==0:
            elapsed=time.monotonic()-started
            progress('training',candidate=name,step=step,total=cfg['updates'],optimized=optimized,no_target=no_target,
                     loss=loss_value,gradient_l2=grad_l2,elapsed_s=elapsed,
                     estimated_candidate_remaining_s=elapsed/max(step-start,1)*(cfg['updates']-step))
        del spatial,values,loss
    cost=folder/f'COST_{time.time_ns()}.json'
    atomic_json({'candidate':candidate,'start_step':start,'final_step':cfg['updates'],'optimized':optimized,
                 'no_target':no_target,'elapsed_s':time.monotonic()-started,'readout_or_action_loss_used':False,
                 'training_states':20,'validation_used_for_fitting':False,
                 'peak_scope':'worker_since_start_not_candidate_increment',**resources()},cost)
    files=[folder/f'step_{step:04d}.png' for step in cfg['checkpoints']]+[latest,cost]
    commit(folder/'COMPLETE.json',files,sid,candidate=candidate,optimized=optimized,no_target=no_target)


def pilot(prep):
    require(complete(ROOT/'reports/MECHANISM_COMPLETE.json',prep['signature']),'Mechanism/cache stage incomplete')
    from pilot_engine import Engine
    from pilot_evaluate import evaluate_candidate,evaluate_baselines
    from pilot_common import observation
    engine=Engine(); obs,language=observation(prep['states'][0])
    atomic_json(engine.validate_rgb(obs,language),ROOT/'reports/PILOT_GPU_RGB_INTERFACE.json')
    discovery=[r for r in prep['states'] if r['split']=='discovery']
    validation=[r for r in prep['states'] if r['split']=='validation']
    # Full VLA calls happen only at final PNG evaluation, not in the 16k optimization updates.
    evaluate_baselines(validation,engine,prep['signature'])
    started=time.monotonic()
    for index,candidate in enumerate(prep['candidates']):
        train_candidate(candidate,discovery,engine,prep['signature'])
        evaluate_candidate(candidate,validation,engine,prep['signature'])
        elapsed=time.monotonic()-started
        progress('candidate_complete',candidate=candidate['candidate_id'],completed=index+1,total=16,
                 elapsed_s=elapsed,estimated_remaining_s=elapsed/(index+1)*(15-index))
    files=[ROOT/'reports/PILOT_GPU_RGB_INTERFACE.json',ROOT/'evaluation/baselines/COMPLETE.json']
    for candidate in prep['candidates']:
        files += [ROOT/'training'/candidate['candidate_id']/'COMPLETE.json',
                  ROOT/'evaluation'/candidate['candidate_id']/'COMPLETE.json']
    commit(ROOT/'reports/PILOT_COMPLETE.json',files,prep['signature'],candidates=16,validation_states=20)
