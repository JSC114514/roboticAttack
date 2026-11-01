# 发布阅读说明：冻结VLA，仅优化共享RGB补丁[3,50,50]、像素范围[0,1]；原生动作损失同时通过DINO和SigLIP反传。outer/inner及Adam更新数分别记录。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""One source-aligned patch trainer for all eligible matrix arms.

One new observation per outer,50source-EOT updates, one cosine scheduler tick
per outer. Model weights are frozen. Each update is one scalar loss backward
through unmodified DINO+SigLIP; no branch replacement or manual gradient mix.
"""
import csv
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import ROOT,config,read,atomic,require,sha,digest,finish,complete,save_torch,Monitor,resources
from .bank import load_record
from .methods import spec,draw,seed_eot,target,loss
from .runtime import Engine,load_wrapper,switch_suite,policy_observation


def save_png(path,patch):
    """Freeze one8-bit RGB artifact using the same round+clamp as source."""
    a=patch.detach().cpu().float().clamp(0,1).permute(1,2,0).mul(255).round().byte().numpy()
    Image.fromarray(a).save(path)


def calibration(engine,rows,initial,monitor,method):
    """One training-only scalar λ; no gradient-ratio machinery or test tuning.

    Use one preselected state per24trainingtasks, source feature cosine loss,
    and median action/absoluteaux loss scales at the common initialPNG/EOT.
    Freeze this coefficient across seeds and all subsequent updates.
    """
    p=ROOT/'data'/('O03_LAMBDA.json' if method['auxiliary']=='dino_cosine' else 'O05_LAMBDA.json')
    if p.exists():return float(read(p)['lambda'])
    chosen=[];seen=set()
    for row in rows:
        key=row['suite'],row['task_id']
        if key not in seen:chosen.append(row);seen.add(key)
    av=[];fv=[]
    with torch.no_grad():
        for i,row in enumerate(chosen):
            obs,lang=load_record(row);obs=policy_observation(obs);switch_suite(engine.w,row['suite'])
            clean,features0=engine.action(obs,lang);seed_eot(42,1,i)
            footprint=engine.patch_footprint(obs) if method['auxiliary']=='dino_affinity' else None
            action,features=engine.action(obs,lang,initial)
            if footprint is not None:features['footprint']=footprint
            from torch.nn.functional import cosine_similarity
            av.append(float((action[...,:1]-target(clean,spec('B01'))[...,:1]).square().mean()))
            _,terms=loss(action,target(clean,method),method,features,features0,1.)
            fv.append(abs(float(terms['auxiliary'])))
    require(float(np.median(fv))>0,'O03 training-only feature calibration degenerates')
    weight=float(np.median(av)/np.median(fv));atomic(dict(lambda_=weight,**{'lambda':weight},states=[r['record_id'] for r in chosen],action_scales=av,auxiliary_scales=fv,rule='median(action_loss)/median(abs(source_cosine_loss))',bank_sha256=read(ROOT/'data/BANK_MANIFEST.json')['bank_sha256']),p)
    return weight


def train(identity,seed,monitor,smoke=False):
    c=config();method=spec(identity)
    eligibility=read(ROOT/'reports/METHOD_ELIGIBILITY.json')
    require(eligibility.get(identity,{}).get('status')!='SKIPPED_PROTOCOL','SKIPPED_PROTOCOL: '+eligibility.get(identity,{}).get('reason',''))
    require(method['sampling']!='clean_dino','Static DINO score lacks matched labels; no substitute sampler allowed')
    folder=ROOT/('outputs/smoke' if smoke else 'outputs/training')/f'{identity}-seed{seed:02d}';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'COMPLETE.json'
    if not smoke and complete(receipt):return
    require(complete(ROOT/'reports/P03_COMPLETE.json'),'P03 real GPU preflight missing')
    require(complete(ROOT/'reports/P02_development_COMPLETE.json'),'P02 development clean reference missing')
    require(complete(ROOT/'reports/SMOKE_16_COMPLETE.json'),'16fixed-native-PNG smoke not completed')
    rows=read(ROOT/'data/BANK_MANIFEST.json')['records'];replay_rows=[];refresh_history=[]
    if method['sampling'].startswith('replay_'):
        require(complete(ROOT/'reports/S03_COMPLETE.json'),'S03 shared replay not frozen')
        replay_rows=read(ROOT/'data/REPLAY_MANIFEST.json')['records']
    w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False)
    from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch,build_uada_adamw,step_uada_patch
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    patch=init_uada_patch((3,50,50),w.device);initial=patch.detach().clone()
    optimizer,scheduler=build_uada_adamw(patch,.002,20,500,weight_decay=0.)
    steps=0;start_outer=1;history=[];used=[];resume=folder/'resume.pt';aux_weight=0.
    if method['auxiliary']:aux_weight=calibration(engine,rows,initial,monitor,method)
    identity_config=dict(replay_sha256=sha(ROOT/'data/REPLAY_MANIFEST.json') if replay_rows else None,protocol_sha256=sha(ROOT/'configs/PROTOCOL.json'),bank_sha256=read(ROOT/'data/BANK_MANIFEST.json')['bank_sha256'],method=method,seed=seed,auxiliary_lambda=aux_weight,backpropagation="native_direct_no_recomputation")
    frozen_identity=digest(identity_config)
    if resume.exists():
        state=torch.load(resume,map_location=w.device,weights_only=False)
        require(state['identity']==frozen_identity,'Resume scientific identity changed')
        patch.data.copy_(state['patch']);initial=state['initial'].to(patch);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);steps=state['steps'];start_outer=state['outer']+1;history=state['history'];used=state['used']
        replay_rows=state.get('replay_rows',replay_rows);refresh_history=state.get('refresh_history',[])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    else:
        atomic(identity_config,folder/'CONFIG.json');save_png(folder/'initial.png',initial)
    atomic(dict(**identity_config,reproduction='ADAPTED_RE_EVALUATION',budget=dict(outer=500,inner=50,optimizer_steps=25000),original_sources=['UADA_OFT_DINO.py','UADA_OFT_SOURCE_COPY.py','appply_random_transform.py']),folder/'METHOD.json')
    total=2 if smoke else 500;monitor.update(phase='training',completed=steps,total=total*50,resumed=steps)
    for outer in range(start_outer,total+1):
        if method['sampling']=='replay_dynamic' and outer==251:
            # Original refresh_after_outer contains250; later source refreshes
            # are outside this fixed500outerbudget, and are not rescaled.
            from .replay import record_score,score_weights
            rgb=patch.detach().float().cpu().clamp(0,1).permute(1,2,0).mul(255).round().byte().numpy()
            state_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            try:
                rescored=[]
                for rr in replay_rows:
                    oo,ll=load_record(rr);oo=policy_observation(oo);switch_suite(w,rr['suite'])
                    rescored.append(dict(rr,**record_score(engine,oo,ll,rgb)))
                replay_rows=score_weights(rescored,c['data_seed']);refresh_history.append(dict(after_outer=250,scored_records=len(rescored),model_calls=3*len(rescored)))
                atomic(dict(rows=replay_rows,refresh_history=refresh_history),folder/'SCORES_AFTER_250.json')
            finally:
                random.setstate(state_rng[0]);np.random.set_state(state_rng[1]);torch.set_rng_state(state_rng[2]);torch.cuda.set_rng_state_all(state_rng[3])
        index=draw(rows,outer,c['data_seed']);row=rows[index]
        if replay_rows and outer%2==0:
            candidates=[i for i,r in enumerate(replay_rows) if (r['suite'],r['task_id'])==(row['suite'],row['task_id'])]
            field={'replay_uniform':None,'replay_dino':'dino_weight','replay_shuffled':'shuffled_weight','replay_dynamic':'dino_weight'}[method['sampling']]
            weights=None if field is None else [replay_rows[i][field] for i in candidates]
            chosen=random.Random(c['data_seed']*1000000+outer).choices(candidates,weights=weights,k=1)[0];row=replay_rows[chosen]
        used.append(row['record_id'])
        obs,language=load_record(row);obs=policy_observation(obs);switch_suite(w,row['suite'])
        with torch.no_grad():clean,clean_features=engine.action(obs,language)
        teacher=target(clean,method);values=[];gradients=[];started=time.monotonic();lr=float(optimizer.param_groups[0]['lr'])
        for inner in range(50):
            seed_eot(seed,outer,inner);optimizer.zero_grad(set_to_none=True)
            footprint=engine.patch_footprint(obs,method['view']) if method['auxiliary']=='dino_affinity' else None
            action,features=engine.action(obs,language,patch,method['view'],True)
            if footprint is not None:features['footprint']=footprint
            objective,metrics=loss(action,teacher,method,features,clean_features,aux_weight)
            require(torch.isfinite(objective),'Nonfinite training loss');objective.backward()
            require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Patch gradient missing/nonfinite')
            gradients.append(float(patch.grad.float().norm()));values.append(float(objective.detach()))
            step_uada_patch(patch,optimizer,w.vla);steps+=1
            del action,features,objective
            monitor.update(completed=steps,outer=outer,inner=inner+1,loss=values[-1],gradient_norm=gradients[-1],lr=lr)
        scheduler.step();engine.assert_frozen()
        history.append(dict(outer=outer,actual_updates=steps,record_id=row['record_id'],suite=row['suite'],task_id=row['task_id'],loss=float(np.mean(values)),gradient_norm=float(np.mean(gradients)),lr=lr,pixel_rms_change=float((patch-initial).square().mean().sqrt()),elapsed_outer_s=time.monotonic()-started))
        print(history[-1],flush=True)
        # Save at completeouter boundaries; never double-count a partialouter on resume.
        state=dict(identity=frozen_identity,outer=outer,steps=steps,patch=patch.detach(),initial=initial.detach(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,used=used,replay_rows=replay_rows,refresh_history=refresh_history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save_torch(state,resume)
        if outer%25==0:save_png(folder/f'outer-{outer:03d}.png',patch)
    require(steps==total*50,'Optimizer budget mismatch')
    save_png(folder/'final.png',patch);save_torch(dict(patch=patch.detach().cpu(),actual_optimizer_steps=steps,identity=identity_config),folder/'final.pt')
    with (folder/'TRAINING.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    finish(receipt,[folder/'CONFIG.json',folder/'METHOD.json',folder/'initial.png',folder/'final.png',folder/'final.pt',folder/'TRAINING.csv'],actual_optimizer_steps=steps,unique_training_records=len(set(used)),**resources())


def run(args):
    with Monitor(f'{args.id}-seed{args.seed:02d}-train') as m:train(args.id,args.seed,m,args.smoke)
