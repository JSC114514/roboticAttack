"""Exact action objectives, constrained adaptation, and resumable adapter training."""
from collections import OrderedDict
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from PIL import Image, ImageEnhance
from wr_common import (ROOT, PROJECT, config, read, require, atomic, save_torch, complete,
    finish, signature, candidate, load_wrapper, switch_suite, resources)
from wr_engine import Engine
from wr_math import objectives, objective, update_dual, learning_rate


class Store:
    """Bounded trajectory cache, never deserialize the complete image pool."""
    def __init__(self,limit=3):self.limit=limit;self.cache=OrderedDict()
    def get(self,path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        if str(p) not in self.cache:
            with np.load(p,allow_pickle=False) as a:self.cache[str(p)]={k:a[k] for k in a.files}
        self.cache.move_to_end(str(p))
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return self.cache[str(p)]

def observation(front,wrist,state):
    # Expert RLDS images are already in policy convention (not raw simulator GL).
    from VLAAttacker.multiview.openvla_oft_experiments import resize_oft_observation_images
    value=dict(full_image=np.asarray(front,np.uint8),wrist_image=np.asarray(wrist,np.uint8),
               state=np.asarray(state,np.float32).copy())
    return resize_oft_observation_images(value,input_views=['front','wrist'],resize_size=224)

def language(value):
    x=value.item() if isinstance(value,np.ndarray) and value.ndim==0 else value
    return x.decode() if isinstance(x,bytes) else str(x)

def expert_pair(ref,store):
    x=store.get(ref['path']);i=int(ref['pair_index'])*2
    obs=[observation(x['front_rgb'][j],x['wrist_rgb'][j],x['proprio'][j]) for j in (i,i+1)]
    return obs,[x['actions'][j] for j in (i,i+1)],language(x['language'])

def camera_pair(ref,store,transform):
    x=store.get(ref['path']);i=int(ref['record_index'])
    native=observation(x['front_rgb'][i],x['wrist_rgb'][i],x['proprio'][i])
    changed=observation(x['front_rgb'][i],x['shifted_wrist'][i,transform],x['proprio'][i])
    return [native,changed],language(x['language'])

def refs(stage,split):
    from wr_data import camera_records
    m=read(ROOT/f'data/{stage}/MANIFEST.json')
    key='train' if split=='train' else 'validation'
    return m['expert'][key],list(camera_records(stage,key))

def values(engine,observations,text,suite):
    switch_suite(engine.wrapper,suite)
    return [engine.value(o,text) for o in observations]

def augment(obs,generator):
    out=copy.deepcopy(obs)
    # Independent draws per camera, fixed stream shared across all augmentation
    # candidates. Native canonical preprocessing still runs after augmentation.
    for key in ('full_image','wrist_image'):
        image=Image.fromarray(out[key]);w,h=image.size
        size=float(generator.uniform(.9,1.))**.5
        cw,ch=max(1,round(w*size)),max(1,round(h*size))
        left=int(generator.integers(0,w-cw+1));top=int(generator.integers(0,h-ch+1))
        image=image.crop((left,top,left+cw,top+ch)).resize((w,h),Image.Resampling.BILINEAR)
        for cls in (ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color):
            image=cls(image).enhance(float(generator.uniform(.9,1.1)))
        out[key]=np.array(image,copy=True)
    return out

def calibration(stage,monitor):
    receipt=ROOT/f'reports/CALIBRATE_{stage}_COMPLETE.json'
    if complete(receipt):return
    if stage=='pilot':
        # The original-model screen tests the motivation before any adaptation.
        # Use only the pre-control visibility audit when counting evidence;
        # failure trajectories themselves never decide which starts are valid.
        from wr_common import cases,profiles
        original={}
        for case in cases(stage):
            for profile in profiles(stage):
                path=ROOT/f'evaluation/{stage}/base'/case['start']/profile['name']/'RESULT.json'
                require(complete(path.with_name('COMPLETE.json')),'Original-model screen is incomplete')
                original[case['start'],profile['name']]=read(path)
        clean_success=sum(v['success'] for (start,p),v in original.items() if p=='native')
        failures=[]
        for (start,p),v in original.items():
            if p=='native' or not original[start,'native']['success'] or v['success']:continue
            visibility=v.get('initial_visibility')
            if visibility is not None and visibility['native_target_pixels']>0 and not visibility['target_lost']:
                failures.append([start,p])
        decision=dict(signature=signature(),clean_success_starts=clean_success,
                      camera_induced_failures_with_initial_target_retained=failures,
                      continue_training=bool(failures),criterion='at_least_one_paired_failure_without_initial_target_loss')
        atomic(decision,ROOT/'results/pilot/MOTIVATION.json')
        require(decision['continue_training'],
                'Original-model screen did not show camera-induced failure with the target retained; see MOTIVATION.json. Training budget was not spent.')
    expert,camera=refs(stage,'train');require(expert and camera,'Empty training pool')
    wrapper=load_wrapper(monitor);engine=Engine(wrapper,gradient_checkpointing=False)
    store=Store();sums=dict(B=0.,R=0.,C=0.);n=0;started=time.monotonic()
    # Check base model response first; no fitting or test data enter normalizers.
    for j,ref in enumerate(expert):
        obs,y,text=expert_pair(ref,store);a=values(engine,obs,text,ref['suite'])
        ys=[torch.as_tensor(v,device=a[0].device,dtype=a[0].dtype) for v in y]
        v=objectives(a[0],a[0],a[0],a[1],ys[0],ys[1])
        sums['B']+=float(v['B']);sums['R']+=float(v['R'])
        if j==0:atomic(engine.gate(obs[0],text,gradient=False),ROOT/f'reports/GATES/{stage}_calibration.json')
        monitor.update(completed=j+1,total=len(expert)+len(camera)*8,phase='calibrate_expert')
    for j,ref in enumerate(camera):
        for k in range(8):
            obs,text=camera_pair(ref,store,k);a=values(engine,obs,text,ref['suite'])
            from wr_math import mse
            sums['C']+=float(mse(a[1],a[0]));n+=1
            monitor.update(completed=len(expert)+n,total=len(expert)+len(camera)*8,phase='calibrate_camera')
    baseline=dict(B=sums['B']/len(expert),R=sums['R']/len(expert),C=sums['C']/n)
    require(all(np.isfinite(v) and v>=0 for v in baseline.values()),'Invalid baseline errors')
    path=ROOT/f'data/{stage}/CALIBRATION.json'
    atomic(dict(signature=signature(),baseline=baseline,expert_pairs=len(expert),camera_pairs=n,
                split='train_only',elapsed_s=time.monotonic()-started,**resources()),path)
    finish(receipt,[path]);return baseline

def rng_state(generator,augment_rng):
    return dict(numpy=generator.bit_generator.state,augment=augment_rng.bit_generator.state,
                python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(v,generator,augment_rng):
    generator.bit_generator.state=v['numpy'];augment_rng.bit_generator.state=v['augment']
    random.setstate(v['python']);torch.set_rng_state(v['torch']);torch.cuda.set_rng_state_all(v['cuda'])

def train(stage,name,monitor):
    spec=candidate(stage,name);c=config();out=ROOT/f'training/{stage}/{name}'
    receipt=out/'COMPLETE.json'
    if complete(receipt):return
    require(complete(ROOT/f'reports/CALIBRATE_{stage}_COMPLETE.json'),'Training calibration incomplete')
    baseline=read(ROOT/f'data/{stage}/CALIBRATION.json')['baseline']
    expert,camera=refs(stage,'train');store=Store();started=time.monotonic()
    random.seed(spec['seed']);np.random.seed(spec['seed']);torch.manual_seed(spec['seed'])
    wrapper=load_wrapper(monitor);engine=Engine(wrapper,seed=spec['seed'])
    optimizer=torch.optim.AdamW(engine.parameters,lr=c['optimizer']['lr'],weight_decay=0.)
    multipliers={k:c['dual']['initial'] for k in ('B','R')};ema=dict(B=0.,R=0.)
    generator=np.random.default_rng(spec['seed']);augment_rng=np.random.default_rng(spec['seed']+1000)
    # Sample order is materialized from a separate deterministic stream. Camera
    # choices never change expert visits between objective families.
    order_rng=np.random.default_rng(spec['seed']);orders=[]
    for _ in range(int(np.ceil(spec['updates']/len(expert)))):orders.extend(order_rng.permutation(len(expert)).tolist())
    order=orders[:spec['updates']]
    start=0;prior_elapsed=0.;checkpoint_path=out/'latest.pt'
    if checkpoint_path.is_file():
        saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        require(saved['signature']==signature() and saved['spec']==spec,'Resume revision/config differs')
        engine.load_state_dict(saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        multipliers=saved['multipliers'];ema=saved['ema'];start=saved['step'];prior_elapsed=saved['gpu_seconds']
        restore_rng(saved['rng'],generator,augment_rng)
    initial_parameters=[p.detach().clone() for p in engine.parameters]
    out.mkdir(parents=True,exist_ok=True)
    obs,_,text=expert_pair(expert[0],store);switch_suite(wrapper,expert[0]['suite'])
    # Includes finite gradients and staged-vs-direct comparison on actual policy.
    atomic(engine.gate(obs[0],text,gradient=True),out/'STARTUP_CHECK.json')
    monitor.update(completed=start,resumed=start,total=spec['updates'],phase='train',candidate=name)
    attempt_log=out/('metrics_'+str(time.time_ns())+'.jsonl')
    for step in range(start,spec['updates']):
        eref=expert[order[step]];eobs,y,text=expert_pair(eref,store)
        if spec['method']=='augment':eobs=[augment(o,augment_rng) for o in eobs]
        ea=values(engine,eobs,text,eref['suite'])
        if spec['method'] in ('consistency','constrained'):
            cref=camera[int(generator.integers(len(camera)))];k=int(generator.integers(8))
            cobs,ctext=camera_pair(cref,store,k);ca=values(engine,cobs,ctext,cref['suite'])
        else:ca=[ea[0],ea[0]];cobs=None
        leaves=[a.detach().requires_grad_(True) for a in [ca[0],ca[1],ea[0],ea[1]]]
        ys=[torch.as_tensor(v,device=leaves[0].device,dtype=torch.float32) for v in y]
        v=objectives(*leaves,*ys);loss=objective(v,baseline,spec,multipliers)
        require(torch.isfinite(loss),'Nonfinite objective')
        gradients=torch.autograd.grad(loss,leaves,allow_unused=True)
        optimizer.zero_grad(set_to_none=True)
        if cobs is not None:
            switch_suite(wrapper,cref['suite'])
            for o,g in zip(cobs,gradients[:2]):
                if g is not None:engine.vjp(o,ctext,g)
        switch_suite(wrapper,eref['suite'])
        for o,g in zip(eobs,gradients[2:]):
            if g is not None:engine.vjp(o,text,g)
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.parameters),'Missing/nonfinite LoRA gradient')
        gradnorm=float(torch.nn.utils.clip_grad_norm_(engine.parameters,c['optimizer']['gradient_clip']))
        lr=learning_rate(step,spec['updates'],c['optimizer'])
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();engine.assert_frozen()
        metrics={k:float(value.detach()) for k,value in v.items()}
        if spec['method']=='constrained':update_dual(multipliers,ema,metrics,baseline,spec['tau'],c['dual'])
        row=dict(step=step+1,loss=float(loss.detach()),lr=lr,gradient_l2=gradnorm,
                 multipliers=dict(multipliers),**metrics,**resources())
        if (step+1)%100==0:
            row['adapter_change_since_attempt_start_l2']=float(torch.stack([
                (p.detach()-initial).square().sum() for p,initial in zip(engine.parameters,initial_parameters)]).sum().sqrt())
        with attempt_log.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        monitor.update(completed=step+1,**row)
        if (step+1)%250==0 or step+1==spec['updates']:
            elapsed=prior_elapsed+time.monotonic()-started
            saved=dict(signature=signature(),spec=spec,step=step+1,adapter=engine.state_dict(),
                       optimizer=optimizer.state_dict(),multipliers=multipliers,ema=ema,
                       rng=rng_state(generator,augment_rng),gpu_seconds=elapsed)
            path=out/f'checkpoint_{step+1:06d}.pt';save_torch(saved,path);save_torch(saved,checkpoint_path)
            if step+1==spec['updates']:save_torch(saved,out/'final.pt')
        if step+1==100:
            atomic(dict(updates=100,seconds=time.monotonic()-started,
                        estimated_total_s=(time.monotonic()-started)*spec['updates']/100,**resources()),out/'THROUGHPUT.json')
    # Fixed final adapter, no development checkpoint/seed selection.
    if start==spec['updates'] and not (out/'final.pt').exists():save_torch(saved,out/'final.pt')
    elapsed=prior_elapsed+time.monotonic()-started
    atomic(dict(candidate=name,spec=spec,updates=spec['updates'],baseline=baseline,
                gpu_seconds=elapsed,trainable_parameters=sum(p.numel() for p in engine.parameters),**resources()),out/'TRAINING.json')
    finish(receipt,[out/'final.pt',out/'TRAINING.json',out/'STARTUP_CHECK.json'],gpu_seconds=elapsed,elapsed_s=elapsed)


def offline(stage,name,engine,monitor):
    """Trajectory-held-out action checks. Never used to choose checkpoints."""
    path=ROOT/f'evaluation/{stage}/{name}/OFFLINE.json'
    receipt=path.with_name('OFFLINE_COMPLETE.json')
    if complete(receipt):return
    expert,camera=refs(stage,'validation');store=Store();rows=[]
    from wr_math import mse
    for ref in expert:
        obs,y,text=expert_pair(ref,store);a=values(engine,obs,text,ref['suite'])
        ys=[torch.as_tensor(v,device=a[0].device) for v in y]
        v=objectives(a[0],a[0],a[0],a[1],ys[0],ys[1])
        rows.append(dict(kind='expert',episode_id=ref['episode_id'],pair_id=ref['pair_id'],
                         suite=ref['suite'],task_id=ref['task_id'],B=float(v['B']),R=float(v['R'])))
    for ref in camera:
        for k in range(8):
            obs,text=camera_pair(ref,store,k);a=values(engine,obs,text,ref['suite'])
            rows.append(dict(kind='camera',episode_id=ref['episode_id'],state_id=ref['state_id'],
                             suite=ref['suite'],task_id=ref['task_id'],transform=k,C=float(mse(a[1],a[0]))))
    atomic(dict(signature=signature(),candidate=name,split='validation',records=rows),path)
    finish(receipt,[path])
