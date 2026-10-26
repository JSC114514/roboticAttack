"""One shared PNG; frozen XYZ readout loss, exact native forward pixels."""
import json
import random
import time
import numpy as np
import torch
from rr_common import (ROOT,PILOT,Pool,target,config,read_json,signature,require,native,
                       atomic_json,atomic_torch,finish,complete,make_engine)
from rr_math import Readout, relation_loss, learning_rate, sample_order


def save(path,patch,optimizer,step,optimized,no_target):
    atomic_torch({'signature':signature(),'step':step,'optimized':optimized,'no_target':no_target,
        'patch':patch.detach().cpu(),'optimizer':optimizer.state_dict(),
        'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
        'schedule':{'type':'candidate_step_warmup_cosine','total':config()['updates'],'warmup':config()['warmup']}},path)


def run(monitor):
    folder=ROOT/'training'
    if complete(folder/'COMPLETE.json'): return
    native.require_gpu('relation_train')
    c=config();pool=Pool('fit');engine=make_engine()
    engine.verify_patch_path(pool[0])
    decoder=Readout(PILOT/'readouts/b22.npz',engine.wrapper.device)
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    initial=native.load_png(ROOT/'data/initial.png',engine.wrapper.device)
    patch=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([patch],lr=c['learning_rate'])
    start=optimized=no_target=0;last=folder/'LAST.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        require(saved['signature']==signature(),'Resume revision mismatch')
        with torch.no_grad():patch.copy_(saved['patch'].to(patch))
        optimizer.load_state_dict(saved['optimizer'])
        start,optimized,no_target=saved['step'],saved['optimized'],saved['no_target']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    # Tiny q(clean) cache; never retain all feature maps/observations in RAM.
    clean_path=ROOT/'data/CLEAN_READOUTS.json'
    if clean_path.exists():
        saved_clean=read_json(clean_path)
        require(saved_clean['signature']==signature(),'Clean relation cache revision changed')
        clean=saved_clean['values']
    else:clean={}
    for i,row in enumerate(pool):
        if row['state_id'] not in clean:
            with torch.no_grad():feature=engine.spatial_observation(row['observation']).to(engine.wrapper.device)
            clean[row['state_id']]=decoder(feature).detach().cpu()[0].tolist()
            atomic_json({'signature':signature(),'values':clean},clean_path)
        monitor.update(phase='cache_clean_readout',completed=i+1,total=len(pool))
    # Test the actual readout path and gradient, not just a generic DINO scalar.
    first=next(r for r in pool if target(r) is not None)
    feature,_=engine.spatial_patch(first['observation'],patch)
    prediction=decoder(feature)
    # Independent NumPy reshape/average reference to the original 4x4 cache.
    raw=feature.detach().float().cpu().numpy().reshape(1,16,16,1024)
    reference_grid=raw.reshape(1,4,4,4,4,1024).mean(axis=(2,4)).reshape(1,16384).astype(np.float16).astype(np.float64)
    with np.load(PILOT/'readouts/b22.npz',allow_pickle=False) as z:
        reference=((reference_grid-z['mean'])/z['scale'])@z['coefficients'][:,:3]+z['intercept'][:3]
    error=float(np.max(np.abs(prediction.detach().cpu().numpy()-reference)))
    require(error<=1e-5,'Differentiable readout differs from original cached readout')
    baseline=torch.tensor(clean[first['state_id']],device=patch.device,dtype=torch.float64)
    delta=torch.tensor(target(first),device=patch.device,dtype=torch.float64)
    scalar=relation_loss(prediction,baseline,delta,c['relation_scale_m'])
    gradient=torch.autograd.grad(scalar,patch)[0]
    require(torch.isfinite(gradient).all() and float(gradient.norm())>0,'Relation loss has no finite RGB gradient')
    atomic_json({'status':'PASS','readout_max_abs_error_m':error,'gradient_l2':float(gradient.norm()),
                 'frozen_readout':True,'no_action_objective':True},ROOT/'reports/READOUT_GATE.json')
    del feature,prediction,scalar,gradient
    log=folder/f'ATTEMPT_{time.time_ns()}.jsonl'
    monitor.update(phase='shared_relation',completed=start,resume_completed=start,total=c['updates'])
    with log.open('x',buffering=1) as stream:
        for number in range(start,c['updates']):
            step=number+1;epoch,offset=divmod(number,len(pool))
            row=pool[sample_order(len(pool),c['seed'],epoch)[offset]]
            lr=learning_rate(step,c['updates'],c['warmup'],c['learning_rate'])
            optimizer.param_groups[0]['lr']=lr
            delta=target(row)
            if delta is None:
                no_target+=1;values={'optimized':False,'no_target':True,'loss':None,'gradient_l2':0.,'pixel_step_l2':0.}
            else:
                optimizer.zero_grad(set_to_none=True);before=patch.detach().clone()
                feature,_=engine.spatial_patch(row['observation'],patch)
                predicted=decoder(feature)
                baseline=torch.tensor(clean[row['state_id']],device=patch.device,dtype=torch.float64)
                desired=torch.tensor(delta,device=patch.device,dtype=torch.float64)
                loss=relation_loss(predicted,baseline,desired,c['relation_scale_m'])
                require(torch.isfinite(loss),'Nonfinite relation loss')
                loss.backward()
                require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Invalid RGB gradient')
                moved=(predicted.detach()-baseline)[0]
                values={'optimized':True,'no_target':False,'loss':float(loss.detach()),
                    'relation_delta_m':moved.cpu().tolist(),'actual_target_delta_m':delta.tolist(),
                    'relation_error_cm':float((moved-desired).norm()*100),
                    'gradient_l2':float(patch.grad.norm())}
                optimizer.step()
                with torch.no_grad():patch.clamp_(0,1)
                optimized+=1
                require(all(p.grad is None and not p.requires_grad for m in
                    (engine.wrapper.vla,engine.wrapper.action_head,engine.wrapper.proprio_projector)
                    if m is not None for p in m.parameters()),'Victim gradients/weights changed')
                values.update(pixel_step_l2=float((patch.detach()-before).norm()),
                    pixel_from_initial_l2=float((patch.detach()-initial).norm()),
                    pixel_mae_255=float((patch.detach()-initial).abs().mean()*255))
                del feature,predicted,loss
            stream.write(json.dumps({'step':step,'state_id':row['state_id'],'learning_rate':lr,**values},allow_nan=False)+'\n')
            if step%10==0:monitor.update(phase='shared_relation',completed=step,total=c['updates'],
                actual_updates=optimized,no_target_count=no_target,**values)
            if step%c['checkpoint_interval']==0 or step==c['updates']:
                save(last,patch,optimizer,step,optimized,no_target)
                native.save_png(patch,folder/f'checkpoint_{step:04d}.png')
    image=folder/'final.png';native.save_png(patch,image)
    finish(folder/'COMPLETE.json',[image,last,clean_path,*folder.glob('ATTEMPT_*.jsonl')],
           candidate_updates=c['updates'],optimized=optimized,no_target=no_target,seed=42,
           fixed_final_checkpoint=True,**native.resources())
