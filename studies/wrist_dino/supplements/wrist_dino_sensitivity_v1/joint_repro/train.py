# 发布阅读说明：冻结VLA，仅优化共享RGB补丁[3,50,50]、像素范围[0,1]；原生动作损失同时通过DINO和SigLIP反传。outer/inner及Adam更新数分别记录。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Single-variable sweeps with true mean observation gradients.

Native DINO+SigLIP forward/backward, source EOT, source AdamW/clamp and scheduler
operators remain unchanged. Long runs have one fixed 2000-outer schedule and
resumable stop points. Batch means use microbatch1, never branch gradient mixing.
"""
import csv
import random
import time
import numpy as np
import torch
from PIL import Image
from .common import ROOT,config,read,atomic,require,sha,digest,finish,complete,save_torch,Monitor,resources
from .bank import load_record
from .methods import spec,seed_eot,target,loss
from .runtime import Engine,load_wrapper,switch_suite,policy_observation
from .sweep import study,resolve,validate,batch_indices,stage_path


def save_png(path,patch):
    a=patch.detach().cpu().float().clamp(0,1).permute(1,2,0).mul(255).round().byte().numpy()
    Image.fromarray(a).save(path)


def export_checkpoint(row,seed,outer,state,params):
    """Immutable PNG/Adam snapshots survive subsequent long-run continuation."""
    label=row['checkpoint_ids'][str(outer)];folder=ROOT/f'outputs/training/{label}-seed{seed}'
    if complete(folder/'COMPLETE.json'):return
    folder.mkdir(parents=True,exist_ok=True)
    atomic(dict(params,checkpoint_outer=outer),folder/'CONFIG.json')
    save_png(folder/'initial.png',state['initial']);save_png(folder/'final.png',state['patch'])
    save_torch(dict(patch=state['patch'].cpu(),actual_optimizer_steps=state['steps']),folder/'final.pt')
    save_torch(state,folder/'CHECKPOINT.pt')
    with (folder/'TRAINING.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(state['history'][0]));writer.writeheader();writer.writerows(state['history'])
    finish(folder/'COMPLETE.json',[folder/n for n in ['CONFIG.json','initial.png','final.png','final.pt','CHECKPOINT.pt','TRAINING.csv']],actual_optimizer_steps=state['steps'],sample_backprops=state['sample_backprops'],unique_training_records=len(set(state['used'])),**resources())


def train(identity,seed,stop,monitor):
    validate();s=study();row=resolve(identity);c=config()
    require(row['block']!='DEFAULT','Default training is read-only')
    require(stop<=row['scheduler_outer'] and str(stop) in row['checkpoint_ids'],'Undeclared stopping point')
    receipt=stage_path(identity,seed,stop)
    if complete(receipt):return
    final_label=row['checkpoint_ids'][str(stop)]
    final_receipt=ROOT/f'outputs/training/{final_label}-seed{seed}/COMPLETE.json'
    if complete(final_receipt):
        # Recovery after successful checkpoint export but before stage receipt:
        # reconstruct only the bookkeeping, without allocating/loading a model.
        saved=read(final_receipt)
        finish(receipt,[final_receipt],actual_optimizer_steps=saved['actual_optimizer_steps'],sample_backprops=saved['sample_backprops'],stopped_outer=stop,scheduler_outer=row['scheduler_outer'])
        return
    folder=ROOT/f'outputs/training/{identity}-seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    method=spec(row['group']);rows=read(ROOT/'data/BANK_MANIFEST.json')['records'];batch=row['batch']
    aux_weight=s['auxiliary_lambda']*row['beta']
    params=dict(protocol_sha256=sha(ROOT/'configs/PROTOCOL.json'),study_sha256=sha(ROOT/'configs/SWEEP.json'),bank_sha256=read(ROOT/'data/BANK_MANIFEST.json')['bank_sha256'],method=method,seed=seed,parameters=row,auxiliary_lambda=aux_weight,backpropagation='native_direct_no_recomputation')
    identity_hash=digest(params);resume=folder/'resume.pt';steps=0;sample_backprops=0;history=[];used=[];start_outer=1
    w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False)
    from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch,build_uada_adamw,step_uada_patch
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    patch=init_uada_patch((3,50,50),w.device);initial=patch.detach().clone()
    optimizer,scheduler=build_uada_adamw(patch,row['lr'],20,row['scheduler_outer'],weight_decay=0.)
    if resume.exists():
        state=torch.load(resume,map_location=w.device,weights_only=False)
        require(state['identity']==identity_hash,'Resume identity changed')
        patch.data.copy_(state['patch']);initial=state['initial'].to(patch);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        steps=state['steps'];sample_backprops=state['sample_backprops'];history=state['history'];used=state['used'];start_outer=state['outer']+1
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        # A kill between resume save and immutable export must not lose a
        # declared500/1000/2000 checkpoint when continuation starts atouter+1.
        if str(state['outer']) in row['checkpoint_ids']:export_checkpoint(row,seed,state['outer'],state,params)
    else:
        actual=initial.cpu().float().permute(1,2,0).mul(255).round().byte().numpy()
        expected=np.asarray(Image.open(ROOT/f'outputs/preflight/initial-seed{seed}.png').convert('RGB'))
        require(np.array_equal(actual,expected),'Initial PNG differs from paired baseline')
        atomic(params,folder/'CONFIG.json');save_png(folder/'initial.png',initial)
    monitor.update(phase='training',completed=steps,total=stop*50,resumed=steps,batch=batch,beta=row['beta'],scheduler_outer=row['scheduler_outer'],reason=row['block']+' single-factor ablation')
    for outer in range(start_outer,stop+1):
        tick=time.monotonic();lr=float(optimizer.param_groups[0]['lr']);indices=batch_indices(rows,outer,batch);prepared=[]
        for index in indices:
            record=rows[index];used.append(record['record_id']);obs,language=load_record(record);obs=policy_observation(obs);switch_suite(w,record['suite'])
            with torch.no_grad():clean,features0=engine.action(obs,language)
            prepared.append((obs,language,target(clean,method),features0))
        combined=[];actions=[];auxiliaries=[];gradients=[]
        for inner in range(50):
            optimizer.zero_grad(set_to_none=True);value=action_value=aux_value=0.
            for member,(obs,language,teacher,donor) in enumerate(prepared):
                # Member0 is exactly source EOT; extra members have fixed
                # independent streams, common to every method/batch prefix.
                seed_eot(seed,outer,inner+100*member)
                prediction,features=engine.action(obs,language,patch,'wrist',True,capture=(steps==0))
                objective,terms=loss(prediction,teacher,method,features,donor,aux_weight)
                require(torch.isfinite(objective),'Nonfinite objective')
                (objective/batch).backward()
                if steps==0:
                    witness={}
                    for branch in ['dino','siglip']:
                        gradient=features[branch][1].grad
                        require(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0,'Missing full branch gradient: '+branch)
                        witness[branch]=float(gradient.float().norm())
                    atomic(dict(member=member,branch_gradient_norms=witness,mask=method['mask'],auxiliary_lambda=aux_weight),folder/f'GRADIENT_MEMBER{member}.json')
                value+=float(objective.detach())/batch
                action_value+=float(terms['action'])/batch;aux_value+=float(terms['auxiliary'])/batch
                sample_backprops+=1;del prediction,features,objective
            require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Nonfinite/missing gradient')
            gradient=float(patch.grad.float().norm());step_uada_patch(patch,optimizer,w.vla);steps+=1
            combined.append(value);actions.append(action_value);auxiliaries.append(aux_value);gradients.append(gradient)
            monitor.update(completed=steps,outer=outer,inner=inner+1,loss=value,action_loss=action_value,auxiliary_loss=aux_value,gradient_norm=gradient,lr=lr,sample_backprops=sample_backprops)
        scheduler.step();engine.assert_frozen()
        require(sample_backprops==steps*batch,'Backprop/update count mismatch')
        require(all(float(x['step'])==steps for x in optimizer.state.values() if 'step' in x),'Adam count mismatch')
        history.append(dict(outer=outer,actual_updates=steps,sample_backprops=sample_backprops,record_ids=[rows[i]['record_id'] for i in indices],loss=float(np.mean(combined)),action_loss=float(np.mean(actions)),auxiliary_loss=float(np.mean(auxiliaries)),gradient_norm=float(np.mean(gradients)),lr=lr,pixel_rms_change=float((patch-initial).square().mean().sqrt()),elapsed_outer_s=time.monotonic()-tick))
        print(history[-1],flush=True)
        state=dict(identity=identity_hash,outer=outer,steps=steps,sample_backprops=sample_backprops,patch=patch.detach(),initial=initial,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,used=used,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        save_torch(state,resume)
        if str(outer) in row['checkpoint_ids']:export_checkpoint(row,seed,outer,state,params)
    require(steps==stop*50,'Optimizer budget mismatch')
    label=row['checkpoint_ids'][str(stop)]
    finish(receipt,[ROOT/f'outputs/training/{label}-seed{seed}/COMPLETE.json'],actual_optimizer_steps=steps,sample_backprops=sample_backprops,stopped_outer=stop,scheduler_outer=row['scheduler_outer'])


def run(args):
    with Monitor(f'{args.id}-seed{args.seed}-train') as monitor:train(args.id,args.seed,args.stop,monitor)
