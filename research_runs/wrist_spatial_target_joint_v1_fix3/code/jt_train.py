"""Shared RGB training; discarded probes are charged, Adam branches are isolated."""
import copy
import json
import random
import time
import numpy as np
import torch
from jt_common import ROOT,source,native,config,signature,read,atomic,atomic_torch,require,complete,finish,eligible,cached,make_engine
from jt_math import hierarchy_weights,feature_loss,realization,choose,lr,counts

def snapshot(patch,optimizer):
    return {'patch':patch.detach().cpu().clone(),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'torch_rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}

def restore(state,patch,optimizer):
    with torch.no_grad():patch.copy_(state['patch'].to(patch.device))
    optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    # Optimizer.load_state_dict may alias the supplied tensors: never pass the branch snapshot itself.
    torch.set_rng_state(state['torch_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])

def sample_indices(rows,seed,total):
    grouped={}
    for i,row in enumerate(rows):grouped.setdefault(row['suite'],{}).setdefault(row['task_id'],[]).append(i)
    suites=config()['suites'];require(set(grouped)==set(suites),'Sample pool lost a suite')
    result=[]
    for step in range(total):
        rng=random.Random(seed*1000000+step);group=grouped[suites[step%len(suites)]]
        result.append(rng.choice(group[rng.choice(sorted(group))]))
    return result

def run(method,monitor):
    folder=ROOT/'training'/method;folder.mkdir(parents=True,exist_ok=True)
    if complete(folder/'COMPLETE.json'):return
    require(complete(ROOT/'reports/TARGETS_COMPLETE.json'),'Targets incomplete')
    stats=read(ROOT/'data/TARGET_STATS.json')
    if not stats['proceed']:
        atomic({'status':'STOPPED_NO_APPROACH_OBSTRUCTION','method':method},folder/'STOPPED.json');return
    native.require_gpu('joint_target_train');cfg=config();seed=cfg['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    entries=eligible('fit');pool=[cached(r) for r in entries];weights=hierarchy_weights(entries)
    shared=torch.load(ROOT/'data/FEATURE_STATS.pt',map_location='cpu',weights_only=False)
    monitor.update(phase='load_frozen_Goal_model',method=method)
    engine=make_engine(checkpointing=True);device=engine.wrapper.device
    initial=native.load_png(ROOT/'data/initial_s42.png',device)
    patch=torch.nn.Parameter(initial.clone());optimizer=torch.optim.Adam([patch],lr=cfg['learning_rate'])
    engine.verify_patch_path(pool[0]['row'])
    if method=='translation_boundary':engine.gate(pool[0]['row'],patch,'translation_far_boundary','joint_run_boundary')
    token_weights=shared['token_weights'].to(device);scale=shared['scale']
    harm=np.stack([p['harm'] for p in pool]);clean_losses=np.zeros_like(harm,dtype=float)
    with torch.no_grad():
        for i,p in enumerate(pool):
            for k in range(4):clean_losses[i,k]=float(feature_loss(p['clean_feature'],p['goals'][k],shared['token_weights'],scale))
    require(np.all(clean_losses>cfg['feature_epsilon']),'Zero geometric target energy')
    order=sample_indices(entries,seed,cfg['updates']);last=folder/'LAST.pt'
    budget=0;chosen=stats['fixed_rule'];selection=[];forward_counts={'dino':0,'siglip':0,'language_model':0}
    handles=[]
    for name,model in [('dino',engine.dino),('siglip',engine.siglip),('language_model',engine.wrapper.vla.language_model)]:
        def count(_m,_a,_name=name):forward_counts[_name]+=1
        handles.append(model.register_forward_pre_hook(count))
    if last.exists():
        value=torch.load(last,map_location='cpu',weights_only=False)
        require(value['signature']==signature() and value['method']==method,'Resume identity differs')
        require(value['counts']==counts(method,value['budget'],cfg),'Checkpoint budget mismatch')
        restore(value['state'],patch,optimizer);budget=value['budget'];chosen=value['chosen_rule'];selection=value['selection']
        forward_counts.update(value['forward_counts'])
        require(int(optimizer.state[patch].get('step',0))==value['counts']['accepted_updates'],'Adam step differs from accepted count')
    else:
        # Interface checks must not change the predetermined optimization RNG state.
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    monitor.update(phase='training',method=method,completed=budget,resume_completed=budget,total=cfg['updates'])
    log=folder/f'ATTEMPT_{time.time_ns()}.jsonl'
    accepted_updates=counts(method,budget,cfg)['accepted_updates']

    def update(index,rule,schedule_step,stream,probe=False):
        nonlocal budget
        payload=pool[index];row=payload['row'];optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=lr(schedule_step,cfg)
        before=patch.detach().clone()
        if method=='translation_boundary':
            gradient,metrics=engine.staged_gradient(row,patch,'translation_far_boundary');patch.grad=gradient
        else:
            feature,_=engine.spatial_patch(row['observation'],patch)
            goal=payload['goals'][rule].to(device=device,dtype=torch.float32)
            loss=feature_loss(feature,goal,token_weights,scale)
            require(torch.isfinite(loss),'Nonfinite feature loss')
            patch.grad=torch.autograd.grad(loss,patch)[0]
            metrics={'loss':float(loss.detach()),'gradient_l2':float(patch.grad.norm())}
        require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Invalid RGB gradient')
        optimizer.step()
        with torch.no_grad():patch.clamp_(0,1)
        budget+=1
        metrics.update(charged_backwards=budget,state_id=row['state_id'],rule=None if method=='translation_boundary' else cfg['rules'][rule],
                       probe=probe,schedule_step=schedule_step,learning_rate=lr(schedule_step,cfg),
                       pixel_step_l2=float((patch.detach()-before).norm()),pixel_from_initial_l2=float((patch.detach()-initial).norm()))
        stream.write(json.dumps(metrics)+'\n')
        monitor.update(completed=budget,**metrics,accepted_updates=accepted_updates,forward_counts=dict(forward_counts))
        if budget%20==0:stream.flush()

    def checkpoint():
        expected=counts(method,budget,cfg)
        require(expected['accepted_updates']==accepted_updates,'Accepted/rejected accounting mismatch')
        require(int(optimizer.state[patch].get('step',0))==accepted_updates,'Selected Adam moments do not match accepted updates')
        state={'signature':signature(),'method':method,'budget':budget,'counts':expected,'chosen_rule':chosen,
               'selection':selection,'state':snapshot(patch,optimizer),'forward_counts':forward_counts}
        path=folder/f'checkpoint_{budget:06d}.pt';atomic_torch(state,path)
        native.save_png(patch,path.with_suffix('.png'))
        finish(path.with_suffix('.json'),[path,path.with_suffix('.png')],method=method,budget=budget,counts=expected)
        atomic_torch(state,last)

    try:
        with log.open('x') as stream:
            while budget<cfg['updates']:
                start=budget
                if method in ('joint','realizability_only') and start<cfg['adaptive_budget']:
                    require(start%cfg['cycle_budget']==0,'Resume inside unsealed cycle')
                    parent=snapshot(patch,optimizer);branches=[];scores=np.zeros(4);details=[]
                    for k in range(4):
                        restore(parent,patch,optimizer)
                        for j in range(cfg['probe_updates']):
                            # Identical observations, RNG and LR sequence in all four branches.
                            update(order[start+j],k,start+j+1,stream,probe=True)
                        losses=[]
                        with torch.no_grad():
                            for payload in pool:
                                feature,_=engine.spatial_patch(payload['row']['observation'],patch)
                                losses.append(float(feature_loss(feature,payload['goals'][k].to(device=device,dtype=torch.float32),token_weights,scale)))
                        q=realization(losses,clean_losses[:,k],cfg['feature_epsilon'])
                        score=float(np.sum(weights*q*(harm[:,k] if method=='joint' else 1)))
                        scores[k]=score;branches.append(snapshot(patch,optimizer))
                        details.append({'rule':cfg['rules'][k],'score':score,'realization_mean':float(weights@q),'losses':losses,'realization':q.tolist()})
                    chosen=int(np.argmax(scores));restore(branches[chosen],patch,optimizer)
                    accepted_updates+=cfg['probe_updates']
                    selection.append({'cycle_start_budget':start,'probe_charged':20,'retained_rule':cfg['rules'][chosen],
                                      'details':details,'uses_validation':False,
                                      'all_scores_tied':bool(np.all(scores==scores[0])),
                                      'tie_rule_used':bool(np.count_nonzero(scores==scores.max())>1)})
                    atomic({'signature':signature(),'method':method,'selection':selection},folder/'SELECTION_HISTORY.json')
                # Fixed-harm arm always keeps the single strongest FIT teacher; dynamic arms lock after 3000.
                end=min(start+cfg['cycle_budget'],cfg['updates'])
                while budget<end:
                    update(order[budget],chosen,budget+1,stream)
                    accepted_updates+=1
                checkpoint()
        require(budget==4000,'Final budget must be exactly 4000')
        final=folder/'checkpoint_004000.png'
        finish(folder/'COMPLETE.json',[folder/'checkpoint_004000.json',last],method=method,counts=counts(method,budget,cfg),final_png=str(final))
        monitor.update(completed=budget,accepted_updates=accepted_updates,phase='training_complete')
    finally:
        for handle in handles:handle.remove()

def select(monitor):
    if complete(ROOT/'reports/SELECT_COMPLETE.json'):return
    stats=read(ROOT/'data/TARGET_STATS.json')
    if not stats['proceed']:return
    chosen={}
    for method in config()['methods']:
        folder=ROOT/'training'/method;require(complete(folder/'COMPLETE.json'),'Incomplete method: '+method)
        receipt=folder/'checkpoint_004000.json';require(complete(receipt),'Final checkpoint missing')
        path=receipt.with_suffix('.png');value=torch.load(receipt.with_suffix('.pt'),map_location='cpu',weights_only=False)
        from jt_common import sha
        chosen[method]={'path':str(path),'png_sha256':sha(path),'budget':4000,'chosen_rule':value['chosen_rule'],'counts':value['counts']}
    atomic({'signature':signature(),'candidates':chosen,'selection_criterion':'predeclared_final_4000_no_validation_selection'},ROOT/'data/SELECTED.json')
    finish(ROOT/'reports/SELECT_COMPLETE.json',[ROOT/'data/SELECTED.json',*[ROOT/'training'/m/'COMPLETE.json' for m in config()['methods']]])
