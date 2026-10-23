"""Pure CPU geometry, hierarchy weights and fixed-budget accounting."""
import math
import numpy as np

def translations(target, eef, distance=.02, epsilon=1e-8):
    unit=np.asarray(target,dtype=float)-np.asarray(eef,dtype=float);unit[2]=0
    length=np.linalg.norm(unit)
    if length<=epsilon: return None
    unit/=length;side=np.cross([0.,0.,1.],unit)
    return np.stack([-distance*unit,distance*unit,distance*side,-distance*side])

def hierarchy_weights(rows):
    suites=sorted({r['suite'] for r in rows});weights=[]
    for row in rows:
        tasks={r['task_id'] for r in rows if r['suite']==row['suite']}
        n=sum(r['suite']==row['suite'] and r['task_id']==row['task_id'] for r in rows)
        weights.append(1/(len(suites)*len(tasks)*n))
    return np.asarray(weights,dtype=float)

def choose(harm, realization, weights, mode):
    if mode=='joint': score=(harm*realization*weights[:,None]).sum(0)
    elif mode=='realizability_only': score=(realization*weights[:,None]).sum(0)
    else: score=(harm*weights[:,None]).sum(0)
    return int(np.argmax(score)),score

def lr(charged, cfg):
    if charged<=cfg['warmup']:return cfg['learning_rate']*charged/cfg['warmup']
    return cfg['learning_rate']*.5*(1+math.cos(math.pi*(charged-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))

def counts(method, charged, cfg):
    dynamic=method in ('joint','realizability_only')
    cycles=min(charged,cfg['adaptive_budget'])//cfg['cycle_budget'] if dynamic else 0
    rejected=cycles*3*cfg['probe_updates']
    return {'charged_backwards':charged,'accepted_updates':charged-rejected,'discarded_probe_updates':rejected}

def feature_loss(feature, goal, token_weights, scale):
    return (((feature.float().reshape_as(goal)-goal.float()).square().mean(-1))*token_weights).sum()/scale

def realization(loss, clean_loss, epsilon=1e-12):
    return np.clip(1-np.asarray(loss)/np.maximum(np.asarray(clean_loss),epsilon),0,1)
