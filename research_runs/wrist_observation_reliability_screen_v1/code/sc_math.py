"""Control-step delays use real frame history; image faults never change state/time."""
import copy
import numpy as np

KEYS={'front':'full_image','wrist':'wrist_image'}


def condition(name):
    if name=='clean': return dict(view=None,delay=0,occlusion=False)
    view,kind=name.split('_',1)
    if view not in KEYS: raise ValueError(name)
    if kind in ('delay1','delay2'): return dict(view=view,delay=int(kind[-1]),occlusion=False)
    if kind=='occlusion': return dict(view=view,delay=0,occlusion=True)
    raise ValueError(name)


def schedule(reference_steps, fractions=(.2,.5,.8), chunk=8):
    return sorted(set(max(0,int(reference_steps*f)//chunk*chunk) for f in fractions))


def active(step,onsets,duration=8): return any(s<=step<s+duration for s in onsets)


def deploy(history,name,step,onsets,size=56,fill=128,duration=8):
    spec=condition(name); now=history[-1]; out={k:np.array(v,copy=True) for k,v in now['observation'].items()}
    source=now; applied=False
    if spec['view']:
        if len(history)<=spec['delay']: raise RuntimeError('Real delay history missing')
        source=history[-1-spec['delay']]
        key=KEYS[spec['view']]; out[key]=np.array(source['observation'][key],copy=True)
        if spec['occlusion'] and active(step,onsets,duration):
            y=(out[key].shape[0]-size)//2; x=(out[key].shape[1]-size)//2
            out[key][y:y+size,x:x+size]=fill; applied=True
    return out,source,spec['delay']>0 or applied


def pose_error(current,source):
    shift=float(np.linalg.norm(np.asarray(current['xyz'])-np.asarray(source['xyz'])))
    a=np.asarray(current['rotation']).reshape(3,3);b=np.asarray(source['rotation']).reshape(3,3)
    angle=float(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1)))
    return shift,angle


def action_difference(a,b,stats,used):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    width=np.asarray(stats['q99'])-np.asarray(stats['q01']);mask=np.asarray(stats.get('mask',[True]*6+[False]),bool)
    scale=np.where(mask,2/np.maximum(width,1e-8),1)
    delta=(a-b)*scale
    return dict(next_xyz_normalized_l2=float(np.linalg.norm(delta[0,:3])),
        executed_xyz_normalized_rmse=float(np.sqrt(np.mean(delta[:used,:3]**2))),
        executed_rotation_normalized_rmse=float(np.sqrt(np.mean(delta[:used,3:6]**2))),
        executed_gripper_mae=float(np.mean(np.abs(delta[:used,6]))),
        all8_xyz_normalized_rmse=float(np.sqrt(np.mean(delta[:,:3]**2))))


def paired_counts(clean,changed):
    keys=sorted(set(clean)&set(changed))
    lost=sum(clean[k] and not changed[k] for k in keys);rescued=sum(not clean[k] and changed[k] for k in keys)
    return dict(pairs=len(keys),new_failures=lost,rescued=rescued,
        SR_difference=(rescued-lost)/len(keys) if keys else None)
