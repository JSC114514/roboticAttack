"""Pure CPU protocol math, with no policy/simulator imports."""
import numpy as np

VIEWS = ('front','wrist')
FAULTS = tuple(v+'_'+f for v in VIEWS for f in ('blur','lag'))

def nonattack_conditions():
    return ['clean',*[f'a_{f}_{g}' for f in FAULTS for g in ('raw','attenuate_damaged','attenuate_other')]]

def patch_conditions():
    return ['clean','b_clean_replay',*[f'b_{p}_{mode}' for p in ('relation','uada7') for mode in
             ('native','first_chunk','replay','random','random_clean')]]

def all_conditions(): return list(dict.fromkeys(nonattack_conditions()+patch_conditions()))

def condition_spec(name):
    if name=='clean': return dict(route='shared',fault=None,gate=None,patch=None,camera='native',first_chunk=False)
    if name.startswith('a_'):
        _,view,fault,gate=name.split('_',3)
        if view+'_'+fault not in FAULTS or gate not in ('raw','attenuate_damaged','attenuate_other'): raise ValueError(name)
        return dict(route='nonattack',fault=fault,damaged=view,gate=gate,patch=None,camera='native',first_chunk=False)
    if name=='b_clean_replay': return dict(route='patch',fault=None,gate=None,patch=None,camera='replay',first_chunk=False)
    _,patch,mode=name.split('_',2)
    if patch not in ('relation','uada7') or mode not in ('native','first_chunk','replay','random','random_clean'): raise ValueError(name)
    return dict(route='patch',fault=None,gate=None,patch=None if mode=='random_clean' else patch,
                camera='random' if mode in ('random','random_clean') else 'replay' if mode=='replay' else 'native',
                first_chunk=mode=='first_chunk',camera_source=patch)

def blend(value,mean,alpha):
    if not 0<=alpha<=1: raise ValueError('Invalid alpha')
    if value.shape!=mean.shape: raise ValueError('Feature layouts differ')
    return mean+alpha*(value-mean)

def grid(tokens):
    x=np.asarray(tokens)
    if x.shape[-2]!=256: raise ValueError('Expected 16x16 spatial tokens')
    x=x.reshape(16,16,x.shape[-1]);return x.reshape(4,4,4,4,-1).mean((1,3)).reshape(-1)

def blur(image,sigma=2.,kernel=13):
    from scipy.ndimage import convolve1d
    x=np.arange(kernel)-(kernel-1)/2;w=np.exp(-.5*(x/sigma)**2);w/=w.sum()
    im=np.asarray(image)
    if im.shape!=(224,224,3) or im.dtype!=np.uint8 or kernel%2!=1: raise ValueError('Expected RGB224/odd kernel')
    out=convolve1d(convolve1d(im.astype(np.float64),w,axis=0,mode='reflect'),w,axis=1,mode='reflect')
    return np.clip(np.rint(out),0,255).astype(np.uint8)

def rotation_angle(a,b):
    return float(np.arccos(np.clip((np.trace(np.asarray(a).reshape(3,3).T@np.asarray(b).reshape(3,3))-1)/2,-1,1)))

def rotation_axis(axis,angle):
    a=np.asarray(axis,float);a/=np.linalg.norm(a);x,y,z=a
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)

def random_pose(position,rotation,delta_norm,angle,seed):
    if delta_norm<0 or not 0<=angle<=np.pi+1e-8: raise ValueError('Invalid pose amplitude')
    rng=np.random.default_rng(seed);direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    axis=rng.normal(size=3)
    return np.asarray(position)+delta_norm*direction,np.asarray(rotation).reshape(3,3)@rotation_axis(axis,angle)

def residual(previous,current,delta_proprio,model):
    z0=(previous-model['pca_mean'])@model['components'].T
    z1=(current-model['pca_mean'])@model['components'].T
    x=np.r_[z0,delta_proprio];xn=(x-model['xmean'])/model['xscale']
    pred=xn@model['coef']+model['ymean']
    return float(np.linalg.norm((z1-pred)/model['residual_scale']))

def bootstrap(rows,seed=42,draws=2000):
    if not rows:return None
    groups={}
    for r in rows: groups.setdefault(r['task_id'],[]).append(r['difference'])
    keys=sorted(groups);rng=np.random.default_rng(seed);out=[]
    for _ in range(draws):
        out.append(np.mean([np.mean(rng.choice(groups[key],len(groups[key]),replace=True)) for key in rng.choice(keys,len(keys),replace=True)]))
    return np.quantile(out,[.025,.975]).tolist()

def auc(labels,scores):
    from scipy.stats import rankdata
    y=np.asarray(labels,dtype=bool);s=np.asarray(scores,float)
    positive=int(y.sum());negative=len(y)-positive
    if positive==0 or negative==0:return None
    ranks=rankdata(s,method='average')
    return float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative))
