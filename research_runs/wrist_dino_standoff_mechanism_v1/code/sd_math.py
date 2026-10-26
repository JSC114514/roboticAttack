"""Metric geometry and episode-level statistics; no simulator/model imports."""
import numpy as np

def radial_delta(target,eef,distance=.05,sign=-1):
    r=np.asarray(target,dtype=float)-np.asarray(eef,dtype=float);d=float(np.linalg.norm(r))
    if not np.isfinite(r).all() or d<=1e-9: raise ValueError('Undefined target-to-EEF direction')
    return sign*distance*r/d

def choose(candidates,bins):
    result={};missing={}
    for name,b in bins.items():
        valid=[r for r in candidates if r['visible'] and not r['contact'] and not r['grasp']
            and b['minimum']<=r['distance_m']<=b['maximum'] and not r.get('task_done',False)]
        if valid:result[name]=min(valid,key=lambda r:(abs(r['distance_m']-b['preferred']),r['decision_block']))
        else:missing[name]='no_visible_contact_free_native_decision_in_distance_bin'
    return result,missing

def aabb_center(vertices):
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError('Invalid collision geometry')
    return (v.min(0)+v.max(0))/2

def trajectory_metrics(rows,band=(.04,.06),target=.05):
    d=np.asarray([r['distance_m'] for r in rows]);eef=np.asarray([r['eef_xyz'] for r in rows])
    inside=(d>=band[0])&(d<=band[1]);best=run=0;best_end=0
    # Row zero is the initial observation, not an executed control step.
    for i,flag in enumerate(inside[1:],1):
        run=run+1 if flag else 0
        if run>best:best,best_end=run,i
    start=best_end-best if best else 0
    movement=float(np.linalg.norm(np.diff(eef[start:best_end+1],axis=0),axis=1).sum()) if best else 0.
    return {'start_distance_m':float(d[0]),'end_distance_m':float(d[-1]),
        'start_error_m':float(abs(d[0]-target)),'end_error_m':float(abs(d[-1]-target)),
        'error_reduction_m':float(abs(d[0]-target)-abs(d[-1]-target)),
        'mean_error_m':float(np.abs(d-target).mean()),'minimum_distance_m':float(d.min()),
        'longest_band_dwell_steps':best,'band_dwell_path_m':movement,
        'band_dwell_net_m':float(np.linalg.norm(eef[best_end]-eef[start])) if best else 0.,
        'executed_steps':len(rows)-1,'entered_band':bool(inside.any()),
        'net_radial_change_m':float(d[-1]-d[0])}

def paired_arrays(current,reference,rgb_limit=1e-4):
    result={}
    for k,v in current.items():
        v=np.asarray(v);r=np.asarray(reference[k]);same=v.shape==r.shape
        nrmse=float(np.sqrt(np.mean(((v.astype(float)-r.astype(float))/255.)**2))) if same and k in ('front','wrist') else None
        exact=bool(same and np.array_equal(v,r))
        result[k]={'exact':exact,'shape_match':same,'rgb_nrmse':nrmse,
                   'pass':exact or (nrmse is not None and nrmse<=rgb_limit)}
    return result

def hierarchical_interval(frame,metric,draws=2000,seed=42):
    # Equal task weighting; near/far starts are averaged within trajectory first.
    groups=[]
    for _,task in frame.groupby(['suite','task_id']):
        v=task.groupby('episode_id')[metric].mean().dropna().to_numpy()
        if len(v):groups.append(v)
    if not groups:return {'mean':None,'low':None,'high':None,'tasks':0,'trajectories':0}
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        samples.append(np.mean([rng.choice(groups[i],len(groups[i]),replace=True).mean()
            for i in rng.integers(len(groups),size=len(groups))]))
    return {'mean':float(np.mean([v.mean() for v in groups])),
        'low':float(np.quantile(samples,.025)),'high':float(np.quantile(samples,.975)),
        'tasks':len(groups),'trajectories':sum(len(v) for v in groups)}
