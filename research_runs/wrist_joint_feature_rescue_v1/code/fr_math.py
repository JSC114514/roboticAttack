"""Pure selection/accounting. A restored branch is not a deployable defense."""
import numpy as np

MODES=('native','dino','siglip','both')
def conditions(config):
    rows=[dict(name='clean',patch=None,mode='native',path=None,sha256=None)]
    for p in config['patches']:
        for mode in MODES:
            rows.append(dict(name=p['name']+'__'+mode,patch=p['name'],mode=mode,path=p['path'],sha256=p['sha256']))
    return rows

def branches(mode):
    if mode not in MODES:raise ValueError('Unknown restoration mode')
    return ('dino','siglip') if mode=='both' else () if mode=='native' else (mode,)

def paired_outcomes(clean,raw,restored):
    keys=sorted(set(clean)&set(raw)&set(restored))
    failures=[k for k in keys if clean[k] and not raw[k]]
    return dict(pairs=len(keys),clean_success_pairs=sum(bool(clean[k]) for k in keys),
        patch_induced_failures=len(failures),rescued_induced_failures=sum(bool(restored[k]) for k in failures),
        raw_failures_rescued=sum(not raw[k] and restored[k] for k in keys),
        raw_successes_lost=sum(raw[k] and not restored[k] for k in keys),
        rescue_fraction=sum(bool(restored[k]) for k in failures)/len(failures) if failures else None,
        SR_difference=float(np.mean([int(restored[k])-int(raw[k]) for k in keys])) if keys else None)

def interval(rows,draws=2000,seed=42):
    if not rows:return None
    groups={}
    for r in rows:groups.setdefault(r['suite'],{}).setdefault(r['task_id'],[]).append(r['value'])
    rng=np.random.default_rng(seed);result=[]
    for _ in range(draws):
        means=[]
        for tasks in groups.values():
            keys=sorted(tasks);sample=[]
            for i in rng.integers(len(keys),size=len(keys)):
                values=tasks[keys[i]]
                sample.extend(values[j] for j in rng.integers(len(values),size=len(values)))
            means.append(float(np.mean(sample)))
        result.append(float(np.mean(means)))
    return np.quantile(result,[.025,.975]).tolist()
