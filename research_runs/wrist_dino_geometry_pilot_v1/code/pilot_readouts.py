"""Frozen Discovery-only geometry thermometers, never patch objectives."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from pilot_common import (ROOT,atlas_path,atlas_root,atomic_json,atomic_npz,commit,complete,config,
                          load_json,progress,require)

LABELS=('object_eef_relative_x','object_eef_relative_y','object_eef_relative_z','object_eef_distance')


def linear_coefficients(x,y,alphas):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    mean=x.mean(0); scale=x.std(0); scale=np.where(scale>1e-6,scale,1.)
    xx=(x-mean)/scale; ym=y.mean(0); ys=y.std(0); ys=np.where(ys>1e-8,ys,1.)
    eigen,vectors=eigh(xx@xx.T/x.shape[1],check_finite=False)
    rotated=vectors.T@((y-ym)/ys)
    dual=vectors@(rotated/(np.maximum(eigen,0)[:,None]+np.asarray(alphas)[None]))
    coefficients=(xx.T@dual)/x.shape[1]*ys
    return {'mean':mean,'scale':scale,'coefficients':coefficients,'intercept':ym}


def predict(parameters,x):
    return ((np.asarray(x,dtype=np.float64)-parameters['mean'])/parameters['scale'])@parameters['coefficients']+parameters['intercept']


def pooled(tokens):
    # Exactly the atlas's 4x4 spatial pooling and float16 cache quantization.
    import torch
    atlas_path()
    from atlas_math import pooled_grid
    value=torch.as_tensor(np.asarray(tokens,dtype=np.float32)).reshape(1,256,-1)
    return pooled_grid(value).astype(np.float64)


@lru_cache(maxsize=2)
def monitor_parameters(layer):
    with np.load(ROOT/f'readouts/b{layer:02d}.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def monitor(layer,tokens):
    return predict(monitor_parameters(layer),pooled(tokens)).tolist()


def prepare_readouts(prep):
    if complete(ROOT/'reports/PREPARE_COMPLETE.json',prep['signature']): return
    atlas_path()
    from atlas_probes import fit_ridge
    atlas=atlas_root(); files=[]
    for layer in config()['layers']:
        xs=[]; ys=[]; rows=[]
        for task in range(10):
            folder=atlas/f'data/task_{task:02d}'
            all_rows=load_json(folder/'RECORDS.json')
            selected=[r for r in all_rows if r['split']=='discovery']
            indices=[r['row_index'] for r in selected]
            feature=np.load(folder/f'global/wrist__b{layer:02d}_spatial.npy',mmap_mode='r',allow_pickle=False)
            xs.append(np.asarray(feature[indices],dtype=np.float64))
            ys.extend([[r['truth'][key] for key in LABELS] for r in selected]); rows.extend(selected)
        x=np.concatenate(xs); y=np.asarray(ys)
        require(len(rows)==600 and len({r['episode_id'] for r in rows})==120,'Discovery geometry fit grid changed')
        require(all(r['split']=='discovery' for r in rows),'Validation leakage into readout')
        check,details=fit_ridge(x,y,x[:1],rows,config()['ridge_alphas'])
        parameters=linear_coefficients(x,y,details['alpha'])
        require(np.allclose(predict(parameters,x[:1]),check,atol=1e-8),'Serialized ridge differs from atlas fitting interface')
        path=ROOT/f'readouts/b{layer:02d}.npz'; atomic_npz(path,**parameters); files.append(path)
        meta=ROOT/f'readouts/b{layer:02d}.json'
        atomic_json({'labels':LABELS,'units':'m','layer':layer,'training_split':'discovery','states':600,
                     'episodes':120,'cv':details,'input':'DINO_only_4x4_grid','use':'monitor_only',
                     'training_episode_ids':sorted({r['episode_id'] for r in rows})},meta); files.append(meta)
        progress('prepare_readouts',layer=layer,training_states=600,training_episodes=120)
        del x,y,xs
    manifest=ROOT/'data/COHORT.json'; atomic_json(prep['states'],manifest); files.append(manifest)
    grid=ROOT/'data/CANDIDATES.json'; atomic_json(prep['candidates'],grid); files.append(grid)
    commit(ROOT/'reports/PREPARE_COMPLETE.json',files,prep['signature'],states=40,candidates=16,readouts_fitted_on='discovery_only')
