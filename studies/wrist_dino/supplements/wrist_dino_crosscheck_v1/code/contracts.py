# 发布阅读说明：固定RGB补丁尺寸/位置、原生动作形状与配对噪声地址；不进行按结果选位置或选种子。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Native target interface contracts, detached from old hard-coded grids.

Prediction and action conversion remain in the previously verified native
policies. Only the frozen experiment grid, PNGs and episode identity change.
"""
import numpy as np
from common import ROOT,atomic as atomic_json,read as load_json,sha as sha256_file,require,frozen

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')

def load_frozen():return frozen()

def action_chunk(value,horizon):
    a=np.asarray(value,dtype=np.float32)
    require(a.shape==(horizon,7) and np.isfinite(a).all(),'Native action chunk invalid')
    return a.copy()

def validate_observation(obs):
    for key in ['full_image','wrist_image']:
        a=np.asarray(obs[key]);require(a.dtype==np.uint8 and a.shape==(224,224,3),'Native image invalid: '+key)
    require(np.asarray(obs['state']).shape==(8,) and np.isfinite(obs['state']).all(),'Invalid native proprio')

def paste_wrist(obs,rgb,xy=(26,137)):
    validate_observation(obs);out=dict(obs)
    if rgb is not None:
        require(xy==(26,137) and rgb.shape==(50,50,3) and rgb.dtype==np.uint8,'Changed PNG placement/shape')
        wrist=obs['wrist_image'].copy();wrist[137:187,26:76]=rgb;out['wrist_image']=wrist
    return out

def pi05_payload(obs,instruction):
    validate_observation(obs)
    return {'observation/image':obs['full_image'].copy(),'observation/wrist_image':obs['wrist_image'].copy(),'observation/state':np.asarray(obs['state'],np.float32).copy(),'prompt':str(instruction)}

def paired_noise(task_id,state_index,decision,suite,cfg):
    """Identical diffusion noise per physical start/decision across all PNGs."""
    require(suite in SUITES and state_index in cfg['state_indices'] and decision>=0,'Invalid noise address')
    address=[cfg['noise_seed'],SUITES.index(suite),task_id,state_index,decision]
    model=cfg['models']['pi05']
    return address,np.random.default_rng(np.random.SeedSequence(address)).standard_normal((model['action_horizon'],model['model_action_dim']),dtype=np.float32)
