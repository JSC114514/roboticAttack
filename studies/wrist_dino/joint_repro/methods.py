# 发布阅读说明：定义动作掩码、最远有效端点及DINO辅助。动作张量[B,8,7]；特征[B,N,D]。clean目标detach，预测与当前特征保留梯度。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Single-factor method definitions, deterministic task-balanced sample draws.

Targets and mask changes are independent. All methods share optimizer, EOT,
initialization and500x50budget. No loss contains a manual gradient combination.
"""
import random
import numpy as np
from .common import config,require

BASE=dict(view='wrist',mask=[0],target='opposite',aggregation='mean',auxiliary=None,sampling='clean_uniform')
METHODS={
 'B01':dict(parent=None,changed='reference'),
 'B02':dict(parent='B01',changed='view',view='front'),
 'B03':dict(parent='B01',changed='mask',mask=[0,1,2]),
 'B04':dict(parent='B03',changed='mask',mask=list(range(7))),
 'B05':dict(parent='B03',changed='target',mask=[0,1,2],target='tma_zero'),
 'O01':dict(parent='B01',changed='aggregation',aggregation='max'),
 'O03':dict(parent='B01',changed='auxiliary',auxiliary='dino_cosine'),
 'O05':dict(parent='B01',changed='auxiliary',auxiliary='dino_affinity'),
 'S01':dict(parent='B01',changed='alias'),
 'S02':dict(parent='B01',changed='sampling',sampling='clean_dino'),
 'S04':dict(parent='B01',changed='sampling',sampling='replay_uniform'),
 'S05':dict(parent='S04',changed='sampling',sampling='replay_dino'),
 'S06':dict(parent='S05',changed='sampling',sampling='replay_shuffled'),
 'S07':dict(parent='S05',changed='sampling',sampling='replay_dynamic'),
}


def spec(identity):
    require(identity in METHODS,'Method lacks a source-audited implementable definition: '+identity)
    return dict(BASE,**METHODS[identity])


def audit_differences():
    """Reject scientific changes outside the matrix's single allowed field."""
    rows=[]
    for identity in METHODS:
        child=spec(identity);parent=child['parent']
        if not parent or child['changed']=='alias':continue
        old=spec(parent);diff=[k for k in BASE if old[k]!=child[k]]
        require(diff==[child['changed']],'Single-factor violation: '+identity+' '+str(diff))
        rows.append(dict(id=identity,parent=parent,changed=child['changed'],actual_diff=diff))
    return rows


def opposite(clean):
    """Source boundary rule: normalized XYZ/rot[-1,1], gripper[0,1]."""
    import torch
    low=clean.new_tensor([-1.]*6+[0.]);high=clean.new_tensor([1.]*7)
    return torch.where(clean.detach()>=(low+high)/2,low,high)


def target(clean,method):
    return __import__('torch').zeros_like(clean) if method['target']=='tma_zero' else opposite(clean)


def loss(action,teacher,method,features=None,clean_features=None,aux_weight=0.):
    """Mean/max over8slots, mean overselectedDoFs; scalar additiveaux only."""
    import torch.nn.functional as F
    error=(action[...,method['mask']]-teacher[...,method['mask']]).square().mean(-1)
    main=error.max() if method['aggregation']=='max' else error.mean()
    auxiliary=main.new_zeros(())
    if method['auxiliary']=='dino_cosine':
        # Same mathematical objective as UADA_OFT_FEATURE.feature_disruption_loss.
        a=features['dino'][1];b=clean_features['dino'][1]
        auxiliary=F.cosine_similarity(a.float(),b.float().detach(),dim=-1).mean()-1
    if method['auxiliary']=='dino_affinity':
        from .common import module
        from pathlib import Path
        import sys
        directory=Path(config()['project'])/'research_runs/wrist_dino_spatial_b12_diag500_v1/code'
        sys.path.insert(0,str(directory))
        source=module('jr04_source_affinity',directory/'new_objectives.py')
        auxiliary,_=source.affinity_objective(clean_features['dino'][1],features['dino'][1],features['footprint'],topk=8)
    return main+aux_weight*auxiliary,dict(action=main.detach(),auxiliary=auxiliary.detach())


def draw(rows,outer,seed,weights=None):
    """Cycle shuffled24-task blocks; uniform trajectory/state within a task.

    The task schedule is common to all methods. Only conditional state weights
    change in sampling arms; EOT uses a separate deterministic per-update seed.
    """
    c=config();groups={}
    for i,r in enumerate(rows):groups.setdefault((r['suite'],r['task_id']),[]).append(i)
    tasks=sorted(groups,key=lambda k:(c['suites'].index(k[0]),k[1]));epoch,offset=divmod(outer-1,len(tasks))
    random.Random(seed+epoch).shuffle(tasks);ids=groups[tasks[offset]]
    rng=random.Random(seed*1000000+outer)
    return rng.choices(ids,weights=None if weights is None else [weights[i] for i in ids],k=1)[0]


def seed_eot(seed,outer,inner):
    import torch
    value=seed*10000000+outer*10000+inner
    random.seed(value);np.random.seed(value%(2**32-1));torch.manual_seed(value)
    return value
