# 发布阅读说明：冻结权重、学习率、长度与batch单变量矩阵。长度快照共享2000轮日程，不能与默认500轮日程混用。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Immutable ablation catalog, exact parameter differences and paired draws."""
import random
from pathlib import Path
from .common import ROOT,read,require,sha,complete

def study():return read(ROOT/'configs/SWEEP.json')

def resolve(identity):
    """Length snapshots inherit one shared 2000-outer learning-rate schedule."""
    catalog=study()['catalog']
    if identity in catalog:return catalog[identity]
    for row in catalog.values():
        for outer,label in row['checkpoint_ids'].items():
            if identity==label:return dict(row,checkpoint_outer=int(outer))
    raise RuntimeError('Undeclared sweep identity: '+identity)

def validate():
    s=study();parent=Path(s['parent_root'])
    require(sha(ROOT/'configs/PROTOCOL.json')==s['base_protocol_sha256'],'Base scientific protocol changed')
    for name,h in read(ROOT/'reports/PARENT_SOURCES.json').items():require(sha(parent/name)==h,'Parent source changed: '+name)
    for name in ['runtime.py','bank.py','geometry.py']:require(sha(ROOT/'joint_repro'/name)==sha(parent/'joint_repro'/name),'Native pipeline changed: '+name)
    require(read(ROOT/'data/O03_LAMBDA.json')['lambda']==s['auxiliary_lambda'],'DINO scale recalibrated')
    for name in ['P03_COMPLETE.json','P02_development_COMPLETE.json','SMOKE_16_COMPLETE.json']:require(complete(ROOT/'reports'/name),'Missing source gate '+name)
    for row in s['catalog'].values():
        if row['block']=='DEFAULT':continue
        anchor=s['catalog'][row['group']]
        diff=[k for k in ['lr','batch','beta','scheduler_outer'] if row[k]!=anchor[k]]
        expected=['scheduler_outer'] if row['block']=='T' else [row['changed']]
        require(diff==expected,'Non-single-factor block: '+row['id'])
        if row['block']=='T':require(row['scheduler_outer']==2000,'Length block schedule changed')
    return s

def batch_indices(rows,outer,batch):
    """Member0 matches source B1; others are distinct trajectories of same task.

    Larger batches are prefixes of the same candidate order. Method, seed,
    image transforms and sample weights do not influence this selection.
    """
    from .methods import draw
    from .common import config
    seed=config()['data_seed'];first=draw(rows,outer,seed)
    if batch==1:return [first]
    a=rows[first];candidates=[i for i,r in enumerate(rows) if (r['suite'],r['task_id'])==(a['suite'],a['task_id']) and r['trajectory_id']!=a['trajectory_id']]
    random.Random(seed*1000000+outer+781).shuffle(candidates)
    chosen=[first];seen={a['trajectory_id']}
    for index in candidates:
        trajectory=rows[index]['trajectory_id']
        if trajectory not in seen:chosen.append(index);seen.add(trajectory)
        if len(chosen)>=batch:break
    require(len(chosen)>=batch,'Insufficient distinct trajectories for batch')
    return chosen[:batch]

def stage_path(identity,seed,outer):return ROOT/'reports'/f'{identity}-seed{seed}-train{outer}_COMPLETE.json'


def preflight():
    """CPU-only source/receipt/pool checks; no Torch model or GPU allocation."""
    from .common import atomic,resources
    s=validate();rows=read(ROOT/'data/BANK_MANIFEST.json')['records']
    for outer in range(1,501):
        # Check the largest declared batch now, before any expensive model load.
        indices=batch_indices(rows,outer,8)
        require(len(set(rows[i]['trajectory_id'] for i in indices))==8,'Duplicate trajectory in batch')
        for batch in [1,2,4]:require(batch_indices(rows,outer,batch)==indices[:batch],'Batch prefix mismatch')
    require(len([w for w in s['workers'] if w['gate']=='initial'])==20,'Unexpected core matrix')
    result=dict(status='CPU_PREFLIGHT_PASS',core_workers=20,conditional_workers=12,gpu_tested=False,base_protocol_sha256=s['base_protocol_sha256'],study_sha256=sha(ROOT/'configs/SWEEP.json'),**resources())
    atomic(result,ROOT/'reports/CPU_PREFLIGHT.json');return result
