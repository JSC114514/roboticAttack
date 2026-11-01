# 发布阅读说明：验证单轴/七维×无/有DINO的四格比较及父实验依赖；不伪造缺失父阶段收据。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Frozen factorial supplement: validate reused assets without modifying them.

Only D07 is new. The parent trainer, renderer, EOT, optimizer, action interface
and data reader remain byte-identical. Old receipts are checked against their
linked artifacts; new receipts additionally lock ABLATION.json.
"""
from pathlib import Path
from .common import ROOT,read,sha,require,complete,atomic,finish


def validate():
    """Cheap CPU startup gate; no dataset generation or CUDA initialization."""
    lock=read(ROOT/'configs/ABLATION.json');parent=Path(lock['parent_root'])
    require(sha(ROOT/'configs/PROTOCOL.json')==lock['parent_protocol_sha256'],'Base protocol changed')
    require(sha(parent/'reports/V03_COMPLETE.json')==lock['parent_final_receipt_sha256'],'Parent final receipt changed')
    for name,h in read(ROOT/'reports/PARENT_SOURCES.json').items():
        require(sha(parent/name)==h,'Parent asset changed: '+name)
    for name in ['train.py','runtime.py','evaluate.py','bank.py','geometry.py']:
        require(sha(ROOT/'joint_repro'/name)==sha(parent/'joint_repro'/name),'Shared implementation changed: '+name)
    require(read(ROOT/'data/O03_LAMBDA.json')['lambda']==lock['auxiliary_lambda'],'Frozen lambda changed')
    for name in ['P03_COMPLETE.json','P02_development_COMPLETE.json','SMOKE_16_COMPLETE.json']:
        require(complete(ROOT/'reports'/name),'Invalid reused gate '+name)
    from .methods import spec,BASE
    b=spec('B04');d=spec('D07');o=spec('O03')
    require([k for k in BASE if b[k]!=d[k]]==['auxiliary'],'D07 differs from B04 beyond auxiliary')
    require([k for k in BASE if o[k]!=d[k]]==['mask'],'D07 differs from O03 beyond mask')
    require(lock['seeds']==[42,43,44] and lock['video'] is False,'Frozen execution identity changed')
    return lock


def verify_both_identity(results):
    """Both branches must reproduce clean actions, commands and EEF trajectories.

    Compare complete recorded rollouts after float32 controller conversion, not
    just same-state predictions. This catches the historical float64 interface
    error before a diagnostic is accepted.
    """
    import numpy as np
    rows=[r for r in results if r['intervention']=='both']
    require(len(rows)==80,'Both-restoration cohort incomplete')
    for row in rows:
        case=row['case_id'];clean=ROOT/'outputs/evaluation/development/clean'/case
        restored=ROOT/'outputs/evaluation/mechanism_development'/row['condition']/case
        reference=read(clean/'RESULT.json')
        require(row['success']==reference['success'],'Both restore task differs: '+case)
        for file,keys in [('DECISIONS.json',['actions']),('STEPS.json',['command','eef_after'])]:
            a=read(clean/file);b=read(restored/file)
            require(len(a)==len(b),'Both restore trajectory length differs: '+case)
            for k in keys:require(np.array_equal(np.asarray([x[k] for x in a]),np.asarray([x[k] for x in b])),'Both restore '+k+' differs: '+case)
    atomic(dict(episodes=80,all_task_results_equal=True,all_action_blocks_equal=True,all_controller_commands_equal=True,all_eef_positions_equal=True),ROOT/'reports/M04_BOTH_IDENTITY.json')


def ready():
    """Bounded independent jobs; every seed is evaluated regardless of score."""
    out=[]
    for seed in (42,43,44):
        if complete(ROOT/f'outputs/training/D07-seed{seed}/COMPLETE.json'):
            for scope in ('development','full','confirmation_extra'):
                if not complete(ROOT/f'reports/D07-seed{seed}-{scope}_COMPLETE.json'):
                    # Full can reuse development only after the latter finishes.
                    out.append((f'D07-s{seed}-{scope}',['eval','--id','D07','--seed',str(seed),'--scope',scope],12,6));break
        else:out.append((f'D07-s{seed}-train',['train','--id','D07','--seed',str(seed)],16,6))
    if not complete(ROOT/'reports/M04_COMPLETE.json'):out.append(('M04-restore-O03',['diagnose','--id','M04'],12,6))
    return out


def finished():
    return complete(ROOT/'reports/M04_COMPLETE.json') and all(complete(ROOT/f'reports/D07-seed{s}-{scope}_COMPLETE.json') for s in (42,43,44) for scope in ('development','full','confirmation_extra'))
