# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""V01/V02 frozen selection and bounded seed/full-suite follow-up scheduling."""
from .common import ROOT,config,read,atomic,complete,finish,digest

PRIMARY=['B01','B02','B03','B04','B05','O01','O03','O05','S04','S05','S06','S07']


def primary_complete():
    ready=all(complete(ROOT/'reports'/f'{m}-seed42-development_COMPLETE.json') for m in PRIMARY) and complete(ROOT/'reports/M01_COMPLETE.json') and complete(ROOT/'reports/M02_COMPLETE.json') and complete(ROOT/'reports/M03_INTERACTION_REPLAY_COMPLETE.json')
    if ready and not complete(ROOT/'reports/M03_COMPLETE.json'):
        from .mechanism_analysis import run
        run()
    return ready and complete(ROOT/'reports/M03_COMPLETE.json')


def freeze_selection():
    path=ROOT/'reports/V_SELECTION.json'
    if path.exists():return read(path)
    if not primary_complete():return None
    from .analyze import summarize
    values={r['condition'].removesuffix('-seed42'):r for r in summarize('development')};policy=read(ROOT/'configs/VALIDATION_POLICY.json');chosen=set(policy['core_methods']);baseline=values['B01']['suite_task_equal_casr']
    for name in ['auxiliary_family','sampling_family']:
        candidates=[m for m in policy[name] if values[m]['suite_task_equal_casr'] is not None]
        if not candidates:continue
        best=sorted(candidates,key=lambda m:(-values[m]['suite_task_equal_casr'],m))[0]
        if values[best]['suite_task_equal_casr']>baseline:
            chosen.add(best);chosen.update(policy['required_sampling_controls'].get(best,[]))
    if values['B05']['suite_task_equal_casr']>values['B03']['suite_task_equal_casr']:chosen.add('B05')
    result=dict(methods=sorted(chosen),seeds=[42,43,44],policy=policy,development_results_sha256=digest(values),selection_frozen=True)
    atomic(result,path);return result


def followups():
    selection=freeze_selection()
    if selection is None:return[]
    out=[]
    # Clean full-suite baseline is a P02 asset, independent of training seeds.
    for scope in ['full','confirmation_extra']:
        if not complete(ROOT/'reports'/f'P02_{scope}_COMPLETE.json'):out.append((f'P02-clean-{scope}',['clean-eval','--scope',scope],12,6))
    for method in selection['methods']:
        for seed in [43,44]:
            train=ROOT/'outputs/training'/f'{method}-seed{seed}'/'COMPLETE.json'
            dev=ROOT/'reports'/f'{method}-seed{seed}-development_COMPLETE.json'
            if complete(train) and not complete(dev):out.append((f'{method}-s{seed}-dev',['eval','--id',method,'--seed',str(seed),'--scope','development'],12,3))
            elif not complete(train):out.append((f'{method}-s{seed}-train',['train','--id',method,'--seed',str(seed)],16,12))
    v1_done=all(complete(ROOT/'reports'/f'{m}-seed{s}-development_COMPLETE.json') for m in selection['methods'] for s in [43,44])
    if v1_done:
        if not complete(ROOT/'reports/V01_COMPLETE.json'):finish(ROOT/'reports/V01_COMPLETE.json',[ROOT/'reports/V_SELECTION.json'],methods=selection['methods'],seeds=[43,44])
        for scope in ['full','confirmation_extra']:
            if not complete(ROOT/'reports'/f'P02_{scope}_COMPLETE.json'):continue
            for method in selection['methods']:
                for seed in [42,43,44]:
                    if not complete(ROOT/'reports'/f'{method}-seed{seed}-{scope}_COMPLETE.json'):out.append((f'{method}-s{seed}-{scope}',['eval','--id',method,'--seed',str(seed),'--scope',scope],12,6))
    v2_done=v1_done and all(complete(ROOT/'reports'/f'{m}-seed{s}-{scope}_COMPLETE.json') for m in selection['methods'] for s in [42,43,44] for scope in ['full','confirmation_extra'])
    if v2_done:
        if not complete(ROOT/'reports/V02_COMPLETE.json'):finish(ROOT/'reports/V02_COMPLETE.json',[ROOT/'reports/V_SELECTION.json'],methods=selection['methods'],unique_starts=480)
        if not complete(ROOT/'reports/V03_COMPLETE.json'):
            from .finalize import run
            run()
    return out
