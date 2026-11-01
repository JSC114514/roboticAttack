# 发布阅读说明：同一当前物理状态的clean供体用于指定腕部分支恢复；这些有特权干预不进入原生PNG攻击成绩。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""M01/M02 privileged diagnostics, separate from nativePNG attack performance."""
from pathlib import Path
import numpy as np
from PIL import Image
from .common import ROOT,read,require,complete,finish,atomic,Monitor
from .runtime import load_wrapper,Engine
from .evaluate import cases,episode


def run(args):
    identity=args.id;require(identity=='M04','This supplement only diagnoses frozen O03')
    if complete(ROOT/'reports'/f'{identity}_COMPLETE.json'):return
    with Monitor(identity+'-diagnostic') as monitor:
        require(complete(ROOT/'reports/P02_development_COMPLETE.json'),'Clean reference incomplete')
        donors=['O03']
        for method in donors:require(complete(ROOT/'outputs/training'/f'{method}-seed42'/'COMPLETE.json'),'Missing frozen diagnostic PNG '+method)
        w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False);results=[]
        for method in donors:
            p=ROOT/'outputs/training'/f'{method}-seed42'/'final.png';png=np.asarray(Image.open(p).convert('RGB'),np.uint8)
            conditions=[('wrist','dino'),('wrist','siglip'),('wrist','both')]
            for view,restoration in conditions:
                prefix='M04F32'
                label=f'{prefix}-{method}-{view}'+('-restore-'+restoration if restoration else'')
                for case in cases('development'):
                    results.append(episode(w,engine,r,case,label,42,png,view,monitor,'mechanism_development',intervention=restoration));monitor.update(phase='restoring_'+restoration,completed=len(results),total=240)
        from .supplement import verify_both_identity
        verify_both_identity(results)
        output=ROOT/'reports/M04_RESULTS_FLOAT32.json';atomic(dict(unit=identity,results=results,privileged_diagnostic=True,not_native_attack_score=True,controller_action_dtype='float32'),output)
        finish(ROOT/'reports'/f'{identity}_COMPLETE.json',[output,ROOT/'reports/M04_BOTH_IDENTITY.json'],episodes=len(results))
