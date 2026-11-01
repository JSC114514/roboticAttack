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
    identity=args.id;require(identity in ['M01','M02'],'Unknown diagnostic unit')
    if complete(ROOT/'reports'/f'{identity}_COMPLETE.json'):return
    with Monitor(identity+'-diagnostic') as monitor:
        require(complete(ROOT/'reports/P02_development_COMPLETE.json'),'Clean reference incomplete')
        donors=['B01','B02'] if identity=='M01' else ['B01']
        for method in donors:require(complete(ROOT/'outputs/training'/f'{method}-seed42'/'COMPLETE.json'),'Missing frozen diagnostic PNG '+method)
        w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False);results=[]
        for method in donors:
            p=ROOT/'outputs/training'/f'{method}-seed42'/'final.png';png=np.asarray(Image.open(p).convert('RGB'),np.uint8)
            conditions=[('front' if method=='B01' else 'wrist',None)] if identity=='M01' else [('wrist','dino'),('wrist','siglip'),('wrist','both')]
            for view,restoration in conditions:
                prefix='M02F32' if identity=='M02' else identity
                label=f'{prefix}-{method}-{view}'+('-restore-'+restoration if restoration else'')
                for case in cases('development'):
                    results.append(episode(w,engine,r,case,label,42,png,view,monitor,'mechanism_development',intervention=restoration));monitor.update(completed=len(results),total=160 if identity=='M01' else 240)
        filename='M02_RESULTS_FLOAT32.json' if identity=='M02' else f'{identity}_RESULTS.json'
        output=ROOT/'reports'/filename;atomic(dict(unit=identity,results=results,privileged_diagnostic=identity=='M02',not_native_attack_score=True,controller_action_dtype='float32'),output)
        finish(ROOT/'reports'/f'{identity}_COMPLETE.json',[output],episodes=len(results))
