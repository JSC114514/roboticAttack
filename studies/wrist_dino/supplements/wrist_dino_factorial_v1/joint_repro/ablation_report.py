# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU factorial analysis: preserve seed pairing and complete-episode units.

The factorial interaction asks whether adding DINO has a different effect with
seven-dimensional versus single-axis action supervision. Every bootstrap draw
uses the same tasks/starts for all four cells and all three fixed seeds.
"""
from collections import defaultdict
import numpy as np
from .common import ROOT,read,atomic,finish,require,sha,complete
from .evaluate import cases

METHODS=['B01','O03','B04','D07']
CONTRASTS={'DINO_with_single_axis':[-1,1,0,0],
           'DINO_with_seven_dimensions':[0,0,-1,1],
           'seven_vs_single_with_DINO':[0,-1,0,1],
           'factorial_interaction':[1,-1,-1,1]}


def paired(scope,draws=2000):
    """Return task-equal contrasts and suite-stratified task→start intervals."""
    groups=defaultdict(lambda:defaultdict(list));data=[]
    for case in cases(scope):
        folder=ROOT/'outputs/evaluation'/scope;key=case['case_id']
        if not read(folder/'clean'/key/'RESULT.json')['success']:continue
        vector=[]
        for method in METHODS:
            vector.append(np.mean([not read(folder/f'{method}-seed{s}'/key/'RESULT.json')['success'] for s in (42,43,44)]))
        groups[case['suite']][case['task_id']].append(vector)
    grouped=[[np.asarray(values,float) for _,values in sorted(tasks.items())] for _,tasks in sorted(groups.items())]
    point=np.mean([np.mean([v.mean(0) for v in tasks],0) for tasks in grouped],0)
    rng=np.random.default_rng(20261029);samples=[]
    for _ in range(draws):
        sv=[]
        for tasks in grouped:
            tv=[]
            for index in rng.integers(len(tasks),size=len(tasks)):
                a=tasks[index];tv.append(a[rng.integers(len(a),size=len(a))].mean(0))
            sv.append(np.mean(tv,0))
        samples.append(np.mean(sv,0))
    samples=np.asarray(samples)
    return {name:dict(difference_pp=float(100*point@weights),ci95_pp=(100*np.percentile(samples@weights,[2.5,97.5])).tolist(),draws=draws,seed=20261029) for name,weights in CONTRASTS.items()}


def run():
    """Write final tables/figures once all predeclared work is complete."""
    from .supplement import finished
    require(finished(),'Frozen supplement work incomplete')
    if complete(ROOT/'reports/ABLATION_COMPLETE.json'):return
    from .analyze import summarize
    lines=['# 腕部DINO消融：动作维度与表征扰动','',
        '四个格子：单轴、单轴+DINO、全7维、全7维+DINO。旧三组复用原始完整资产，新组D07三个种子全部报告。',
        '模型、数据池、样本顺序、EOT、优化器和25,000次更新相同；DINO系数固定为原O03的22.855948214853818。','']
    contrasts={};means={};source_files=[]
    for scope in ('development','full','confirmation_extra'):
        rows=summarize(scope);source_files.append(ROOT/f'reports/{scope}_SUMMARY.json')
        lines += ['## '+scope,'','| 方法 | seed | 完成 | 任务成功 | clean成功分母 | 诱发失败 | pooled CASR | 任务等权CASR |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['condition']=='clean':continue
            method,seed=row['condition'].split('-seed')
            lines.append(f'| {method} | {seed} | {row["completed"]}/{row["expected"]} | {row["success"]} | {row["clean_success"]} | {row["induced_failures"]} | {100*row["pooled_casr"]:.2f}% | {100*row["suite_task_equal_casr"]:.2f}% |')
        means[scope]={m:float(np.mean([v['suite_task_equal_casr'] for v in rows if v['condition'].startswith(m+'-seed')])) for m in METHODS}
        if scope!='development':
            contrasts[scope]=paired(scope)
            lines+=['','配对差异按同一任务、起点和种子计算；区间按套件内任务→起点重采样，种子在每次抽样中保持配对。','',
                '| 比较 | 差异（百分点） | 95%区间 |','|---|---:|---|']
            for key,value in contrasts[scope].items():lines.append(f'| {key} | {value["difference_pp"]:.2f} | {value["ci95_pp"][0]:.2f}—{value["ci95_pp"][1]:.2f} |')
    diagnostics=read(ROOT/'reports/M04_RESULTS_FLOAT32.json')['results'];restores={}
    for branch in ('dino','siglip','both'):
        rows=[x for x in diagnostics if x['intervention']==branch];rescued=lost=qualified_failures=0
        for row in rows:
            case=row['case_id'];base=ROOT/'outputs/evaluation/development'
            clean=read(base/'clean'/case/'RESULT.json')['success'];patched=read(base/'O03-seed42'/case/'RESULT.json')['success']
            qualified_failures+=int(clean and not patched)
            rescued+=int(clean and not patched and row['success'])
            lost+=int(clean and patched and not row['success'])
        restores[branch]=dict(episodes=len(rows),task_successes=sum(x['success'] for x in rows),original_clean_qualified_failures=qualified_failures,rescued_failures=rescued,new_losses=lost)
    lines+=['','## O03 seed42分支恢复','',
        '恢复使用同一当前状态的干净供体，属于机制诊断，不计入真实PNG攻击成绩。双分支恢复逐回合验证动作、控制命令、末端轨迹和任务结果与clean一致。','',
        '| 恢复分支 | 任务成功/80 | 原补丁失败中救回 | 新增损失 |','|---|---:|---:|---:|']
    for k,v in restores.items():lines.append(f'| {k} | {v["task_successes"]}/80 | {v["rescued_failures"]}/{v["original_clean_qualified_failures"]} | {v["new_losses"]} |')
    lines+=['','训练和闭环结果不按分数删组；新增Object逐任务差异见full_SUMMARY.json。GPU_COST.csv只记录本次新作业，历史对照成本见父目录。']
    result=ROOT/'reports/FACTORIAL_RESULTS.json';atomic(dict(methods=METHODS,seeds=[42,43,44],task_equal_means=means,paired_contrasts=contrasts,restorations=restores),result)
    path=ROOT/'RESULTS.md';path.write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[];out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
    for scope in ('full','confirmation_extra'):
        fig,ax=plt.subplots(figsize=(7.5,4),layout='constrained')
        rows=read(ROOT/f'reports/{scope}_SUMMARY.json')['conditions']
        for j,seed in enumerate((42,43,44)):
            y=[100*next(v for v in rows if v['condition']==f'{m}-seed{seed}')['suite_task_equal_casr'] for m in METHODS]
            ax.scatter(np.arange(4)+(j-1)*.15,y,label=f'Seed {seed}')
        ax.set(xticks=range(4),xticklabels=['Single axis','Single + DINO','Seven DoF','Seven + DINO'],ylim=(0,105),ylabel='Suite/task-equal CASR (%)',title=scope);ax.legend();ax.grid(axis='y',alpha=.2)
        for suffix in ('png','pdf'):
            f=out/f'{scope}_factorial.{suffix}';fig.savefig(f,dpi=250);figures.append(f)
        plt.close(fig)
    finish(ROOT/'reports/ABLATION_COMPLETE.json',[path,result,ROOT/'GPU_COST.csv',ROOT/'configs/ABLATION.json',ROOT/'reports/M04_RESULTS_FLOAT32.json',ROOT/'reports/M04_BOTH_IDENTITY.json',*source_files,*figures],new_training_seeds=[42,43,44],new_native_episodes=1440,diagnostic_episodes=240)
