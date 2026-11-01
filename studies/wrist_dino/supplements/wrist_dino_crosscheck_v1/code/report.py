# 发布阅读说明：报告完整配对条件的成功数与CASR；原生PNG、恢复诊断与历史动作回放分开计数。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Native task/CASR tables, paired results and actual Slurm allocation costs.

Video replays and privileged branch restorations never enter attack rankings.
Missing or exception episodes are absent, not counted as task failures/successes.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from common import ROOT,frozen,read,atomic,complete

def paired_difference(rows,model,method,reference,draws=2000):
    """Mean seeds per physical start; bootstrap tasks→starts within suite."""
    selected=[r for r in rows if r['model']==model and r['native_png']]
    index={(r['condition'],r['case_id']):r for r in selected};groups={}
    for case in frozen()['cases']:
        key=case['case_id']
        if not index['clean',key]['success']:continue
        delta=np.mean([float(not index[f'{method}-seed{s}',key]['success'])-float(not index[f'{reference}-seed{s}',key]['success']) for s in [42,43,44]])
        groups.setdefault(case['suite'],{}).setdefault(case['task_id'],[]).append(float(delta))
    if not groups:return dict(model=model,method=method,reference=reference,difference=None,ci95=None)
    point=float(np.mean([np.mean([np.mean(v) for v in tasks.values()]) for tasks in groups.values()]))
    rng=np.random.default_rng(42);samples=[]
    for _ in range(draws):
        suite_values=[]
        for tasks in groups.values():
            values=list(tasks.values());means=[]
            for i in rng.integers(0,len(values),len(values)):
                v=np.asarray(values[i]);means.append(float(np.mean(v[rng.integers(0,len(v),len(v))])))
            suite_values.append(float(np.mean(means)))
        samples.append(float(np.mean(suite_values)))
    return dict(model=model,method=method,reference=reference,difference=point,ci95=np.percentile(samples,[2.5,97.5]).tolist(),resampling='suite-stratified task->start; seeds averaged within start')

def aggregate(table,raw):
    """Write formal three-seed summaries only for a complete target grid."""
    output=[];lines=['# 两目标的三种子确认','', 'CASR先任务、再套件等权，最后平均三个训练种子。TaskSR统计全部80物理起点。','', '| 目标 | 方法 | 平均CASR | 种子标准差(pp) | 平均任务成功率 | Spatial | Object | Goal | Long |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    pairs=[]
    for model in ['vla_adapter_pro','pi05']:
        r=[x for x in raw if x['model']==model and x['native_png']]
        if len(r)!=1280:continue
        for label in ['INITIAL','LR004-B01','LR004-O03','B04','W025-D07']:
            suites=[];seed_scores=[]
            for suite in frozen()['config']['suites']:
                scores=[x['casr'] for x in table if x['model']==model and x['suite']==suite and x['condition'].rsplit('-seed',1)[0]==label and x['casr'] is not None]
                suites.append(float(np.mean(scores)) if scores else None)
            for seed in [42,43,44]:
                scores=[x['casr'] for x in table if x['model']==model and x['condition']==f'{label}-seed{seed}' and x['casr'] is not None]
                if scores:seed_scores.append(float(np.mean(scores)))
            learned=[x for x in r if x['condition'].rsplit('-seed',1)[0]==label]
            score=float(np.mean(seed_scores)) if seed_scores else None
            item=dict(model=model,method=label,mean_macro_casr=score,seed_std=float(np.std(seed_scores,ddof=1)) if len(seed_scores)>1 else 0.,mean_task_success_rate=float(np.mean([x['success'] for x in learned])),by_suite=suites,physical_starts=80,training_seeds=[42,43,44]);output.append(item)
            fmt=lambda v:'无分母' if v is None else f'{100*v:.2f}%'
            lines.append(f'| {model} | {label} | {fmt(score)} | {100*item["seed_std"]:.2f} | {fmt(item["mean_task_success_rate"])} | '+' | '.join(fmt(v) for v in suites)+' |')
        for method,base in [('LR004-O03','LR004-B01'),('W025-D07','B04')]:pairs.append(paired_difference(raw,model,method,base))
    if output:
        atomic(dict(rows=output,paired=pairs),ROOT/'reports/AGGREGATED_RESULTS.json')
        lines+=['','## 参数匹配比较','']
        for p in pairs:
            if p['difference'] is not None:lines.append(f'- {p["model"]}，{p["method"]} 对 {p["reference"]}：{100*p["difference"]:+.2f} pp，配对95%区间[{100*p["ci95"][0]:.2f}, {100*p["ci95"][1]:.2f}] pp。')
        (ROOT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

def result_rows():
    out=[]
    for model in ['vla_adapter_pro','pi05']:
        for p in (ROOT/f'outputs/{model}').glob('*/*/RESULT.json'):
            if (p.parent/'COMPLETE.json').exists():out.append(read(p))
    return out

def report(sample):
    a=frozen();rows=result_rows();transfer=[r for r in rows if r['native_png']];mechanism=[r for r in rows if not r['native_png']]
    replays=sum(1 for p in (ROOT/'outputs/video_replay').glob('*/COMPLETE.json'))
    table=[];lines=['# DINO机制补充验证：跨模型原生PNG','', 'CASR仅以同目标模型、同物理起点的clean成功为分母。逐套件先任务等权；三个训练种子重复同一批物理起点。','', '| 模型 | 套件 | 条件 | 任务成功 | clean成功 | 诱发失败 | CASR |','|---|---|---|---:|---:|---:|---:|']
    for model in ['vla_adapter_pro','pi05']:
        for suite in a['config']['suites']:
            group=[r for r in transfer if r['model']==model and r['suite']==suite]
            clean={(r['task_id'],r['state_index']):r for r in group if r['condition']=='clean'}
            for condition in [p['id'] for p in a['config']['patches']]:
                current=[r for r in group if r['condition']==condition]
                if len(current)!=20 or len(clean)!=20:continue
                qualified=[r for r in current if clean[r['task_id'],r['state_index']]['success']]
                tasks=defaultdict(list)
                for r in qualified:tasks[r['task_id']].append(float(not r['success']))
                score=float(np.mean([np.mean(v) for v in tasks.values()])) if tasks else None
                value=dict(model=model,suite=suite,condition=condition,episodes=20,success=sum(r['success'] for r in current),clean_success=len(qualified),induced_failures=sum(not r['success'] for r in qualified),casr=score)
                table.append(value);s='无分母' if score is None else f'{score*100:.2f}%'
                lines.append(f'| {model} | {suite} | {condition} | {value["success"]}/20 | {len(qualified)} | {value["induced_failures"]} | {s} |')
    atomic(dict(rows=table),ROOT/'reports/TRANSFER_RESULTS.json');(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    aggregate(table,transfer)
    accounting={}
    for line in sample['probes']['accounting']['stdout'].splitlines():
        v=line.split('|')
        if len(v)>=6 and v[3].isdigit():accounting[v[0]]=dict(state=v[2],elapsed_s=int(v[3]),tres=v[5])
    submissions=read(ROOT/'reports/SUBMISSIONS.json') if (ROOT/'reports/SUBMISSIONS.json').exists() else []
    costs=[]
    for x in submissions:
        v=accounting.get(x['job_id'])
        if v:costs.append(dict(unit=x['id'],job_id=x['job_id'],gpu_hours=0. if x['resource']['type']=='cpu' else v['elapsed_s']/3600,cpu_only=x['resource']['type']=='cpu',**v))
    atomic(dict(rows=costs,total_gpu_hours=sum(x['gpu_hours'] for x in costs)),ROOT/'reports/GPU_COSTS.json')
    # Restorations report actual clean-qualified rescue, not just feature drift.
    restores=[]
    for model in ['vla_adapter_pro']:
        for suite in a['config']['suites']:
            for label in ['LR004-B01-seed42','LR004-O03-seed42']:
                base={(r['task_id'],r['state_index']):r for r in transfer if r['model']==model and r['suite']==suite and r['condition']==label}
                clean={(r['task_id'],r['state_index']):r for r in transfer if r['model']==model and r['suite']==suite and r['condition']=='clean'}
                for branch in ['dino','siglip','both']:
                    current=[r for r in mechanism if r['model']==model and r['suite']==suite and r['condition']==label+'-restore-'+branch]
                    if len(current)!=20 or len(base)!=20 or len(clean)!=20:continue
                    qualified=[r for r in current if clean[r['task_id'],r['state_index']]['success'] and not base[r['task_id'],r['state_index']]['success']]
                    restores.append(dict(model=model,suite=suite,method=label,branch=branch,success=sum(r['success'] for r in current),original_failures=len(qualified),rescued=sum(r['success'] for r in qualified),new_losses=sum(base[r['task_id'],r['state_index']]['success'] and not r['success'] for r in current)))
    atomic(dict(rows=restores),ROOT/'reports/RESTORATION_RESULTS.json')
    index=[]
    for p in (ROOT/'outputs').glob('*/*/*/RESULT.json'):
        if not (p.parent/'COMPLETE.json').exists():continue
        v=read(p)
        for path,h in v.get('video',{}).get('files',{}).items():index.append(dict(mp4=path,sha256=h,model=v.get('model'),condition=v.get('condition'),case_id=v.get('case_id'),success=v.get('success')))
    for p in (ROOT/'outputs/video_replay').glob('*/RESULT.json'):
        if not (p.parent/'COMPLETE.json').exists():continue
        v=read(p)
        for path,h in v['video']['files'].items():index.append(dict(mp4=path,sha256=h,model='historical_command_replay',condition=v['source']['condition'],case_id=v['source']['case_id'],success=v['source']['success']))
    atomic(dict(rows=index),ROOT/'reports/VIDEO_INDEX.json')
    return dict(transfer_complete=len(transfer),transfer_expected=2560,mechanism_complete=len(mechanism),mechanism_max=480,replays_complete=replays,replays_expected=312,gpu_hours=sum(x['gpu_hours'] for x in costs))
