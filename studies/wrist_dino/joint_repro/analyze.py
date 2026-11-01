# 发布阅读说明：分别计算任务成功率、clean成功起点上的CASR和实际成本；保留缺失及无定义分母，不把缺失视为成功。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU paired reports: actual task success, clean-qualified CASR and GPU cost.

Aggregate per task then per suite, separately from pooled episode counts. Keep
missing episodes outside success/failure denominators, never treat them as
success. Bootstrap resamples tasks then initial states, notframes/actionslots.
"""
from pathlib import Path
import csv
import re
import time
import numpy as np
from .common import ROOT,config,read,atomic,complete,finish,digest
from .evaluate import cases


def gpu_cost(row,allocation=None,requested_gpu=False):
    """Charge allocated GPUs even if startup failed before CUDA initialized.

    Slurm allocation time is authoritative. Process time is an explicitly
    labelled fallback; device initialization is not proof of allocation size.
    A generic GPU TRES takes precedence over its typed entries to avoid
    counting the same card twice.
    """
    if allocation is not None:
        tres=allocation.get('allocated_tres','') or ''
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        count=int(match.group(1)) if match else sum(int(v) for v in re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)',tres))
    else:count=int(bool(row.get('gpu')) or requested_gpu)
    actual=allocation.get('elapsed_s') if allocation is not None else None
    elapsed=actual if actual is not None else row['elapsed_s']
    return dict(allocated_gpu_count=count,gpu_hours=count*elapsed/3600,gpu_cost_time_source='slurm_allocated_elapsed' if actual is not None else 'process_elapsed_fallback')


def bootstrap(rows,seed=42,draws=2000):
    """Paired task→start resampling within fixed suites for CASR."""
    groups={}
    for row in rows:
        if row['clean_success']:groups.setdefault(row['suite'],{}).setdefault(row['task_id'],[]).append(1-int(row['success']))
    if not groups:return None
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        suites=[]
        for tasks in groups.values():
            ids=list(tasks);values=[]
            for idx in rng.integers(0,len(ids),len(ids)):
                a=tasks[ids[idx]];values.append(float(np.mean(np.asarray(a)[rng.integers(0,len(a),len(a))])))
            suites.append(float(np.mean(values)))
        samples.append(float(np.mean(suites)))
    return np.percentile(samples,[2.5,97.5]).tolist()


def summarize(scope='development'):
    starts=cases(scope);groups={};expected={};missing=[];records=[]
    root=ROOT/'outputs/evaluation'/scope
    if root.exists():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():continue
            valid=[]
            selected=[r for r in starts if r['state'] in config()['dev_states'][:2]] if scope=='development' and folder.name.startswith('INITIAL-') else starts
            expected[folder.name]=len(selected)
            for case in selected:
                p=folder/case['case_id'];clean=root/'clean'/case['case_id']
                if not (p/'COMPLETE.json').exists():missing.append(dict(condition=folder.name,case_id=case['case_id']));continue
                value=read(p/'RESULT.json');value['clean_success']=bool(read(clean/'RESULT.json')['success']) if (clean/'COMPLETE.json').exists() else False
                value['clean_reference_available']=(clean/'COMPLETE.json').exists();valid.append(value);records.append(value)
            groups[folder.name]=valid
    summaries=[]
    for name,rows in groups.items():
        cache=ROOT/'reports/stat_cache'/f'{scope}-{name}.json';key=digest(rows)
        if cache.exists() and read(cache)['input_sha256']==key:
            saved=read(cache)['summary'];saved['expected']=expected[name];saved['evaluation_role']='engineering_smoke16' if scope=='development' and name.startswith('INITIAL-') else 'performance'
            summaries.append(saved);continue
        task_rows=[]
        for suite in config()['suites']:
            for task in sorted({r['task_id'] for r in rows if r['suite']==suite}):
                r=[x for x in rows if x['suite']==suite and x['task_id']==task];qualified=[x for x in r if x['clean_success'] and x['clean_reference_available']]
                task_rows.append(dict(condition=name,suite=suite,task_id=task,completed=len(r),success=sum(x['success'] for x in r),clean_success=len(qualified),induced_failures=sum(not x['success'] for x in qualified),casr=sum(not x['success'] for x in qualified)/len(qualified) if qualified else None))
        suites=[]
        for suite in config()['suites']:
            r=[x for x in task_rows if x['suite']==suite];defined=[x['casr'] for x in r if x['casr'] is not None]
            if r:suites.append(dict(suite=suite,task_equal_success=float(np.mean([x['success']/x['completed'] for x in r])),task_equal_casr=float(np.mean(defined)) if defined else None,undefined_casr_tasks=[x['task_id'] for x in r if x['casr'] is None]))
        qualified=[x for x in rows if x['clean_success'] and x['clean_reference_available']];defined=[x['task_equal_casr'] for x in suites if x['task_equal_casr'] is not None]
        summaries.append(dict(condition=name,completed=len(rows),expected=expected[name],evaluation_role='engineering_smoke16' if scope=='development' and name.startswith('INITIAL-') else 'performance',success=sum(x['success'] for x in rows),clean_success=len(qualified),induced_failures=sum(not x['success'] for x in qualified),pooled_casr=sum(not x['success'] for x in qualified)/len(qualified) if qualified else None,suite_task_equal_casr=float(np.mean(defined)) if defined else None,casr_ci95=bootstrap(rows) if name!='clean' else [0.,0.],by_suite=suites,by_task=task_rows))
    for summary in summaries:
        atomic(dict(input_sha256=digest(groups[summary['condition']]),summary=summary),ROOT/'reports/stat_cache'/f'{scope}-{summary["condition"]}.json')
    accounting={}
    audit=ROOT/'reports/LAST_RESOURCE_AUDIT.json'
    if audit.exists():
        for line in read(audit)['probes']['accounting']['stdout'].splitlines():
            a=line.split('|')
            if len(a)>=6 and a[3].isdigit():accounting[a[0]]=dict(elapsed_s=int(a[3]),state=a[2],allocated_tres=a[5])
    costs=[]
    invalid=read(ROOT/'reports/INVALIDATED_RESULTS.json') if (ROOT/'reports/INVALIDATED_RESULTS.json').exists() else []
    validity={str(r['job_id']):r['status'] for r in invalid}
    submission_path=ROOT/'reports/SUBMISSIONS.json'
    gpu_jobs={str(r['job_id']) for r in read(submission_path) if r.get('resource',{}).get('type')} if submission_path.exists() else set()
    for p in sorted((ROOT/'reports/costs').glob('*.json')):
        r=read(p);allocation=accounting.get(str(r.get('job_id')));a=allocation or {};costs.append(dict(slurm_elapsed_s=a.get('elapsed_s'),slurm_state=a.get('state'),allocated_tres=a.get('allocated_tres'),stage=r['name'],job_id=r.get('job_id'),status=r['status'],result_validity=validity.get(str(r.get('job_id')),'NOT_INVALIDATED'),elapsed_s=r['elapsed_s'],gpu=r.get('gpu'),**gpu_cost(r,allocation,str(r.get('job_id')) in gpu_jobs),rss_hwm_bytes=r.get('rss_hwm_bytes'),cuda_peak_allocated_bytes=r.get('cuda_peak_allocated_bytes'),error=r.get('error')))
    if costs:
        with (ROOT/'GPU_COST.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(costs[0]));w.writeheader();w.writerows(costs)
    atomic(dict(scope=scope,conditions=summaries,missing=missing,costs=costs,time_unix=time.time()),ROOT/'reports'/f'{scope}_SUMMARY.json')
    lines=['# 统一协议复核：阶段结果','',f'评测范围：{scope}。任务成功率越高越好，CASR越高表示补丁诱发失败越多。','', '| 条件 | 完成/计划 | 任务成功 | clean成功分母 | 诱发失败 | pooled CASR | 套件/任务等权CASR |','|---|---:|---:|---:|---:|---:|---:|']
    def pct(x):return '未定义' if x is None else f'{100*x:.2f}%'
    for s in summaries:
        label=s['condition']+('（16回合接口冒烟）' if s['evaluation_role']=='engineering_smoke16' else '')
        lines.append(f'| {label} | {s["completed"]}/{s["expected"]} | {s["success"]} | {s["clean_success"]} | {s["induced_failures"]} | {pct(s["pooled_casr"])} | {pct(s["suite_task_equal_casr"])} |')
    lines+=['','未完成回合保留缺失记录，不当作成功。内部恢复条件属于独立机制诊断，不并入原生PNG攻击排名。','', '## 方法状态', (ROOT/'METHOD_PROVENANCE.md').read_text()]
    # Scoped reports are immutable with respect to other evaluation scopes:
    # a full-suite summary must not replace the development results navigation.
    (ROOT/'reports'/f'{scope}_RESULTS.md').write_text('\n'.join(lines)+'\n')
    if scope=='development':(ROOT/'QUICK_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summaries


def run(args):
    print(__import__('json').dumps(summarize(args.scope),ensure_ascii=False,indent=2))
