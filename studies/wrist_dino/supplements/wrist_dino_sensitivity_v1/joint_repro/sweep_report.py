# 发布阅读说明：开发集选配置，确认集不选种子或检查点；报告固定三个种子与任务→起点配对差异。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Partial/full ablation tables; aggregate tasks before suites, not action slots."""
import csv
import time
import numpy as np
from .common import ROOT,read,atomic,complete,config,require,sha
from .sweep import study,resolve
from .evaluate import cases


def metrics(identity,seed=42,scope='development'):
    """Return None until every expected native episode and clean pair exists."""
    rows=[];clean_rows=[]
    condition='clean' if identity=='clean' else f'{identity}-seed{seed}'
    for case in cases(scope):
        folder=ROOT/'outputs/evaluation'/scope/condition/case['case_id']
        clean=ROOT/'outputs/evaluation'/scope/'clean'/case['case_id']
        if not (folder/'COMPLETE.json').is_file() or not (clean/'COMPLETE.json').is_file():return None
        row=read(folder/'RESULT.json');reference=read(clean/'RESULT.json')
        require(row['seed']==reference['seed']==case['seed'],'Unpaired evaluation seed')
        require(row['checkpoint']==reference['checkpoint']==config()['model_dir'],'Unpaired checkpoint')
        rows.append(row);clean_rows.append(reference)
    tasks=[];suites=[]
    for suite in config()['suites']:
        for task in sorted({r['task_id'] for r in rows if r['suite']==suite}):
            indices=[i for i,r in enumerate(rows) if r['suite']==suite and r['task_id']==task]
            valid=[i for i in indices if clean_rows[i]['success']]
            tasks.append(dict(suite=suite,task_id=task,n=len(indices),task_success=sum(rows[i]['success'] for i in indices),clean_success=len(valid),failures=sum(not rows[i]['success'] for i in valid),casr=sum(not rows[i]['success'] for i in valid)/len(valid) if valid else None))
        values=[r['casr'] for r in tasks if r['suite']==suite and r['casr'] is not None]
        suites.append(dict(suite=suite,casr=float(np.mean(values)) if values else None))
    qualified=[i for i,r in enumerate(clean_rows) if r['success']]
    return dict(identity=identity,seed=seed,scope=scope,count=len(rows),success=sum(r['success'] for r in rows),clean_success=len(qualified),failures=sum(not rows[i]['success'] for i in qualified),pooled_casr=sum(not rows[i]['success'] for i in qualified)/len(qualified) if qualified else None,macro_casr=float(np.mean([r['casr'] for r in suites if r['casr'] is not None])),by_suite=suites,by_task=tasks)


def trained_labels():
    for row in study()['catalog'].values():
        for label in row['checkpoint_ids'].values():
            if complete(ROOT/f'outputs/training/{label}-seed42/COMPLETE.json'):yield label


def training_seconds(identity,seed=42):
    """Recorded phase time breaks score ties; no hidden test-set selection."""
    row=resolve(identity);path=ROOT/f'outputs/training/{identity}-seed{seed}/TRAINING.csv'
    if not path.exists():return float('inf')
    with path.open() as stream:return sum(float(r['elapsed_outer_s']) for r in csv.DictReader(stream))


def report():
    rows=[]
    for label in sorted(trained_labels()):
        result=metrics(label)
        if result is not None:rows.append(result)
    atomic(dict(time_unix=time.time(),rows=rows),ROOT/'reports/SCREEN_RESULTS.json')
    fields=['identity','group','block','lr','batch','beta','scheduler_outer','checkpoint_outer','success','count','failures','clean_success','macro_casr','pooled_casr']
    with (ROOT/'reports/SCREEN_RESULTS.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for result in rows:
            row=resolve(result['identity']);writer.writerow({**{k:row.get(k) for k in fields if k in row},**{k:result[k] for k in fields if k in result},'checkpoint_outer':row.get('checkpoint_outer',500)})
    lines=['# 单变量消融结果','', 'CASR以同一起点clean成功为分母；主指标先任务等权、再套件等权。','', '| 配置 | 成功/回合 | 诱发失败/clean成功 | 等权CASR | Spatial | Object | Goal | Long |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[f'{100*x["casr"]:.2f}%' if x['casr'] is not None else '无分母' for x in r['by_suite']]
        lines.append(f'| {r["identity"]} | {r["success"]}/{r["count"]} | {r["failures"]}/{r["clean_success"]} | {100*r["macro_casr"]:.2f}% | '+' | '.join(values)+' |')
    lines+=['','所有候选按预定配置报告。参数选择只使用development，holdout不参与选择。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return rows


def cost_report():
    """Use sacct allocation seconds once per job, including failed startup jobs."""
    from .analyze import gpu_cost,share_job_cost
    path=ROOT/'reports/LAST_RESOURCE_AUDIT.json';accounting={}
    if path.exists():
        for line in read(path)['probes']['accounting']['stdout'].splitlines():
            a=line.split('|')
            if len(a)>=6 and a[3].isdigit():accounting[a[0]]=dict(elapsed_s=int(a[3]),state=a[2],allocated_tres=a[5])
    submissions=read(ROOT/'reports/SUBMISSIONS.json') if (ROOT/'reports/SUBMISSIONS.json').exists() else []
    jobs={str(r['job_id']) for r in submissions};rows=[];seen=set()
    for p in sorted((ROOT/'reports/costs').glob('*.json')):
        a=read(p);job=str(a.get('job_id'));allocation=accounting.get(job);seen.add(job)
        rows.append(dict(job_id=job,stage=a['name'],status=a['status'],elapsed_s=a['elapsed_s'],slurm_elapsed_s=allocation['elapsed_s'] if allocation else None,**gpu_cost(a,allocation,job in jobs),rss_hwm_bytes=a.get('rss_hwm_bytes'),cuda_peak_allocated_bytes=a.get('cuda_peak_allocated_bytes'),cuda_peak_reserved_bytes=a.get('cuda_peak_reserved_bytes'),error=a.get('error')))
    for job in sorted(jobs-seen):
        a=accounting.get(job)
        if not a:continue
        rows.append(dict(job_id=job,stage='allocation_without_finished_phase',status=a['state'],elapsed_s=a['elapsed_s'],slurm_elapsed_s=a['elapsed_s'],**gpu_cost(dict(elapsed_s=a['elapsed_s']),a,True),rss_hwm_bytes=None,cuda_peak_allocated_bytes=None,cuda_peak_reserved_bytes=None,error=None))
    rows=share_job_cost(rows)
    atomic(dict(rows=rows,total_gpu_hours=sum(r['gpu_hours'] for r in rows),time_unix=time.time()),ROOT/'reports/GPU_COSTS.json')
    if rows:
        with (ROOT/'GPU_COST.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def paired_interval(method,reference,seeds,draws=2000):
    """Average seeds per paired start; bootstrap tasks→starts within each suite."""
    grouped={};rng=np.random.default_rng(42)
    for case in cases('holdout'):
        base=ROOT/'outputs/evaluation/holdout';name=case['case_id']
        if not read(base/'clean'/name/'RESULT.json')['success']:continue
        values=[]
        for seed in seeds:
            a=read(base/f'{method}-seed{seed}'/name/'RESULT.json')
            b=read(base/f'{reference}-seed{seed}'/name/'RESULT.json')
            values.append(float(not a['success'])-float(not b['success']))
        grouped.setdefault(case['suite'],{}).setdefault(case['task_id'],[]).append(float(np.mean(values)))
    if not grouped:return None
    samples=[]
    for _ in range(draws):
        suite_values=[]
        for tasks in grouped.values():
            ids=list(tasks);values=[]
            for chosen in rng.integers(0,len(ids),len(ids)):
                a=np.asarray(tasks[ids[chosen]]);values.append(float(np.mean(a[rng.integers(0,len(a),len(a))])))
            suite_values.append(float(np.mean(values)))
        samples.append(float(np.mean(suite_values)))
    return np.percentile(samples,[2.5,97.5]).tolist()



def aggregate_seeds(rows):
    """Descriptive mean/std over training seeds, preserving task-equal CASR.

    The three seeds repeat the same80physical starts. They are not treated as
    240 independent starts; paired uncertainty is handled by paired_interval.
    """
    output=[]
    for identity in sorted({r['identity'] for r in rows}):
        selected=[r for r in rows if r['identity']==identity]
        values=[r['macro_casr'] for r in selected]
        suites=[]
        for suite in config()['suites']:
            a=[s['casr'] for r in selected for s in r['by_suite'] if s['suite']==suite and s['casr'] is not None]
            suites.append(dict(suite=suite,mean_casr=float(np.mean(a)) if a else None))
        output.append(dict(identity=identity,seeds=[r['seed'] for r in selected],starts_per_seed=selected[0]['count'],clean_success_starts=selected[0]['clean_success'],mean_task_success_rate=float(np.mean([r['success']/r['count'] for r in selected])),mean_macro_casr=float(np.mean(values)),std_macro_casr=float(np.std(values,ddof=1)) if len(values)>1 else 0.,mean_pooled_casr=float(np.mean([r['pooled_casr'] for r in selected])),by_suite=suites))
    return output

def confirmation_report():
    """Final report is frozen only when every declared seed/start is complete."""
    if complete(ROOT/'reports/FINAL_COMPLETE.json'):return ROOT/'reports/CONFIRMATION_RESULTS.json'
    selection_path=ROOT/'reports/FROZEN_SELECTION.json'
    if not selection_path.exists():return None
    selection=read(selection_path);labels=['INITIAL',*selection['selected']];rows=[]
    for label in labels:
        for seed in selection['seeds']:
            value=metrics(label,seed,'holdout')
            if value is None:return None
            rows.append(value)
    from .controller import matched_reference
    pairs=[]
    for group in ['O03','D07']:
        method=selection['winners'][group];reference=matched_reference(method,selection['selected'])
        a=[r['macro_casr'] for r in rows if r['identity']==method];b=[r['macro_casr'] for r in rows if r['identity']==reference]
        pairs.append(dict(method=method,reference=reference,delta_macro=float(np.mean(a)-np.mean(b)),paired_ci95=paired_interval(method,reference,selection['seeds'])))
    clean_reference=metrics('clean',42,'holdout')
    aggregated=aggregate_seeds(rows)
    path=ROOT/'reports/CONFIRMATION_RESULTS.json'
    atomic(dict(clean_reference=clean_reference,rows=rows,seed_aggregates=aggregated,paired_differences=pairs,selection=selection,time_unix=time.time()),path)
    lines=['# 配置冻结后的三种子确认','', '四套件8任务、初态20–29、80起点；逐种子完整结果。CASR分母仅同起点clean成功。','', '| 配置 | 种子 | 成功/80 | 新增失败/clean成功 | 等权CASR | Spatial | Object | Goal | Long |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[f'{100*x["casr"]:.2f}%' if x['casr'] is not None else '无分母' for x in r['by_suite']]
        lines.append(f'| {r["identity"]} | {r["seed"]} | {r["success"]}/{r["count"]} | {r["failures"]}/{r["clean_success"]} | {100*r["macro_casr"]:.2f}% | '+' | '.join(values)+' |')
    averages=['# 冻结配置：三种子平均结果','', '每个种子使用相同80起点，先任务、再套件等权。标准差描述三个训练种子的变化；配对区间按任务→起点计算。','', '| 配置 | 平均任务成功率 | 平均等权CASR | 种子标准差(pp) | Spatial | Object | Goal | Long |','|---|---:|---:|---:|---:|---:|---:|---:|']
    averages.insert(2,f'clean参考：{clean_reference["success"]}/{clean_reference["count"]}成功（{100*clean_reference["success"]/clean_reference["count"]:.2f}%）；相同物理起点参考对全部训练种子共用。')
    for r in aggregated:
        suite_values=[f'{100*s["mean_casr"]:.2f}%' if s['mean_casr'] is not None else '无分母' for s in r['by_suite']]
        averages.append(f'| {r["identity"]} | {100*r["mean_task_success_rate"]:.2f}% | {100*r["mean_macro_casr"]:.2f}% | {100*r["std_macro_casr"]:.2f} | '+' | '.join(suite_values)+' |')
    lines=averages+['']+lines
    lines+=['','## 参数匹配的DINO辅助收益','']
    for p in pairs:
        interval=p['paired_ci95'];lines.append(f'- {p["method"]} 对 {p["reference"]}：等权CASR差 {100*p["delta_macro"]:.2f} pp；任务→起点配对95%区间 [{100*interval[0]:.2f}, {100*interval[1]:.2f}] pp。')
    lines+=['','先对每个起点平均三个种子的配对差，再在套件内按任务→起点重采样。完整开发网格见RESULTS.md；实际成本见GPU_COST.csv。']
    (ROOT/'CONFIRMATION.md').write_text('\n'.join(lines)+'\n')
    return path
