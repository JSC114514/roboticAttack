"""Conditional CASR, complete pairs, and transparent coverage/engineering failures."""
from collections import Counter
import fcntl
import numpy as np
import pandas as pd
from sr_common import (ROOT,config,episodes,records,cache_receipt,complete,finish,read_json,atomic_json,atomic_parquet,require,signature)

def coverage(monitor=None):
    require(all(complete(cache_receipt(e['episode_id'])) for e in episodes()),'Capture incomplete')
    out=[]
    for cohort in ('fit','verify'):
        rows=list(records(cohort))
        for s in config()['suites']:
            group=[r for r in rows if r['suite']==s];eligible=[r for r in group if not r['teacher']['no_target']]
            out.append({'cohort':cohort,'suite':s,'frames':len(group),'eligible':len(eligible),
                'eligible_tasks':sorted({r['task_id'] for r in eligible}),
                'eligible_trajectories':len({r['episode_id'] for r in eligible}),
                'reasons':dict(Counter(r['teacher']['no_target_reason'] for r in group if r['teacher']['no_target']))})
    p=ROOT/'results/COVERAGE.json';atomic_json({'signature':signature(),'rows':out},p)
    missing=[r['suite'] for r in out if r['cohort']=='fit' and r['eligible']==0]
    require(not missing,'Goal clean approach coverage absent in '+repr(missing)+'; training stopped, inspect COVERAGE.json')
    finish(ROOT/'reports/COVERAGE_COMPLETE.json',[p])

def paired_interval(pairs):
    if not pairs:return None
    rng=np.random.default_rng(42);tasks=sorted({(p['suite'],p['task_id']) for p in pairs});values=[]
    groups={t:[p['failure_difference'] for p in pairs if (p['suite'],p['task_id'])==t] for t in tasks}
    for _ in range(config()['bootstrap_draws']):
        draw=[]
        for i in rng.integers(0,len(tasks),len(tasks)):
            g=groups[tasks[i]];draw.extend(rng.choice(g,len(g),replace=True).tolist())
        values.append(np.mean(draw))
    return np.quantile(values,[.025,.975]).tolist()

def run(monitor=None,final=False):
    (ROOT/'reports/locks').mkdir(parents=True,exist_ok=True)
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/locks/report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[];failures=[]
        worker_receipts=[ROOT/f'evaluation/worker_{i}_COMPLETE.json' for i in range(2)]
        fully_complete=all(complete(p) for p in worker_receipts)
        do_intervals=final or fully_complete
        for p in sorted((ROOT/'evaluation/episodes').glob('*/*/COMPLETE.json')):
            if complete(p):rows.append(read_json(p.parent/'EPISODE.json'))
        for p in sorted((ROOT/'reports/COSTS').glob('*.json')):
            v=read_json(p)
            if v.get('status')=='FAILED':failures.append({'path':str(p),'stage':v.get('stage'),'error':v.get('error'),'elapsed_s':v.get('elapsed_s')})
        clean={(r['suite'],r['task_id'],r['state_index']):r for r in rows if r['condition']=='clean'}
        eligible={k for k,v in clean.items() if v['success']}
        conditions=[*[f'initial_s{s}' for s in config()['seeds']],'uada500',
            *[f'{m}_s{s}' for s in config()['seeds'] for m in config()['methods']]]
        summary=[];lookup={(r['suite'],r['task_id'],r['state_index'],r['condition']):r for r in rows}
        for suite in ['all']+config()['suites']:
            starts={k for k in eligible if suite=='all' or k[0]==suite}
            clean_rows=[v for k,v in clean.items() if suite=='all' or k[0]==suite]
            for cond in conditions:
                paired=[lookup[(*k,cond)] for k in starts if (*k,cond) in lookup]
                summary.append({'suite':suite,'condition':cond,'clean_completed':len(clean_rows),'clean_successes':len(starts),
                    'paired_completed':len(paired),'induced_failures':sum(r['failure'] for r in paired),
                    'casr':sum(r['failure'] for r in paired)/len(paired) if paired else None,
                    'complete_pairing':len(paired)==len(starts),
                    'casr_task_start_bootstrap_95ci':paired_interval([{'suite':r['suite'],'task_id':r['task_id'],'failure_difference':r['failure']} for r in paired]) if do_intervals else None})
        contrasts=[]
        for seed in config()['seeds']:
            for other in (f'initial_s{seed}',f'translation_far_boundary_s{seed}','uada500'):
                treatment=f'radial_reverse_s{seed}';pairs=[]
                for k in sorted(eligible):
                    if (*k,treatment) in lookup and (*k,other) in lookup:
                        pairs.append({'suite':k[0],'task_id':k[1],'state_index':k[2],
                            'failure_difference':lookup[(*k,treatment)]['failure']-lookup[(*k,other)]['failure']})
                contrasts.append({'treatment':treatment,'comparator':other,'paired_starts':len(pairs),
                    'failure_rate_difference':float(np.mean([p['failure_difference'] for p in pairs])) if pairs else None,
                    'task_start_bootstrap_95ci':paired_interval(pairs) if do_intervals else None})
        temporary=ROOT/'results/.CASR.csv.tmp';pd.DataFrame(summary).to_csv(temporary,index=False)
        temporary.replace(ROOT/'results/CASR.csv')
        if rows:atomic_parquet(pd.DataFrame(rows),ROOT/'results/ROLLOUTS.parquet')
        atomic_json({'signature':signature(),'status':'COMPLETE' if fully_complete else 'PARTIAL','checkpoint':config()['checkpoint'],
            'episodes_completed':len(rows),'clean_starts_completed':len(clean),'clean_success_starts':len(eligible),
            'contrasts':contrasts,'engineering_failures':failures},ROOT/'results/STATUS.json')
        text=['# 混合数据／冻结Goal：空间接近阻断',
            f'状态：{"全部完成" if fully_complete else "部分完成"}；完成{len(rows)}回合；clean起点{len(clean)}/80，其中成功{len(eligible)}。',
            '数据覆盖四套件；主模型固定Goal。CASR仅以同起点clean成功、该条件已完成的配对为分母；缺失回合不计成功。',
            '原Spatial UADA500是迁移部署参考；同池平移远边界是预算/梯度/DoF匹配对照，不冒称原专家参考UADA。',
            '控制目标的5mm是OSC期望位移，不能写成实际末端退后5mm；实际运动见TRACE与STEPS。',
            '本轮无DINO专用损失或分支限制，不据此声称DINO独有因果机制。','',
            '|套件|条件|完成配对/clean成功|失败|CASR|','|---|---|---:|---:|---:|']
        for v in summary:
            score='—' if v['casr'] is None else f"{v['casr']:.1%}"
            text.append(f"|{v['suite']}|{v['condition']}|{v['paired_completed']}/{v['clean_successes']}|{v['induced_failures']}|{score}|")
        text+=['','## 配对差异（方法失败率减对照失败率）']
        for v in contrasts:text.append(str(v))
        text+=['','## 覆盖及失败','所有工程失败和重试成本见STATUS.json与reports/COSTS；有效训练样本和缺失原因见COVERAGE.json。',
            '多个相邻帧/动作槽位不作为独立重复；两个种子均完整报告，不按测试表现挑种子。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        if final:
            if fully_complete:finish(ROOT/'results/COMPLETE.json',[*worker_receipts,ROOT/'results/CASR.csv',ROOT/'results/STATUS.json',ROOT/'results/RESULTS.md'])
        if monitor:monitor.update(phase='reported',completed=len(rows),total=640)
