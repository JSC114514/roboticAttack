"""Nonblocking partial/final reporting. Only complete, matched episodes enter CASR."""
import csv
from pathlib import Path
import numpy as np
from rr_common import ROOT,config,read_json,atomic_json,complete,finish,episode_path,signature


def interval(values,units,draws=2000):
    if not values:return {'mean':None,'ci95':None,'tasks':0,'trajectories':0}
    grouped={}
    for value,(task,episode) in zip(values,units):
        grouped.setdefault(task,{}).setdefault(episode,[]).append(value)
    arrays={t:np.array([np.mean(x) for x in eps.values()]) for t,eps in grouped.items()}
    tasks=sorted(arrays);rng=np.random.default_rng(42);estimates=[]
    for _ in range(draws):
        chosen=rng.choice(tasks,len(tasks),replace=True)
        estimates.append(np.mean([rng.choice(arrays[t],len(arrays[t]),replace=True).mean() for t in chosen]))
    return {'mean':float(np.mean([a.mean() for a in arrays.values()])),
            'ci95':[float(x) for x in np.quantile(estimates,[.025,.975])],
            'tasks':len(tasks),'trajectories':sum(map(len,arrays.values()))}


def write_csv(path,rows):
    if not rows:return
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def run(monitor,interim=False):
    prefix='INTERIM' if interim else 'RESULTS';c=config()
    gate_path=ROOT/'reports/DIRECTION_COMPLETE.json'
    gate=read_json(gate_path) if complete(gate_path) else None
    stopped=gate is not None and not gate['passed']
    pairs=[];finished_episodes=0
    for task in c['eval_tasks']:
        for state in c['eval_states']:
            parts={}
            for name in c['conditions']:
                receipt=episode_path(task,state,name)
                if complete(receipt):
                    parts[name]=read_json(receipt.parent/'EPISODE.json');finished_episodes+=1
            if len(parts)==len(c['conditions']):pairs.append({'task':task,'state':state,'parts':parts})
    eligible=[x for x in pairs if x['parts']['clean']['success']==1]
    casr=[]
    for name in c['conditions'][1:]:
        vals=[float(x['parts'][name]['failure']) for x in eligible]
        units=[(x['task'],str(x['state'])) for x in eligible]
        summary=interval(vals,units,c['bootstrap_draws'])
        initial=[x['parts']['initial']['failure'] for x in eligible]
        casr.append({'condition':name,'failures':sum(vals),'clean_successes':len(vals),
            'casr':sum(vals)/len(vals) if vals else None,
            'new_failures_vs_initial':sum(bool(y) and not bool(z) for y,z in zip(vals,initial)),
            'recovered_vs_initial':sum(not bool(y) and bool(z) for y,z in zip(vals,initial)),
            'task_equal_casr':summary['mean'],'ci95_low':summary['ci95'][0] if vals else None,
            'ci95_high':summary['ci95'][1] if vals else None})
    offline=[]
    for path in sorted((ROOT/'evaluation/offline').glob('*_COMPLETE.json')):
        if complete(path):offline.extend(read_json(path.with_name(path.name.replace('_COMPLETE',''))))
    offline_summary={}
    for name in c['conditions']:
        rows=[r for r in offline if r['condition']==name and not r['no_target']]
        units=[(r['task_id'],r['episode_id']) for r in rows]
        offline_summary[name]={key:interval([r[key] for r in rows],[*units],c['bootstrap_draws'])
            for key in ('relation_error_cm','first_action_delta_l2','chunk_translation_delta_l2')}
        directional=[r for r in rows if r['first_action_projection'] is not None]
        offline_summary[name]['first_action_projection']=interval([r['first_action_projection'] for r in directional],
            [(r['task_id'],r['episode_id']) for r in directional],c['bootstrap_draws'])
        if name=='relation':
            offline_summary[name]['restore_clean_dino_first_delta_l2']=interval(
                [r['restore_clean_dino_first_delta_l2'] for r in rows],units,c['bootstrap_draws'])
    trained=complete(ROOT/'training/COMPLETE.json')
    finished=stopped or (trained and len(pairs)==10 and len(offline)==240)
    status='STOPPED_DIRECTION_NOT_SUPPORTED' if stopped else 'COMPLETE' if finished else 'PARTIAL_OR_PENDING'
    failures=[]
    for path in sorted((ROOT/'reports/COSTS').glob('*.json')):
        v=read_json(path)
        if v.get('status')=='FAILED':failures.append({'worker':v.get('worker'),'error':v.get('error'),'elapsed_s':v.get('elapsed_s')})
    if failures and not finished:status='PARTIAL_OR_FAILED'
    value={'signature':signature(),'status':status,'interim':interim,'gate':gate,
        'training':read_json(ROOT/'training/COMPLETE.json') if trained else None,
        'training_progress':read_json(ROOT/'reports/PROGRESS/train_0.json') if (ROOT/'reports/PROGRESS/train_0.json').exists() else None,
        'offline_states':len(offline)//4,'expected_offline_states':60,'episodes':finished_episodes,
        'expected_episodes':40,'fully_paired_starts':len(pairs),'clean_success_starts':len(eligible),
        'casr':casr,'offline':offline_summary,'failed_attempts':failures,
        'scope':'Spatial development set; probe manipulation is not proof of causal semantics; no cross-task universality'}
    status_path=ROOT/f'results/{"INTERIM_STATUS" if interim else "STATUS"}.json';atomic_json(value,status_path)
    csv_path=ROOT/f'results/{prefix}_CASR.csv';write_csv(csv_path,casr)
    lines=['# DINO关系读出目标小试', '',f'状态：{status}；离线{len(offline)//4}/60；完整配对{len(pairs)}/10；回合{finished_episodes}/40。',
        '固定Spatial、50×50腕部PNG、无EOT。关系读数、原生动作与闭环失败分别解释；本开发集已查看，不称独立确认。','']
    if gate:
        lines += [f'方向预算门槛：{"通过" if gate["passed"] else "未通过"}；适用{gate["applicable"]}/12；关系响应中位数={gate["relation_median"]}；相对随机优势中位数={gate["advantage_median"]}。',
                  '这是预定探索性门槛，不是显著性检验或DINO唯一控制机制的证明。','']
    if stopped:lines+=['未训练PNG：当前冻结读出方向未显示预定动作优势。没有自动换方向或追加预算。','']
    if trained:
        train=value['training'];lines += [f'训练：{train["candidate_updates"]}次候选，{train["optimized"]}次实际更新，{train["no_target"]}次无目标；固定最终检查点。','']
    lines += ['|条件|失败/clean成功|CASR|新增失败/恢复，相对初始PNG|','|---|---:|---:|---:|']
    for r in casr:
        score=f'{100*r["casr"]:.1f}%' if r['casr'] is not None else 'NA'
        lines.append(f'|{r["condition"]}|{int(r["failures"])}/{r["clean_successes"]}|{score}|{r["new_failures_vs_initial"]}/{r["recovered_vs_initial"]}|')
    lines+=['','任务→轨迹/起点重采样区间及离线厘米误差见STATUS.json；动作指令不是机械臂实际位移。',
            '同范数随机方向在连续特征中严格处于读出零空间；模型dtype量化后的实际范数及读出泄漏见DIRECTION.json。',
            '工程失败、缺失回合不算攻击成功。启动检查、30秒心跳、RSS/CUDA峰值与失败成本见reports。']
    report=ROOT/f'results/{prefix}.md';temporary=report.with_suffix('.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(report)
    if finished and not interim:finish(ROOT/'results/COMPLETE.json',[report,status_path,csv_path],scientific_status=status)
    monitor.update(phase='interim_report' if interim else 'final_report',scientific_status=status,
                   completed=finished_episodes,total=40)
