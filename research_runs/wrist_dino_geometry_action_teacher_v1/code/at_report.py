"""Paired completed starts only; missing/failed work never becomes a task failure."""
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd
from at_common import ROOT,config,read_json,complete,atomic_json,atomic_parquet,finish,module,source

def run(monitor=None,final=False):
    if complete(ROOT/'results/COMPLETE.json'):return
    lockpath=ROOT/'reports/locks/report.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX | (0 if final else fcntl.LOCK_NB))
        except BlockingIOError:return
        _run(monitor,final)

def _run(monitor,final):
    from at_evaluate import conditions
    folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
    try:expected=list(conditions())
    except (FileNotFoundError,RuntimeError):expected=[]
    rows=[];errors=[]
    for p in sorted((ROOT/'evaluation/episodes').glob('*/*/COMPLETE.json')):
        try:
            if complete(p):rows.append(read_json(p.parent/'EPISODE.json'))
        except Exception as error:errors.append({'receipt':str(p),'error':repr(error)})
    allrows=pd.DataFrame(rows);paired=pd.DataFrame();scores=[]
    helper=module('at_readonly_statistics',source.MATCH/'code/report.py')
    if rows and expected:
        paired=allrows.pivot(index=['suite','task_id','state_index'],columns='condition',values='success')
        paired=paired.reindex(columns=expected).dropna()
        for suite in ['all',*config()['suites']]:
            part=paired if suite=='all' else paired[paired.index.get_level_values('suite')==suite]
            eligible=part[part.clean==1]
            for condition in expected:
                if condition=='clean':continue
                switches=(eligible[condition]==0).astype(float).reset_index()
                groups=[g[condition].to_numpy() for _,g in switches.groupby(['suite','task_id'])]
                seed=43 if condition.endswith('s43') else 42
                initial=f'initial_s{seed}'
                scores.append({'suite':suite,'condition':condition,'paired_starts':len(part),
                    'clean_successes':len(eligible),'failure_switches':int((eligible[condition]==0).sum()),
                    'CASR_pooled':float((eligible[condition]==0).mean()) if len(eligible) else None,
                    'failures_beyond_initial':int(((eligible[condition]==0)&(eligible[initial]==1)).sum()) if initial in eligible else None,
                    **helper.interval(groups,draws=config()['bootstrap_draws'])})
        atomic_parquet(allrows,folder/'ROLLOUTS.parquet');paired.to_csv(folder/'PAIRS.csv')
    pd.DataFrame(scores).to_csv(folder/'CASR.csv',index=False)
    offline=[]
    for p in sorted((ROOT/'evaluation/offline').glob('*.json')):
        if complete(p):offline.append(pd.read_parquet(p.with_suffix('.parquet')))
    if offline:
        frame=pd.concat(offline,ignore_index=True);atomic_parquet(frame,folder/'OFFLINE.parquet')
        helper.summarize(frame,['condition'],['target_mse','translation_delta_l2','direction_cos','direction_positive',
            'q','orthogonal','target_distance','restore_clean_dino_delta_l2','restore_clean_siglip_delta_l2']).to_csv(folder/'OFFLINE_SUMMARY.csv',index=False)
    training={}
    for p in sorted((ROOT/'training').glob('*/COMPLETE.json')):
        if complete(p):training[p.parent.name]=read_json(p)
    selected=read_json(ROOT/'data/SELECTED.json') if (ROOT/'data/SELECTED.json').exists() else None
    done=bool(len(paired)==80 and len(expected)==8 and len(rows)==640 and len(offline)==240 and len(training)==4)
    full_budget=bool(len(training)==4 and all(v.get('budget_complete',False) for v in training.values()))
    status={'status':'COMPLETE' if done else 'PARTIAL_OR_FAILED','signature':__import__('at_common').signature(),
        'closedloop_episodes':len(rows),'expected_closedloop_episodes':640,'fully_paired_starts':len(paired),
        'offline_observations':len(offline),'conditions':expected,'errors':errors,'training':training,
        'selection':selected,'fixed_checkpoint':'libero_spatial','primary':config()['primary'],
        'independent_confirmation':False,'full_training_budget_complete':full_budget,'snapshot_final_worker':bool(final)}
    atomic_json(status,folder/'STATUS.json')
    lines=['# DINO几何动作教师：原生PNG长实验','',
        f'状态：{status["status"]}。已完成 {len(rows)}/640 回合；完整配对起点 {len(paired)}/80；离线验证 {len(offline)}/240。',
        '固定Spatial模型，24任务训练240观测、跨轨迹验证240观测；闭环40任务初态0/1。',
        '主结果预指定geometry_teacher_s42；seed43仅作重复性检查。未按验证或闭环成绩选择检查点。',
        '几何教师只替换腕部DINO，训练匹配未来8槽平移动作；部署是原生双分支PNG。',
        'clean_far_boundary是clean参考的同池远边界基线，不是原专家参考UADA；uada500是历史Spatial UADA500 PNG。',
        '初始补丁两个种子分别配对。已查看的历史场景不称独立确认。动作命令偏差不能当机械臂位移。',
        f'四组是否全部完成24000次候选更新：{full_budget}。',
        '实际评测检查点：'+', '.join(f'{k}={v["step"]}次候选/{v["optimized"]}次实际更新' for k,v in (selected or {}).get('candidates',{}).items()),'',
        '|套件|条件|失败切换/clean成功|CASR|超出初始补丁的失败|','|---|---|---:|---:|---:|']
    for v in scores:
        if not v['paired_starts']:continue
        score='NA' if v['CASR_pooled'] is None else f'{v["CASR_pooled"]:.1%}'
        lines.append(f'|{v["suite"]}|{v["condition"]}|{v["failure_switches"]}/{v["clean_successes"]}|{score}|{v["failures_beyond_initial"]}|')
    lines+=['','所有比率仅来自该表共同完整配对起点；缺失、工程失败和正在运行回合不算攻击成功。',
        '预算、选定检查点、未完成组见STATUS.json；成本、30秒心跳及启动检查见reports。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if done:
        artifacts=[folder/p for p in ('RESULTS.md','STATUS.json','CASR.csv','PAIRS.csv','ROLLOUTS.parquet','OFFLINE.parquet','OFFLINE_SUMMARY.csv')]
        finish(folder/'COMPLETE.json',artifacts)
    if monitor:monitor.update(phase='report_written',completed=len(rows),total=640)
