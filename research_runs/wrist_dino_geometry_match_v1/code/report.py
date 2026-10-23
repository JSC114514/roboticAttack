"""Hierarchical paired reporting, including explicit missing/failed stages."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from common import ROOT,cfg,read_json,atomic_json,atomic_parquet,finish,complete,require,signature

def interval(groups,draws=1000):
    values=[np.asarray(v,dtype=float) for v in groups if len(v)]
    values=[v[np.isfinite(v)] for v in values];values=[v for v in values if len(v)]
    if not values:return {'mean':None,'ci95_low':None,'ci95_high':None,'tasks':0}
    counts=np.array([len(v) for v in values]);n=len(values);width=int(counts.max())
    matrix=np.zeros((n,width))
    for i,v in enumerate(values):matrix[i,:len(v)]=v
    rng=np.random.default_rng(42);tasks=rng.integers(n,size=(draws,n));sizes=counts[tasks]
    picks=(rng.random((draws,n,width))*sizes[...,None]).astype(int)
    sampled=matrix[tasks[...,None],picks];mask=np.arange(width)[None,None,:]<sizes[...,None]
    distribution=((sampled*mask).sum(-1)/sizes).mean(-1)
    return {'mean':float(np.mean([v.mean() for v in values])),
            'ci95_low':float(np.quantile(distribution,.025)),'ci95_high':float(np.quantile(distribution,.975)),
            'tasks':n}

def summarize(frame,keys,metrics):
    output=[]
    for group,rows in frame.groupby(keys,dropna=False):
        if not isinstance(group,tuple):group=(group,)
        for metric in metrics:
            if metric not in rows:continue
            trajectories=rows.groupby(['suite','task_id','episode_id'])[metric].mean().reset_index()
            groups=[v[metric].dropna().to_numpy() for _,v in trajectories.groupby(['suite','task_id'])]
            output.append({**dict(zip(keys,group)),'metric':metric,**interval(groups),
                           'observations':len(rows),'undefined':int(rows[metric].isna().sum())})
    return pd.DataFrame(output)

def casr(frame):
    output=[]
    for split,split_rows in frame.groupby('split'):
        paired=split_rows.pivot(index=['suite','task_id','state_index'],columns='condition',values='success')
        require(not paired.isna().any().any(),'Unpaired/missing rollout conditions')
        eligible=paired[paired.clean==1]
        for condition in sorted(set(paired.columns)-{'clean'}):
            switches=(eligible[condition]==0).astype(float).reset_index()
            groups=[v[condition].to_numpy() for _,v in switches.groupby(['suite','task_id'])]
            output.append({'split':split,'condition':condition,'eligible_clean_starts':len(eligible),
                           'failure_switches':int(switches[condition].sum()),**interval(groups)})
    return pd.DataFrame(output)

def check_sources(receipts,phase):
    errors=[]
    for path in receipts:
        try:
            if not complete(path):errors.append(str(path.relative_to(ROOT)))
        except Exception as error:errors.append(str(error))
    if errors:
        atomic_json({'status':'FAILED_OR_INCOMPLETE','signature':signature(),'missing_or_invalid':errors,
                     'scientific_result_available':False},ROOT/f'results/{phase}/STATUS.json')
        (ROOT/f'results/{phase}').mkdir(parents=True,exist_ok=True)
        (ROOT/f'results/{phase}/RESULTS.md').write_text('本轮未完整完成，不能报告成功率。\n\n'+ '\n'.join(errors)+'\n')
        raise RuntimeError('Incomplete '+phase+' stages; see results/'+phase+'/STATUS.json')

def pilot_report(monitor):
    receipts=[ROOT/f'training/pilot/{k}/COMPLETE.json' for k in ('individual','projection','full')]
    receipts += [ROOT/'evaluation/pilot/ORACLE_COMPLETE.json',ROOT/'evaluation/pilot/CHECK_COMPLETE.json']
    check_sources(receipts,'pilot');folder=ROOT/'results/pilot';folder.mkdir(parents=True,exist_ok=True)
    rows=pd.read_parquet(ROOT/'evaluation/pilot/PNG_CHECK.parquet')
    oracle=pd.read_parquet(ROOT/'evaluation/pilot/ORACLE.parquet')
    summary=summarize(rows,['cohort','condition'],['q','q_delta','projection_loss','orthogonal','target_distance',
                        'first_translation_l2','direction_positive','direction_cos',
                        'restore_dino_translation_l2','restore_siglip_translation_l2'])
    ideal=summarize(oracle,['cohort'],['ideal_direction_l2','dino_plus_change_l2','dino_half_change_l2','native_rgb_direction_l2'])
    summary.to_csv(folder/'PNG_SUMMARY.csv',index=False);ideal.to_csv(folder/'ORACLE_SUMMARY.csv',index=False)
    lines=['# DINO几何匹配：今晚的断点验证','',
           '固定Spatial checkpoint。独立补丁仅报告拟合图像；共享补丁另报不同轨迹验证。',
           '数据：24任务，每组48条轨迹/48张图；无目标记录保留。',
           '完整目标损失=投影损失+正交残差。两种损失的候选更新预算及初始PNG配对。','',
           '下表为任务等权均值，完整区间见CSV。动作输出是命令，不是机械臂实际位移。','',
           '|组别|条件|目标距离|正交残差|动作方向一致率|','|---|---|---:|---:|---:|']
    for (cohort,condition),group in summary.groupby(['cohort','condition']):
        v=dict(zip(group.metric,group['mean']))
        def fmt(k):return 'NA' if pd.isna(v.get(k)) else f'{v[k]:.4f}'
        lines.append(f'|{cohort}|{condition}|{fmt("target_distance")}|{fmt("orthogonal")}|{fmt("direction_positive")}|')
    lines += ['', '解释顺序：先比较理想DINO目标的动作响应，再比较单图/共享补丁拟合，最后比较共享补丁跨轨迹表现。',
              '单图优于共享支持共享冲突；完整匹配优于投影支持原损失约束不足。小预算未实现不等于不可实现。',
              '本轮不报告CASR，不把内部替换当作RGB补丁结果。夜间方法固定为full_s42，full_s43复核，不按本轮验证选种子。']
    report=folder/'RESULTS.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json({'status':'COMPLETE','signature':signature(),'closed_loop_tested':False},folder/'STATUS.json')
    finish(folder/'COMPLETE.json',[report,folder/'PNG_SUMMARY.csv',folder/'ORACLE_SUMMARY.csv',folder/'STATUS.json'],
           primary_night_method='full_s42',closed_loop_tested=False)
    monitor.update(phase='pilot_report_written',completed=1,total=1)

def night_report(monitor):
    receipts=[ROOT/f'evaluation/night/{s}_COMPLETE.json' for s in cfg()['suites']]
    check_sources(receipts,'night');folder=ROOT/'results/night';folder.mkdir(parents=True,exist_ok=True)
    rolls=pd.concat([pd.read_parquet(ROOT/f'evaluation/night/{s}_rollouts.parquet') for s in cfg()['suites']],ignore_index=True)
    scores=casr(rolls);scores.to_csv(folder/'CASR.csv',index=False)
    offline=pd.concat([pd.read_parquet(ROOT/f'evaluation/night/{s}_{model}_offline.parquet')
                       for s in cfg()['suites'] for model in ('fixed_spatial','native_suite')],ignore_index=True)
    summary=summarize(offline,['checkpoint','condition'],['q_delta','orthogonal','target_distance',
                          'direction_positive','direction_cos','first_translation_l2',
                          'restore_dino_translation_l2','restore_siglip_translation_l2'])
    summary.to_csv(folder/'OFFLINE_SUMMARY.csv',index=False)
    path=folder/'ROLLOUTS.parquet';atomic_parquet(rolls,path)
    lines=['# DINO完整几何匹配：夜间结果','',
           '主结果预先固定full_s42；full_s43只作重复性检查。没有根据测试集挑选种子。',
           'validation使用初态48；heldout使用本轮未参与训练/选优的初态0、1；所有clean重新执行。',
           '任务划分复用历史实验，因此这不是全新未接触任务的确认性试验。',
           'projection_s42复用旧PNG，但与其他条件在本轮重新配对评测。UADA是部署参考，训练池/预算不同。','',
           '|划分|条件|CASR|95%区间|失败切换/clean成功|','|---|---|---:|---|---:|']
    for row in scores.to_dict('records'):
        val='NA' if pd.isna(row['mean']) else f'{row["mean"]:.2%}'
        ci='NA' if pd.isna(row['ci95_low']) else f'{row["ci95_low"]:.2%}–{row["ci95_high"]:.2%}'
        lines.append(f'|{row["split"]}|{row["condition"]}|{val}|{ci}|{row["failure_switches"]}/{row["eligible_clean_starts"]}|')
    lines += ['', '固定Spatial和原生suite模型的离线结果分开报告；不把模型迁移差异解释为空间信息本身的差异。',
              '表示变化、定向动作变化、真实闭环失败分别下结论。全部指标见OFFLINE_SUMMARY.csv。',
              'RSS、CUDA峰值、失败成本和每worker心跳见reports/COSTS、reports/PROGRESS。']
    report=folder/'RESULTS.md';report.write_text('\n'.join(lines)+'\n')
    atomic_json({'status':'COMPLETE','signature':signature(),'primary_candidate':'full_s42',
                 'test_used_for_selection':False},folder/'STATUS.json')
    finish(folder/'COMPLETE.json',[report,path,folder/'CASR.csv',folder/'OFFLINE_SUMMARY.csv',folder/'STATUS.json'],
           primary_candidate='full_s42',test_used_for_selection=False)
    monitor.update(phase='night_report_written',completed=1,total=1)
