"""H1: mild faults matter; H2: wrist/front asymmetry; H3: motion-conditioned response."""
from pathlib import Path
import numpy as np
from sc_common import (ROOT,PROJECT,config,read,sha,atomic,require,verify,complete,finish,
    episode_receipt,episode_files,module)


def table(path,rows):
    import csv
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def interval(rows,draws=None):
    if not rows:return None
    c=config();groups={}
    for row in rows:groups.setdefault(row['suite'],{}).setdefault(row['task_id'],[]).append(row['value'])
    rng=np.random.default_rng(c['bootstrap_seed']);values=[]
    for _ in range(draws or c['bootstrap_draws']):
        suite_values=[]
        for suite,tasks in groups.items():
            keys=sorted(tasks);sample=[]
            for i in rng.integers(len(keys),size=len(keys)):
                v=tasks[keys[i]];sample.extend(v[j] for j in rng.integers(len(v),size=len(v)))
            suite_values.append(np.mean(sample))
        values.append(np.mean(suite_values))
    return np.quantile(values,[.025,.975]).tolist()


def rank_correlation(x,y):
    import pandas as pd
    a=np.asarray(x,float);b=np.asarray(y,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])


def run(monitor):
    import pandas as pd
    m=verify();rows=[];motion=[];action_rows=[]
    for case in m['cases']:
        for cond in m['conditions']:
            receipt=episode_receipt(case,cond['name'])
            if not complete(receipt):continue
            files=episode_files(receipt);meta=read(files['EPISODE.json'])
            require(meta['checkpoint']==m['checkpoint'] and meta['rollout_seed']==case['rollout_seed'],'Episode identity differs')
            rows.append(meta)
            tr=pd.read_parquet(files['TRACE.parquet']);active=tr[tr.fault_active.astype(bool)]
            action_rows.append(dict(suite=case['suite'],task_id=case['task_id'],start=case['start'],condition=cond['name'],
                active_decisions=len(active),total_decisions=len(tr),
                next_xyz_normalized_l2=float(active.next_xyz_normalized_l2.mean()) if len(active) else None,
                executed_xyz_normalized_rmse=float(active.executed_xyz_normalized_rmse.mean()) if len(active) else None,
                mean_front_added_age_s=float(tr.front_added_age_s.mean()),mean_wrist_added_age_s=float(tr.wrist_added_age_s.mean()),
                mean_front_camera_shift_m=float(tr.front_camera_shift_m.mean()),mean_wrist_camera_shift_m=float(tr.wrist_camera_shift_m.mean())))
            if 'PROBES.parquet' in files:
                motion.extend([{**case,**v} for v in pd.read_parquet(files['PROBES.parquet']).to_dict('records')])
    lookup={(v['start'],v['condition']):v for v in rows};summaries=[];contrasts=[]
    for cond in m['conditions']:
        for suite in config()['suites']+['ALL_SUITES']:
            cases=[c for c in m['cases'] if suite=='ALL_SUITES' or c['suite']==suite]
            current=[lookup[(c['start'],cond['name'])] for c in cases if (c['start'],cond['name']) in lookup]
            pairs=[(c,lookup[(c['start'],'clean')],lookup[(c['start'],cond['name'])]) for c in cases
                if (c['start'],'clean') in lookup and (c['start'],cond['name']) in lookup]
            values=[dict(suite=c['suite'],task_id=c['task_id'],value=b['success']-a['success']) for c,a,b in pairs]
            summaries.append(dict(condition=cond['name'],suite=suite,completed=len(current),expected=len(cases),missing=len(cases)-len(current),
                task_successes=sum(v['success'] for v in current),task_SR=np.mean([v['success'] for v in current]).item() if current else None,
                paired_starts=len(pairs),new_failures=sum(a['success'] and not b['success'] for c,a,b in pairs),
                rescued=sum(not a['success'] and b['success'] for c,a,b in pairs),
                SR_difference_vs_clean=float(np.mean([v['value'] for v in values])) if values else None,
                SR_difference_CI95=interval(values),first_contact_count=sum(v['first_target_contact_step'] is not None for v in current)))
    for fault in ('delay1','delay2','occlusion'):
        for suite in config()['suites']+['ALL_SUITES']:
            pairs=[]
            for c in m['cases']:
                if suite!='ALL_SUITES' and c['suite']!=suite:continue
                a=lookup.get((c['start'],'front_'+fault));b=lookup.get((c['start'],'wrist_'+fault))
                if a and b:pairs.append(dict(suite=c['suite'],task_id=c['task_id'],value=a['success']-b['success']))
            contrasts.append(dict(fault=fault,suite=suite,pairs=len(pairs),
                additional_wrist_failure_rate=float(np.mean([v['value'] for v in pairs])) if pairs else None,CI95=interval(pairs)))
    # Reduce each probe series to one start-level value before correlating or bootstrapping.
    motion_summary=[];motion_cases=[]
    for view in ('front','wrist'):
        for lag in (1,2):
            start_rows=[]
            for c in m['cases']:
                part=[v for v in motion if v['start']==c['start'] and v['view']==view and v['delay_steps']==lag]
                if part:start_rows.append(dict(suite=c['suite'],task_id=c['task_id'],start=c['start'],view=view,delay_steps=lag,
                    camera_shift_m=float(np.mean([v['camera_shift_m'] for v in part])),
                    camera_rotation_rad=float(np.mean([v['camera_rotation_rad'] for v in part])),
                    next_xyz_normalized_l2=float(np.mean([v['next_xyz_normalized_l2'] for v in part])),probe_points=len(part)))
            motion_cases.extend(start_rows)
            motion_summary.append(dict(view=view,delay_steps=lag,starts=len(start_rows),
                camera_shift_action_spearman=rank_correlation([v['camera_shift_m'] for v in start_rows],[v['next_xyz_normalized_l2'] for v in start_rows]),
                camera_rotation_action_spearman=rank_correlation([v['camera_rotation_rad'] for v in start_rows],[v['next_xyz_normalized_l2'] for v in start_rows])))
    folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
    for name,values in [('EPISODES',rows),('SUMMARY',summaries),('VIEW_CONTRASTS',contrasts),('ACTION_RESPONSES',action_rows),
                        ('MOTION_PROBES',motion),('MOTION_STARTS',motion_cases),('MOTION_SUMMARY',motion_summary)]:table(folder/(name+'.csv'),values)
    status=dict(status='COMPLETE' if len(rows)==224 else 'INCOMPLETE',completed=len(rows),expected=224,missing=224-len(rows),
        clean_completed=sum(v['condition']=='clean' for v in rows),training_performed=False)
    atomic(status,folder/'STATUS.json')
    all_rows=[v for v in summaries if v['suite']=='ALL_SUITES']
    def percent(v):return '—' if v is None else f'{100*v:.2f}%'
    lines=['# 常见观测退化：腕部可靠性立项筛查','',f"完成 {len(rows)}/224 回合。联合OFT，四套件16任务、初态10/11；完整任务，非64步短窗口。",'',
        '|条件|完成|任务成功率|相对clean变化|新增失败|恢复成功|','|---|---:|---:|---:|---:|---:|']
    for v in all_rows:lines.append(f"|{v['condition']}|{v['completed']}/32|{percent(v['task_SR'])}|{percent(v['SR_difference_vs_clean'])}|{v['new_failures']}|{v['rescued']}|")
    lines+=['','H1：SUMMARY.csv比较轻度延迟/短时遮挡与clean。缺失不计成功，clean失败起点保留。',
        'H2：VIEW_CONTRASTS.csv中正值表示腕部比前视更易造成失败；同起点、同延迟和相同遮挡时序。',
        'H3：MOTION_PROBES.csv来自clean轨迹控制步16/40/64的同状态离线对照，不受退化分支后续运动影响。',
        'MOTION_STARTS.csv先把多个探针归并到起点；MOTION_SUMMARY.csv报告相机运动与首动作偏差的相关性，完整任务失败另由闭环结果判断。','']
    total={v['condition']:v for v in all_rows};asym=[v for v in contrasts if v['suite']=='ALL_SUITES']
    if len(rows)!=224:decision='等待缺失回合；不以部分结果决定训练恢复模块。'
    else:
        harmful=[v for v in summaries if v['suite']!='ALL_SUITES' and v['condition']!='clean' and v['SR_difference_vs_clean'] is not None and v['SR_difference_vs_clean']<0]
        if not harmful:decision='当前轻度退化未降低任务成功率：不进入恢复模块训练。'
        elif len({v['suite'] for v in harmful})<2:decision='影响目前集中在一个套件：先核对任务差异，不启动通用恢复模块训练。'
        elif not any(v['additional_wrist_failure_rate'] is not None and v['additional_wrist_failure_rate']>0 for v in asym):decision='退化有跨套件影响，但没有总体腕部劣势：考虑通用多视角可靠性问题，不预设腕部专用方法。'
        else:decision='出现跨套件退化及腕部劣势：结合逐任务差异、区间和运动探针讨论局部修复小试。'
    lines+=['下一步建议：'+decision,'','本轮不训练模块，也不估计自然部署中故障发生频率。模型覆盖目前限于联合OFT；跨模型复现作为后续步骤。',
        '视频集中在research_videos/wrist_observation_reliability_screen_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/。',
        '每个视频显示策略可用的对应视角输入：延迟画面来自真实历史采样，遮挡只出现在预定窗口。','']
    (folder/'RESULTS.md').write_text('\n'.join(lines));atomic(dict(status=status['status'],recommendation=decision),folder/'RECOMMENDATION.json')
    if len(rows)==224:finish(folder/'COMPLETE.json',list(folder.glob('*.csv'))+[folder/'RESULTS.md',folder/'STATUS.json',folder/'RECOMMENDATION.json'])
    monitor.update(completed=len(rows),total=224,phase=status['status']);return status


def videos(monitor):
    verify();exporter=module('reliability_singleview_exporter',PROJECT/'research_tools/export_singleview_videos.py')
    out=PROJECT/'research_videos/wrist_observation_reliability_screen_v1_singleview'
    result=exporter.export_run(ROOT,out)
    atomic(dict(status='COMPLETE' if result['episodes']==224 else 'PARTIAL',completed=result['episodes'],expected=224,video_root=str(out)),ROOT/'reports/VIDEO_STATUS.json')
    monitor.update(completed=result['episodes'],total=224,phase='video_export_complete')
