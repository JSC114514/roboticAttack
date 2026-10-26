"""Task/start-level restoration outcomes; oracle interventions stay separate from PNG results."""
from pathlib import Path
import numpy as np
from fr_common import ROOT,PROJECT,config,read,sha,atomic,require,verify,complete,finish,episode_receipt,episode_files,module
from fr_math import paired_outcomes,interval

def table(path,rows):
    import csv
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.tmp').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)

def run(monitor):
    import pandas as pd
    m=verify();c=config();rows=[];action_rows=[];lookup={};costs=[]
    for case in m['cases']:
        for condition in m['conditions']:
            receipt=episode_receipt(case,condition['name'])
            if not complete(receipt):continue
            files=episode_files(receipt);meta=read(files['EPISODE.json'])
            require(meta['checkpoint']==m['checkpoint'] and meta['rollout_seed']==case['rollout_seed'],'Episode identity differs')
            require(meta['state_index']==10,'Validation state was used')
            reused='REUSE.json' in files
            row={**meta,'patch':condition['patch'],'restoration':condition['mode'],'reused_clean':reused,
                'same_state_oracle':condition['mode']!='native'}
            rows.append(row);lookup[(case['start'],condition['name'])]=row
            trace=pd.read_parquet(files['TRACE.parquet'])
            n=np.asarray(trace['executed_slots'],float)
            def executed(key):
                return float(np.sqrt(np.average(np.asarray(trace[key],float)**2,weights=n))) if key in trace and sum(n)>0 else None
            action_rows.append(dict(suite=case['suite'],task_id=case['task_id'],start=case['start'],condition=condition['name'],
                decisions=len(trace),executed_slots=int(sum(n)),executed_xyz_normalized_rmse=executed('executed_xyz_normalized_rmse'),
                executed_rotation_normalized_rmse=executed('executed_rotation_normalized_rmse'),
                mean_dino_used_delta_rms=float(trace.dino_used_delta_rms.mean()) if 'dino_used_delta_rms' in trace else None,
                mean_siglip_used_delta_rms=float(trace.siglip_used_delta_rms.mean()) if 'siglip_used_delta_rms' in trace else None))
    summaries=[];effects=[];task_rows=[]
    for condition in m['conditions']:
        name=condition['name']
        for suite in c['suites']+['ALL_SUITES']:
            cases=[case for case in m['cases'] if suite=='ALL_SUITES' or case['suite']==suite]
            current=[lookup[(case['start'],name)] for case in cases if (case['start'],name) in lookup]
            pairs=[(case,lookup[(case['start'],'clean')],lookup[(case['start'],name)]) for case in cases
                   if (case['start'],'clean') in lookup and (case['start'],name) in lookup]
            eligible=[(case,a,b) for case,a,b in pairs if a['success']]
            induced=sum(not b['success'] for _,_,b in eligible)
            values=[dict(suite=case['suite'],task_id=case['task_id'],value=int(b['success'])-int(a['success'])) for case,a,b in pairs]
            summaries.append(dict(condition=name,suite=suite,patch=condition['patch'],restoration=condition['mode'],
                oracle_intervention=condition['mode']!='native',completed=len(current),expected=len(cases),missing=len(cases)-len(current),
                task_successes=sum(r['success'] for r in current),task_SR=float(np.mean([r['success'] for r in current])) if current else None,
                clean_success_pairs=len(eligible),failures_on_clean_success_starts=induced,
                failure_fraction_on_clean_success=induced/len(eligible) if eligible else None,
                SR_difference_vs_clean=float(np.mean([r['value'] for r in values])) if values else None,
                difference_CI95=interval(values,c['bootstrap_draws'],c['bootstrap_seed']),
                first_contact_count=sum(r['first_target_contact_step'] is not None for r in current),
                grasp_count=sum(r['first_target_grasp_step'] is not None for r in current)))
    for patch in c['patches']:
        raw_name=patch['name']+'__native'
        for mode in ('dino','siglip','both'):
            name=patch['name']+'__'+mode
            for suite in c['suites']+['ALL_SUITES']:
                cases=[case for case in m['cases'] if suite=='ALL_SUITES' or case['suite']==suite]
                clean={case['start']:lookup[(case['start'],'clean')]['success'] for case in cases if (case['start'],'clean') in lookup}
                raw={case['start']:lookup[(case['start'],raw_name)]['success'] for case in cases if (case['start'],raw_name) in lookup}
                restored={case['start']:lookup[(case['start'],name)]['success'] for case in cases if (case['start'],name) in lookup}
                values=[dict(suite=case['suite'],task_id=case['task_id'],value=int(restored[case['start']])-int(raw[case['start']]))
                    for case in cases if case['start'] in raw and case['start'] in restored and case['start'] in clean]
                effects.append(dict(patch=patch['name'],restoration=mode,suite=suite,**paired_outcomes(clean,raw,restored),
                    difference_CI95=interval(values,c['bootstrap_draws'],c['bootstrap_seed'])))
                if suite!='ALL_SUITES':
                    for case in cases:
                        s=case['start']
                        task_rows.append(dict(patch=patch['name'],restoration=mode,suite=suite,task_id=case['task_id'],start=s,
                            clean_success=clean.get(s),raw_success=raw.get(s),restored_success=restored.get(s),
                            complete_pair=s in clean and s in raw and s in restored))
    for path in sorted((ROOT/'reports/COSTS').glob('*.json')):
        v=read(path);costs.append(dict(stage=v['stage'],status=v['status'],elapsed_s=v['elapsed_s'],
            process_rss_hwm_bytes=v.get('process_rss_hwm_bytes'),cuda_peak_allocated_bytes=v.get('cuda_peak_allocated_bytes'),
            cuda_peak_reserved_bytes=v.get('cuda_peak_reserved_bytes'),error=v.get('error')))
    gates=[read(p) for p in (ROOT/'reports/REUSE_GATES').glob('*.json')]
    full=len(rows)==144
    status=dict(status='COMPLETE' if full else 'PARTIAL',completed=len(rows),expected=144,missing=144-len(rows),
        reused_clean=sum(r['reused_clean'] for r in rows),GPU_start_gates=len(gates),training_performed=False)
    folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
    for name,values in [('EPISODES',rows),('SUMMARY',summaries),('RESTORATION_EFFECTS',effects),
        ('PER_TASK',task_rows),('ACTION_RESPONSES',action_rows),('COSTS',costs)]:table(folder/(name+'.csv'),values)
    totals=[r for r in summaries if r['suite']=='ALL_SUITES']
    all_effects=[r for r in effects if r['suite']=='ALL_SUITES']
    identity_mismatches=[dict(start=case['start'],patch=p['name']) for case in m['cases'] for p in c['patches']
        if (case['start'],'clean') in lookup and (case['start'],p['name']+'__both') in lookup
        and lookup[(case['start'],'clean')]['success']!=lookup[(case['start'],p['name']+'__both')]['success']]
    recommendation=dict(status=status['status'],action='Discuss repair location; no training is scheduled',
        oracle_only=True,both_restore_outcome_mismatches=identity_mismatches,full_pair_results=all_effects)
    if not full:recommendation['action']='Complete missing paired episodes before choosing a repair location'
    elif identity_mismatches:recommendation['action']='Investigate full-restoration rollout identity before module training'
    atomic(status,folder/'STATUS.json');atomic(recommendation,folder/'RECOMMENDATION.json')
    def pct(v):return '—' if v is None else f'{100*v:.2f}%'
    lines=['# 联合OFT：修复位置与任务恢复','',f"完成{len(rows)}/144；复用clean {status['reused_clean']}；初态10，初态11未使用。",'',
        '|条件|完成|任务成功率|clean成功起点上的失败|首次接触|抓持|','|---|---:|---:|---:|---:|---:|']
    for r in totals:lines.append(f"|{r['condition']}|{r['completed']}/16|{pct(r['task_SR'])}|{r['failures_on_clean_success_starts']}/{r['clean_success_pairs']}|{r['first_contact_count']}|{r['grasp_count']}|")
    lines+=['','|补丁|恢复位置|补丁新增失败中的救回|原成功被损害|任务成功率差|','|---|---|---:|---:|---:|']
    for r in all_effects:lines.append(f"|{r['patch']}|{r['restoration']}|{r['rescued_induced_failures']}/{r['patch_induced_failures']}|{r['raw_successes_lost']}|{pct(r['SR_difference'])}|")
    lines+=['','原生PNG与特权恢复分开解释。干净供体均来自当前受损分支同一观测，恢复结果不是已部署防护成绩。',
        '全部16起点保留，包括clean失败；缺失不计成功。区间按套件分层、任务→起点重采样，动作槽位不充当独立样本。',
        '双分支恢复与同状态clean动作每次决策检查1e-5；整回合结果不一致数：'+str(len(identity_mismatches))+'。',
        '复用clean的计时属于历史成本；COSTS.csv仅汇总本轮worker成本，失败尝试也保留。',
        '视频：research_videos/wrist_joint_feature_rescue_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/。',
        '下一步：'+recommendation['action']]
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if full:finish(folder/'COMPLETE.json',list(folder.glob('*.csv'))+[folder/'RESULTS.md',folder/'STATUS.json',folder/'RECOMMENDATION.json'])
    monitor.update(phase=status['status'],completed=len(rows),total=144);return status

def videos(monitor):
    verify();exporter=module('feature_rescue_video_exporter',PROJECT/'research_tools/export_singleview_videos.py')
    out=PROJECT/'research_videos/wrist_joint_feature_rescue_v1_singleview'
    result=exporter.export_run(ROOT,out)
    atomic(dict(status='COMPLETE' if result['episodes']==144 else 'PARTIAL',completed=result['episodes'],expected=144,video_root=str(out)),
        ROOT/'reports/VIDEO_STATUS.json')
    monitor.update(phase='video_export_complete',completed=result['episodes'],total=144)
