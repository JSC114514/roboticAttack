"""Matched final outcomes after finite pulses; incomplete branches stay missing."""
import csv
import numpy as np
from dp_common import (ROOT, PROJECT, config, read, atomic, verify, complete, finish,
                       episode_receipt, episode_files, eligible, module)
from fr_math import interval, paired_outcomes


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def run(monitor):
    import pandas as pd
    m = verify(); c = config(); rows = []; lookup = {}; missing = []; costs = []; actions = []
    available = [case for case in m['cases'] if eligible(case)]
    unavailable = [case for case in m['cases'] if not eligible(case)]
    for case in available:
        for condition in m['conditions']:
            name = condition['name']; receipt = episode_receipt(case, name)
            if not complete(receipt):
                missing.append(dict(start=case['start'], condition=name)); continue
            files = episode_files(receipt); meta = read(files['EPISODE.json'])
            if (meta['state_index'], meta['checkpoint'], meta['rollout_seed']) != (11,m['checkpoint'],case['rollout_seed']):
                raise RuntimeError('Episode provenance differs')
            rows.append(meta); lookup[(case['start'], name)] = meta
            trace = pd.read_parquet(files['TRACE.parquet'])
            active = trace[trace.pulse_active]
            n = np.asarray(active.executed_slots, float)
            def executed(key):
                return float(np.sqrt(np.average(np.asarray(active[key],float)**2,weights=n))) if len(n) and sum(n)>0 else None
            actions.append(dict(start=case['start'], suite=case['suite'], task_id=case['task_id'], condition=name,
                pulse_executed_slots=int(sum(n)), executed_xyz_normalized_rmse=executed('executed_xyz_normalized_rmse'),
                executed_rotation_normalized_rmse=executed('executed_rotation_normalized_rmse'),
                mean_dino_used_delta_rms=float(active.dino_used_delta_rms.mean()) if len(n) else None,
                mean_siglip_used_delta_rms=float(active.siglip_used_delta_rms.mean()) if len(n) else None))
    summaries = []; effects = []; paired = []
    for condition in m['conditions']:
        name = condition['name']; duration = condition['duration']; clean_name = f'clean_{duration:02d}'
        for suite in c['suites']+['ALL_SUITES']:
            cases = [case for case in available if suite=='ALL_SUITES' or case['suite']==suite]
            current = [lookup[(case['start'],name)] for case in cases if (case['start'],name) in lookup]
            pairs = [(lookup[(case['start'],clean_name)],lookup[(case['start'],name)]) for case in cases
                     if (case['start'],clean_name) in lookup and (case['start'],name) in lookup]
            clean_success = [(a,b) for a,b in pairs if a['success']]
            failures = sum(not b['success'] for a,b in clean_success)
            summaries.append(dict(condition=name, pulse_duration=duration, mode=condition['mode'], suite=suite,
                expected=len(cases), completed=len(current), missing=len(cases)-len(current),
                task_successes=sum(x['success'] for x in current),
                task_SR=float(np.mean([x['success'] for x in current])) if current else None,
                clean_success_pairs=len(clean_success), induced_failures=failures,
                CASR=failures/len(clean_success) if clean_success else None,
                removal_reached=sum(x['removal_reached'] for x in current),
                successes_before_or_at_removal=sum(x['success'] and not x['post_removal_steps'] for x in current),
                failures_after_removal=sum(not x['success'] and x['removal_reached'] for x in current),
                actual_png_steps=sum(x['actual_png_steps'] for x in current)))
    for duration in c['durations']:
        names = {key:f'{key}_{duration:02d}' for key in c['modes']}
        for restored in ('dino','siglip'):
            for suite in c['suites']+['ALL_SUITES']:
                cases = [case for case in available if suite=='ALL_SUITES' or case['suite']==suite]
                def outcomes(key):
                    return {case['start']:lookup[(case['start'],names[key])]['success'] for case in cases
                            if (case['start'],names[key]) in lookup}
                clean = outcomes('clean'); raw = outcomes('png'); fixed = outcomes(restored)
                values = [dict(suite=case['suite'],task_id=case['task_id'],
                               value=int(fixed[case['start']])-int(raw[case['start']]))
                          for case in cases if case['start'] in clean and case['start'] in raw and case['start'] in fixed]
                counts = paired_outcomes(clean,raw,fixed)
                grouped = {}
                for value in values: grouped.setdefault(value['suite'],[]).append(value['value'])
                pooled_difference = counts.pop('SR_difference')
                suite_difference = float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
                effects.append(dict(pulse_duration=duration,restoration=restored,suite=suite,**counts,
                    SR_difference=suite_difference,SR_difference_pooled=pooled_difference,
                    difference_aggregation='equal_suite_means_of_complete_pairs',suites_with_pairs=len(grouped),
                    difference_CI95=interval(values,c['bootstrap_draws'],c['bootstrap_seed'])))
                if suite != 'ALL_SUITES':
                    for case in cases:
                        start = case['start']
                        paired.append(dict(start=start,suite=suite,task_id=case['task_id'],pulse_duration=duration,
                            restoration=restored,clean_success=clean.get(start),png_success=raw.get(start),
                            restored_success=fixed.get(start),complete_pair=start in clean and start in raw and start in fixed))
    for p in sorted((ROOT/'reports/COSTS').glob('*.json')):
        v = read(p)
        costs.append({k:v.get(k) for k in ('stage','status','elapsed_s','process_rss_hwm_bytes','cuda_peak_allocated_bytes',
                                          'cuda_peak_reserved_bytes','error')})
    attempts = [dict(path=str(p.relative_to(ROOT)),**read(p)) for p in (ROOT/'episodes').glob('*/*/attempt_*/FAILURE.json')]
    full = len(rows) == m['runnable']
    status = dict(status='COMPLETE' if full else 'PARTIAL', completed=len(rows),expected=m['expected'],
                  runnable=m['runnable'],missing=len(missing),unavailable=len(unavailable)*len(m['conditions']),
                  failed_attempts=len(attempts),training_performed=False,automatic_next_stage=False)
    folder = ROOT/'results'; folder.mkdir(parents=True,exist_ok=True)
    for name, data in [('EPISODES',rows),('SUMMARY',summaries),('DINO_EFFECTS',effects),('PAIRED_OUTCOMES',paired),
                       ('ACTION_RESPONSES',actions),('COSTS',costs),('FAILED_ATTEMPTS',attempts),('MISSING',missing),
                       ('UNAVAILABLE',[dict(start=x['start'],reason=x['eligibility']) for x in unavailable])]:
        table(folder/(name+'.csv'),data)
    atomic(status,folder/'STATUS.json')
    atomic(dict(status=status['status'],decision='Discuss finite-pulse DINO effects before any predictor or patch training',
                effects=[x for x in effects if x['suite']=='ALL_SUITES']),folder/'RECOMMENDATION.json')
    def pct(value):return '—' if value is None else f'{100*value:.2f}%'
    lines = ['# DINO短时干预：移除补丁后的任务结果','',
        f"完成{len(rows)}/{m['runnable']}可执行分支；计划上限{m['expected']}；不可用{status['unavailable']}；缺失{len(missing)}。",'',
        '|条件|完成|任务成功|CASR|移除后失败|','|---|---:|---:|---:|---:|']
    for row in summaries:
        if row['suite']=='ALL_SUITES':
            lines.append(f"|{row['condition']}|{row['completed']}/{row['expected']}|{row['task_successes']}（{pct(row['task_SR'])}）|{pct(row['CASR'])}|{row['failures_after_removal']}|")
    lines += ['', '|干预步数|恢复分支|PNG新增失败|救回|新增损害|成功率差|95%配对区间|',
              '|---:|---|---:|---:|---:|---:|---|']
    for row in effects:
        if row['suite']=='ALL_SUITES':
            lines.append(f"|{row['pulse_duration']}|{row['restoration']}|{row['patch_induced_failures']}|{row['rescued_induced_failures']}|{row['raw_successes_lost']}|{pct(row['SR_difference'])}|{row['difference_CI95']}|")
    lines += ['', 'clean前缀使用保存命令重放；首次接触前决策起点不按攻击响应选择。',
        '每个干预分支的干净供体来自该分支当前状态。8/32步后移除PNG和内部替换，继续原生策略；总预算保持套件原上限。',
        '干预期间已成功的回合单列，缺失回合不计成功。DINO/SigLIP恢复属于机制诊断；PNG条件为真实输入。',
        '分别统计两种时长；恢复成功率差及区间均按有完整配对的套件等权，另存合并配对差。区间按套件分层、任务→起点重采样，动作槽位和视频帧不充当样本。',
        '本轮不训练后果预测器或新补丁，不自动追加实验。',
        '视频：research_videos/wrist_joint_dino_pulse_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if full:
        finish(folder/'COMPLETE.json',list(folder.glob('*.csv'))+[folder/'STATUS.json',folder/'RESULTS.md',folder/'RECOMMENDATION.json'])
    monitor.update(phase=status['status'],completed=len(rows),total=m['runnable'])
    return status


def videos(monitor):
    m = verify()
    exporter = module('dino_pulse_video_exporter', PROJECT/'research_tools/export_singleview_videos.py')
    out = PROJECT/'research_videos/wrist_joint_dino_pulse_v1_singleview'
    result = exporter.export_run(ROOT, out)
    atomic(dict(status='COMPLETE' if result['episodes']==m['runnable'] else 'PARTIAL',completed=result['episodes'],
                expected=m['runnable'],video_root=str(out)),ROOT/'reports/VIDEO_STATUS.json')
    monitor.update(phase='videos',completed=result['episodes'],total=m['runnable'])
