"""Paired task-level reporting. Never count action chunks as independent runs."""
import numpy as np
import pandas as pd
from protocol import (ROOT, config, require, read_json, atomic_json, atomic_parquet,
                      complete, finish, episode_path)

def paired_scores(frame):
    c = config()
    require(len(frame) == 60 and not frame.duplicated(['task_id', 'state_index', 'condition']).any(),
            'Missing/duplicate episodes')
    pairs = frame.pivot(index=['task_id', 'state_index'], columns='condition', values='success').sort_index()
    require(set(pairs.columns) == set(c['conditions']) and len(pairs) == 12 and not pairs.isna().any().any(),
            'Unpaired conditions')
    require(all(pairs.loc[t].index.tolist() == c['states'] for t in c['tasks']), 'Task/state mismatch')
    eligible = pairs[pairs.clean == 1]
    result = []
    rng = np.random.default_rng(42)
    tasks = np.array(c['tasks'])
    for condition in c['conditions'][1:]:
        failure = 1 - eligible[condition]
        random_failure = 1 - eligible.initial
        boot, diff = [], []
        for _ in range(c['bootstrap_draws']):
            selected = []
            for task in rng.choice(tasks, len(tasks), replace=True):
                group = eligible.loc[eligible.index.get_level_values('task_id') == task]
                if len(group):
                    selected.append(group.iloc[rng.integers(len(group), size=len(group))])
            if selected:
                sample = pd.concat(selected)
                boot.append(float((1 - sample[condition]).mean()))
                diff.append(float((sample.initial - sample[condition]).mean()))
        result.append({'condition': condition, 'clean_successes': len(eligible),
                       'failures': int(failure.sum()), 'CASR': float(failure.mean()) if len(eligible) else None,
                       'CASR_ci95_low': float(np.quantile(boot, .025)) if boot else None,
                       'CASR_ci95_high': float(np.quantile(boot, .975)) if boot else None,
                       'successes_all12': int(pairs[condition].sum()),
                       'failures_beyond_random': int(((failure == 1) & (random_failure == 0)).sum()),
                       'random_only_failures': int(((failure == 0) & (random_failure == 1)).sum()),
                       'failures_overlap_random': int(((failure == 1) & (random_failure == 1)).sum()),
                       'CASR_difference_vs_random': float((failure - random_failure).mean()) if len(eligible) else None,
                       'difference_ci95_low': float(np.quantile(diff, .025)) if diff else None,
                       'difference_ci95_high': float(np.quantile(diff, .975)) if diff else None})
    return pairs, pd.DataFrame(result)

def run_report(monitor):
    missing = []
    rows = []
    for task in config()['tasks']:
        for state in config()['states']:
            for condition in config()['conditions']:
                receipt = episode_path(task, state, condition)
                try:
                    require(complete(receipt), 'Missing episode')
                    rows.append(read_json(receipt.parent / 'EPISODE.json'))
                except Exception as error:
                    missing.append({'task': task, 'state': state, 'condition': condition, 'error': repr(error)})
    folder = ROOT / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    if missing:
        atomic_json({'status': 'FAILED_OR_INCOMPLETE', 'completed_episodes': len(rows), 'missing': missing}, folder / 'STATUS.json')
        (folder / 'RESULTS.md').write_text(f'闭环评测未完整完成：{len(rows)}/60 回合。不能给出完整比较结论。\n详见 STATUS.json 和失败成本日志。\n')
        raise RuntimeError('Incomplete rollout set; no missing episodes silently excluded')
    frame = pd.DataFrame(rows)
    pairs, summary = paired_scores(frame)
    atomic_parquet(frame, folder / 'ROLLOUTS.parquet')
    pairs.to_csv(folder / 'PAIRS.csv')
    summary.to_csv(folder / 'CASR.csv', index=False)
    metrics = []
    for row in rows:
        trace = pd.read_parquet(episode_path(row['task_id'], row['state_index'], row['condition']).parent / 'TRACE.parquet')
        for r in trace.to_dict('records'):
            metrics.append({k: row[k] for k in ('task_id', 'state_index', 'condition')} | {
                k: r[k] for k in ('decision_block', 'executed_slots', 'first_translation_delta_l2',
                                 'executed_translation_delta_l2_mean', 'executed_rotation_delta_l2_mean',
                                 'executed_gripper_delta_abs_mean')})
    atomic_parquet(pd.DataFrame(metrics), folder / 'ACTION_DIAGNOSTICS.parquet')
    n = int((pairs.clean == 1).sum())
    lines = ['# 现有几何补丁的小规模闭环核验', '',
             f'固定 Spatial checkpoint；6 个任务 × 初态44/45。重新执行 clean 成功 {n}/12。',
             '两张共享补丁均来自同一 seed42、4800次更新的短验证。无新增训练、EOT或内部特征替换。', '',
             '|条件|失败切换/clean成功|CASR|随机补丁成功而该条件失败|与随机失败重合|',
             '|---|---:|---:|---:|---:|']
    for r in summary.to_dict('records'):
        score = 'NA' if pd.isna(r['CASR']) else f'{r["CASR"]:.1%}'
        lines.append(f'|{r["condition"]}|{r["failures"]}/{n}|{score}|{r["failures_beyond_random"]}|{r["failures_overlap_random"]}|')
    lines += ['', 'CASR与随机的配对差值及任务→初态重采样区间见 CASR.csv；12组起点的小样本区间只作探索性描述。',
              '所有条件完整保留；clean失败不计入CASR分母，仍在PAIRS.csv报告。',
              '动作诊断逐回合记录，不把重复动作块当作独立样本。同观测clean预测从不执行。',
              '本轮可回答既有PNG是否造成闭环任务失败，不能单凭此比较证明失败由几何信息介导。',
              '这些是原训练任务；44/45未参与本轮短补丁训练/验证，但在旧长训练池中，不宣称全新留出或任意环境通用。',
              'UADA是既有部署参考，训练预算与数据不同；不据此声称公平训练方法排名。',
              '后续先检查失败轨迹及随机补丁重合，再决定新目标；本入口不启动任何长训练。',
              '状态与每30秒心跳见reports/PROGRESS；RSS/CUDA峰值、吞吐及失败重试成本见reports/COSTS。']
    (folder / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    atomic_json({'status': 'COMPLETE', 'episodes': 60, 'clean_successes': n,
                 'new_training_submitted': False, 'closed_loop_tested': True}, folder / 'STATUS.json')
    finish(folder / 'COMPLETE.json', [folder / name for name in ('RESULTS.md', 'ROLLOUTS.parquet', 'PAIRS.csv',
           'CASR.csv', 'ACTION_DIAGNOSTICS.parquet', 'STATUS.json')])
    monitor.update(phase='report_written', completed=60, total=60)
