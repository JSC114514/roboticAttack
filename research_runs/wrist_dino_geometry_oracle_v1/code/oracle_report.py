"""Complete paired outcomes, intervention coverage and qualified interpretation."""
import numpy as np
import pandas as pd
from oracle_common import (ROOT, CLOSED, MATCH, config, native, load_module, require, read_json,
                           atomic_json, atomic_parquet, finish, complete, episode_path)

def summarize(frame):
    c = config()
    require(len(frame) == 30 and not frame.duplicated(['task_id', 'state_index', 'condition']).any(),
            'Incomplete/duplicate oracle episodes')
    pairs = frame.pivot(index=['task_id', 'state_index'], columns='condition', values='success').sort_index()
    require(len(pairs) == 10 and set(pairs.columns) == set(c['conditions']) and not pairs.isna().any().any(),
            'Missing paired oracle condition')
    require(all(pairs.loc[t].index.tolist() == c['states'] for t in c['tasks']), 'Unexpected oracle start set')
    eligible = pairs[pairs.clean == 1]
    interval = load_module('oracle_readonly_intervals', MATCH / 'code/report.py').interval
    output = []
    for condition in c['conditions'][1:]:
        failures = (1 - eligible[condition]).reset_index()
        groups = [rows[condition].to_numpy() for _, rows in failures.groupby('task_id')]
        stats = interval(groups, draws=c['bootstrap_draws'])
        rows = frame[frame.condition == condition]
        output.append({'condition': condition, 'failure_switches': int(failures[condition].sum()),
                       'eligible_clean_starts': len(eligible),
                       'pooled_CASR': float(failures[condition].mean()) if len(eligible) else None,
                       'task_equal_CASR': stats['mean'], 'ci95_low': stats['ci95_low'], 'ci95_high': stats['ci95_high'],
                       'tasks_with_clean_success': stats['tasks'], 'successes_all10': int(rows.success.sum()),
                       'available_target_decisions': int(rows.available_target_decisions.sum()),
                       'total_decisions': int(rows.decision_blocks.sum()),
                       'intervened_executed_slots': int(rows.intervened_executed_slots.sum()),
                       'executed_steps': int(rows.executed_steps.sum()),
                       'intervention_step_fraction': float(rows.intervened_executed_slots.sum() / rows.executed_steps.sum())})
    return pairs, pd.DataFrame(output)

def run(monitor):
    rows, missing = [], []
    for task in config()['tasks']:
        for state in config()['states']:
            for condition in config()['conditions']:
                receipt = episode_path(task, state, condition)
                try:
                    require(complete(receipt), 'Episode missing')
                    rows.append(read_json(receipt.parent / 'EPISODE.json'))
                except Exception as error:
                    missing.append({'task_id': task, 'state_index': state, 'condition': condition, 'error': repr(error)})
    folder = ROOT / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    if missing:
        atomic_json({'status': 'FAILED_OR_INCOMPLETE', 'completed_episodes': len(rows), 'missing': missing}, folder / 'STATUS.json')
        (folder / 'RESULTS.md').write_text(f'理想DINO几何干预尚未完整完成：{len(rows)}/30回合。不能以缺失任务后的子集给出最终结论。\n详见STATUS.json及失败成本。\n')
        raise RuntimeError('Oracle incomplete; missing/engineering failures are not attack successes')
    frame = pd.DataFrame(rows)
    pairs, scores = summarize(frame)
    atomic_parquet(frame, folder / 'ROLLOUTS.parquet')
    pairs.to_csv(folder / 'PAIRS.csv')
    scores.to_csv(folder / 'CASR.csv', index=False)
    trace_rows = []
    for meta in rows:
        path = episode_path(meta['task_id'], meta['state_index'], meta['condition']).parent / 'TRACE.parquet'
        trace = pd.read_parquet(path)
        for row in trace.to_dict('records'):
            trace_rows.append({k: meta[k] for k in ('task_id', 'state_index', 'condition')} | row)
    atomic_parquet(pd.DataFrame(trace_rows), folder / 'TRACES.parquet')
    old_clean_agrees = all(read_json(CLOSED / f'evaluation/episodes/t{r["task_id"]:02d}_s{r["state_index"]}_clean/EPISODE.json')['success'] == r['success']
                           for r in rows if r['condition'] == 'clean')
    count = int((pairs.clean == 1).sum())
    lines = ['# 当前几何目标的理想DINO闭环验证', '',
             f'固定Spatial checkpoint；5任务×初态44/45，30回合完整配对。本轮clean成功{count}/10。',
             '每次决策在原生相机采样时，仅移动目标可视几何，物理状态不动；将对应腕部DINO空间输出完整替换到原生策略。',
             '前视、腕部SigLIP、语言、本体状态和模型权重不变。世界X正负平移量沿用旧规则：30°方位弦长、上限30mm，非固定30mm。',
             '**这是内部机制干预结果，不是RGB补丁结果，也不是任意DINO干预的效果上限。**', '',
             '|条件|失败切换/clean成功|CASR（起点计数）|实际执行槽位干预覆盖率|',
             '|---|---:|---:|---:|']
    for row in scores.to_dict('records'):
        score = 'NA' if pd.isna(row['pooled_CASR']) else f'{row["pooled_CASR"]:.1%}'
        lines.append(f'|{row["condition"]}|{row["failure_switches"]}/{count}|{score}|{row["intervention_step_fraction"]:.1%}|')
    if scores.failure_switches.sum() == 0:
        lines += ['', '本组起点、当前目标幅度及实际覆盖范围内，理想DINO目标未造成任务失败。不能把RGB补丁拟合不足作为唯一解释；不据此否定其他空间目标或DINO。']
    else:
        lines += ['', '当前几何目标在部分回合可造成任务失败；需检查轨迹中作用阶段及覆盖，再决定如何训练共享RGB补丁。不能将内部干预成功视作补丁已经实现。']
    lines += ['', '没有有效可见目标时保留原生动作，明确报告原因，不删除难样本；表示无变化与目标不可见在TRACE中可区分。',
              '完整动作8×7、实际执行槽位、同观测clean参考、目标实体/位移/可见性及相机时钟见TRACES.parquet；命令不当作实际位移。',
              'CASR.csv另报任务等权均值及任务→初态重采样区间；只有5任务/10起点，区间作探索性描述，不设置效果门槛。',
              f'本轮clean与此前10组配对clean成功结果一致：{old_clean_agrees}。旧RGB/UADA结果只作历史背景，本轮不重测这些条件。',
              '这些起点来自已查看的开发集；此次干预沿当前轨迹更新，不宣称跨任务、跨模型或普适效果。',
              '不会自动训练补丁、放大位移或扩展任务。每30秒心跳及成功/失败尝试成本见reports/PROGRESS和reports/COSTS。']
    (folder / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    atomic_json({'status': 'COMPLETE', 'episodes': 30, 'clean_successes': count,
                 'closed_loop_tested': True, 'scope': 'internal_DINO_mechanism_not_RGB_patch',
                 'historical_clean_success_agrees': old_clean_agrees}, folder / 'STATUS.json')
    finish(folder / 'COMPLETE.json', [folder / p for p in ('RESULTS.md', 'STATUS.json', 'ROLLOUTS.parquet',
                                                          'PAIRS.csv', 'CASR.csv', 'TRACES.parquet')])
    monitor.update(phase='report_written', completed=30, total=30)
