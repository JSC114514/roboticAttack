"""CPU gates, frozen selection and paired/task-clustered exploratory reports."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from pilot_common import (ROOT, atomic_json, atomic_parquet, budget, cfg, digest, key, load_json,
                          parent_root, prepared, require, sha256, shards, signature, specs)
from pilot_run import completed_episode, verify_screen


def episodes(stage, selection_hash='none'):
    rows, receipts = [], {}
    for suite, task in shards(stage):
        folder = ROOT / 'episodes' / stage / suite / ('task_' + str(task).zfill(2))
        for spec in specs(stage, task):
            episode_folder = folder / key(spec)
            receipt = completed_episode(episode_folder, signature(), selection_hash)
            require(receipt is not None, 'Missing episode: ' + str(episode_folder))
            rows.append({**receipt['row'], 'episode_folder': str(episode_folder)})
            path = episode_folder / 'COMPLETE.json'
            receipts[str(path.relative_to(ROOT))] = sha256(path)
    return pd.DataFrame(rows), receipts


def screens():
    results, assets = [], {}
    for suite, task in shards('calibration'):
        folder = ROOT / 'episodes/calibration' / suite / ('task_' + str(task).zfill(2))
        checked = verify_screen(folder)
        assets[str((folder / 'SCREEN_STATUS.json').relative_to(ROOT))] = sha256(folder / 'SCREEN_STATUS.json')
        for name in checked['assets']:
            if name.endswith('.json'):
                results.append(load_json(folder / name))
            path = folder / name
            assets[str(path.relative_to(ROOT))] = checked['assets'][name]
    return results, assets


def derivative_quality(screen, candidate):
    c = cfg()
    groups = ['translation', 'rotation', 'gripper'] if candidate == 'all' else [candidate]
    passing = []
    for endpoint in ['clean', 'patched']:
        for group in groups:
            values = [row for row in screen['finite_differences'] if row['endpoint'] == endpoint and row['group'] == group]
            passing.append(any(row['nonzero'] and row['sign_agrees'] and row['relative_error'] is not None and
                               row['relative_error'] <= c['finite_difference_error_limit'] for row in values))
    return float(np.mean(passing))


def choose(frame, screen_rows, family, suite, patch):
    subset = frame[(frame.suite == suite) & (frame.patch == patch) & (frame.family == family) & (frame.strength == 1.)]
    scores = []
    for group in cfg()[family + '_candidates']:
        values = subset[subset.group == group]
        own = values[~values.random_control.astype(bool)]
        control = values[values.random_control.astype(bool)]
        pairs = own.merge(control, on=['suite', 'task_id', 'state_index', 'patch', 'decision_block'], suffixes=('_own', '_random'))
        if len(pairs):
            valid = pairs.action_gap_l2_own.gt(1e-8) & pairs.action_gap_closing_fraction_own.notna() & pairs.action_gap_closing_fraction_random.notna()
            valid &= pairs.norm_match_ok_random.fillna(False).astype(bool)
            if family == 'spatial':
                valid &= pairs.selected_token_count_own.gt(0)
            else:
                valid &= pairs.control_rank_own.gt(0)
            pairs = pairs[valid]
        quality = 1.
        if family == 'functional':
            matching = [screen for screen in screen_rows if screen['context']['suite'] == suite and screen['context']['patch'] == patch]
            quality = float(np.mean([derivative_quality(screen, group) for screen in matching])) if matching else 0.
        eligible = (len(pairs) >= 2 and pairs.task_id.nunique() == 2 and
                    quality >= cfg()['finite_difference_min_valid_fraction'])
        if len(pairs):
            score = float((pairs.action_gap_closing_fraction_own - pairs.action_gap_closing_fraction_random).groupby(pairs.task_id).mean().mean())
            own_effect = float(pairs.action_gap_closing_fraction_own.groupby(pairs.task_id).mean().mean())
        else:
            score, own_effect = None, None
        scores.append({'group': group, 'eligible': bool(eligible), 'matched_sites': len(pairs),
                       'tasks': sorted(pairs.task_id.unique().astype(int).tolist()) if len(pairs) else [],
                       'score': score, 'own_gap_closing': own_effect, 'finite_difference_passing_fraction': quality})
    available = [row for row in scores if row['eligible']]
    require(available, 'No valid ' + family + ' candidate for ' + suite + '/' + patch + ': ' + json.dumps(scores))
    best = sorted(available, key=lambda row: (-row['score'], row['group']))[0]
    return {**best, 'all_scores': scores, 'positive_screen_evidence': bool(best['score'] > 0),
            'interpretation': 'exploratory candidate; no closed-loop result used for selection'}


def select():
    prepared()
    path = ROOT / 'reports/SELECTION.json'
    if path.exists():
        old = load_json(path)
        require(old['signature'] == signature() and old['status'] == 'FROZEN', 'Frozen selection provenance changed')
        require(all(sha256(ROOT / name) == value for name, value in old['calibration_assets'].items()), 'Frozen calibration changed')
        print('SELECTION_ALREADY_FROZEN', flush=True)
        return
    raw, episode_assets = episodes('calibration')
    native = raw[raw.kind == 'native']
    require(len(raw) == 160 and len(native) == 64, 'Calibration episode budget mismatch')
    require(all(row['status'] == 'REPRODUCTION_PASS' for row in native.reproduction), 'Reproduction gate failed')
    values, screen_assets = screens()
    frame = pd.DataFrame([record for screen in values for record in screen['records']])
    groups = {}
    for suite in cfg()['suites']:
        for patch in cfg()['patches']:
            groups[suite + '|' + patch] = {family: choose(frame, values, family, suite, patch)
                                          for family in ['spatial', 'functional']}
    atomic_parquet(frame, ROOT / 'reports/SAME_STATE_SCREEN.parquet')
    fd = pd.DataFrame([{**screen['context'], **item} for screen in values for item in screen['finite_differences']])
    atomic_parquet(fd, ROOT / 'reports/FINITE_DIFFERENCES.parquet')
    native_seconds_per_block = float(np.median(native.elapsed_s / native.decision_blocks))
    vjp_seconds = float(np.median([s['gradient_summary']['endpoint_vjp_elapsed_s'] for s in values]))
    old = pd.read_parquet(parent_root() / 'results/BATCH500_ROLLOUTS.parquet')
    estimated_blocks = 0
    functional_sites = 0
    for suite, task in shards('main'):
        for spec in specs('main', task):
            match = old[(old.suite == suite) & (old.task_id == task) & (old.state_index == spec['state_index']) & (old.condition == spec['patch'])]
            require(len(match) == 1, 'Missing old timing reference')
            estimated_blocks += int(match.iloc[0].decision_blocks)
            functional_sites += int(spec['kind'].startswith('functional'))
    # These are service-time estimates; they exclude queues, model loading and longer rescued episodes.
    low = (estimated_blocks * native_seconds_per_block + 432 * 4 * 2 * native_seconds_per_block + functional_sites * vjp_seconds) / 3600
    timing = {'native_median_s_per_block': native_seconds_per_block, 'endpoint_vjp_median_s': vjp_seconds,
              'remaining_gpu_service_hours_estimate_low': low,
              'remaining_gpu_service_hours_estimate_high': low * 2,
              'four_gpu_ideal_wall_hours_low': low / 4,
              'excludes': ['queue', 'model_load', 'task_load_imbalance', 'rescued_episode_lengthening'],
              'not_a_deadline_guarantee': True}
    result = {'status': 'FROZEN', 'signature': signature(), 'created_unix': time.time(),
              'selection_rule': 'task-equal mean(normalized action gap closing - matched random); deterministic lexical tie break',
              'selection_tasks': [0, 1], 'selection_state': 40, 'selection_blocks': cfg()['screen_blocks'],
              'groups': groups, 'calibration_assets': {**episode_assets, **screen_assets},
              'control_reuse_gate': {'native_reproductions': 64, 'complete_trajectory_reproduction': True,
                                     'scope': 'calibration tasks 0/1, both states, clean and all 3 patches; other controls reused with this validation scope'},
              'timing': timing, 'new_episode_budget': budget(), 'development_data': True}
    atomic_json(result, path)
    atomic_json(timing, ROOT / 'reports/TIMING.json')
    print('SELECTION_FROZEN ' + json.dumps({'groups': {name: {f: v['group'] for f, v in row.items()} for name, row in groups.items()},
                                          'timing': timing}), flush=True)


def clustered_interval(frame, value, rng, repetitions=None):
    require(len(frame) > 0, 'Cannot bootstrap empty data')
    means = frame.groupby('task_id')[value].mean().to_numpy(dtype=float)
    samples = rng.choice(means, size=(repetitions or cfg()['bootstrap_replicates'], len(means)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return float(means.mean()), float(low), float(high)


def analyze():
    prepared()
    selection = load_json(ROOT / 'reports/SELECTION.json')
    require(selection['signature'] == signature() and selection['status'] == 'FROZEN', 'Invalid selection')
    calibration, ca = episodes('calibration')
    main, ma = episodes('main', sha256(ROOT / 'reports/SELECTION.json'))
    require(len(calibration) + len(main) == 592, 'New episode grid incomplete')
    all_rows = pd.concat([calibration, main], ignore_index=True)
    old = pd.read_parquet(parent_root() / 'results/BATCH500_ROLLOUTS.parquet')
    controls = old[old.condition.isin(['clean', *cfg()['patches']])].copy()
    clean = controls[controls.condition == 'clean'][['suite', 'task_id', 'state_index', 'success']].rename(columns={'success': 'clean_success'})
    native = controls[controls.condition != 'clean'][['suite', 'task_id', 'state_index', 'condition', 'success']].rename(columns={'condition': 'patch', 'success': 'native_success'})
    paired = all_rows.merge(clean, on=['suite', 'task_id', 'state_index'], validate='many_to_one')
    paired = paired.merge(native, on=['suite', 'task_id', 'state_index', 'patch'], how='left', validate='many_to_one')
    paired['reference_success'] = np.where(paired.kind == 'whole_transplant', paired.clean_success, paired.native_success)
    paired['delta_success'] = paired.success - paired.reference_success
    paired['delta_vs_png_success'] = paired.success - paired.native_success
    paired['conditional_failure'] = np.where(paired.clean_success == 1, 1 - paired.success, np.nan)
    paired['native_conditional_failure'] = np.where(paired.clean_success == 1, 1 - paired.native_success, np.nan)
    rng, summaries = np.random.default_rng(cfg()['random_seed']), []
    for (suite, cohort, patch, kind), group in paired[paired.patch != 'clean'].groupby(['suite', 'cohort', 'patch', 'kind']):
        mean, lo, hi = clustered_interval(group, 'delta_success', rng)
        eligible = group[group.clean_success == 1]
        summaries.append({'suite': suite, 'cohort': cohort, 'patch': patch, 'kind': kind, 'episodes': len(group),
                          'tasks': group.task_id.nunique(), 'successes': int(group.success.sum()),
                          'primary_reference': 'clean' if kind == 'whole_transplant' else 'original_png',
                          'reference_successes': int(group.reference_success.sum()),
                          'native_successes': int(group.native_success.sum()), 'success_delta_pp': mean * 100,
                          'task_bootstrap_low_pp': lo * 100, 'task_bootstrap_high_pp': hi * 100,
                          'clean_success_episodes': len(eligible),
                          'CASR_percent': None if len(eligible) == 0 else float(eligible.groupby('task_id').conditional_failure.mean().mean() * 100),
                          'native_CASR_percent': None if len(eligible) == 0 else float(eligible.groupby('task_id').native_conditional_failure.mean().mean() * 100),
                          'window_not_reached': int(group.window_not_reached.sum()),
                          'effective_intervention_blocks': int(group.effective_intervention_blocks.sum())})
    summary = pd.DataFrame(summaries)
    contrasts = []
    for family in ['functional', 'spatial']:
        a, b = family + '_early', family + '_random_early'
        sensitive = paired[paired.kind == a]
        random_control = paired[paired.kind == b]
        match = sensitive.merge(random_control, on=['suite', 'task_id', 'state_index', 'patch', 'cohort'], suffixes=('_sensitive', '_random'), validate='one_to_one')
        match['paired_success_difference'] = match.success_sensitive - match.success_random
        for (suite, cohort, patch), group in match.groupby(['suite', 'cohort', 'patch']):
            mean, lo, hi = clustered_interval(group, 'paired_success_difference', rng)
            contrasts.append({'suite': suite, 'cohort': cohort, 'patch': patch, 'family': family,
                              'pairs': len(group), 'tasks': group.task_id.nunique(),
                              'sensitive_minus_random_success_pp': mean * 100, 'task_bootstrap_low_pp': lo * 100,
                              'task_bootstrap_high_pp': hi * 100,
                              'sensitive_only_success': int(((group.success_sensitive == 1) & (group.success_random == 0)).sum()),
                              'random_only_success': int(((group.success_sensitive == 0) & (group.success_random == 1)).sum())})
    trace_records = []
    for row in all_rows.to_dict('records'):
        with (Path(row['episode_folder']) / row['trace']).open() as stream:
            for line in stream:
                trace_records.append({**json.loads(line), **{field: row[field] for field in ['suite', 'task_id', 'state_index', 'patch', 'kind', 'cohort']}})
    traces = pd.DataFrame(trace_records)
    active = traces[traces.intervention_active.astype(bool)]
    monitor = active.groupby(['suite', 'cohort', 'patch', 'kind']).agg(
        intervention_blocks=('decision_block', 'count'), median_dino_change_l2=('actual_change_l2', 'median'),
        median_action_change_l2=('action_change_l2', 'median'),
        median_action_gap_closing=('action_gap_closing_fraction', 'median')).reset_index()
    outputs = {'PILOT_ROLLOUTS.parquet': paired, 'PILOT_TRACES.parquet': traces,
               'PILOT_SUMMARY.csv': summary, 'MATCHED_CONTROLS.csv': pd.DataFrame(contrasts), 'MONITOR_SUMMARY.csv': monitor,
               'REUSED_CONTROLS.parquet': controls}
    files = {}
    for name, frame in outputs.items():
        path = ROOT / 'results' / name
        if name.endswith('.parquet'):
            atomic_parquet(frame, path)
        else:
            # Atomic CSV without serializing via shell text.
            temporary = path.with_suffix(path.suffix + '.tmp')
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
        files[name] = sha256(path)
    text = ['# DINO 闭环机制小规模验证', '', '完整完成 592 个新回合；没有训练新补丁。', '',
            '这批结果属于单训练种子、已使用开发初态上的探索性机制证据。确认任务 7–9 未参与本次分组选择，但并非未见最终测试集。', '',
            'CASR 越高表示干预后的任务失败越多；success_delta_pp 通常相对于原 PNG 轨迹，正值表示恢复。whole_transplant 的主参照是 clean，负值表示移植导致任务损害；表内 primary_reference 明确标记。', '',
            '随机对照在各自当前物理状态内匹配 token 数/子空间秩及 BF16 实际改变量，分支轨迹不同后不宣称状态或幅度跨轨迹严格相同。', '',
            '整体移植检验充分性参照，整体恢复检验恢复参照；部分替换也可能产生不自然的混合表征。VJP 仅定义局部候选方向。', '',
            '32 步窗口未到达的回合保留在预定比较中，并单列暴露情况。功能基在窗口开始时估计，窗口内固定；窗口结束后回到原生 PNG 推理。', '',
            '95% 区间按任务聚类 bootstrap；主比较每套件 5 个任务，确认比较只有 3 个任务，不支持稳定优越性或唯一根因的强结论。', '',
            '## 结果索引', '', '- PILOT_SUMMARY.csv：所有条件、原 PNG 对照、CASR 和恢复差异。',
            '- MATCHED_CONTROLS.csv：功能/空间分量与匹配随机对照的成败比较。',
            '- MONITOR_SUMMARY.csv / PILOT_TRACES.parquet：干预命中、归一化动作、反归一化与夹爪二值化后的动作、像素、模拟器状态和窗口追踪。',
            '- ../reports/SELECTION.json / FINITE_DIFFERENCES.parquet：冻结分组、有限差分、复现门槛与计时。',
            '- ../episodes/：每回合 trace.jsonl、窗口特征、VJP、失败堆栈与完整性收据。', '']
    result_path = ROOT / 'results/RESULTS.md'
    result_path.write_text('\n'.join(text))
    files['RESULTS.md'] = sha256(result_path)
    atomic_json({'status': 'COMPLETE', 'signature': signature(), 'episodes': 592, 'new_training': 0,
                 'assets': files, 'episode_receipts': {**ca, **ma}, 'selection_sha256': sha256(ROOT / 'reports/SELECTION.json'),
                 'independent_patch_training_seeds': 1, 'development_states': [40, 41],
                 'finished_unix': time.time()}, ROOT / 'results/STATUS.json')
    print('PILOT_COMPLETE 592 episodes; results/RESULTS.md', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['select', 'analyze'])
    args = parser.parse_args()
    try:
        select() if args.action == 'select' else analyze()
    except Exception as error:
        import traceback
        atomic_json({'status': 'FAILED', 'signature': signature(), 'action': args.action,
                     'error': repr(error), 'traceback': traceback.format_exc()}, ROOT / 'reports' / (args.action.upper() + '_FAILED.json'))
        raise


if __name__ == '__main__':
    main()
