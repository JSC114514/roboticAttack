"""Paired episode accounting and task/start bootstrap, never frame-level counts."""
import csv
from pathlib import Path
import numpy as np
from jr_common import (ROOT, PROJECT, config, read, sha, atomic, require, complete,
                       finish, verify, episode_receipt, episode_files, module)
from jr_math import paired_summary, suite_equal


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with temp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def estimates(rows, draws, seed):
    """Rows carry paired numerator/denominator; tasks then starts stay together."""
    if not rows: return None, None
    groups = {}
    for r in rows: groups.setdefault(r['task_id'], []).append(r)
    tasks = sorted(groups); rng = np.random.default_rng(seed); values = []
    for _ in range(draws):
        numerator = denominator = 0.
        for i in rng.integers(len(tasks), size=len(tasks)):
            g = groups[tasks[i]]
            for j in rng.integers(len(g), size=len(g)):
                numerator += g[j]['num']; denominator += g[j]['den']
        values.append(numerator / denominator if denominator else np.nan)
    valid = np.asarray(values); valid = valid[np.isfinite(valid)]
    return np.quantile(valid, [.025, .975]).tolist() if len(valid) else None, np.asarray(values)


def collect(m):
    rows = []
    for case in m['cases']:
        for cond in m['conditions']:
            p = episode_receipt(case, cond['name'])
            if not complete(p): continue
            meta = read(episode_files(p)['EPISODE.json'])
            require(meta['rollout_seed'] == case['rollout_seed'] and meta['checkpoint'] == m['checkpoint']
                    and meta['png_sha256'] == cond['sha256'], 'Episode identity differs')
            rows.append(meta)
    return rows


def summarize(m, rows):
    c = config(); out = []; clean = {r['start']: r for r in rows if r['condition'] == 'clean'}
    for cond in m['conditions']:
        per_suite = []; casr_samples = []; sr_samples = []
        for suite in c['suites']:
            seed = c['bootstrap_seed'] + 1009*c['suites'].index(suite)
            part = [r for r in rows if r['condition'] == cond['name'] and r['suite'] == suite]
            info = paired_summary(part, clean, 10)
            sr_ci, sr = estimates([dict(task_id=r['task_id'], num=r['success'], den=1) for r in part], c['bootstrap_draws'], seed)
            eligible = [r for r in part if r['start'] in clean]
            casr_ci, ca = estimates([dict(task_id=r['task_id'],
                num=(1-r['success'])*int(clean[r['start']]['success']), den=int(clean[r['start']]['success']))
                for r in eligible], c['bootstrap_draws'], seed)
            # Complete-case denominators are reported; missing episodes never count as success.
            row = dict(condition=cond['name'], suite=suite, **info, SR_CI95=sr_ci, CASR_CI95=casr_ci,
                       complete_suite=info['missing'] == 0)
            out.append(row); per_suite.append(row); sr_samples.append(sr); casr_samples.append(ca)
        def mean_ci(samples):
            if any(v is None for v in samples): return None
            values = np.mean(samples, axis=0); values = values[np.isfinite(values)]
            return np.quantile(values, [.025, .975]).tolist() if len(values) else None
        out.append(dict(condition=cond['name'], suite='EQUAL_SUITE_MEAN', completed=sum(r['completed'] for r in per_suite),
            expected=40, missing=sum(r['missing'] for r in per_suite),
            task_SR=suite_equal({r['suite']: r['task_SR'] for r in per_suite}, c['suites']),
            CASR=suite_equal({r['suite']: r['CASR'] for r in per_suite}, c['suites']),
            SR_CI95=mean_ci(sr_samples), CASR_CI95=mean_ci(casr_samples),
            complete_suite=all(r['complete_suite'] for r in per_suite)))
    return out


def contrasts(rows, conditions):
    c = config(); result = []; index = {(r['start'], r['condition']): r for r in rows}
    comparisons = []
    for seed in (42, 43):
        left = f'matched_relation_s{seed}'
        comparisons += [(left, f'{other}_s{seed}') for other in ('shuffled_relation','uada_clean_xyz','uada_clean_7dof','translation_far_boundary','initial')]
        comparisons += [(f'geometry_action_geometry_teacher_s{seed}', f'geometry_action_clean_far_boundary_s{seed}')]
    comparisons += [('geometry_match_full', 'geometry_match_projection')]
    comparisons += [(f'geometry_select_{k}', 'geometry_select_translation_boundary') for k in ('joint','fixed_harm','realizability_only')]
    comparisons += [(v['name'], 'initial_s'+str(v['seed'])) for v in conditions if v['kind']=='trained']
    comparisons = list(dict.fromkeys(comparisons))
    for left, right in comparisons:
        for suite in c['suites']:
            pairs = []
            for clean in rows:
                if clean['condition'] != 'clean' or clean['suite'] != suite: continue
                a, b = index.get((clean['start'], left)), index.get((clean['start'], right))
                if a is not None and b is not None:
                    pairs.append(dict(task_id=clean['task_id'], num=(b['success']-a['success'])*int(clean['success']), den=int(clean['success'])))
            ci, _ = estimates(pairs, c['bootstrap_draws'], c['bootstrap_seed'])
            denominator=sum(p['den'] for p in pairs)
            result.append(dict(left=left, right=right, suite=suite, complete_clean_success_pairs=denominator,
                CASR_difference=sum(p['num'] for p in pairs)/denominator if denominator else None, CI95=ci))
    return result


def historical_goal(m, rows):
    """Verify historical start/PNG evidence; attack changes use common clean successes."""
    base = PROJECT / 'research_runs/wrist_spatial_relation_fair_v1'
    old_manifest = read(m['historical_goal_manifest']); by_start = {v['start']: v for v in old_manifest['cases']}
    out = []; grouped = {}; index = {(r['start'], r['condition']): r for r in rows}
    for case in m['cases']:
        if case['suite'] != 'libero_goal': continue
        old_case = by_start.get(case['start']); now_clean = index.get((case['start'], 'clean'))
        if old_case is None or now_clean is None: continue
        require(old_case['rollout_seed'] == case['rollout_seed'] and old_case['initial_file'] == case['initial_file'], 'Historical seed/initial file differs')
        clean_path = base / 'episodes' / case['start'] / 'clean/COMPLETE.json'
        def load_old(p):
            v = read(p); require(v['status'] == 'COMPLETE', 'Historical episode unfinished')
            needed = {Path(k).name: base / k for k in v['artifacts'] if Path(k).name in ('EPISODE.json', 'START.npz')}
            for k, h in v['artifacts'].items():
                if Path(k).name in needed: require(sha(base/k) == h, 'Historical episode evidence changed')
            return read(needed['EPISODE.json']), needed
        old_clean, old_files = load_old(clean_path)
        current_start = episode_files(episode_receipt(case, 'clean'))['START.npz']
        with np.load(old_files['START.npz'], allow_pickle=False) as old, np.load(current_start, allow_pickle=False) as now:
            start_ok = all(k in old and np.array_equal(now[k], old[k]) for k in ('sim_state', 'proprio', 'front', 'wrist'))
        for cond in m['conditions']:
            name = cond['name']; current = index.get((case['start'], name)); old_path = base / 'episodes' / case['start'] / name / 'COMPLETE.json'
            if current is None or not old_path.exists() or name not in old_manifest['conditions']: continue
            path = old_manifest['conditions'][name]
            if (sha(path) if path else None) != cond['sha256']: continue
            previous, _ = load_old(old_path)
            require(previous['rollout_seed'] == case['rollout_seed'], 'Historical episode seed changed')
            common = bool(old_clean['success'] and now_clean['success'] and start_ok)
            out.append(dict(start=case['start'], task_id=case['task_id'], condition=name, physical_camera_start_match=start_ok,
                old_Goal_success=previous['success'], joint_success=current['success'],
                old_clean_success=old_clean['success'], joint_clean_success=now_clean['success'],
                common_clean_success=common,
                induced_failure_change=(previous['success']-current['success']) if common else None))
            grouped.setdefault(name, []).append(out[-1])
    summary = []
    for name, items in grouped.items():
        valid = [v for v in items if v['physical_camera_start_match']]
        common = [v for v in valid if v['common_clean_success']]
        summary.append(dict(condition=name, paired_starts=len(valid), reset_mismatches=len(items)-len(valid),
            old_Goal_task_SR=sum(v['old_Goal_success'] for v in valid)/len(valid) if valid else None,
            joint_task_SR=sum(v['joint_success'] for v in valid)/len(valid) if valid else None,
            common_clean_successes=len(common),
            CASR_change_common_clean=sum(v['induced_failure_change'] for v in common)/len(common) if common else None))
    return out, summary


def run(monitor):
    m = verify(); rows = collect(m); expected = m['total_episodes']; summary = summarize(m, rows)
    paired = contrasts(rows, m['conditions']); historical, history_summary = historical_goal(m, rows)
    folder = ROOT / 'results'; folder.mkdir(parents=True, exist_ok=True)
    table(folder / 'EPISODES.csv', rows); table(folder / 'CASR.csv', summary)
    table(folder / 'PAIRED_CONTRASTS.csv', paired); table(folder / 'MODEL_COMPARISON_EPISODES.csv', historical)
    table(folder / 'MODEL_COMPARISON.csv', history_summary)
    status = dict(status='COMPLETE' if len(rows) == expected else 'INCOMPLETE', completed=len(rows), expected=expected,
                  missing=expected-len(rows), completed_clean=sum(r['condition']=='clean' for r in rows),
                  weights='fixed joint model; old PNG transfer only', second_batch='prepared, not submitted')
    atomic(status, folder / 'STATUS.json')
    def pct(x): return '—' if x is None else f'{x*100:.2f}%'
    lines = ['# 联合OFT旧补丁复测', '', f"完成 {len(rows)}/{expected} 回合；clean先完成40个起点。", '',
        '表中CASR以该方法已完成、且clean成功的配对起点为分母；缺失回合单列。四套件均值按套件等权。', '',
        '| 条件 | Spatial CASR | Object CASR | Goal CASR | Long CASR | 等权均值 | 完成 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cond in m['conditions']:
        part = [v for v in summary if v['condition']==cond['name']]
        lines.append('| '+cond['name']+' | '+' | '.join(pct(v['CASR']) for v in part)+f" | {part[-1]['completed']}/40 |")
    lines += ['', 'CASR.csv同时列出任务成功率、clean成功分母和任务→起点重采样区间；PAIRED_CONTRASTS.csv给出同起点差异。',
        '本轮初态2，每任务只有一个起点，区间主要反映任务间变化。', '',
        'MODEL_COMPARISON.csv仅比较来源一致、物理状态与双视角缓存一致的历史Goal起点；攻击差值使用两模型共同clean成功起点。', '',
        '22张补丁按各自原训练池、更新次数、源模型解释。该实验测量固定旧PNG在联合模型上的迁移；重训优先级由配套方法与基线的差异决定。', '',
        '视频：research_videos/wrist_joint_checkpoint_patch_retest_v1_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/。',
        '接口错误、资源峰值与失败成本见reports/GATES、reports/COSTS；第二批初态3尚未提交。', '']
    (folder / 'RESULTS.md').write_text('\n'.join(lines))
    if len(rows) == expected:
        finish(folder / 'COMPLETE.json', [folder/n for n in ('EPISODES.csv','CASR.csv','PAIRED_CONTRASTS.csv',
            'MODEL_COMPARISON.csv','MODEL_COMPARISON_EPISODES.csv','STATUS.json','RESULTS.md')])
    monitor.update(phase=status['status'], completed=len(rows), total=expected)
    return status


def videos(monitor):
    m = verify(); output = PROJECT / 'research_videos/wrist_joint_checkpoint_patch_retest_v1_singleview'
    exporter = module('joint_singleview_exporter', PROJECT / 'research_tools/export_singleview_videos.py')
    exporter.sha = sha
    result = exporter.export_run(ROOT, output)
    atomic(dict(status='COMPLETE' if result['episodes']==m['total_episodes'] else 'PARTIAL',
        completed=result['episodes'], expected=m['total_episodes'], video_root=str(output)), ROOT / 'reports/VIDEO_STATUS.json')
    monitor.update(completed=result['episodes'], total=m['total_episodes'], phase='videos_exported')
