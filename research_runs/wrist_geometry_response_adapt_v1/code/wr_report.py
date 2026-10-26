"""Paired task-success summaries; observations and action slots are never replicates."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import random
import statistics


def harmonic(clean, perturbed):
    return 0.0 if clean + perturbed == 0 else 2 * clean * perturbed / (clean + perturbed)


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def case_key(row):
    return (row['suite'], int(row.get('task', row.get('task_id', -1))),
            int(row.get('init', row.get('state_index', row.get('init_id', -1)))), row['profile'])


def validate_rows(rows):
    """Reject duplicate completed records instead of silently choosing an attempt."""
    result, seen = [], set()
    for raw in rows:
        row = dict(raw)
        key = (row['candidate'],) + case_key(row)
        if key in seen:
            raise ValueError('Duplicate episode: ' + repr(key))
        seen.add(key)
        if not isinstance(row.get('success'), bool):
            raise ValueError('Completed episode requires boolean success: ' + repr(key))
        result.append(row)
    return result


def suite_equal(rows):
    """Equal suite, then task, then initial-state weights for represented units."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['suite']][int(row.get('task', row.get('task_id', -1)))].append(float(row['success']))
    return _mean(_mean(_mean(v) for v in tasks.values()) for tasks in groups.values())


def summarize(rows, expected, profiles):
    """Expected is candidate -> list of suite/task/init/profile dictionaries."""
    rows = validate_rows(rows)
    by_candidate = defaultdict(list)
    for row in rows:
        by_candidate[row['candidate']].append(row)
    output = {}
    for candidate, cases in expected.items():
        current = by_candidate[candidate]
        wanted = {case_key(c) for c in cases}
        present = {case_key(c) for c in current}
        extra = present - wanted
        if extra:
            raise ValueError(f'Unexpected evaluation cases for {candidate}: {sorted(extra)}')
        missing = wanted - present
        per_profile = {}
        for profile in profiles:
            subset = [r for r in current if r['profile'] == profile]
            required = sum(c['profile'] == profile for c in cases)
            per_profile[profile] = dict(completed=len(subset), expected=required,
                successes=sum(r['success'] for r in subset), success_rate=suite_equal(subset),
                complete=len(subset) == required and required > 0)
        clean_name = 'native' if 'native' in profiles else 'clean'
        clean = per_profile.get(clean_name, {}).get('success_rate')
        adverse = [per_profile[p]['success_rate'] for p in profiles if p != clean_name]
        robust = _mean(adverse) if adverse and all(x is not None for x in adverse) else None
        completed = not missing and bool(wanted)
        score = harmonic(clean, robust) if completed and clean is not None and robust is not None else None
        output[candidate] = dict(complete=completed, completed=len(current), expected=len(wanted),
            missing=[list(x) for x in sorted(missing)], per_profile=per_profile,
            clean_success_rate=clean, perturbed_success_rate=robust, selection_score=score)
    return output


def select_configs(summary, candidates, gpu_seconds=None):
    """Select each tunable family only when every preregistered candidate finished."""
    gpu_seconds = gpu_seconds or {}
    families = defaultdict(list)
    for candidate in candidates:
        name = candidate.get('name', candidate.get('id'))
        if name != 'base':
            families[candidate['method']].append(name)
    selection, pending = {}, {}
    for family, ids in families.items():
        unavailable = [c for c in ids if c not in summary or not summary[c]['complete']]
        if unavailable:
            pending[family] = unavailable
            continue
        selection[family] = max(ids, key=lambda c: (
            summary[c]['selection_score'], summary[c]['clean_success_rate'],
            -float(gpu_seconds.get(c, math.inf)), c))
    return dict(selected=selection, pending=pending, complete=bool(families) and not pending)


def paired_bootstrap(rows, candidate, reference='base', repetitions=2000, seed=42):
    """Paired stratified task -> start bootstrap, with the four profiles kept together.

    Input is one checkpoint per candidate. Multi-seed aggregation is implemented
    separately by the caller so a checkpoint seed never becomes a fresh episode.
    """
    selected = {case_key(r): r for r in rows if r['candidate'] == candidate}
    baseline = {case_key(r): r for r in rows if r['candidate'] == reference}
    paired = sorted(selected.keys() & baseline.keys())
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for key in paired:
        suite, task, initial, profile = key
        groups[suite][task][initial].append(float(selected[key]['success']) - float(baseline[key]['success']))
    if not paired:
        return dict(paired_episodes=0, delta=None, ci95=None)
    rng = random.Random(seed)
    def aggregate(sample):
        return statistics.mean(statistics.mean(statistics.mean(statistics.mean(v) for v in starts.values())
            for starts in tasks.values()) for tasks in sample.values())
    actual = aggregate(groups)
    draws = []
    for _ in range(repetitions):
        suite_values = []
        for tasks in groups.values():
            task_ids = list(tasks)
            task_values = []
            for task in rng.choices(task_ids, k=len(task_ids)):
                starts = tasks[task]
                initial_ids = list(starts)
                task_values.append(statistics.mean(statistics.mean(starts[i])
                    for i in rng.choices(initial_ids, k=len(initial_ids))))
            suite_values.append(statistics.mean(task_values))
        draws.append(statistics.mean(suite_values))
    draws.sort()
    return dict(paired_episodes=len(paired), delta=actual,
                ci95=[draws[int(.025 * (repetitions - 1))], draws[int(.975 * (repetitions - 1))]])


def method_summary(rows, candidates, repetitions=2000, seed=42):
    """Multi-seed method means and paired crossed seed/task/start uncertainty."""
    base = {case_key(r): r for r in rows if r['candidate'] == 'base'}
    methods = defaultdict(list)
    lookup = defaultdict(dict)
    for row in rows: lookup[row['candidate']][case_key(row)] = row
    for c in candidates:
        if c['name'] != 'base': methods[c['method']].append(c['name'])
    output = {}
    for method, names in methods.items():
        clean, corrupt = [], []
        for name in names:
            records = list(lookup[name].values())
            clean.append(suite_equal(r for r in records if r['profile'] in ('native','clean')))
            profiles = sorted({r['profile'] for r in records} - {'native','clean'})
            corrupt.append(_mean(suite_equal(r for r in records if r['profile']==p) for p in profiles))
        common = set(base)
        for name in names: common &= set(lookup[name])
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for key in sorted(common):
            s,t,i,p = key
            groups[s][t][i].append(key)
        rng = random.Random(seed)
        def value(sample_names, randomize):
            suites = []
            for tasks in groups.values():
                task_ids = list(tasks)
                chosen = rng.choices(task_ids,k=len(task_ids)) if randomize else task_ids
                values = []
                for t in chosen:
                    ids = list(tasks[t]); initials = rng.choices(ids,k=len(ids)) if randomize else ids
                    values.append(statistics.mean(statistics.mean(
                        statistics.mean(float(lookup[n][k]['success'])-float(base[k]['success'])
                                        for k in tasks[t][i]) for n in sample_names) for i in initials))
                suites.append(statistics.mean(values))
            return _mean(suites)
        draws = sorted(value(rng.choices(names,k=len(names)),True) for _ in range(repetitions)) if common else []
        output[method] = dict(configurations=names,seed_count=len(names),
            clean_success_rate=_mean(v for v in clean if v is not None),
            perturbed_success_rate=_mean(v for v in corrupt if v is not None),
            complete_pair_count_per_seed=len(common),paired_delta=value(names,False),
            paired_ci95=[draws[int(.025*(repetitions-1))],draws[int(.975*(repetitions-1))]] if draws else None)
    return output


def write_report(output, rows, expected, profiles, candidates, stage, gpu_seconds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = validate_rows(rows)
    summaries = summarize(rows, expected, profiles)
    selection = select_configs(summaries, candidates, gpu_seconds) if stage == 'pilot' else None
    result = dict(stage=stage, status='COMPLETE' if all(v['complete'] for v in summaries.values()) else 'INCOMPLETE',
                  metric='robot_task_success_rate', suites_equal_weighted=True,
                  candidates=summaries, selection=selection,
                  paired_vs_base={c: paired_bootstrap(rows, c) for c in summaries if c != 'base'})
    native_rows=[r for r in rows if r['profile'] in ('native','clean')]
    changed_rows=[r for r in rows if r['profile'] not in ('native','clean')]
    result['paired_vs_base_by_domain']={c:dict(native=paired_bootstrap(native_rows,c),
        camera=paired_bootstrap(changed_rows,c)) for c in summaries if c!='base'}
    # The main scientific comparison is against ordinary augmentation and action
    # consistency, with the same starts. Base-only comparisons cannot answer it.
    if selection and selection['complete'] and set(('bc','augment','consistency','constrained')).issubset(selection['selected']):
        chosen=selection['selected'];main=chosen['constrained']
        result['selected_method_vs_controls']={method:dict(
            native=paired_bootstrap(native_rows,main,chosen[method]),
            camera=paired_bootstrap(changed_rows,main,chosen[method]))
            for method in ('bc','augment','consistency')}
    if stage == 'main': result['methods'] = method_summary(rows,candidates)
    (output / 'SUMMARY.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    if selection is not None:
        # Incomplete family selection cannot be consumed by the main launcher.
        (output / 'SELECTION.json').write_text(json.dumps(selection, indent=2) + '\n')
    with (output / 'EPISODES.csv').open('w', newline='') as f:
        fields = ['candidate', 'suite', 'task', 'init', 'profile', 'success', 'elapsed_seconds']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# 腕部视觉适配结果', '', f'阶段：{stage}；状态：{result["status"]}。指标为机器人任务成功率，越高越好。', '',
             '| 配置 | 完成/计划回合 | 原生成功率 | 相机扰动平均成功率 | 调参分数 |', '|---|---:|---:|---:|---:|']
    pct = lambda x: '—' if x is None else f'{100*x:.2f}%'
    for name, item in summaries.items():
        lines.append(f'| {name} | {item["completed"]}/{item["expected"]} | {pct(item["clean_success_rate"])} | {pct(item["perturbed_success_rate"])} | {pct(item["selection_score"])} |')
    lines += ['', '成功率按套件、任务、起点等权汇总。缺失回合单独列出，不计作成功或失败；不完整配置不参与调参选择。', '',
              '配对区间按套件内任务→起点重采样，保持同一起点的条件配对；动作和视频帧不是独立样本。']
    if result.get('selected_method_vs_controls'):
        lines += ['', '## 约束适配与普通训练方法的直接比较', '',
                  '| 对照 | 原生成功率差值 | 相机扰动成功率差值 | 扰动差值95%区间 |',
                  '|---|---:|---:|---|']
        for name,value in result['selected_method_vs_controls'].items():
            ci=value['camera']['ci95']
            interval='—' if ci is None else f'[{pct(ci[0])}, {pct(ci[1])}]'
            lines.append(f'| {name} | {pct(value["native"]["delta"])} | {pct(value["camera"]["delta"])} | {interval} |')
    if selection is not None:
        lines += ['', '## 小规模配置选择', '', '```json', json.dumps(selection, ensure_ascii=False, indent=2), '```', '',
                  '本阶段结束于报告，不自动提交正式训练。']
    if stage == 'main':
        lines += ['', '## 跨训练种子汇总', '', '| 方法 | 种子数 | 原生成功率 | 扰动成功率 | 相对原模型配对差值及95%区间 |', '|---|---:|---:|---:|---|']
        for name,item in result['methods'].items():
            ci=item['paired_ci95']; interval='—' if ci is None else f'[{pct(ci[0])}, {pct(ci[1])}]'
            lines.append(f'| {name} | {item["seed_count"]} | {pct(item["clean_success_rate"])} | {pct(item["perturbed_success_rate"])} | {pct(item["paired_delta"])} {interval} |')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return result


def run(stage='pilot', monitor=None):
    from wr_common import ROOT, candidates, cases, complete, signature, config, atomic
    from wr_orchestrate import profiles_for
    profiles = profiles_for(stage)
    configs = [dict(name='base', method='base', seed=0)] + candidates(stage)
    expected = {c['name']: [{**case, 'profile': profile} for case in cases(stage) for profile in profiles]
                for c in configs}
    rows = []
    for path in sorted((ROOT/'evaluation'/stage).glob('*/*/*/RESULT.json')):
        receipt = path.with_name('COMPLETE.json')
        if not complete(receipt):
            continue
        row = json.loads(path.read_text())
        if row.get('signature') != signature():
            raise ValueError('Evaluation signature differs: '+str(path))
        row['task'] = row['task_id']; row['init'] = row['state_index']
        row['elapsed_seconds'] = row.get('elapsed_s')
        rows.append(row)
    costs = {}
    for candidate in configs:
        folder = ROOT/'training'/stage/candidate['name']
        for name in ('COMPLETE.json', 'METRICS.json'):
            path = folder/name
            if path.exists():
                payload = json.loads(path.read_text())
                for field in ('gpu_seconds', 'elapsed_s', 'elapsed_seconds'):
                    if field in payload:
                        costs[candidate['name']] = float(payload[field]); break
    result = write_report(ROOT/'results'/stage, rows, expected, profiles, configs, stage, costs)
    offline={}
    for item in configs:
        path=ROOT/f'evaluation/{stage}/{item["name"]}/OFFLINE.json'
        if not complete(path.with_name('OFFLINE_COMPLETE.json')):continue
        data=json.loads(path.read_text())['records'];metrics={}
        for metric in ('B','R','C'):
            groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
            for record in data:
                if metric in record:groups[record['suite']][record['task_id']][record['episode_id']].append(record[metric])
            metrics[metric]=_mean(_mean(_mean(_mean(v) for v in episodes.values())
                for episodes in tasks.values()) for tasks in groups.values())
        offline[item['name']]=metrics
    result['offline_validation']=offline
    atomic(result,ROOT/'results'/stage/'SUMMARY.json')
    if offline:
        with (ROOT/'results'/stage/'RESULTS.md').open('a') as f:
            f.write('\n## 跨轨迹离线核验\n\nB为专家动作误差，R为专家动作变化误差，C为同状态相机变化引起的动作差异，均为按有效范围归一化的均方误差，越低越好。\n\n| 配置 | B | R | C |\n|---|---:|---:|---:|\n')
            for name,value in offline.items():
                f.write('| '+name+' | '+' | '.join('—' if value[k] is None else f'{value[k]:.6g}' for k in ('B','R','C'))+' |\n')
    motivation=ROOT/'results/pilot/MOTIVATION.json'
    if stage=='pilot' and motivation.is_file():
        result['motivation']=json.loads(motivation.read_text())
        atomic(result,ROOT/'results'/stage/'SUMMARY.json')
        if not result['motivation']['continue_training']:
            with (ROOT/'results'/stage/'RESULTS.md').open('a') as f:
                f.write('\n原模型未出现“目标在起点仍可见、改变相机后诱发失败”的配对证据，本轮停在原模型筛查，没有消耗适配训练预算。\n')
    if result['selection'] is not None:
        atomic({**result['selection'], 'signature':signature()}, ROOT/'results'/stage/'SELECTION.json')
    for level in ('suite', 'task'):
        grouped = defaultdict(list)
        for row in rows:
            key = (row['candidate'], row['suite'], row['profile'])
            if level == 'task': key += (row['task'],)
            grouped[key].append(row)
        with (ROOT/'results'/stage/(level.upper()+'.csv')).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['candidate','suite','profile']+(['task'] if level=='task' else [])+['completed','successes','success_rate'])
            for key, subset in sorted(grouped.items()):
                writer.writerow(list(key)+[len(subset), sum(r['success'] for r in subset), suite_equal(subset)])
    return result
