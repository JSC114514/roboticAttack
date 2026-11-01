#!/usr/bin/env python3
"""Recompute native PNG counts/CASR from the released episode CSVs on CPU.

Each row is one actual condition/start evaluation. Seeds are separate patches
on shared physical starts. CASR is computed against the same model/start/reset
seed, then averaged across tasks and suites. Internal restorations are excluded.
Historical scope memberships are kept separate; development reused within full
is never summed into a count of new physical starts.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def load_rows(root=HERE):
    """Read small tables, requiring unique study/scope/model/condition/start keys."""
    rows = []
    seen = set()
    for path in sorted((root / 'results/episodes').rglob('*.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[k] for k in ['study', 'scope', 'model', 'condition', 'case_id'])
                if key in seen:
                    raise ValueError('Duplicate released evaluation: ' + str(key))
                seen.add(key)
                row['success'] = bool(int(row['success']))
                row['native_png'] = bool(int(row['native_png']))
                rows.append(row)
    return rows


def summarize(rows, study, scope, model):
    """Return per-PNG rows, explicitly retaining undefined clean denominators."""
    selected = [r for r in rows if r['study'] == study and r['scope'] == scope
                and r['model'] == model and r['native_png']]
    clean = {r['case_id']: r for r in selected if r['condition'] == 'clean'}
    # The sensitivity development cohort explicitly reuses the parent's clean.
    if study == 'sensitivity' and scope == 'development':
        for row in rows:
            if (row['study'], row['scope'], row['model'], row['condition']) == (
                    'main', 'development', model, 'clean'):
                clean.setdefault(row['case_id'], row)
    groups = defaultdict(list)
    for row in selected:
        groups[row['condition']].append(row)
    output = []
    for condition, data in sorted(groups.items()):
        qualified = []
        for row in data:
            reference = clean.get(row['case_id'])
            if reference is None:
                raise ValueError('Missing same-cohort clean: ' + row['case_id'])
            if reference['reset_seed'] != row['reset_seed']:
                raise ValueError('Unpaired reset seed: ' + row['case_id'])
            if reference['success']:
                qualified.append(row)
        tasks = defaultdict(list)
        for row in qualified:
            tasks[row['suite'], row['task_id']].append(float(not row['success']))
        suites = defaultdict(list)
        for (suite, _), values in tasks.items():
            suites[suite].append(statistics.mean(values))
        macro = statistics.mean(statistics.mean(v) for v in suites.values()) if suites else None
        output.append(dict(condition=condition, completed=len(data),
                           success=sum(r['success'] for r in data),
                           clean_success=len(qualified),
                           induced_failures=sum(not r['success'] for r in qualified),
                           macro_casr=macro))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['main', 'factorial', 'sensitivity', 'crosscheck'], default='main')
    parser.add_argument('--scope', default='full')
    parser.add_argument('--model', default='joint_oft')
    args = parser.parse_args()
    rows = summarize(load_rows(), args.study, args.scope, args.model)
    if not rows:
        parser.error('No released cohort matches these arguments')
    print('| Condition | Complete | Task success | Clean denominator | Induced failures | Macro CASR |')
    print('|---|---:|---:|---:|---:|---:|')
    for row in rows:
        score = 'undefined' if row['macro_casr'] is None else f'{100 * row["macro_casr"]:.2f}%'
        print(f'| {row["condition"]} | {row["completed"]} | {row["success"]} | '
              f'{row["clean_success"]} | {row["induced_failures"]} | {score} |')


if __name__ == '__main__':
    main()
