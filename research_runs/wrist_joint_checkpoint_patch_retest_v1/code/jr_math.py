"""Pure CPU policy/episode accounting: failures and missing rows stay distinct."""
import math
import numpy as np


def paired_summary(rows, clean, expected):
    pairs = [(r, clean[r['start']]) for r in rows if r['start'] in clean]
    successes = sum(bool(r['success']) for r in rows)
    eligible = [(r, c) for r, c in pairs if c['success']]
    failures = sum(not r['success'] for r, c in eligible)
    return dict(completed=len(rows), expected=expected, missing=expected-len(rows), robot_successes=successes,
                task_SR=successes / len(rows) if rows else None, clean_success_pairs=len(eligible),
                induced_failures=failures, CASR=failures / len(eligible) if eligible else None)


def hierarchical_interval(rows, statistic, draws=2000, seed=42):
    """Resample task then starts; every sampled start retains all its conditions."""
    if not rows: return None
    groups = {}
    for r in rows: groups.setdefault((r['suite'], r['task_id']), {}).setdefault(r['start'], []).append(r)
    keys = sorted(groups); rng = np.random.default_rng(seed); values = []
    for _ in range(draws):
        sample = []
        for index in rng.integers(0, len(keys), len(keys)):
            starts = list(groups[keys[index]].values())
            for j in rng.integers(0, len(starts), len(starts)): sample.extend(starts[j])
        v = statistic(sample)
        if v is not None and math.isfinite(v): values.append(v)
    return np.quantile(values, [.025, .975]).tolist() if values else None


def suite_equal(values, suites):
    """An all-suite mean exists only when every suite has its own denominator."""
    if any(values.get(s) is None for s in suites): return None
    return sum(values[s] for s in suites) / len(suites)


def weighted_shards(cases, count, max_steps):
    if not 1 <= count <= 4: raise ValueError('One to four workers required')
    bins = [[] for _ in range(count)]; loads = [0] * count
    for c in sorted(cases, key=lambda c: (-max_steps[c['suite']], c['start'])):
        i = min(range(count), key=lambda j: (loads[j], j)); bins[i].append(c); loads[i] += max_steps[c['suite']]
    return bins


def paste(observation, patch, xy=(26, 137)):
    out = {k: np.array(v, copy=True) for k, v in observation.items()}
    if patch is None: return out
    x, y = xy
    if out['wrist_image'].shape != (224, 224, 3) or np.asarray(patch).shape != (50, 50, 3): raise ValueError('RGB shape differs')
    out['wrist_image'][y:y+50, x:x+50] = patch
    return out
