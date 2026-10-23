"""Matched-length controller targets; no simulator or model imports."""
import numpy as np


def rotation_between(source, target):
    a, b = np.asarray(source, float), np.asarray(target, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite direction')
    if not (np.isclose(np.linalg.norm(a), 1) and np.isclose(np.linalg.norm(b), 1)):
        raise ValueError('Directions must be unit vectors')
    cosine = np.clip(a @ b, -1., 1.)
    if cosine > 1 - 1e-12:
        return np.eye(3)
    if cosine < -1 + 1e-12:
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis); axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def matched_targets(clean, original, unit, donor_unit, bound=.05):
    """Rotate the displacement, then shrink BOTH arms to their joint feasible ball.

    The shared factor is chosen per observation across all eight slots. No
    post-hoc clipping: that would break equal displacement norms. Boundary
    degeneracies remain recorded, including alpha=0.
    """
    clean, original = np.asarray(clean, float), np.asarray(original, float)
    if clean.shape != (1, 8, 3) or original.shape != clean.shape:
        raise ValueError('Expected 1x8x3 commands')
    if not np.isfinite(clean).all() or not np.isfinite(original).all():
        raise ValueError('Nonfinite commands')
    if max(abs(clean).max(), abs(original).max()) > bound + 1e-8:
        raise ValueError('Command outside native controller box')
    clean = clean.clip(-bound, bound)
    delta = original - clean
    rotated = delta @ rotation_between(unit, donor_unit).T
    alpha = 1.
    for change in (delta, rotated):
        for value, movement in zip(clean.flat, change.flat):
            if movement > 1e-14:
                alpha = min(alpha, (bound - value) / movement)
            elif movement < -1e-14:
                alpha = min(alpha, (-bound - value) / movement)
    alpha = max(0., min(1., alpha))
    # A tiny interior offset handles serialized float32 rounding at the boundary.
    if alpha < 1: alpha *= 1 - 1e-6
    targets = [clean + alpha * change for change in (delta, rotated)]
    for target in targets:
        if abs(target).max() > bound + 1e-10:
            raise ValueError('Matched target violated bounds')
    lengths = [np.linalg.norm(target - clean, axis=-1) for target in targets]
    if not np.allclose(*lengths, atol=1e-10, rtol=1e-8):
        raise ValueError('Matched perturbation lengths differ')
    return targets[0].astype('float32'), targets[1].astype('float32'), alpha


def donor_map(rows, seed=20261021):
    """Suite-blocked episode derangement; donors differ in trajectory AND task.

    Preserve frame rank when present, otherwise use the nearest available rank.
    No Validation observations or action responses enter assignment.
    """
    rng = np.random.default_rng(seed); suites = {}
    for row in rows:
        suites.setdefault(row['suite'], {}).setdefault(row['episode_id'], []).append(row)
    result = {}
    for suite, episodes in sorted(suites.items()):
        names = sorted(episodes)
        for _ in range(10000):
            perm = rng.permutation(len(names))
            if all(episodes[names[i]][0]['task_id'] != episodes[names[j]][0]['task_id']
                   for i, j in enumerate(perm)):
                break
        else:
            raise ValueError('Cannot derange tasks: ' + suite)
        for i, j in enumerate(perm):
            donor = episodes[names[j]]
            for row in episodes[names[i]]:
                choice = min(donor, key=lambda r: (abs(r['frame_rank'] - row['frame_rank']), r['state_id']))
                result[row['state_id']] = choice
    return result
