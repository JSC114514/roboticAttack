"""Explicit legacy PNG inventory: no history scan or implicit best checkpoint."""
from pathlib import Path
from jr_common import PROJECT, read, sha, require


def inventory():
    root = PROJECT / 'research_runs'
    fair = read(root / 'wrist_spatial_relation_fair_v1/data/MANIFEST.json')
    rows = []
    def add(name, path, family, source_model, observations, updates, seed, core=False, proof=None, kind='trained'):
        p = Path(path)
        require(p.is_file(), 'Missing PNG: ' + str(p))
        h = sha(p); receipt = Path(proof) if proof else p.with_suffix('.json')
        source_budget = {}; initial_source = None; initial_hash = None
        if kind == 'trained':
            require(receipt.is_file(), 'Missing final PNG receipt: ' + str(receipt))
            v = read(receipt); require(v.get('status') == 'COMPLETE', 'Unfinished PNG: ' + name)
            # Verify the exact exported PNG; do not trust an aggregate WAITING status.
            candidates = [(k, value) for k, value in v.get('artifacts', {}).items() if Path(k).name == p.name]
            require(any(value == h for k, value in candidates), 'PNG not attested by source completion: ' + name)
            source_budget = {k: v[k] for k in ('step','optimized','no_target','budget','counts','candidate_updates') if k in v}
            candidates = [a / 'data' / f'initial_s{seed}.png' for a in p.parents if (a / 'data' / f'initial_s{seed}.png').is_file()]
            require(candidates, 'Source initialization unavailable: '+name)
            initial_source = str(candidates[0]); initial_hash = sha(initial_source)
            require(initial_hash == sha(root / 'wrist_spatial_approach_pilot_v1/data' / f'initial_s{seed}.png'), 'Source initial PNG differs: '+name)
        else: receipt = None
        rows.append(dict(name=name, path=str(p), sha256=h, family=family, source_model=source_model,
                         training_observations=observations, updates=updates, seed=seed, core=core,
                         source_budget=source_budget, initial_source=initial_source, initial_sha256=initial_hash,
                         kind=kind, receipt=str(receipt) if receipt else None))
    for seed in (42, 43):
        for method, family in [('matched_relation','correct_relation'), ('shuffled_relation','shuffled_relation'),
                               ('translation_far_boundary','physical_xyz_boundary'), ('uada_clean_xyz','uada_xyz'),
                               ('uada_clean_7dof','uada_all_actions')]:
            name = method + '_s' + str(seed)
            add(name, fair['conditions'][name], family, 'libero_goal', 217, 4000, seed, True)
        add('earlier_radial_s' + str(seed), root / f'wrist_spatial_approach_pilot_v1/training/radial_reverse_s{seed}/checkpoint_004000.png',
            'earlier_approach_blocking', 'libero_goal', 217, 4000, seed)
    for method in ('joint', 'fixed_harm', 'realizability_only', 'translation_boundary'):
        add('geometry_select_' + method, root / f'wrist_spatial_target_joint_v1_fix3/training/{method}/checkpoint_004000.png',
            'geometry_selection_' + method, 'libero_goal', 47, 4000, 42)
    for seed, updates in ((42, 20000), (43, 19000)):
        for method in ('geometry_teacher', 'clean_far_boundary'):
            name = method + '_s' + str(seed)
            selected = read(root / 'wrist_dino_geometry_action_teacher_v1/data/SELECTED.json')['candidates'][name]
            require(selected['step'] == updates and selected['matched_budget'], 'Long-run selected budget changed')
            add('geometry_action_' + name, selected['path'], 'geometry_action_' + method,
                'libero_spatial', 240, updates, seed)
    for method in ('projection', 'full'):
        p = root / f'wrist_dino_geometry_match_v1/training/pilot/{method}'
        v = read(p / 'COMPLETE.json'); require(v['candidate_updates'] == v['optimized'] == 4800, 'Pilot budget changed')
        add('geometry_match_' + method, p / 'final.png', 'geometry_match_' + method,
            'libero_spatial', 48, 4800, 42, proof=p / 'COMPLETE.json')
    require(len(rows) == 22 and sum(r['core'] for r in rows) == 10, 'Unexpected learned-patch budget')
    for seed in (42, 43):
        name = 'initial_s' + str(seed); p = Path(fair['conditions'][name])
        for directory in ('wrist_spatial_approach_pilot_v1', 'wrist_dino_geometry_action_teacher_v1', 'wrist_dino_geometry_match_v1'):
            require(sha(root / directory / 'data' / (name + '.png')) == sha(p), 'Initial PNG differs across families')
        add(name, p, 'initial', None, None, 0, seed, kind='initial')
    p = root / 'wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png'
    add('historical_uada500', p, 'historical_reference', 'libero_spatial', 250, '500_outer', None, kind='historical_reference')
    require(len(rows) == 25 and len({r['sha256'] for r in rows}) == 25, 'Duplicate PNG; update unique episode budget explicitly')
    return [dict(name='clean', path=None, sha256=None, family='clean', core=False, kind='clean',
                 source_model=None, training_observations=None, updates=0, seed=None, receipt=None,
                 source_budget={}, initial_source=None, initial_sha256=None)] + rows


def confirmation_conditions(rows):
    return [r['name'] for r in rows if r['core'] or r['kind'] in ('clean', 'initial', 'historical_reference')]
