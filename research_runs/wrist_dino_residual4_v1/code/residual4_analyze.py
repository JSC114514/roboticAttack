"""CPU analysis: four records, two physical states, fixed stop rule."""
from __future__ import annotations
import argparse
import json
import statistics
import time
import traceback
from pathlib import Path
from residual4_common import (ROOT, atomic_json, cases, cfg, completed_episode, episode_folder,
    load_json, prepared, require, sha256, signature, specs)


def read_trace(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def decide(rows, quality_ok):
    require(len(rows) == 4 and {r['case_id'] for r in rows} == {'c03', 'c04', 'c05', 'c09'}, 'Four-case grid required')
    require(len({r['state_key'] for r in rows}) == 2, 'Exactly two physical states required')
    for row in rows:
        require(row['failure'] in [0, 1] and row['success'] == 1-row['failure'], 'Invalid binary outcome')
    failed = [r for r in rows if r['failure']]
    states = sorted({r['state_key'] for r in failed})
    decision = ('STOP_DIAGNOSTIC_QUALITY' if not quality_ok else
                'NEXT_PIXEL_FEASIBILITY' if len(states) == 2 else
                'STOP_TWO_STATE_REQUIREMENT' if states else 'STOP_SINGLE_COMPONENT_ROUTE')
    return {'decision': decision, 'quality_pass': quality_ok, 'failed_patch_state_records': len(failed),
            'failed_physical_initial_states': len(states), 'physical_initial_states_total': 2,
            'failed_cases': [r['case_id'] for r in failed], 'failed_state_keys': states,
            'automatic_training': False, 'automatic_additional_episodes': False,
            'pixel_reachability_tested': False, 'suite_wide_asr_computed': False}


def quality_check(case, row, traces):
    import numpy as np
    issues = []
    require(row['direction'] == 'inject' and row['component'] == 'residual_full', 'Wrong new intervention')
    require(len(traces) == row['decision_blocks'] and
            sum(t['executed_slots'] for t in traces) == row['executed_steps'], 'Receipt/trace count mismatch')
    require([t['decision_block'] for t in traces] == list(range(len(traces))), 'Noncontiguous trace')
    require(bool(traces), 'Empty completed trace')
    ref = case['references']['functional_injection']
    baseline = load_json(ref['receipt'])['row']
    old_first = read_trace(ref['trace'])[0]
    for key in ['sim_state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
        a = np.asarray(row['initial_after_wait_state'][key], dtype=np.float64)
        b = np.asarray(baseline['initial_after_wait_state'][key], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-8, rtol=0):
            issues.append('initial_physical_state_mismatch:' + key)
    action_error = float(np.max(np.abs(np.asarray(traces[0]['clean_normalized_action']) -
                                      np.asarray(old_first['clean_normalized_action']))))
    if action_error > cfg()['native_action_atol']:
        issues.append('block_zero_clean_action_mismatch')
    if traces[0].get('validation', {}).get('status') != 'NATIVE_AND_IDENTITY_PASS':
        issues.append('missing_native_identity_validation')
    for t in traces:
        require(t['intervention_active'] and t['basis_refreshed'] and
                t['basis_derived_at_block'] == t['decision_block'], 'Basis was frozen instead of refreshed')
        require(t['amplitude_policy'] == 'unscaled_original_complement' and t['norm_gain'] == 1.,
                'Residual was rescaled')
        if not t['norm_match_ok'] or t['norm_match_relative_error'] > cfg()['norm_match_relative_tolerance']:
            issues.append('post_cast_residual_norm_error:block_' + str(t['decision_block']))
        if t['basis_orthogonality_max_error'] > .002:
            issues.append('basis_orthogonality_error:block_' + str(t['decision_block']))
        applied = np.asarray(t['applied_actions'])
        predicted = np.asarray(t['predicted_processed_action_chunk'])[:t['executed_slots']]
        require(np.array_equal(applied, predicted), 'Executed action log differs from command prefix')
    finite = []
    checks = traces[0].get('finite_differences', [])
    for endpoint in ['clean', 'patched']:
        for group in ['translation', 'rotation', 'gripper']:
            values = [c for c in checks if c['endpoint'] == endpoint and c['group'] == group]
            require(len(values) == 2 and {v['relative_step'] for v in values} ==
                    set(cfg()['finite_difference_relative_steps']), 'Missing finite-difference grid')
            good = any(c['nonzero'] and c['sign_agrees'] and c['relative_error'] is not None and
                       c['relative_error'] <= cfg()['finite_difference_error_limit'] for c in values)
            finite.append({'endpoint': endpoint, 'group': group, 'pass': good})
    return {'case_id': case['id'], 'issues': issues, 'block_zero_clean_action_max_error': action_error,
            'initial_observation_hash_drift': row['initial_observation_hashes'] != baseline['initial_observation_hashes'],
            'intervention_blocks': len(traces), 'zero_effect_blocks': sum(t['actual_change_l2'] == 0 for t in traces),
            'maximum_post_cast_norm_error': max(t['norm_match_relative_error'] for t in traces),
            'actual_residual_full_delta_ratio_median': statistics.median(t['actual_change_fraction_of_delta'] for t in traces),
            'executed_gripper_flips': sum(t['executed_gripper_flips'] for t in traces),
            'finite_checks': finite}


def analyze():
    prepared()
    sig = signature()
    rows, quality, comparisons, assets, missing = [], [], [], {}, []
    for case in cases():
        spec = specs(case)[0]
        folder = episode_folder(case, spec)
        receipt = completed_episode(folder, sig)
        if receipt is None:
            missing.append(case['id'])
            continue
        row = receipt['row']
        require(all(row[k] == v for k, v in spec.items()) and all(row[k] == case[k]
            for k in ['suite', 'task_id', 'state_index', 'state_key', 'rollout_seed']), 'Case pairing changed')
        traces = read_trace(folder / row['trace'])
        rows.append(row)
        quality.append(quality_check(case, row, traces))
        reference = {k: load_json(v['receipt'])['row'] for k, v in case['references'].items()}
        comparisons.append({'case': case['id'], 'suite': case['suite'], 'state_key': case['state_key'],
            'patch': case['patch'], 'whole_injection_failure_reused': reference['whole_injection']['failure'],
            'functional_injection_failure_reused': reference['functional_injection']['failure'],
            'small_orthogonal_injection_failure_reused': reference['small_orthogonal_injection']['failure'],
            'residual_original_amplitude_failure_new': row['failure'], 'steps_new': row['executed_steps']})
        assets[str((folder / 'COMPLETE.json').relative_to(ROOT))] = sha256(folder / 'COMPLETE.json')
        for path, h in receipt['assets'].items():
            assets[str((folder / path).relative_to(ROOT))] = h
    if missing:
        result = {'status': 'INCOMPLETE', 'signature': sig, 'episodes': len(rows), 'missing': missing,
                  'decision': 'INCOMPLETE', 'automatic_training': False}
        atomic_json(result, ROOT / 'results/STATUS.json')
        print(json.dumps(result, indent=2))
        return result
    finite_groups = {}
    for group in ['translation', 'rotation', 'gripper']:
        values = [v for q in quality for v in q['finite_checks'] if v['group'] == group]
        finite_groups[group] = {'checks': len(values), 'passed': sum(v['pass'] for v in values)}
    quality_ok = not any(q['issues'] for q in quality) and all(
        v['checks'] == 8 and v['passed']/v['checks'] >= .9 for v in finite_groups.values())
    decision = decide(rows, quality_ok)
    documents = {'DECISION.json': decision, 'COMPARISONS.json': comparisons,
                 'QUALITY.json': {'pass': quality_ok, 'cases': quality, 'finite_groups': finite_groups},
                 'TIMING.json': {'episode_s_sum': sum(r['elapsed_s'] for r in rows),
                    'workers': {c['id']: load_json(ROOT / 'episodes' / c['id'] / 'STATUS.json').get('elapsed_s') for c in cases()}}}
    for name, value in documents.items():
        atomic_json(value, ROOT / 'results' / name)
    text = ['# Four-episode DINO residual closure', '', '**Decision: ' + decision['decision'] + '**', '',
        'Four selected patch×state records, TWO independent physical initial states. This is not suite-wide ASR.', '',
        '| Case | Suite | Patch | Whole failure (reused) | Functional failure (reused) | Small residual failure (reused) | Original residual failure (NEW) |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        text.append(f"| {r['case']} | {r['suite']} | {r['patch']} | {r['whole_injection_failure_reused']} | {r['functional_injection_failure_reused']} | {r['small_orthogonal_injection_failure_reused']} | {r['residual_original_amplitude_failure_new']} |")
    text += ['', f"Retained failures: {decision['failed_patch_state_records']}/4 records; {decision['failed_physical_initial_states']}/2 physical states.",
        'Quality pass: ' + str(quality_ok), '',
        'No training or additional episodes started. NEXT_PIXEL_FEASIBILITY only permits a bounded real-PNG feasibility proposal.',
        'All sources and reused receipts/traces are pinned in ../SOURCE_PROVENANCE.json and ../CASES.json.', '']
    path = ROOT / 'results/RESULTS.md'
    path.write_text('\n'.join(text))
    result = {'status': 'COMPLETE', 'signature': sig, 'episodes': 4, **decision, 'episode_assets': assets,
              'assets': {name: sha256(ROOT / 'results' / name) for name in [*documents, 'RESULTS.md']},
              'finished_unix': time.time()}
    atomic_json(result, ROOT / 'results/STATUS.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ['episode_assets','assets']}, indent=2))
    return result


def main():
    argparse.ArgumentParser().parse_args()
    try:
        if analyze()['status'] != 'COMPLETE':
            raise SystemExit(2)
    except Exception as error:
        atomic_json({'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(),
                     'failed_unix': time.time()}, ROOT / 'reports/ANALYZE_FAILED.json')
        raise


if __name__ == '__main__':
    main()
