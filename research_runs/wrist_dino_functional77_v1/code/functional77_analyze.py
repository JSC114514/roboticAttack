"""CPU-only paired analysis with a predeclared practical target-selection gate."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

from functional77_common import (ROOT, atomic_json, cases, cfg, completed_episode, episode_folder,
    episode_key, load_json, prepared, require, sha256, signature, specs)


def read_trace(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def outcome_table(rows, case_grid):
    """Ten fixed slots; changed fresh controls earn no efficacy credit."""
    controls, indexed = {}, {}
    for row in rows:
        key = (row['case_id'], row['direction'], row['component'])
        require(key not in indexed, 'Duplicate episode row')
        require(row['success'] in [0, 1] and row['failure'] == 1 - row['success'], 'Invalid binary outcome')
        indexed[key] = row
        if row['direction'] == 'native' and row['component'] == 'clean':
            require(row['state_key'] not in controls, 'Duplicated shared clean control')
            controls[row['state_key']] = row
    output = []
    for case in case_grid:
        clean = controls[case['state_key']]
        png = indexed[(case['id'], 'native', 'png')]
        record = {k: case[k] for k in ['id', 'suite', 'task_id', 'state_index', 'patch', 'state_key']}
        record.update(fresh_clean_success=clean['success'], fresh_png_failure=png['failure'],
                      historical_clean_success=1, historical_png_failure=1, historical_whole_restore_success=1,
                      eligible=bool(clean['success'] and png['failure']))
        record['control_change_reason'] = ('none' if record['eligible'] else
            'fresh_clean_failed' if not clean['success'] else 'fresh_png_no_longer_failed')
        for direction in ['restore', 'inject']:
            for component in ['functional', 'random', 'orthogonal']:
                row = indexed[(case['id'], direction, component)]
                record[direction + '_' + component + '_success'] = row['success']
                record[direction + '_' + component + '_failure'] = row['failure']
        output.append(record)
    require(len(rows) == 77 and len(output) == 10 and len(controls) == 7, 'Incomplete factorial/control grid')
    return output


def evaluate_gate(table, quality_ok, gate_config):
    require(len(table) == gate_config['denominator'] == 10, 'Gate requires ten fixed case slots')
    contrasts = []
    for direction in ['restore', 'inject']:
        outcome = 'success' if direction == 'restore' else 'failure'
        for control in gate_config['controls']:
            favorable, unfavorable = [], []
            for row in table:
                if not row['eligible']:
                    continue
                functional = row[direction + '_functional_' + outcome]
                comparison = row[direction + '_' + control + '_' + outcome]
                if functional > comparison:
                    favorable.append(row)
                elif functional < comparison:
                    unfavorable.append(row)
            net = len(favorable) - len(unfavorable)
            suites = sorted({row['suite'] for row in favorable})
            passed = net >= gate_config['minimum_net_favorable_episodes_per_contrast'] and len(suites) >= gate_config['minimum_favorable_suites_per_contrast']
            contrasts.append({'direction': direction, 'control': control,
                'favorable_cases': [r['id'] for r in favorable], 'unfavorable_cases': [r['id'] for r in unfavorable],
                'net_favorable': net, 'fixed_denominator': 10, 'net_fraction': net / 10,
                'favorable_suites': suites, 'pass': passed})
    efficacy_ok = all(row['pass'] for row in contrasts)
    decision = ('STOP_DIAGNOSTIC_QUALITY' if not quality_ok else
                'TRAINING_TARGET_PASS' if efficacy_ok else 'STOP_TARGET_NOT_SUPPORTED')
    return {'decision': decision, 'quality_pass': quality_ok, 'efficacy_pass': efficacy_ok,
            'eligible_cases': sum(row['eligible'] for row in table), 'fixed_case_denominator': 10,
            'contrasts': contrasts, 'automatically_start_training': False}


def summarize_quality(rows, traces_by_key):
    import numpy as np
    clean = {r['state_key']: r for r in rows if r['direction'] == 'native' and r['component'] == 'clean'}
    issues, fd = [], []
    initial_image_drift, rank_zero = [], []
    active_count = parity = matched = 0
    intervention_episodes = 0
    for row in rows:
        key = (row['case_id'], row['direction'], row['component'])
        traces = traces_by_key[key]
        require(len(traces) == row['decision_blocks'] and sum(t['executed_slots'] for t in traces) == row['executed_steps'],
                'Committed trace count differs from receipt')
        require([t['decision_block'] for t in traces] == list(range(len(traces))), 'Noncontiguous decision trace')
        if not traces:
            issues.append({'episode': list(key), 'issue': 'empty_trace'})
            continue
        reference = clean[row['state_key']]
        for field in ['sim_state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
            a = np.asarray(row['initial_after_wait_state'][field], dtype=np.float64)
            b = np.asarray(reference['initial_after_wait_state'][field], dtype=np.float64)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-8):
                issues.append({'episode': list(key), 'issue': 'fresh_initial_physical_state_mismatch', 'field': field})
        if row['initial_observation_hashes'] != reference['initial_observation_hashes']:
            # Physical reset is the quality gate. Renderer pixel variation is exposed,
            # not confused with historical full-trajectory numerical identity.
            initial_image_drift.append(list(key))
        if row['direction'] == 'native':
            require(not any(t['intervention_active'] for t in traces), 'Native control was intervened on')
            continue
        intervention_episodes += 1
        validation = traces[0].get('validation', {})
        if validation.get('status') == 'NATIVE_AND_IDENTITY_PASS':
            parity += 1
        else:
            issues.append({'episode': list(key), 'issue': 'missing_native_identity_parity'})
        if not traces[0].get('finite_differences'):
            issues.append({'episode': list(key), 'issue': 'missing_initial_finite_differences'})
        for trace in traces:
            require(trace['intervention_active'] and trace['basis_refreshed'] and
                    trace['basis_derived_at_block'] == trace['decision_block'], 'Basis was not refreshed at every decision')
            active_count += 1
            matched += int(trace['norm_match_ok'])
            if not trace['norm_match_ok'] or trace['norm_match_relative_error'] > cfg()['norm_match_relative_tolerance']:
                issues.append({'episode': list(key), 'block': trace['decision_block'], 'issue': 'norm_matching_failed'})
            if trace['functional_rank'] == 0:
                rank_zero.append({'episode': list(key), 'block': trace['decision_block']})
            if trace['basis_orthogonality_max_error'] > .002:
                issues.append({'episode': list(key), 'block': trace['decision_block'], 'issue': 'basis_orthogonality_failed'})
            if trace['decision_block'] == 0:
                grouped = {}
                for check in trace['finite_differences']:
                    grouped.setdefault((check['endpoint'], check['group']), []).append(check)
                require(set(grouped) == {(ep, g) for ep in ['clean', 'patched']
                        for g in ['translation', 'rotation', 'gripper']}, 'Endpoint finite-check grid changed')
                for (endpoint, group), values in grouped.items():
                    require({v['relative_step'] for v in values} == set(cfg()['finite_difference_relative_steps']),
                            'Finite-check step grid changed')
                    passed = any(v['nonzero'] and v['sign_agrees'] and v['relative_error'] is not None and
                                 v['relative_error'] <= cfg()['finite_difference_error_limit'] for v in values)
                    fd.append({'episode': list(key), 'endpoint': endpoint, 'group': group,
                               'pass': passed, 'steps': values})
    finite_groups = {}
    for group in ['translation', 'rotation', 'gripper']:
        points = [v for v in fd if v['group'] == group]
        fraction = sum(v['pass'] for v in points) / len(points) if points else 0.
        finite_groups[group] = {'endpoint_checks': len(points), 'passed': sum(v['pass'] for v in points),
                                'fraction': fraction, 'pass': fraction >= cfg()['finite_difference_min_best_of_two_fraction_per_group']}
    require(intervention_episodes == 60, 'Incomplete intervention cohort')
    return {'pass': not issues and all(v['pass'] for v in finite_groups.values()) and parity == 60 and matched == active_count,
            'issues': issues, 'native_identity_episodes_passed': parity, 'intervention_blocks': active_count,
            'norm_match_blocks_passed': matched, 'finite_difference_groups': finite_groups,
            'failed_finite_checks': [v for v in fd if not v['pass']],
            'zero_rank_blocks': rank_zero, 'fresh_initial_observation_hash_drift': initial_image_drift,
            'historical_bitwise_trajectory_match_required': False}


def historical_control_audit(new_row, new_traces, reference):
    import numpy as np
    receipt = load_json(reference['receipt'])
    old_row = receipt['row']
    old_traces = read_trace(Path(reference['receipt']).parent / old_row['trace'])
    first, maximum, applied_max, gripper_flips = None, 0., 0., 0
    for new, old in zip(new_traces, old_traces):
        error = float(np.max(np.abs(np.asarray(new['predicted_environment_action_chunk']) - np.asarray(old['predicted_environment_action_chunk']))))
        maximum = max(maximum, error)
        if first is None and error > 0:
            first = new['decision_block']
        a, b = np.asarray(new['applied_actions']), np.asarray(old['applied_actions'])
        length = min(len(a), len(b))
        if length:
            applied_max = max(applied_max, float(np.max(np.abs(a[:length] - b[:length]))))
            gripper_flips += int(np.count_nonzero(a[:length, -1] != b[:length, -1]))
    return {'old_success': old_row['success'], 'fresh_success': new_row['success'],
            'old_steps': old_row['executed_steps'], 'fresh_steps': new_row['executed_steps'],
            'old_blocks': old_row['decision_blocks'], 'fresh_blocks': new_row['decision_blocks'],
            'aligned_predicted_action_max_error': maximum, 'aligned_applied_action_max_error': applied_max,
            'aligned_executed_gripper_flips': gripper_flips, 'first_action_difference_block': first,
            'tail_not_aligned': len(new_traces) != len(old_traces), 'blocking': False}


def monitoring_summary(rows, traces_by_key):
    output = []
    for direction in ['restore', 'inject']:
        for component in ['functional', 'random', 'orthogonal']:
            selected = [r for r in rows if r['direction'] == direction and r['component'] == component]
            points = [t for r in selected for t in traces_by_key[(r['case_id'], direction, component)]]
            stats = {'direction': direction, 'component': component, 'episodes': len(selected), 'blocks': len(points),
                     'successes': sum(r['success'] for r in selected), 'failures': sum(r['failure'] for r in selected),
                     'executed_gripper_flips': sum(p['executed_gripper_flips'] for p in points),
                     'episode_elapsed_s': sum(r['elapsed_s'] for r in selected)}
            for name in ['functional_rank', 'sensitive_fraction_of_delta', 'actual_change_fraction_of_delta',
                         'norm_match_relative_error', 'recipient_action_change_l2',
                         'actual_change_outside_functional_l2', 'executed_recipient_translation_change_l2',
                         'executed_recipient_rotation_change_l2', 'endpoint_vjp_elapsed_s',
                         'translation_endpoint_vjp_cosine', 'rotation_endpoint_vjp_cosine', 'gripper_endpoint_vjp_cosine']:
                values = [p[name] for p in points if p.get(name) is not None]
                stats[name + '_median'] = statistics.median(values) if values else None
            output.append(stats)
    return output


def report_markdown(table, gate, quality, monitoring):
    lines = ['# Functional DINO target validation', '', '**Decision: ' + gate['decision'] + '**', '',
             '77 fresh episodes: 60 interventions, 7 shared clean controls, 10 actual-PNG controls. No training submitted.', '',
             'In restore columns 1 means success; in injection columns 1 means failure. F/R/O = functional/random/orthogonal.', '',
             '| Case | Suite | Task/state | PNG | Clean success | PNG failure | Restore F/R/O | Inject F/R/O | Eligible |',
             '|---|---|---|---|---:|---:|---|---|---|']
    for row in table:
        restore = '/'.join(str(row['restore_' + c + '_success']) for c in cfg()['components'])
        inject = '/'.join(str(row['inject_' + c + '_failure']) for c in cfg()['components'])
        lines.append(f"| {row['id']} | {row['suite']} | {row['task_id']}/{row['state_index']} | {row['patch']} | {row['fresh_clean_success']} | {row['fresh_png_failure']} | {restore} | {inject} | {row['eligible']} |")
    lines += ['', 'Shared clean states are shown per case in this table but executed only seven times.', '',
              '| Direction | Matched control | Favorable | Unfavorable | Net / fixed 10 | Favorable suites | Pass |',
              '|---|---|---:|---:|---|---|---|']
    for r in gate['contrasts']:
        lines.append(f"| {r['direction']} | {r['control']} | {len(r['favorable_cases'])} | {len(r['unfavorable_cases'])} | {r['net_favorable']}/10 | {', '.join(r['favorable_suites'])} | {r['pass']} |")
    lines += ['', 'Quality pass: ' + str(quality['pass']) + '. Detailed checks: QUALITY.json.',
              'Changed fresh controls: ' + ', '.join(r['id'] + ':' + r['control_change_reason'] for r in table if not r['eligible']), '',
              'The decision selects or rejects a functional training target. It is not a deployable patch comparison against UADA.',
              'Raw receipts/traces: ../episodes/c00…c09/. Input cases and old receipt paths: ../CASES.json.',
              'Historical drift audit: HISTORICAL_AUDIT.json. Internal monitoring: MONITORING.json.', '']
    return '\n'.join(lines)


def analyze():
    prepared()
    sig = signature()
    rows, traces, assets, missing = [], {}, {}, []
    for case in cases():
        for spec in specs(case):
            folder = episode_folder(case, spec)
            receipt = completed_episode(folder, sig)
            if receipt is None:
                missing.append(case['id'] + '/' + episode_key(spec))
                continue
            row = receipt['row']
            require(all(row[k] == spec[k] for k in spec), 'Receipt does not match planned condition')
            require(row['suite'] == case['suite'] and row['task_id'] == case['task_id'] and
                    row['state_index'] == case['state_index'] and row['rollout_seed'] == case['rollout_seed'], 'Receipt pairing mismatch')
            rows.append(row)
            key = (case['id'], spec['direction'], spec['component'])
            traces[key] = read_trace(folder / row['trace'])
            assets[str((folder / 'COMPLETE.json').relative_to(ROOT))] = sha256(folder / 'COMPLETE.json')
            for name, h in receipt['assets'].items():
                assets[str((folder / name).relative_to(ROOT))] = h
    if missing:
        result = {'status': 'INCOMPLETE', 'signature': sig, 'episodes': len(rows), 'missing': missing,
                  'decision': 'INCOMPLETE', 'training_started': False}
        atomic_json(result, ROOT / 'results/STATUS.json')
        print(json.dumps(result, indent=2))
        return result
    table = outcome_table(rows, cases())
    quality = summarize_quality(rows, traces)
    gate = evaluate_gate(table, quality['pass'], cfg()['gate'])
    monitoring = monitoring_summary(rows, traces)
    clean = {r['state_key']: r for r in rows if r['component'] == 'clean'}
    indexed = {(r['case_id'], r['direction'], r['component']): r for r in rows}
    audit = []
    for case in cases():
        for name, row in [('clean', clean[case['state_key']]), ('png', indexed[(case['id'], 'native', 'png')])]:
            key = (row['case_id'], row['direction'], row['component'])
            audit.append({'case': case['id'], 'control': name,
                          **historical_control_audit(row, traces[key], case['history'][name])})
    files = {'OUTCOMES.json': table, 'QUALITY.json': quality, 'DECISION.json': gate,
             'MONITORING.json': monitoring, 'HISTORICAL_AUDIT.json': audit,
             'TIMING.json': {'episode_elapsed_s_sum': sum(r['elapsed_s'] for r in rows),
                            'worker_elapsed_s': {case['id']: load_json(ROOT / 'episodes' / case['id'] / 'STATUS.json').get('elapsed_s')
                                                 for case in cases()}, 'queue_time_included': False}}
    for name, value in files.items():
        atomic_json(value, ROOT / 'results' / name)
    path = ROOT / 'results/RESULTS.md'
    path.write_text(report_markdown(table, gate, quality, monitoring))
    result_paths = [*files, 'RESULTS.md']
    result = {'status': 'COMPLETE', 'signature': sig, 'episodes': 77, 'decision': gate['decision'],
              'quality_pass': quality['pass'], 'eligible_cases': gate['eligible_cases'],
              'training_started': False, 'finished_unix': time.time(), 'episode_assets': assets,
              'assets': {name: sha256(ROOT / 'results' / name) for name in result_paths}}
    atomic_json(result, ROOT / 'results/STATUS.json')
    print(json.dumps({k: v for k, v in result.items() if k not in ['assets', 'episode_assets']}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    try:
        result = analyze()
        if result['status'] != 'COMPLETE':
            raise SystemExit(2)
    except Exception as error:
        atomic_json({'status': 'FAILED', 'error': repr(error), 'traceback': traceback.format_exc(),
                     'failed_unix': time.time()}, ROOT / 'reports/ANALYZE_FAILED.json')
        raise


if __name__ == '__main__':
    main()
