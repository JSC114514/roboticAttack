import copy
from pathlib import Path

import pytest

import functional77_common as common
import functional77_analyze as analysis


def rows_fixture():
    rows = []
    for case in common.cases():
        for spec in common.specs(case):
            success = (1 if spec['component'] == 'clean' else 0) if spec['direction'] == 'native' else (
                int(spec['component'] == 'functional') if spec['direction'] == 'restore' else int(spec['component'] != 'functional'))
            rows.append({**spec, **{k: case[k] for k in ['suite', 'task_id', 'state_index', 'state_key', 'rollout_seed']},
                         'success': success, 'failure': 1 - success, 'decision_blocks': 1, 'executed_steps': 8,
                         'initial_after_wait_state': {'sim_state': [0., 1., 2.], 'robot0_eef_pos': [0., 0., 0.],
                             'robot0_eef_quat': [0., 0., 0., 1.], 'robot0_gripper_qpos': [0., 0.]},
                         'initial_observation_hashes': {'full_image': 'f', 'wrist_image': 'w', 'state': 's'}})
    return rows


def traces_fixture(rows):
    checks = [{'endpoint': ep, 'group': group, 'relative_step': step, 'nonzero': True,
               'sign_agrees': True, 'relative_error': .1} for ep in ['clean', 'patched']
              for group in ['translation', 'rotation', 'gripper'] for step in [.002, .01]]
    return {(r['case_id'], r['direction'], r['component']): [{'decision_block': 0, 'executed_slots': 8,
        'intervention_active': r['direction'] != 'native', 'basis_refreshed': True, 'basis_derived_at_block': 0,
        'validation': {'status': 'NATIVE_AND_IDENTITY_PASS'}, 'finite_differences': copy.deepcopy(checks),
        'norm_match_ok': True, 'norm_match_relative_error': .001, 'functional_rank': 6,
        'basis_orthogonality_max_error': 1e-6}] for r in rows}


def test_budget_unique_controls_case_pairing_and_four_suite_coverage():
    rows = rows_fixture()
    assert len(rows) == common.budget()['total_new'] == 77
    assert len({(r['case_id'], r['direction'], r['component']) for r in rows}) == 77
    assert sum(r['component'] == 'clean' for r in rows) == 7
    assert sum(r['component'] == 'png' for r in rows) == 10
    assert {r['suite'] for r in rows} == set(common.cfg()['suites'])
    assert [common.episode_key(s) for s in common.specs(common.cases()[0])] == [common.episode_key(s) for s in common.specs(common.cases()[0])]


def test_gate_uses_all_four_exact_component_contrasts():
    table = analysis.outcome_table(rows_fixture(), common.cases())
    result = analysis.evaluate_gate(table, True, common.cfg()['gate'])
    assert result['decision'] == 'TRAINING_TARGET_PASS'
    assert {(c['direction'], c['control']) for c in result['contrasts']} == {
        (d, c) for d in ['restore', 'inject'] for c in ['random', 'orthogonal']}
    assert all(c['net_favorable'] == 10 for c in result['contrasts'])
    assert not result['automatically_start_training']


@pytest.mark.parametrize('direction,control', [(d, c) for d in ['restore', 'inject'] for c in ['random', 'orthogonal']])
def test_any_failed_comparison_stops_candidate(direction, control):
    table = analysis.outcome_table(rows_fixture(), common.cases())
    outcome = 'success' if direction == 'restore' else 'failure'
    for row in table:
        row[direction + '_' + control + '_' + outcome] = row[direction + '_functional_' + outcome]
    result = analysis.evaluate_gate(table, True, common.cfg()['gate'])
    assert result['decision'] == 'STOP_TARGET_NOT_SUPPORTED'


def test_net_discordances_not_just_count_of_positive_cases():
    table = analysis.outcome_table(rows_fixture(), common.cases())
    for i, row in enumerate(table):
        row['restore_functional_success'] = int(i < 4)
        row['restore_random_success'] = int(4 <= i < 6)
    result = analysis.evaluate_gate(table, True, common.cfg()['gate'])
    selected = next(r for r in result['contrasts'] if r['direction'] == 'restore' and r['control'] == 'random')
    assert len(selected['favorable_cases']) == 4 and len(selected['unfavorable_cases']) == 2
    assert selected['net_favorable'] == 2 and not selected['pass']


def test_single_suite_positive_cases_do_not_pass():
    table = analysis.outcome_table(rows_fixture(), common.cases())
    for row in table:
        if row['suite'] != 'libero_10':
            row['restore_random_success'] = row['restore_functional_success']
    result = analysis.evaluate_gate(table, True, common.cfg()['gate'])
    selected = next(r for r in result['contrasts'] if r['direction'] == 'restore' and r['control'] == 'random')
    assert selected['net_favorable'] == 4 and not selected['pass']


def test_changed_clean_or_png_stays_in_denominator_and_gets_no_credit():
    rows = rows_fixture()
    for row in rows:
        if row['case_id'] == 'c00' and row['component'] == 'clean':
            row['success'], row['failure'] = 0, 1
        if row['case_id'] == 'c01' and row['component'] == 'png':
            row['success'], row['failure'] = 1, 0
    table = analysis.outcome_table(rows, common.cases())
    assert table[0]['control_change_reason'] == 'fresh_clean_failed'
    assert table[1]['control_change_reason'] == 'fresh_png_no_longer_failed'
    assert not table[2]['eligible'], 'c02 shares c00 clean control'
    result = analysis.evaluate_gate(table, True, common.cfg()['gate'])
    assert result['eligible_cases'] == 7
    assert all(c['fixed_denominator'] == 10 and c['net_favorable'] == 7 for c in result['contrasts'])


def test_quality_cannot_be_overridden_by_good_outcomes():
    table = analysis.outcome_table(rows_fixture(), common.cases())
    assert analysis.evaluate_gate(table, False, common.cfg()['gate'])['decision'] == 'STOP_DIAGNOSTIC_QUALITY'


def test_quality_reads_all_sixty_gradients_best_of_two_steps_and_current_physical_reset():
    rows = rows_fixture()
    traces = traces_fixture(rows)
    # Deliberately fail the small BF16 step; larger actual cast-aware step passes.
    for value in traces.values():
        for check in value[0]['finite_differences']:
            if check['relative_step'] == .002:
                check.update(nonzero=False, sign_agrees=False, relative_error=1.)
    result = analysis.summarize_quality(rows, traces)
    assert result['pass'] and result['native_identity_episodes_passed'] == 60
    assert all(c['endpoint_checks'] == c['passed'] == 120 for c in result['finite_difference_groups'].values())
    rows[-1]['initial_after_wait_state']['robot0_eef_pos'][0] = .01
    assert not analysis.summarize_quality(rows, traces)['pass']


def test_missing_gradient_check_grid_or_frozen_basis_is_rejected():
    rows = rows_fixture()
    traces = traces_fixture(rows)
    point = next(v[0] for key, v in traces.items() if key[1] != 'native')
    point['basis_derived_at_block'] = -1
    with pytest.raises(RuntimeError, match='refreshed'):
        analysis.summarize_quality(rows, traces)


def test_duplicates_cannot_inflate_effect_count():
    rows = rows_fixture()
    with pytest.raises(RuntimeError, match='Duplicate'):
        analysis.outcome_table([*rows, rows[0]], common.cases())


def test_incomplete_analysis_does_not_evaluate_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis, 'prepared', lambda: {})
    monkeypatch.setattr(analysis, 'signature', lambda: 'synthetic')
    monkeypatch.setattr(analysis, 'episode_folder', lambda c, s: tmp_path / c['id'] / common.episode_key(s))
    monkeypatch.setattr(analysis, 'evaluate_gate', lambda *args: pytest.fail('Incomplete run evaluated efficacy'))
    result = analysis.analyze()
    assert result['status'] == result['decision'] == 'INCOMPLETE'
    assert result['episodes'] == 0 and len(result['missing']) == 77
