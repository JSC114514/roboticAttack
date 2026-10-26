import importlib.util
import json
from types import SimpleNamespace
import pytest
import torch
import numpy as np
import residual4_common as common
import residual4_math as math
import residual4_analyze as analysis
from residual4_run import episode


def test_original_residual_is_twelve_not_the_old_matched_five():
    clean = torch.zeros(1, 1, 4)
    patched = torch.tensor([[[3., 4., 12., 0.]]])
    rows = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    replacement, stats = math.component_replacement(clean, patched, rows, 'residual_full', 'inject', 7)
    assert torch.equal(replacement, torch.tensor([[[0., 0., 12., 0.]]]))
    old, _ = common.frozen_module('functional77_math').component_replacement(clean, patched, rows, 'orthogonal', 'inject', 7)
    assert math.norm(old) == 5 and stats['actual_change_l2'] == 12
    assert stats['norm_gain'] == 1 and stats['target_change_l2'] == 12
    assert stats['actual_change_functional_projection_l2'] == 0


def test_bf16_actual_consumed_feature_equals_unscaled_original_residual():
    gen = torch.Generator().manual_seed(47)
    clean = torch.randn(1, 256, 1024, generator=gen).bfloat16()
    patched = (clean.float() + .5 * torch.randn(clean.shape, generator=gen)).bfloat16()
    rows = common.frozen_module('functional77_math').random_basis(6, clean.numel(), 12, 'cpu')
    replacement, info = math.component_replacement(clean, patched, rows, 'residual_full', 'inject', 0)
    delta = patched.float() - clean.float()
    # Preserve the registered FP32 operation order D_clean + (delta - F):
    # reassociating it as (D_clean + delta) - F can cross BF16 rounding ties.
    expected = (clean.float() + (delta - math.project(delta, rows))).bfloat16()
    assert torch.equal(replacement, expected)
    assert .99 < info['actual_change_fraction_of_delta'] < 1.01
    assert info['norm_match_ok'] and info['norm_gain'] == 1
    assert not torch.cuda.is_initialized()


def test_zero_rank_is_full_shift_and_zero_residual_is_identity():
    clean, patched = torch.zeros(1, 1, 4), torch.tensor([[[1., 0., 0., 0.]]])
    full, _ = math.component_replacement(clean, patched, torch.empty(0, 4), 'residual_full', 'inject', 0)
    zero, info = math.component_replacement(clean, patched, torch.tensor([[1., 0., 0., 0.]]), 'residual_full', 'inject', 0)
    assert torch.equal(full, patched) and torch.equal(zero, clean)
    assert info['control_degenerate'] and info['actual_change_l2'] == 0


@pytest.mark.parametrize('component,direction', [('orthogonal', 'inject'), ('residual_full', 'restore')])
def test_other_conditions_are_rejected(component, direction):
    with pytest.raises(RuntimeError, match='Only one new condition'):
        math.component_replacement(torch.zeros(4), torch.ones(4), torch.empty(0, 4), component, direction, 0)


def test_four_records_are_only_two_independent_initial_states():
    assert common.budget()['total_new'] == 4
    assert len({c['state_key'] for c in common.cases()}) == 2
    assert sum(c['suite'] == 'libero_10' for c in common.cases()) == 3
    assert all(len(common.specs(c)) == 1 for c in common.cases())


@pytest.mark.parametrize('failures,expected', [([0,0,0,0], 'STOP_SINGLE_COMPONENT_ROUTE'),
    ([1,1,1,0], 'STOP_TWO_STATE_REQUIREMENT'), ([0,0,0,1], 'STOP_TWO_STATE_REQUIREMENT'),
    ([1,0,0,1], 'NEXT_PIXEL_FEASIBILITY'), ([1,1,1,1], 'NEXT_PIXEL_FEASIBILITY')])
def test_frozen_gate_counts_physical_states_not_three_correlated_long_records(failures, expected):
    rows = [{**c, 'case_id':c['id'], 'failure':f, 'success':1-f} for c,f in zip(common.cases(),failures)]
    result = analysis.decide(rows, True)
    assert result['decision'] == expected
    assert not result['automatic_training'] and not result['automatic_additional_episodes']
    assert not result['pixel_reachability_tested']
    assert analysis.decide(rows, False)['decision'] == 'STOP_DIAGNOSTIC_QUALITY'


def test_duplicate_or_missing_case_cannot_produce_decision():
    rows = [{**c, 'case_id':c['id'], 'failure':1, 'success':0} for c in common.cases()]
    with pytest.raises(RuntimeError, match='Four-case'):
        analysis.decide([*rows[:3], rows[0]], True)
    with pytest.raises(RuntimeError, match='Four-case'):
        analysis.decide(rows[:3], True)


def synthetic_helpers():
    path = common.FROZEN_ROOT / 'tests/test_rollout.py'
    spec = importlib.util.spec_from_file_location('frozen77_synthetic_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_rollout_refreshes_original_residual_logs_actual_actions_and_restarts(tmp_path):
    fake = synthetic_helpers()
    wrapper, env = fake.wrapper(), fake.FakeEnv()
    probe = fake.probe(wrapper)
    case = common.cases()[0]
    spec = common.specs(case)[0]
    progress = SimpleNamespace(update=lambda *args, **kwargs: None)
    args = (env, wrapper, probe, SimpleNamespace(language='controlled task'), np.zeros(3),
        np.ones((50,50,3),dtype=np.uint8)*200, {'max_steps_by_suite':{case['suite']:24}}, case, spec, tmp_path, progress)
    first, assets = episode(*args, model=fake.FakeRuntime, process_action=fake.processed)
    frozen_bytes = (tmp_path / first['trace']).read_bytes()
    traces = [json.loads(s) for s in frozen_bytes.splitlines()]
    assert first['executed_steps'] == 17 and [t['executed_slots'] for t in traces] == [8,8,1]
    assert all(t['norm_gain'] == 1 and t['basis_derived_at_block'] == i for i,t in enumerate(traces))
    assert all(t['amplitude_policy'] == 'unscaled_original_complement' for t in traces)
    assert all(t['policy_rgb_changed_pixels'] == 0 for t in traces)
    assert all(np.array_equal(t['applied_actions'],t['predicted_processed_action_chunk'][:t['executed_slots']]) for t in traces)
    assert not probe.engine.dino._forward_hooks and not probe.engine.siglip._forward_hooks
    reference = tmp_path / 'reference.json'
    common.atomic_json({'row': first}, reference)
    quality_case = {**case, 'references': {'functional_injection': {
        'receipt': str(reference), 'trace': str(tmp_path / first['trace'])}}}
    quality = analysis.quality_check(quality_case, first, traces)
    assert not quality['issues'] and quality['block_zero_clean_action_max_error'] == 0
    invalid = json.loads(json.dumps(traces))
    invalid[0]['norm_gain'] = .03
    with pytest.raises(RuntimeError, match='rescaled'):
        analysis.quality_check(quality_case, first, invalid)
    second, _ = episode(*args, model=fake.FakeRuntime, process_action=fake.processed)
    assert second['initial_after_wait_state'] == first['initial_after_wait_state']
    assert len(list(tmp_path.glob('attempt_*'))) == 2 and (tmp_path / first['trace']).read_bytes() == frozen_bytes
    assert not torch.cuda.is_initialized()


def test_incomplete_analysis_never_evaluates_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis, 'prepared', lambda: {})
    monkeypatch.setattr(analysis, 'signature', lambda: 'synthetic')
    monkeypatch.setattr(analysis, 'episode_folder', lambda c,s:tmp_path/c['id'])
    monkeypatch.setattr(analysis, 'decide', lambda *args:pytest.fail('Incomplete efficacy evaluation'))
    result = analysis.analyze()
    assert result['status'] == 'INCOMPLETE' and len(result['missing']) == 4
