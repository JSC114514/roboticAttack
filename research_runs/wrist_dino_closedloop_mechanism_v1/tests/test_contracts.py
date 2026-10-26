"""CPU behavioral checks; these do not stand in for a real GPU/EGL pilot."""
from __future__ import annotations

import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

import pilot_common as common
import pilot_math as math
import pilot_pipeline as pipeline
import pilot_run as runner


def test_grid_keeps_pairs_task_split_and_gpu_budget():
    assert common.budget()['total_new'] == 592
    for stage in ['calibration', 'main']:
        for _, task in common.shards(stage):
            values = common.specs(stage, task)
            assert len({common.key(value) for value in values}) == len(values)
            if stage == 'main':
                assert {v['state_index'] for v in values} == {41}
                for patch in common.cfg()['patches']:
                    kinds = {v['kind'] for v in values if v['patch'] == patch}
                    assert {'functional_early', 'functional_random_early'} <= kinds
    assert not set(common.cfg()['calibration_tasks']) & set(common.cfg()['main_tasks'] + common.cfg()['confirmation_tasks'])


def test_orthogonal_decomposition_preserves_actual_delta_and_rank():
    vectors = [torch.tensor([1., 2., 0., 0.]), torch.tensor([2., 4., 0., 0.]), torch.tensor([0., 0., 1., 0.])]
    rows = math.basis(vectors)
    delta = torch.tensor([2., -1., 3., 4.])
    sensitive = math.project(delta, rows)
    orthogonal = delta - sensitive
    assert len(rows) == 2
    assert torch.allclose(sensitive + orthogonal, delta)
    assert torch.allclose(rows @ orthogonal, torch.zeros(2), atol=1e-5)
    assert math.norm(math.basis([torch.zeros(4)])) == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_control_matches_post_cast_norm_and_is_reproducible(dtype):
    generator = torch.Generator().manual_seed(8)
    base = torch.randn((1, 256, 4), generator=generator).to(dtype)
    direction = torch.randn(base.shape, generator=generator)
    result, report = math.matched_change(base, direction, 2.)
    assert result.dtype == dtype
    assert report['norm_match_ok']
    assert abs(math.norm(result.float() - base.float()) - 2.) <= .06
    assert torch.equal(math.random_basis(3, 1024, 34, 'cpu'), math.random_basis(3, 1024, 34, 'cpu'))
    assert torch.allclose(math.random_basis(3, 1024, 34, 'cpu') @ math.random_basis(3, 1024, 34, 'cpu').T,
                          torch.eye(3), atol=1e-5)


def test_zero_delta_is_explicit_identity_and_zero_control_is_rejected():
    base = torch.ones(8, dtype=torch.bfloat16)
    result, report = math.matched_change(base, torch.zeros(8), 0.)
    assert torch.equal(result, base) and report['control_degenerate']
    with pytest.raises(RuntimeError, match='zero random'):
        math.matched_change(base, torch.zeros(8), 1.)
    assert torch.equal(math.apply_delta(base, torch.randn(8), 0), base)


def test_footprint_uses_consumed_pixels_and_excludes_front_changes():
    clean = torch.zeros((1, 12, 224, 224), dtype=torch.bfloat16)
    patched = clean.clone()
    patched[:, 6:9, 141:151, 31:41] = 1
    support = math.footprint(clean, patched)
    assert support.sum() == 1 and support.reshape(16, 16)[10, 2]
    for group in common.cfg()['spatial_candidates'][1:]:
        assert not (math.spatial_mask(group, support) & support).any()
    assert math.random_mask(int(support.sum()), 123, 'cpu').sum() == support.sum()
    patched[:, 0, 0, 0] = 1
    with pytest.raises(RuntimeError, match='Front'):
        math.footprint(clean, patched)


def hook_engine():
    frozen = common.parent_module('engine')
    result = object.__new__(frozen.OFTEngine)
    result.dino, result.siglip = torch.nn.Identity(), torch.nn.Identity()
    return result


def test_real_hook_targets_second_view_and_preserves_siglip():
    engine = hook_engine()
    front, wrist, donor = torch.zeros(1, 256, 3), torch.ones(1, 256, 3), torch.full((1, 256, 3), 4.)
    with engine.hooks(replacement=donor) as captured:
        actual_front = engine.dino(front)
        engine.siglip(front)
        actual_wrist = engine.dino(wrist)
        actual_siglip = engine.siglip(wrist)
    assert torch.equal(actual_front, front) and torch.equal(actual_wrist, donor)
    assert torch.equal(actual_siglip, wrist) and torch.equal(captured['siglip'], wrist)
    assert len(engine.dino._forward_hooks) == len(engine.siglip._forward_hooks) == 0


def test_real_hook_cleanup_on_exception_and_wrong_call_count():
    engine = hook_engine()
    with pytest.raises(ValueError):
        with engine.hooks():
            raise ValueError('injected failure')
    assert not engine.dino._forward_hooks and not engine.siglip._forward_hooks
    with pytest.raises(RuntimeError, match='Front then Wrist'):
        with engine.hooks():
            engine.dino(torch.zeros(1, 256, 3))
    assert not engine.dino._forward_hooks and not engine.siglip._forward_hooks


def test_endpoint_vjp_matches_finite_differences_without_retained_gradient_accumulation():
    from pilot_probe import Probe
    generator = torch.Generator().manual_seed(17)
    weights = torch.randn((1024, 56), generator=generator) * .03
    class Engine:
        def forward(self, inputs, proprio, replacement=None, leaf=False):
            feature = inputs['feature'] if replacement is None else replacement
            feature = feature.detach().requires_grad_(True) if leaf else feature
            return torch.tanh(feature.flatten() @ weights).reshape(1, 8, 7), {'dino': feature, 'siglip': torch.zeros(1)}, {}
    probe = object.__new__(Probe)
    probe.engine = Engine()
    clean = torch.randn((1, 256, 4), generator=generator) * .1
    patched = clean + torch.randn(clean.shape, generator=generator) * .05
    pair = {'clean_dino': clean, 'patched_dino': patched, 'clean_inputs': {'feature': clean},
            'patched_inputs': {'feature': patched}, 'proprio': np.zeros(8),
            'clean_siglip': torch.zeros(1), 'patched_siglip': torch.zeros(1)}
    pair['clean_action'] = probe.engine.forward(pair['clean_inputs'], None)[0]
    pair['patched_action'] = probe.engine.forward(pair['patched_inputs'], None)[0]
    rows, checks, report, gradients = probe.functional(pair, finite=True)
    assert len(rows['all']) >= 3 and len(gradients) == 6
    assert all(row['relative_error'] < .01 and row['sign_agrees'] for row in checks)
    assert clean.grad is None and patched.grad is None
    replacement, details = probe.replacement(pair, 'functional', 'all', 1., 79, rows=rows['all'], random_control=True)
    own, _ = probe.replacement(pair, 'functional', 'all', 1., 79, rows=rows['all'])
    assert details['norm_match_ok'] and details['control_rank'] == len(rows['all'])
    assert abs(math.norm(replacement - patched) - math.norm(own - patched)) < .03 * math.norm(own - patched)


def test_reproduction_rejects_drift_in_actions_and_trajectory_length():
    row = {'success': 1, 'executed_steps': 8, 'decision_blocks': 1, 'rollout_seed': 6}
    trace = {'decision_block': 0, 'executed_slots': 8, 'step_end': 8,
             'predicted_environment_action_chunk': np.zeros((8, 7)).tolist()}
    assert runner.verify_reproduction(row, [trace], row, [trace])['status'] == 'REPRODUCTION_PASS'
    changed = {**trace, 'predicted_environment_action_chunk': np.ones((8, 7)).tolist()}
    with pytest.raises(RuntimeError, match='drift'):
        runner.verify_reproduction(row, [changed], row, [trace])
    with pytest.raises(RuntimeError, match='executed_steps'):
        runner.verify_reproduction({**row, 'executed_steps': 7}, [trace], row, [trace])


def test_episode_resume_checks_assets_and_selection_before_skipping(tmp_path):
    asset = tmp_path / 'trace.jsonl'
    asset.write_text('{}\n')
    common.atomic_json({'signature': 'sig', 'selection_sha256': 'selection', 'assets': {'trace.jsonl': common.sha256(asset)}}, tmp_path / 'COMPLETE.json')
    assert runner.completed_episode(tmp_path, 'sig', 'selection')
    with pytest.raises(RuntimeError, match='provenance'):
        runner.completed_episode(tmp_path, 'sig', 'other')
    asset.write_text('changed')
    with pytest.raises(RuntimeError, match='assets'):
        runner.completed_episode(tmp_path, 'sig', 'selection')


def test_closed_loop_window_returns_to_native_and_preserves_trace(tmp_path, monkeypatch):
    import sys
    from pilot_probe import Probe
    observation = {'full_image': np.zeros((224, 224, 3), dtype=np.uint8),
                   'wrist_image': np.zeros((224, 224, 3), dtype=np.uint8), 'state': np.zeros(8)}
    raw = {'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': np.array([0., 0., 0., 1.]), 'robot0_gripper_qpos': np.zeros(2)}
    class Environment:
        count = 0
        def get_sim_state(self):
            return np.array([self.count], dtype=float)
        def step(self, action):
            self.count += 1
            return raw, 0., False, {}
    env = Environment()
    def reset(env, initial, seed, waits):
        env.count = 0
        return raw
    model = SimpleNamespace(reset_wait=reset, observation=lambda raw, runtime: observation)
    monkeypatch.setattr(runner, 'parent_module', lambda name: model)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    def processed(action):
        value = np.asarray(action).copy()
        value[-1] = -1. if value[-1] > 0 else 1.
        return value
    monkeypatch.setitem(sys.modules, 'VLAAttacker.multiview.openvla_oft_experiments', SimpleNamespace(process_libero_action=processed))
    class SmallProbe:
        replacement = Probe.replacement
        def paste(self, obs, rgb):
            result = {**obs, 'wrist_image': obs['wrist_image'].copy()}
            result['wrist_image'][137:187, 26:76] = rgb
            return result
        def pair(self, obs, task, rgb):
            support = torch.zeros(256, dtype=torch.bool)
            support[:16] = True
            return {'clean_dino': torch.zeros(1, 256, 3), 'patched_dino': torch.ones(1, 256, 3),
                    'clean_siglip': torch.zeros(1), 'patched_siglip': torch.ones(1), 'support': support,
                    'clean_action': torch.zeros(1, 8, 7), 'patched_action': torch.full((1, 8, 7), .2)}
        def validate(self, pair):
            return {'status': 'CPU_FAKE_INTERFACE'}
        def intervention(self, pair, replacement):
            return replacement.mean().expand(1, 8, 7) * .2
        def environment_actions(self, value):
            return value.numpy()[0].copy()
    wrapper = SimpleNamespace(gen_cfg=SimpleNamespace(task_suite_name='libero_spatial'),
                              predict_action=lambda obs, task: np.full((8, 7), .2, dtype=np.float32))
    spec = {'kind': 'spatial_early', 'patch': common.cfg()['patches'][0], 'state_index': 41, 'cohort': 'main'}
    progress = SimpleNamespace(update=lambda *args, **kwargs: None)
    row, traces, assets = runner.episode(env, wrapper, SmallProbe(), SimpleNamespace(language='test'), None,
        123, np.full((50, 50, 3), 255, dtype=np.uint8), {'max_steps_by_suite': {'libero_spatial': 40}},
        spec, {'spatial': {'group': 'footprint'}}, tmp_path, progress)
    assert row['intervention_blocks'] == row['effective_intervention_blocks'] == 4
    assert len(traces) == 5 and [t['intervention_active'] for t in traces] == [True] * 4 + [False]
    assert traces[0]['predicted_environment_action_chunk'][0][0] < .2
    assert np.isclose(traces[4]['predicted_environment_action_chunk'][0][0], .2)
    assert len(traces[0]['applied_actions']) == 8 and 'clean_processed_action_chunk' in traces[0]
    assert traces[0]['policy_rgb_changed_pixels'] == 2500
    assert all(path.is_file() for path in assets)
    assert len((tmp_path / row['trace']).read_text().splitlines()) == 5


def test_selection_is_frozen_by_matched_effect_and_rejects_bad_derivatives():
    from pilot_analyze import choose
    suite, patch = common.cfg()['suites'][0], common.cfg()['patches'][0]
    records, screens = [], []
    for task in [0, 1]:
        context = {'suite': suite, 'task_id': task, 'state_index': 40, 'patch': patch, 'decision_block': 0}
        for group in common.cfg()['functional_candidates']:
            for control in [False, True]:
                records.append({**context, 'family': 'functional', 'group': group, 'strength': 1.,
                                'random_control': control, 'norm_match_ok': True, 'control_rank': 2,
                                'action_gap_l2': 1., 'action_gap_closing_fraction': .1 if control else (.8 if group == 'rotation' else .3)})
        screens.append({'context': context, 'finite_differences': [
            {'endpoint': endpoint, 'group': group, 'nonzero': True, 'sign_agrees': True, 'relative_error': .01}
            for endpoint in ['clean', 'patched'] for group in ['translation', 'rotation', 'gripper']]})
    frame = pd.DataFrame(records)
    result = choose(frame, screens, 'functional', suite, patch)
    assert result['group'] == 'rotation' and result['positive_screen_evidence']
    for screen in screens:
        for row in screen['finite_differences']:
            row['sign_agrees'] = False
    with pytest.raises(RuntimeError, match='No valid functional'):
        choose(frame, screens, 'functional', suite, patch)


def test_submission_explicit_dependencies_duplicate_guard_and_no_cancellation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'prepare', lambda: {'signature': 'fixed'})
    queued = []
    monkeypatch.setattr(pipeline, 'queue', lambda: queued)
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=str(len(commands)) + '\n', stderr='')
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    pipeline.submit()
    assert len(commands) == 5 and all(command[0] == 'sbatch' for command in commands)
    assert '--dependency=afterok:1' in commands[1]
    assert '--dependency=afterok:4' in commands[4]
    queued.append({'job_id': '2', 'gres': 'gpu:4090:1'})
    with pytest.raises(RuntimeError, match='already has active'):
        pipeline.submit()
    assert len(commands) == 5


def test_submission_timeout_is_not_retried_blindly(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'prepare', lambda: {'signature': 'fixed'})
    monkeypatch.setattr(pipeline, 'queue', lambda: [])
    def timeout(command, **kwargs):
        raise pipeline.subprocess.TimeoutExpired(command, 40)
    monkeypatch.setattr(pipeline.subprocess, 'run', timeout)
    with pytest.raises(RuntimeError, match='uncertain'):
        pipeline.submit()
    assert common.load_json(tmp_path / 'reports/SUBMISSION.json')['status'] == 'SUBMISSION_UNCERTAIN'
    with pytest.raises(RuntimeError, match='may have succeeded'):
        pipeline.submit()


def test_retry_does_not_allocate_gpu_for_verified_complete_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'prepare', lambda: {'signature': 'fixed'})
    monkeypatch.setattr(pipeline, 'queue', lambda: [])
    monkeypatch.setattr(pipeline, 'pending_shards', lambda stage, signature: [])
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=str(len(commands)), stderr='')
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_run)
    pipeline.submit()
    assert len(commands) == 3 and all(not any(item.startswith('--array') for item in command) for command in commands)
    receipt = common.load_json(tmp_path / 'reports/SUBMISSION.json')
    assert {row['stage'] for row in receipt['skipped_stages']} == {'calibration', 'main'}


def test_pipeline_default_only_prepares_and_does_not_submit(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'prepare', lambda: calls.append('prepare'))
    monkeypatch.setattr(pipeline, 'submit', lambda: pytest.fail('Unexpected submission'))
    monkeypatch.setattr(pipeline, 'queue', lambda: pytest.fail('Unexpected scheduler call'))
    monkeypatch.setattr('sys.argv', ['pilot_pipeline.py'])
    pipeline.main()
    assert calls == ['prepare']


def test_report_full_grid_serialization_pairing_and_monitoring(tmp_path, monkeypatch):
    import pilot_analyze as analyze
    monkeypatch.setattr(analyze, 'ROOT', tmp_path)
    monkeypatch.setattr(analyze, 'prepared', lambda: None)
    monkeypatch.setattr(analyze, 'signature', lambda: 'fixed')
    monkeypatch.setattr(analyze, 'parent_root', lambda: tmp_path / 'old')
    common.atomic_json({'status': 'FROZEN', 'signature': 'fixed'}, tmp_path / 'reports/SELECTION.json')
    previous = []
    for suite in common.cfg()['suites']:
        for task in range(10):
            for state in [40, 41]:
                for condition in ['clean', *common.cfg()['patches']]:
                    previous.append({'suite': suite, 'task_id': task, 'state_index': state, 'condition': condition,
                                     'success': int(condition == 'clean')})
    common.atomic_parquet(pd.DataFrame(previous), tmp_path / 'old/results/BATCH500_ROLLOUTS.parquet')
    frames = {}
    for stage in ['calibration', 'main']:
        rows = []
        for suite, task in common.shards(stage):
            for spec in common.specs(stage, task):
                episode_folder = tmp_path / stage / suite / str(task) / common.key(spec)
                episode_folder.mkdir(parents=True)
                active = spec['kind'] != 'native'
                trace = {'decision_block': 0, 'intervention_active': active, 'actual_change_l2': 2. if active else None,
                         'action_change_l2': .5 if active else None, 'action_gap_closing_fraction': .8 if active else None,
                         'before': {'sim_state': [0., 1.]}, 'after': {'sim_state': [1., 2.]},
                         'predicted_environment_action_chunk': np.zeros((8, 7)).tolist()}
                (episode_folder / 'trace.jsonl').write_text(json.dumps(trace) + '\n')
                rows.append({**spec, 'suite': suite, 'task_id': task,
                             'success': int(spec['kind'] == 'functional_early'),
                             'episode_folder': str(episode_folder), 'trace': 'trace.jsonl',
                             'window_not_reached': False, 'effective_intervention_blocks': int(active)})
        frames[stage] = pd.DataFrame(rows)
    monkeypatch.setattr(analyze, 'episodes', lambda stage, *args: (frames[stage], {}))
    analyze.analyze()
    status = common.load_json(tmp_path / 'results/STATUS.json')
    assert status['episodes'] == 592 and status['status'] == 'COMPLETE'
    contrast = pd.read_csv(tmp_path / 'results/MATCHED_CONTROLS.csv')
    functional = contrast[contrast.family == 'functional']
    assert functional.sensitive_minus_random_success_pp.eq(100).all()
    assert set(functional['pairs']) == {3, 5}
    assert len(pd.read_parquet(tmp_path / 'results/PILOT_TRACES.parquet')) == 592


def test_no_cuda_was_initialized_by_cpu_contract_checks():
    assert not torch.cuda.is_initialized()
