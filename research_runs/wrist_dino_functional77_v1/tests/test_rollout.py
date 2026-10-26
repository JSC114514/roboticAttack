"""Synthetic nonlinear head + real frozen hook lifecycle; no VLA model or renderer."""
import ast
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import functional77_common as common
from functional77_probe import Probe
from functional77_run import episode


def frozen_hook_method():
    # Execute the pinned native hook function itself without importing its heavy
    # model/runtime dependencies. This detects cleanup and call-order regressions.
    source = ast.parse((common.parent_root() / 'code/engine.py').read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'OFTEngine')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'hooks')
    namespace = {'contextmanager': contextmanager, 'require': common.require}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<frozen native hooks>', 'exec'), namespace)
    return namespace['hooks']


def features(obs):
    base = torch.linspace(-.4, .4, 256 * 8).reshape(1, 256, 8)
    base = base + float(obs['state'][0]) * .01
    amount = float(np.asarray(obs['wrist_image']).mean()) / 255
    token = torch.sin(torch.arange(256).float() * .31).reshape(1, 256, 1)
    delta = amount * (token + torch.linspace(.2, .9, 8)[None, None])
    return base + delta, torch.ones_like(base) * amount * .3


def action_head(dino, siglip):
    x = dino.mean(dim=1)
    action = torch.stack([torch.sin(x[:, 0]), torch.tanh(x[:, 1]), x[:, 2].square(),
        x[:, 3] * x[:, 4], torch.cos(x[:, 4]), x[:, 5] * x[:, 6], torch.tanh(x[:, 7])], dim=-1)
    return (action[:, None] + siglip.mean() * .02).expand(1, 8, 7).float()


class FakeEngine:
    hooks = frozen_hook_method()

    def __init__(self):
        self.dino, self.siglip = torch.nn.Identity(), torch.nn.Identity()
        self.dino.num_prefix_tokens = 5
        self.fail = False

    def base(self, record):
        obs = record['observation']
        dino, siglip = features(obs)
        pixels = torch.zeros(1, 12, 224, 224)
        pixels[:, 6:9, 137:187, 26:76] = float(np.asarray(obs['wrist_image']).mean()) / 255
        return {'pixel_values': pixels, 'input_ids': torch.tensor([[1, 2, 3]]),
                'features': dino, 'siglip': siglip}, np.asarray(obs['state']).copy()

    def forward(self, inputs, proprio, replacement=None, leaf=False):
        with self.hooks(replacement, leaf) as captures:
            self.dino(inputs['features'])
            self.siglip(inputs['siglip'])
            dino = self.dino(inputs['features'])
            siglip = self.siglip(inputs['siglip'])
            if self.fail:
                raise RuntimeError('synthetic head failure')
            action = action_head(dino, siglip)
        return action, captures, {}


def unnormalize(values, _key):
    out = np.asarray(values, dtype=np.float32).copy()
    out[:, 6] = (out[:, 6] + 1) / 2
    return out


def wrapper():
    def predict(obs, instruction):
        dino, siglip = features(obs)
        return unnormalize(action_head(dino, siglip).detach().numpy()[0], 'fake')
    return SimpleNamespace(gen_cfg=SimpleNamespace(unnorm_key='fake'),
        vla=SimpleNamespace(_unnormalize_actions=unnormalize), predict_action=predict)


def probe(w):
    result = Probe.__new__(Probe)
    result.wrapper, result.engine = w, FakeEngine()
    def paste(obs, rgb):
        image = obs['wrist_image'].copy()
        image[1:3, 1:3] = int(rgb.mean())
        return {**obs, 'wrist_image': image}
    result.paste = paste
    return result


def processed(action):
    result = np.asarray(action, dtype=np.float32).copy()
    result[-1] = -np.sign(2 * result[-1] - 1)
    return result


class FakeEnv:
    def reset(self, initial):
        self.steps = 0
        self.position = np.asarray(initial, dtype=float).copy()
        return self.raw()

    def raw(self):
        return {'robot0_eef_pos': self.position.copy(), 'robot0_eef_quat': np.array([0., 0., 0., 1.]),
                'robot0_gripper_qpos': np.array([0., 0.])}

    def get_sim_state(self):
        return np.concatenate(([self.steps], self.position))

    def step(self, action):
        self.position += np.asarray(action[:3]) * .01
        self.steps += 1
        return self.raw(), 0., self.steps >= 17, {}


class FakeRuntime:
    @staticmethod
    def reset_wait(env, initial, seed, waits):
        return env.reset(initial)

    @staticmethod
    def observation(raw, runtime):
        return {'full_image': np.zeros((4, 4, 3), dtype=np.uint8),
                'wrist_image': np.ones((4, 4, 3), dtype=np.uint8) * 5,
                'state': np.concatenate([raw['robot0_eef_pos'], np.zeros(5)]).astype(np.float32)}


@pytest.mark.parametrize('direction,component', [('native', 'clean'), ('native', 'png'),
    *[(d, c) for d in ['restore', 'inject'] for c in ['functional', 'random', 'orthogonal']]])
def test_full_trajectory_consumes_actual_actions_refreshes_basis_and_records_end_slots(tmp_path, direction, component):
    w, env = wrapper(), FakeEnv()
    p = probe(w)
    case = common.cases()[0]
    spec = {'case_id': case['id'], 'direction': direction, 'component': component}
    progress = SimpleNamespace(update=lambda *args, **kwargs: None)
    row, assets = episode(env, w, p, SimpleNamespace(language='controlled task'), np.zeros(3),
        np.ones((50, 50, 3), dtype=np.uint8) * 200,
        {'max_steps_by_suite': {case['suite']: 24}}, case, spec, tmp_path / 'trial', progress,
        model=FakeRuntime, process_action=processed)
    from functional77_analyze import read_trace
    traces = read_trace(tmp_path / 'trial' / row['trace'])
    assert row['success'] == 1 and row['executed_steps'] == 17 and row['decision_blocks'] == 3
    assert [t['executed_slots'] for t in traces] == [8, 8, 1]
    assert all(np.array_equal(np.asarray(t['applied_actions']),
               np.asarray(t['predicted_processed_action_chunk'])[:t['executed_slots']]) for t in traces)
    if direction != 'native':
        assert all(t['basis_refreshed'] and t['basis_derived_at_block'] == i for i, t in enumerate(traces))
        assert traces[0]['validation']['status'] == 'NATIVE_AND_IDENTITY_PASS'
        assert len(traces[0]['finite_differences']) == 12
        assert all(t['norm_match_ok'] for t in traces)
        assert traces[0]['actual_change_l2'] > 0
        if component == 'functional':
            assert traces[0]['recipient_action_change_l2'] > 0
        assert row['intervention_blocks'] == row['effective_intervention_blocks'] == 3
        assert len(assets) == 2, 'One block-zero full-vector asset plus trace'
        assert traces[-1]['executed_gripper_flips'] <= 1
    else:
        assert row['intervention_blocks'] == 0
    assert all(p.exists() for p in assets)
    assert not p.engine.dino._forward_hooks and not p.engine.siglip._forward_hooks
    assert not torch.cuda.is_initialized()


def test_real_native_hook_cleanup_on_downstream_exception():
    w = wrapper()
    p = probe(w)
    obs = FakeRuntime.observation({'robot0_eef_pos': np.zeros(3)}, {})
    pair = p.pair(obs, 'test', np.ones((50, 50, 3), dtype=np.uint8) * 200)
    p.engine.fail = True
    with pytest.raises(RuntimeError, match='synthetic head failure'):
        p.intervention(pair, pair['clean_dino'])
    assert not p.engine.dino._forward_hooks and not p.engine.siglip._forward_hooks


def test_failed_attempt_is_preserved_and_restart_begins_from_initial_state(tmp_path):
    w, env = wrapper(), FakeEnv()
    p = probe(w)
    native_predict = w.predict_action
    calls = []
    def fail_second(obs, instruction):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('injected mid-episode failure')
        return native_predict(obs, instruction)
    w.predict_action = fail_second
    case = common.cases()[0]
    spec = {'case_id': case['id'], 'direction': 'native', 'component': 'png'}
    args = (env, w, p, SimpleNamespace(language='test'), np.zeros(3),
            np.ones((50, 50, 3), dtype=np.uint8) * 200,
            {'max_steps_by_suite': {case['suite']: 24}}, case, spec, tmp_path,
            SimpleNamespace(update=lambda *args, **kwargs: None))
    with pytest.raises(RuntimeError, match='mid-episode'):
        episode(*args, model=FakeRuntime, process_action=processed)
    old_trace = next(tmp_path.glob('attempt_*/trace.jsonl'))
    frozen = old_trace.read_bytes()
    assert len(frozen.splitlines()) == 1
    w.predict_action = native_predict
    row, assets = episode(*args, model=FakeRuntime, process_action=processed)
    assert old_trace.read_bytes() == frozen
    assert len(list(tmp_path.glob('attempt_*'))) == 2
    assert row['initial_after_wait_state']['sim_state'] == [0., 0., 0., 0.]
    assert row['executed_steps'] == 17
