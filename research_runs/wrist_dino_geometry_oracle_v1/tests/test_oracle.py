"""CPU checks for geometry rules, hook lifetime, paired execution and submission."""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import oracle_common as common
import oracle_camera as camera
import oracle_rollout as rollout
import oracle_report as report
import oracle_cli as cli

class OracleTests(unittest.TestCase):
    def test_original_geometry_rules(self):
        g = common.geometry()
        for target in ([.2, .1, .02], [.005, .001, .1]):
            deltas, metadata = g.paired_deltas(target, [0, 0, 0])
            radius = np.linalg.norm(np.asarray(target)[:2])
            expected = min(.03, 2 * radius * np.sin(np.pi / 12))
            np.testing.assert_allclose(deltas['translation_plus'], [expected, 0, 0], atol=1e-12)
            np.testing.assert_allclose(deltas['translation_minus'], [-expected, 0, 0], atol=1e-12)
            self.assertAlmostEqual(metadata['matched_displacement_m'], expected)

    def test_scene_positions_and_render_function_restored_after_error(self):
        geom = SimpleNamespace(objtype=5, objid=7, pos=np.array([1., 2., 3.]))
        scene = SimpleNamespace(ngeom=1, geoms=[geom])
        def failure(view, current):
            np.testing.assert_allclose(current.geoms[0].pos, [1.03, 2, 3])
            raise ValueError('render failed')
        module = SimpleNamespace(mjr_render=failure)
        with patch.dict(sys.modules, {'mujoco': module}):
            with self.assertRaises(ValueError):
                with common.geometry().SceneTranslation(scene, [7], [.03, 0, 0]):
                    module.mjr_render(None, scene)
        self.assertIs(module.mjr_render, failure)
        np.testing.assert_array_equal(geom.pos, [1, 2, 3])

    def test_camera_callback_restored_on_failure(self):
        original = lambda cache: np.zeros((2, 2, 3), np.uint8)
        sensor = SimpleNamespace(_sensor=original)
        env = SimpleNamespace(env=SimpleNamespace(_observables={'robot0_eye_in_hand_image': sensor}))
        with patch.object(camera, 'fingerprint', return_value={}), patch.object(camera, 'capture', side_effect=ValueError('capture failed')):
            with self.assertRaises(ValueError):
                with camera.Camera(env) as sampler:
                    sampler.arm('test')
                    sensor._sensor({})
        self.assertIs(sensor._sensor, original)

    def test_camera_rejects_stale_or_changed_frame(self):
        original = lambda cache: np.zeros((2, 2, 3), np.uint8)
        sensor = SimpleNamespace(_sensor=original)
        env = SimpleNamespace(env=SimpleNamespace(_observables={'robot0_eye_in_hand_image': sensor}),
                              sim=SimpleNamespace(data=SimpleNamespace(time=1.)))
        with patch.object(camera, 'fingerprint', return_value={}), patch.object(camera, 'capture', return_value={}):
            with camera.Camera(env) as sampler:
                sampler.arm('a')
                pixels = sensor._sensor({})
                self.assertEqual(sampler.take({'robot0_eye_in_hand_image': pixels}, 'a')['key'], 'a')
                with self.assertRaises(RuntimeError): sampler.take({'robot0_eye_in_hand_image': pixels}, 'b')
                with self.assertRaises(RuntimeError): sampler.take({'robot0_eye_in_hand_image': pixels + 1}, 'a')

    def test_hook_installed_after_hard_reset_and_only_last_wait_armed(self):
        events = []
        original = lambda cache: np.zeros((2, 2, 3), np.uint8)
        sensor = SimpleNamespace(_sensor=original)
        class Env:
            def __init__(self):
                self.env = SimpleNamespace(_observables={'robot0_eye_in_hand_image': sensor})
                self.sim = SimpleNamespace(data=SimpleNamespace(time=0.))
            def seed(self, seed): events.append(('seed', seed))
            def set_init_state(self, initial): events.append('set_init'); return {}
            def get_sim_state(self): return np.array([1., 2.])
            def step(self, action):
                self.sim.data.time += 1
                return {'robot0_eye_in_hand_image': sensor._sensor({})}, 0, False, {}
        env = Env()
        replacement = lambda cache: np.full((2, 2, 3), 2, np.uint8)
        def reset(env):
            events.append('reset')
            sensor._sensor = replacement
        runtime = SimpleNamespace(bind_environment_context=lambda env: events.append('bind'), reset_environment=reset,
            renderer_metadata=lambda env: None, validate_camera_frames=lambda raw: None)
        with patch.object(camera, 'fingerprint', return_value={}), patch.object(camera, 'capture', return_value={}):
            with rollout.reset_camera(env, np.array([1., 2.]), 17, runtime, {'num_steps_wait': 10}, lambda: [0]*7) as (raw, sampler):
                self.assertEqual(sampler.captures, 1)
                sampler.take(raw, 'decision_0')
        self.assertIs(sensor._sensor, replacement)
        self.assertEqual(events, [('seed', 17), 'bind', 'reset', 'set_init'])

    def test_no_visible_target_kept(self):
        image = np.zeros((3, 3, 3), np.uint8)
        g = SimpleNamespace(task_candidates=lambda env: ([{'name': 'target'}], []),
                            render=lambda env, entity, delta: (image, np.zeros((3, 3), bool)))
        with patch.object(camera, 'geometry', return_value=g):
            result = camera.capture(None, image)
        self.assertTrue(result['no_target'])
        self.assertEqual(result['reason'], 'no_visible_task_relevant_entity')
        np.testing.assert_array_equal(result['images']['plus'], image)

    def test_numeric_pairing_diagnostics(self):
        a = {'sim_state': np.array([1., 2.]), 'wrist': np.zeros((2, 2, 3), np.uint8)}
        b = {'sim_state': np.array([1., 2. + 1e-9]), 'wrist': a['wrist'].copy()}
        b['wrist'][0, 0, 0] = 1
        diff = rollout.compare_start(a, b)
        self.assertFalse(diff['sim_state']['equal'])
        self.assertEqual(diff['wrist']['different_elements'], 1)
        self.assertEqual(diff['wrist']['max_abs_difference'], 1)

    def frame(self):
        return pd.DataFrame([{'task_id': t, 'state_index': s, 'condition': c, 'success': 1,
                              'available_target_decisions': 8, 'decision_blocks': 10,
                              'intervened_executed_slots': 64 if c != 'clean' else 0, 'executed_steps': 80}
            for t in common.config()['tasks'] for s in [44, 45] for c in common.config()['conditions']])

    def test_paired_scores_and_coverage(self):
        f = self.frame()
        f.loc[(f.task_id == 0) & (f.state_index == 44) & (f.condition == 'clean'), 'success'] = 0
        f.loc[(f.task_id == 1) & (f.state_index == 44) & (f.condition == 'dino_plus'), 'success'] = 0
        with patch.object(report, 'config', return_value={**common.config(), 'bootstrap_draws': 10}):
            pairs, scores = report.summarize(f)
        plus = scores.set_index('condition').loc['dino_plus']
        self.assertEqual(plus.eligible_clean_starts, 9)
        self.assertEqual(plus.failure_switches, 1)
        self.assertAlmostEqual(plus.pooled_CASR, 1/9)
        self.assertEqual(plus.available_target_decisions, 80)
        self.assertAlmostEqual(plus.intervention_step_fraction, .8)

    def test_incomplete_and_duplicate_sets_rejected(self):
        f = self.frame()
        with self.assertRaises(RuntimeError): report.summarize(f.iloc[:-1])
        with self.assertRaises(RuntimeError): report.summarize(pd.concat([f.iloc[:-1], f.iloc[:1]]))

    def test_zero_clean_successes_undefined(self):
        f = self.frame()
        f.loc[f.condition == 'clean', 'success'] = 0
        with patch.object(report, 'config', return_value={**common.config(), 'bootstrap_draws': 10}):
            _, scores = report.summarize(f)
        self.assertTrue(scores.pooled_CASR.isna().all())

    def test_one_gpu_submission_and_no_training(self):
        commands = cli.commands(True)
        self.assertEqual([stage for stage, _ in commands], ['rollout', 'report'])
        self.assertFalse(any('--array' in arg for _, command in commands for arg in command))
        self.assertIn('--dependency=afterany:<ROLLOUT_JOB_ID>', commands[1][1])
        self.assertNotIn('train', ' '.join(arg for _, command in commands for arg in command))
        self.assertEqual([stage for stage, _ in cli.commands(False)], ['report'])
        text = (common.ROOT/'slurm/rollout.sbatch').read_text()
        for option in ('--partition=rtx4090', '--gres=gpu:4090:1', '--mem=12G', '--cpus-per-task=4'):
            self.assertIn(option, text)
        self.assertNotIn('--gres', (common.ROOT/'slurm/report.sbatch').read_text())

    def test_worker_refuses_local_gpu_execution(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError): cli.worker('rollout')

    def test_input_changes_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input'
            path.write_text('old')
            manifest = {'signature': common.signature(), 'inputs': {str(path): common.sha(path)}}
            common.verify_inputs(manifest)
            path.write_text('new')
            with self.assertRaises(RuntimeError): common.verify_inputs(manifest)

    def test_execution_uses_intervened_actions_and_truncates_last_chunk(self):
        self.fake_episode(10, None, False, [8, 2])

    def test_no_target_falls_back_to_native_without_dropping(self):
        self.fake_episode(10, None, True, [8, 2])

    def test_success_stops_mid_chunk(self):
        self.fake_episode(16, 3, False, [3])

    def fake_episode(self, limit, success_at, no_target, slots):
        class Env:
            def __init__(self): self.actions = []
            def raw(self): return {'robot0_eef_pos': np.array([len(self.actions), 0, 0]), 'robot0_gripper_qpos': np.zeros(2)}
            def get_sim_state(self): return np.zeros(3)
            def step(self, action):
                self.actions.append(action)
                return self.raw(), 0, len(self.actions) == success_at, {}
        env = Env()
        obs = {'full_image': np.zeros((224, 224, 3), np.uint8), 'wrist_image': np.zeros((224, 224, 3), np.uint8), 'state': np.zeros(8)}
        runtime = SimpleNamespace(observation=lambda raw, config: obs)
        payload = {'no_target': no_target, 'visible_pixels': {'plus': 10, 'minus': 10}, 'sample_time_s': 0.,
                   'images': {'plus': obs['wrist_image'], 'minus': obs['wrist_image']}}
        sampler = SimpleNamespace(take=lambda raw, key: payload, arm=lambda key: None, captures=1, elapsed_s=.1)
        @contextmanager
        def reset(*args): yield env.raw(), sampler
        class Engine:
            def forward(self, obs, language, replacements=None):
                dino = np.array([[[0.]]], np.float32) if replacements is None else replacements['dino'].reshape(1,1,1)
                return np.full((8,7), float(dino.item()), np.float32), {'dino': dino, 'siglip': np.zeros(1)}
            def spatial_observation(self, obs): return torch.ones((1,1), dtype=torch.float32)
        module = SimpleNamespace(process_libero_action=lambda x: x, get_libero_dummy_action=lambda: [0]*7)
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'VLAAttacker.multiview.openvla_oft_experiments': module}), \
             patch.object(rollout, 'reset_camera', reset), patch.object(rollout, 'config', return_value={**common.config(), 'max_steps': limit}), \
             patch.object(rollout, 'pairing_gate', return_value=Path(folder)/'START.npz'), \
             patch.object(rollout, 'cf_observation', return_value=obs):
            # START location must be relative to ROOT, just like the real call.
            with patch.object(rollout, 'ROOT', Path(folder)):
                result, traces, images = rollout.episode(env, SimpleNamespace(language='test'), np.zeros(3), 17,
                    'dino_plus', 0, 44, Engine(), runtime, SimpleNamespace(update=lambda **kw: None), Path(folder), {'done': True})
            self.assertEqual([r['executed_slots'] for r in traces], slots)
            self.assertEqual(len(env.actions), sum(slots))
            self.assertEqual(result['success'], int(success_at is not None))
            self.assertTrue(all(np.all(np.asarray(a) == (0 if no_target else 1)) for a in env.actions))
            self.assertEqual(result['intervened_executed_slots'], 0 if no_target else sum(slots))
            self.assertTrue(all(np.asarray(r['same_observation_clean_actions']).sum() == 0 for r in traces))

if __name__ == '__main__':
    unittest.main()
