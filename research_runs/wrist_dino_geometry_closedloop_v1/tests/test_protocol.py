"""CPU regression checks for paired rollouts, execution and submission limits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import protocol
import rollout
import closed_report
import cli

class ProtocolTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{'task_id': t, 'state_index': s, 'condition': c, 'success': 1}
                             for t in protocol.config()['tasks'] for s in [44, 45]
                             for c in protocol.config()['conditions']])

    def test_pairing_and_random_overlap(self):
        f = self.frame()
        # First clean fails: none of its patch outcomes can count toward CASR.
        f.loc[(f.task_id == 0) & (f.state_index == 44), 'success'] = 0
        f.loc[(f.task_id == 1) & (f.state_index == 44) & f.condition.isin(['initial', 'full']), 'success'] = 0
        f.loc[(f.task_id == 2) & (f.state_index == 45) & (f.condition == 'full'), 'success'] = 0
        f.loc[(f.task_id == 4) & (f.state_index == 45) & (f.condition == 'initial'), 'success'] = 0
        settings = {**protocol.config(), 'bootstrap_draws': 12}
        with patch.object(closed_report, 'config', return_value=settings):
            _, scores = closed_report.paired_scores(f)
        full = scores.set_index('condition').loc['full']
        self.assertEqual(full.clean_successes, 11)
        self.assertEqual(full.failures, 2)
        self.assertEqual(full.failures_beyond_random, 1)
        self.assertEqual(full.random_only_failures, 1)
        self.assertEqual(full.failures_overlap_random, 1)
        self.assertAlmostEqual(full.CASR, 2 / 11)

    def test_missing_or_duplicate_conditions_rejected(self):
        f = self.frame()
        with self.assertRaises(RuntimeError):
            closed_report.paired_scores(f.iloc[:-1])
        with self.assertRaises(RuntimeError):
            closed_report.paired_scores(pd.concat([f.iloc[:-1], f.iloc[:1]]))

    def test_no_clean_successes_is_undefined(self):
        f = self.frame()
        f.loc[f.condition == 'clean', 'success'] = 0
        with patch.object(closed_report, 'config', return_value={**protocol.config(), 'bootstrap_draws': 3}):
            _, scores = closed_report.paired_scores(f)
        self.assertTrue(scores.CASR.isna().all())
        self.assertTrue((scores.clean_successes == 0).all())

    def test_workers_and_seeded_order(self):
        c = protocol.config()
        self.assertEqual(sorted(t for group in c['workers'] for t in group), c['tasks'])
        self.assertEqual(len(set(t for group in c['workers'] for t in group)), 6)
        self.assertEqual(rollout.ordered_conditions(0, 44), rollout.ordered_conditions(0, 44))
        self.assertEqual(set(rollout.ordered_conditions(0, 44)), set(c['conditions']))

    def test_submission_only_three_eval_workers_and_cpu_report(self):
        commands = cli.commands([0, 1, 2])
        self.assertEqual([x[0] for x in commands], ['rollout', 'report'])
        self.assertIn('--array=0,1,2%3', commands[0][1])
        self.assertIn('--dependency=afterany:<ROLLOUT_JOB_ID>', commands[1][1])
        self.assertNotIn('train', ' '.join(' '.join(c) for _, c in commands))
        self.assertEqual([s for s, _ in cli.commands([])], ['report'])
        self.assertIn('--array=2%3', cli.commands([2])[0][1])

    def test_queue_failure_never_submits(self):
        with patch.object(cli, 'prepare'), patch.object(cli, 'queue', side_effect=RuntimeError('unavailable')), \
             patch.object(cli.subprocess, 'run') as submit:
            with self.assertRaises(RuntimeError):
                cli.submit()
            submit.assert_not_called()

    def test_resource_scripts_match_budget(self):
        text = (protocol.ROOT / 'slurm/rollout.sbatch').read_text()
        for option in ('--partition=rtx4090', '--gres=gpu:4090:1', '--mem=12G', '--cpus-per-task=4'):
            self.assertIn(option, text)
        text = (protocol.ROOT / 'slurm/report.sbatch').read_text()
        self.assertIn('--mem=4G', text)
        self.assertIn('--cpus-per-task=2', text)
        self.assertNotIn('--gres', text)

    def test_input_artifact_change_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'input'
            p.write_text('old')
            manifest = {'signature': protocol.signature(), 'inputs': {str(p): protocol.sha(p)}}
            protocol.verify_inputs(manifest)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):
                protocol.verify_inputs(manifest)

    def test_patch_only_modifies_fixed_wrist_region(self):
        obs = {'full_image': np.zeros((224, 224, 3), np.uint8),
               'wrist_image': np.zeros((224, 224, 3), np.uint8), 'state': np.arange(8)}
        out = protocol.native.paste_numpy(obs, np.full((50, 50, 3), 255, np.uint8))
        self.assertEqual(np.count_nonzero(out['wrist_image']), 50 * 50 * 3)
        self.assertTrue((out['wrist_image'][137:187, 26:76] == 255).all())
        self.assertFalse(obs['wrist_image'].any())
        np.testing.assert_array_equal(out['state'], obs['state'])
        np.testing.assert_array_equal(out['full_image'], obs['full_image'])

    def test_reference_never_executed_and_last_chunk_truncated(self):
        self.run_mock_episode(limit=10, finish_step=None, expected_slots=[8, 2])

    def test_success_stops_mid_chunk(self):
        self.run_mock_episode(limit=16, finish_step=3, expected_slots=[3])

    def run_mock_episode(self, limit, finish_step, expected_slots):
        class Env:
            def __init__(self): self.steps = []
            def raw(self):
                return {'robot0_eef_pos': np.array([len(self.steps), 0, 0]),
                        'robot0_gripper_qpos': np.array([0, 0])}
            def get_sim_state(self): return np.array([0., 1.])
            def step(self, action):
                self.steps.append(action)
                return self.raw(), 0, len(self.steps) == finish_step, {}
        obs = {'full_image': np.zeros((224, 224, 3), np.uint8),
               'wrist_image': np.zeros((224, 224, 3), np.uint8), 'state': np.zeros(8)}
        env = Env()
        def reset(e, initial, seed, waits):
            self.assertEqual(seed, 17)
            self.assertEqual(waits, 10)
            self.assertFalse(e.steps)
            return e.raw()
        runtime = SimpleNamespace(reset_wait=reset, observation=lambda raw, cfg: copy.deepcopy(obs))
        class Wrapper:
            def predict_action(self, current, language):
                return np.ones((8, 7), np.float32) if current['wrist_image'].any() else np.zeros((8, 7), np.float32)
        settings = {**protocol.config(), 'max_steps': limit}
        with tempfile.TemporaryDirectory() as folder:
            result, traces, images = rollout.run_episode(env, Wrapper(), SimpleNamespace(language='test'),
                np.zeros(2), 17, np.ones((50, 50, 3), np.uint8), SimpleNamespace(update=lambda **kw: None),
                Path(folder), runtime=runtime, process_action=lambda x: x, settings=settings)
            self.assertEqual([r['executed_slots'] for r in traces], expected_slots)
            self.assertEqual(len(env.steps), sum(expected_slots))
            self.assertTrue(all(np.asarray(action).all() for action in env.steps))
            self.assertTrue(all(np.asarray(r['same_observation_clean_actions']).sum() == 0 for r in traces))
            self.assertEqual(result['success'], int(finish_step is not None))
            self.assertTrue(all(path.exists() for path in images))

if __name__ == '__main__':
    unittest.main()
