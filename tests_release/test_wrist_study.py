"""Publication checks grounded in completed counts, source AST and PNG bytes.

These tests run on CPU and do not import a VLA, allocate CUDA, or submit Slurm.
They verify the exported per-episode table against the original summary fields.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/wrist_dino'
spec = importlib.util.spec_from_file_location('released_wrist_summary', STUDY / 'summarize_results.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class WristStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = summary.load_rows()

    def test_full_counts_reproduce_original_summary(self):
        original = json.loads((STUDY / 'reports/full_SUMMARY.json').read_text())['conditions']
        actual = {r['condition']: r for r in summary.summarize(self.rows, 'main', 'full', 'joint_oft')}
        for source in original:
            result = actual[source['condition']]
            for name in ['completed', 'success', 'clean_success', 'induced_failures']:
                self.assertEqual(result[name], source[name])
            self.assertAlmostEqual(result['macro_casr'], source['suite_task_equal_casr'], places=12)

    def test_confirmation_is_same_eighty_starts_for_each_seed(self):
        data = summary.summarize(self.rows, 'sensitivity', 'holdout', 'joint_oft')
        for seed in [42, 43, 44]:
            result = next(r for r in data if r['condition'] == f'LR004-O03-seed{seed}')
            self.assertEqual((result['completed'], result['success'], result['clean_success']), (80, 0, 71))
            self.assertEqual(result['macro_casr'], 1)

    def test_crosscheck_and_restorations_not_mixed(self):
        selected = [r for r in self.rows if r['study'] == 'crosscheck']
        self.assertEqual(sum(r['native_png'] for r in selected), 2560)
        self.assertEqual(sum(not r['native_png'] for r in selected), 480)
        original = json.loads((STUDY / 'supplements/wrist_dino_crosscheck_v1/reports/TRANSFER_RESULTS.json').read_text())['rows']
        for model in ['vla_adapter_pro', 'pi05']:
            data = summary.summarize(self.rows, 'crosscheck', 'transfer', model)
            self.assertEqual(len(data), 16)
            self.assertTrue(all(r['completed'] == 80 for r in data))
            # CSV data contain exactly the same per-suite successes as the report.
            for expected in [r for r in original if r['model'] == model]:
                cases = [r for r in selected if r['model'] == model and r['suite'] == expected['suite']
                         and r['condition'] == expected['condition'] and r['native_png']]
                self.assertEqual(len(cases), 20)
                self.assertEqual(sum(r['success'] for r in cases), expected['success'])

    def test_final_png_bytes_match_manifest(self):
        manifest = json.loads((STUDY / 'patches/MANIFEST.json').read_text())
        for row in manifest:
            path = STUDY / row['png']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['sha256'])
            raw = path.read_bytes()
            self.assertEqual(raw[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual((int.from_bytes(raw[16:20], 'big'), int.from_bytes(raw[20:24], 'big')), (50, 50))
            self.assertEqual(raw[24:26], b'\x08\x02')  # 8-bit RGB, not a resized preview.

    def test_new_release_has_no_cluster_jobs_or_large_runtime_assets(self):
        forbidden = {'.pt', '.pth', '.npy', '.npz', '.safetensors', '.mp4', '.webm', '.sbatch'}
        for path in STUDY.rglob('*'):
            if not path.is_file():
                continue
            self.assertNotIn(path.suffix, forbidden, str(path))
            self.assertFalse(path.name.startswith('submit'), str(path))
            self.assertNotIn(path.name, ['controller.py', 'monitor_snapshot.py', 'nightly.py', 'session_review.py'])

    def test_direct_entry_help_uses_no_external_assets(self):
        result = subprocess.run([sys.executable, '-B', str(STUDY / 'run_experiment.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--run-root', result.stdout)

    def test_direct_entry_passes_original_command_and_workspace(self):
        entry_spec = importlib.util.spec_from_file_location('wrist_public_entry', STUDY / 'run_experiment.py')
        entry = importlib.util.module_from_spec(entry_spec)
        entry_spec.loader.exec_module(entry)
        common = types.SimpleNamespace(ROOT=None, PYTHON=None)
        target = types.SimpleNamespace(run=Mock())
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'argv', [
                'run_experiment.py', 'main', 'eval', '--run-root', folder, '--id', 'B04', '--seed', '43']), \
                patch.object(entry.importlib, 'import_module', side_effect=[common, target]), \
                patch.object(sys, 'path', list(sys.path)):
            entry.main()
            args = target.run.call_args.args[0]
            self.assertEqual((args.command, args.id, args.seed), ('eval', 'B04', 43))
            self.assertEqual(common.ROOT, Path(folder).resolve())

    def test_factorial_entry_does_not_retrain_reused_baselines(self):
        entry_spec = importlib.util.spec_from_file_location('wrist_factorial_entry', STUDY / 'run_experiment.py')
        entry = importlib.util.module_from_spec(entry_spec)
        entry_spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as folder, patch.object(sys, 'argv', [
                'run_experiment.py', 'factorial', 'train', '--run-root', folder, '--id', 'B01']), \
                patch.object(entry.importlib, 'import_module', return_value=types.SimpleNamespace()), \
                patch.object(sys, 'path', list(sys.path)), patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:
                entry.main()
            self.assertEqual(error.exception.code, 2)

    def test_missing_clean_and_wrong_reset_are_not_counted(self):
        row = next(r for r in self.rows if r['study'] == 'main' and r['scope'] == 'full'
                   and r['condition'] == 'B01-seed42')
        with self.assertRaisesRegex(ValueError, 'Missing same-cohort clean'):
            summary.summarize([row], 'main', 'full', 'joint_oft')
        clean = dict(row, condition='clean', success=True, reset_seed='wrong')
        with self.assertRaisesRegex(ValueError, 'Unpaired reset seed'):
            summary.summarize([row, clean], 'main', 'full', 'joint_oft')


if __name__ == '__main__':
    unittest.main()
