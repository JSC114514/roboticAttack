"""CPU checks for publication provenance and exact diagnostic result accounting."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_annotation_changes_no_executable_ast(self):
        """The saved original-AST digest must match every annotated copy."""
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
        for name in manifest['annotated_files']:
            row = manifest['source_files'][name]
            ast_bytes = ast.dump(ast.parse((ROOT / name).read_bytes()), include_attributes=False).encode()
            self.assertEqual(hashlib.sha256(ast_bytes).hexdigest(), row['source_ast_sha256'], name)
            self.assertNotEqual(row['sha256'], row['original_sha256'], name)

    def test_diagnostic_counts_are_branches_not_casr(self):
        """Keep 8-step branch accounting distinct from full-rollout success."""
        result = json.loads((ROOT / 'results/camera_task_response_summary.json').read_text())
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertFalse(result['short_task_flag_is_full_rollout_success'])
        self.assertEqual(sum(r['completed'] for r in result['counts'].values()), 279)
        self.assertTrue(all(r['missing'] == 0 and r['invalid'] == 6 for r in result['counts'].values()))

    def test_raw_configurations_are_bound_to_sources(self):
        """Every documented configuration must have a source hash and readable file."""
        manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())['source_files']
        catalog = json.loads((ROOT / 'docs/EXPERIMENT_CATALOG.json').read_text())
        for run in catalog:
            for name in run['config_files']:
                self.assertIn(name, manifest)
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), manifest[name]['sha256'])

    def test_current_research_models_use_joint_checkpoint(self):
        """Adaptation model identity and diagnostic source must agree."""
        adapt = json.loads((ROOT / 'research_runs/wrist_geometry_response_adapt_v1/config.json').read_text())
        diag = json.loads((ROOT / 'research_runs/wrist_camera_task_response_diag_v1/config.json').read_text())
        self.assertTrue(adapt['model_dir'].endswith('openvla-7b-oft-finetuned-libero-spatial-object-goal-10'))
        self.assertEqual(diag['source_run'], 'research_runs/wrist_geometry_response_adapt_v1')
        self.assertEqual(diag['chunk'], 8)


if __name__ == '__main__':
    unittest.main()
