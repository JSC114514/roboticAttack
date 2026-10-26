import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import diag_report
import diag_run


MODELS = ['base', 'consistency_w1_s42', 'consistency_w10_s42']


def fixture():
    conditions = [dict(name=f't{t}_c{c}', target_shift_m=t, camera_shift_m=c, valid=True, reasons=[])
                  for t in (-.02, 0., .02) for c in (-.02, 0., .02)]
    anchors = [dict(episode_id=f'e{i}', suite='spatial', task_id=0, selected={'frame': 1}, conditions=copy.deepcopy(conditions)) for i in range(2)]
    results = {}
    for anchor in anchors:
        for m, gain in zip(MODELS, (1, 2, 3)):
            for c in conditions:
                t, cam = c['target_shift_m'], c['camera_shift_m']
                results[m, anchor['episode_id'], c['name']] = dict(executed_steps=8, approach_m=gain*t+10*cam,
                    eef_displacement_m=[gain*t, cam, 0.], actions=[[gain*t]*7]*8, normalized_actions=[[t]*7]*8)
    return dict(status='COMPLETE', anchors=anchors), results


class ReportTests(unittest.TestCase):
    def test_absolute_csv_keeps_values_and_invalid_missing(self):
        manifest, results = fixture()
        key = ('base', 'e0', 't-0.02_c-0.02')
        results[key].update(initial_distance_m=.5, final_distance_m=.3, approach_m=.2,
                            first_contact_step=3, first_grasp_step=None, executed_steps=8)
        manifest['anchors'][0]['conditions'][1].update(valid=False, reasons=['out of view', 'collision'])
        del results['base', 'e0', 't-0.02_c0.02']
        rows = diag_report.branch_rows(manifest, results, MODELS)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'BRANCHES.csv'
            diag_report.write_branches(path, rows)
            with path.open() as stream: written = list(csv.DictReader(stream))
        row = next(r for r in written if (r['model'], r['episode_id'], r['condition']) == key)
        self.assertEqual(float(row['approach_m']), .2)
        self.assertEqual(row['contact_observed'], '1')
        self.assertEqual(row['grasp_observed'], '0')
        invalid = next(r for r in written if r['model'] == 'base' and r['status'] == 'INVALID')
        self.assertEqual(json.loads(invalid['invalid_reasons']), ['out of view', 'collision'])
        self.assertEqual(invalid['approach_m'], '')
        counts = diag_report.absolute_by_task(rows)['base/spatial/t0']
        self.assertEqual((counts['completed'], counts['invalid'], counts['missing']), (16, 1, 1))
        self.assertEqual(counts['contact_grasp_denominator'], 16)

    def test_sealed_step_distance_curves(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            branch = root/'evaluation/base/e0/c0'; branch.mkdir(parents=True)
            steps = branch/'attempt/STEPS.parquet'; steps.parent.mkdir()
            pd.DataFrame({'step':[1, 2], 'distance_m':[.4, .3]}).to_parquet(steps, index=False)
            (branch/'COMPLETE.json').write_text(json.dumps({'artifacts':{str(steps.relative_to(root)):'sealed'}}))
            curves = diag_report.distance_figures({('base', 'e0', 'c0'):{'initial_distance_m':.5}}, root, ['base'])
            self.assertEqual(curves['errors'], [])
            self.assertEqual(len(curves['files']), 2)
            for path in curves['files']: self.assertGreater((root/path).stat().st_size, 100)

    def test_video_full_decode_and_bad_frame_count(self):
        import imageio.v2 as imageio
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder); root = project/'run'
            branch = root/'evaluation/base/e0/c0'; branch.mkdir(parents=True)
            for view in ('front', 'wrist'):
                with imageio.get_writer(branch/(view+'.mp4'), fps=20, macro_block_size=1) as writer:
                    for value in (0, 80, 160): writer.append_data(np.full((224, 224, 3), value, dtype=np.uint8))
            result = {'success_flag':False, 'executed_steps':2, 'fps':20}
            exported = diag_report.publish_videos({('base', 'e0', 'c0'):result}, root, project)
            self.assertEqual(exported['errors'], [])
            self.assertEqual(exported['status'], 'COMPLETE')
            self.assertEqual(len(exported['exports']), 2)
            result['executed_steps'] = 3
            bad = diag_report.publish_videos({('base', 'e0', 'c0'):result}, root, project)
            self.assertEqual(bad['status'], 'INCOMPLETE')
            self.assertEqual(len(bad['errors']), 2)

    def test_signed_paired_interaction(self):
        manifest, results = fixture()
        report = diag_report.aggregate(manifest, results, MODELS, draws=20)
        self.assertEqual(report['status'], 'COMPLETE')
        estimates = report['estimates']
        positive = estimates['physical/t0.02_c0.02/consistency_w1_s42/approach_m/interaction_vs_base']
        negative = estimates['physical/t-0.02_c0.02/consistency_w1_s42/approach_m/interaction_vs_base']
        self.assertAlmostEqual(positive['mean'], .02)
        self.assertAlmostEqual(negative['mean'], -.02)
        self.assertEqual(positive['n_trajectories'], 2)
        self.assertEqual(positive['task_denominators'], {'spatial/t0': 2})
        camera = estimates['camera/t0.02_c0.02/consistency_w1_s42/approach_m/interaction_vs_base']
        self.assertAlmostEqual(camera['mean'], 0.)

    def test_missing_branch_excludes_trajectory_in_all_models(self):
        manifest, results = fixture()
        del results[MODELS[2], 'e0', 't0.0_c0.02']
        report = diag_report.aggregate(manifest, results, MODELS, draws=20)
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['counts'][MODELS[2]]['missing'], 1)
        estimate = report['estimates']['physical/t0.02_c0.02/consistency_w1_s42/approach_m/interaction_vs_base']
        self.assertEqual(estimate['n_trajectories'], 1)

    def test_short_chunk_is_engineering_missing_and_excluded_in_all_models(self):
        manifest, results = fixture()
        results[MODELS[2], 'e0', 't0.0_c0.02']['executed_steps'] = 7
        report = diag_report.aggregate(manifest, results, MODELS, draws=20)
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['counts'][MODELS[2]]['missing'], 1)
        self.assertEqual(report['counts'][MODELS[2]]['invalid'], 0)
        self.assertEqual(report['invalid_results'][0]['reason'], 'executed_steps_must_equal_8')
        for model in MODELS:
            estimate = report['estimates'][f'physical/t0.02_c0.02/{model}/approach_m/response']
            self.assertEqual(estimate['n_trajectories'], 1)

    def test_invalid_is_separate_from_missing(self):
        manifest, results = fixture()
        manifest['anchors'][0]['conditions'][0]['valid'] = False
        report = diag_report.aggregate(manifest, results, MODELS, draws=5)
        self.assertEqual(report['counts']['base']['invalid'], 1)
        self.assertEqual(report['counts']['base']['missing'], 0)
        self.assertEqual(report['counts']['base']['completed'], 17)

    def test_missing_manifest_and_no_anchor(self):
        result = diag_report.aggregate(dict(status='MISSING', anchors=[]), {}, MODELS, draws=5)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(result['estimates'], {})

    def test_task_weighting_and_bootstrap_reproducibility(self):
        rows = [dict(suite='a', task_id=0, value=0.)]*9 + [dict(suite='a', task_id=1, value=4.)]
        first = diag_report._estimate(rows, 100)
        self.assertEqual(first['mean'], 2.)
        self.assertEqual(first, diag_report._estimate(rows, 100))


class RunnerTests(unittest.TestCase):
    def test_stale_cpu_report_never_blocks_gpu_resume(self):
        report = ['1', 'PENDING', diag_run.PREFIX+'oldhash_report', 'DependencyNeverSatisfied']
        gpu = ['2', 'RUNNING', diag_run.PREFIX+'newhash_evaluate', 'gpu1']
        self.assertEqual(diag_run.blocking_jobs([report]), ([], [report]))
        self.assertEqual(diag_run.blocking_jobs([report, gpu]), ([gpu], [report]))

    def test_dag_resources_and_resume(self):
        import diag_common
        with mock.patch.object(diag_common, 'complete', return_value=False), mock.patch.object(diag_common, 'config', return_value={'models':MODELS}), mock.patch.object(diag_common, 'signature', return_value='abcd'):
            plan = diag_run.build_plan()
        self.assertEqual([s['name'] for s in plan['stages']], ['prepare', 'evaluate', 'report'])
        self.assertEqual(plan['stages'][0]['concurrency'], 1)
        evaluation, report = plan['stages'][1:]
        self.assertEqual(len(evaluation['units']), 3)
        self.assertEqual(evaluation['concurrency'], 2)
        self.assertEqual(report['parents'], ['prepare', 'evaluate'])
        command = diag_run.slurm_command(evaluation, Path('/tmp/plan.json'), 'abcd', ['123'])
        self.assertIn('--array=0-2%2', command)
        self.assertIn('--time=02:00:00', command)
        self.assertIn('--mem=12G', command)
        self.assertIn('--cpus-per-task=4', command)
        command = diag_run.slurm_command(report, Path('/tmp/plan.json'), 'abcd', ['123', '124'])
        self.assertIn('--dependency=afterany:123:124', command)
        self.assertIn('--time=00:30:00', command)
        with mock.patch.object(diag_common, 'complete', return_value=True), mock.patch.object(diag_common, 'config', return_value={'models':MODELS}), mock.patch.object(diag_common, 'signature', return_value='abcd'):
            self.assertEqual([s['name'] for s in diag_run.build_plan()['stages']], ['report'])


if __name__ == '__main__': unittest.main()
