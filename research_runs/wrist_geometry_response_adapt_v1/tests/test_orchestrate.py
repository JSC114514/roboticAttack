"""CPU-only scheduling and report contracts. No Slurm submission or GPU loading."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import io
import os
import subprocess
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import wr_common as common
import wr_orchestrate as orchestration
import wr_report as report


class SchedulingTests(unittest.TestCase):
    def test_pilot_dag_and_resources(self):
        # This test describes a fresh run, regardless of real completed assets.
        with patch.object(common,'complete',return_value=False):
            plan = orchestration.build_plan('pilot',2)
        stages = {x['name']:x for x in plan['stages']}
        self.assertEqual(stages['prepare-expert']['memory_gb'],4)
        self.assertEqual(stages['prepare-expert']['cpus'],2)
        self.assertEqual(len(stages['train']['units']),8)
        self.assertEqual(stages['train']['memory_gb'],16)
        self.assertEqual(stages['evaluate']['memory_gb'],12)
        self.assertEqual(stages['evaluate']['concurrency'],2)
        self.assertIn('calibrate',stages['train']['parents'])
        self.assertIn('base-eval',stages['train']['parents'])
        self.assertEqual(stages['report']['after'],'afterany')
        self.assertNotIn('main',[x.get('stage') for x in plan['stages']])

    def test_resource_limit(self):
        with self.assertRaises(ValueError): orchestration.build_plan('pilot',3)

    def test_wrap_quotes_and_dependencies(self):
        item=dict(name='train',units=[{},{}],memory_gb=16,cpus=4,gpu=True,concurrency=2,after='afterok')
        cmd=orchestration.slurm_command(item,Path('/tmp/plan with spaces.json'),'pilot','abc',['123'])
        self.assertIn('--gres=gpu:4090:1',cmd)
        self.assertIn('--partition=rtx4090',cmd)
        self.assertIn('--dependency=afterok:123',cmd)
        self.assertIn('--array=0-1%2',cmd)
        self.assertIn("'/tmp/plan with spaces.json'",cmd[-1])

    def test_resume_omits_completed_stages(self):
        with patch.object(common,'complete',return_value=True):
            plan=orchestration.build_plan('pilot',2)
        self.assertEqual([s['name'] for s in plan['stages']],['report'])

    def test_active_queue_prevents_duplicate(self):
        with patch.object(orchestration,'check'),patch.object(orchestration,'queue',return_value=[['123','RUNNING','wradapt1_x','']]),patch.object(orchestration,'build_plan') as build:
            with contextlib.redirect_stdout(io.StringIO()):orchestration.submit('pilot',2)
            build.assert_not_called()

    def test_tracked_pending_report_does_not_block_resume(self):
        prior=dict(stage='pilot',signature='46859e45abc',jobs={'report':'150772'})
        report=['150772','PENDING','wradapt1_46859e45_pilot_report','(Priority)']
        self.assertEqual(orchestration.resume_jobs([report],'pilot',prior),([],[report]))

    def test_pending_report_does_not_hide_compute_job(self):
        prior=dict(stage='pilot',signature='46859e45abc',jobs={'report':'150772'})
        report=['150772','PENDING','wradapt1_46859e45_pilot_report','(Priority)']
        gpu=['150767','PENDING','wradapt1_46859e45_pilot_prepare-camera','(Priority)']
        self.assertEqual(orchestration.resume_jobs([report,gpu],'pilot',prior),([gpu],[report]))

    def test_running_or_untracked_report_still_blocks(self):
        prior=dict(stage='pilot',signature='46859e45abc',jobs={'report':'150772'})
        for row in (['150772','RUNNING','wradapt1_46859e45_pilot_report','node'],
                    ['150999','PENDING','wradapt1_46859e45_pilot_report','(Priority)']):
            with self.subTest(row=row):
                self.assertEqual(orchestration.resume_jobs([row],'pilot',prior),([row],[]))

    def test_resume_with_only_pending_report_reaches_submission(self):
        prior=dict(stage='pilot',signature='46859e45abc',jobs={'report':'150772'})
        queued=[['150772','PENDING','wradapt1_46859e45_pilot_report','(Priority)']]
        with patch.object(orchestration,'check'),patch.object(orchestration,'queue',return_value=queued),patch.object(orchestration,'_read',return_value=prior),patch.object(orchestration,'build_plan',return_value={'stages':[]}) as build,patch.object(orchestration,'_atomic'):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                orchestration.submit('pilot',2)
            build.assert_called_once_with('pilot',2)
            self.assertIn('PENDING_CPU_REPORT_DOES_NOT_BLOCK_RESUME',output.getvalue())

    def test_shell_sets_renderer_to_requested_gpu(self):
        # Execute the actual shell setup without starting Python, Slurm or EGL.
        setup=(orchestration.ROOT/'run.sh').read_text().split('\nexec ',1)[0]
        probe=setup+'\nprintf "%s\\n" "$OFT_STAGE_GPU" "$MUJOCO_EGL_DEVICE_ID" "$CUDA_VISIBLE_DEVICES" "$MUJOCO_GL" "$PYOPENGL_PLATFORM" "$_CHECK_PEFT"\n'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'probe.sh';path.write_text(probe)
            for inherited in (None,'3090'):
                with self.subTest(inherited=inherited):
                    env=dict(os.environ,SLURM_JOB_ID='test-only',CUDA_VISIBLE_DEVICES='3',MUJOCO_EGL_DEVICE_ID='0',_CHECK_PEFT='1')
                    env.pop('OFT_STAGE_GPU',None)
                    if inherited is not None:env['OFT_STAGE_GPU']=inherited
                    result=subprocess.run(['bash',str(path)],env=env,capture_output=True,text=True,check=True)
                    self.assertEqual(result.stdout.splitlines(),['4090','3','3','egl','egl','0'])
        self.assertEqual(common.config()['resources']['gres'],'gpu:4090:1')

    def test_shell_rejects_multiple_gpu_mapping(self):
        setup=(orchestration.ROOT/'run.sh').read_text().split('\nexec ',1)[0]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'probe.sh';path.write_text(setup+'\n')
            result=subprocess.run(['bash',str(path)],env=dict(os.environ,SLURM_JOB_ID='test-only',CUDA_VISIBLE_DEVICES='0,1'),capture_output=True,text=True)
            self.assertEqual(result.returncode,11)
            self.assertIn('Exactly one allocated GPU mapping required',result.stderr)


class ReportTests(unittest.TestCase):
    def rows(self):
        return [dict(candidate=c,suite='suite',task=0,init=i,profile=p,success=(c=='best' or p=='native'))
            for c in ('base','best','weak') for i in (1,2) for p in ('native','xminus','xplus','zminus','zplus')]

    def test_missing_excluded_from_selection(self):
        rows=self.rows(); cases=[{k:r[k] for k in ('suite','task','init','profile')} for r in rows if r['candidate']=='base']
        profiles=['native','xminus','xplus','zminus','zplus']
        rows=[r for r in rows if not(r['candidate']=='best' and r['init']==2)]
        summary=report.summarize(rows,{'best':cases,'weak':cases,'base':cases},profiles)
        self.assertIsNone(summary['best']['selection_score'])
        selection=report.select_configs(summary,[dict(name='best',method='a'),dict(name='weak',method='a')])
        self.assertFalse(selection['complete']);self.assertEqual(selection['selected'],{})

    def test_harmonic_and_tie_cost(self):
        self.assertAlmostEqual(report.harmonic(1,.5),2/3)
        entry=dict(complete=True,selection_score=.8,clean_success_rate=1.)
        chosen=report.select_configs({'a':entry,'b':entry},[dict(name='a',method='m'),dict(name='b',method='m')],{'a':10,'b':20})
        self.assertEqual(chosen['selected']['m'],'a')

    def test_duplicate_rejected(self):
        row=self.rows()[0]
        with self.assertRaises(ValueError): report.validate_rows([row,row])

    def test_bootstrap_pairing(self):
        result=report.paired_bootstrap(self.rows(),'best',repetitions=100)
        self.assertEqual(result['paired_episodes'],10)
        self.assertAlmostEqual(result['delta'],.8)
        self.assertEqual(result['ci95'],[.8,.8])

    def test_no_completed_rows(self):
        with tempfile.TemporaryDirectory() as d:
            case=dict(suite='s',task_id=0,state_index=1,profile='native')
            result=report.write_report(d,[],{'base':[case]},['native'],[dict(name='base',method='base')],'pilot')
            self.assertEqual(result['status'],'INCOMPLETE')
            self.assertIsNone(result['candidates']['base']['clean_success_rate'])


if __name__=='__main__': unittest.main()
