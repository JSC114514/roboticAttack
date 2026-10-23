"""CPU contracts for nonlinear staged gradients, targets and Slurm scheduling."""
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from at_math import objective,far_boundary,learning_rate,paired_checkpoint
from at_engine import Engine,ref,install_checkpointing
import at_cli
import at_train
import at_report
import at_common
import at_evaluate

def pixels(obs,p,wrapper):
    return torch.cat([torch.zeros(1,6,2,2),p[None],p[None]],dim=1),p

def features(x):return torch.sin(x).flatten(2).transpose(1,2)
def siglip(x):return torch.cos(x*.7).flatten(2).transpose(1,2)
def head(d,s):
    v=torch.tanh(d.mean(1)+.6*s.mean(1))
    a=torch.cat([v,v.square(),v[...,:1].sigmoid()],-1)
    return a[:,None,:].expand(1,8,7)

class FakeEngine:
    wrapper=None
    spatial=lambda self,x:features(x[:,6:9])
    siglip=staticmethod(siglip)
    def action(self,obs,language,pixels,replacements=None):
        caps={'dino':self.spatial(pixels),'siglip':self.siglip(pixels[:,9:12])}
        caps.update(replacements or {})
        return head(caps['dino'],caps['siglip']),caps

class GradientTests(unittest.TestCase):
    def test_nonlinear_staged_vjp_both_methods_matches_direct(self):
        for method in ('geometry_teacher','clean_far_boundary'):
            with self.subTest(method=method):
                p=torch.linspace(.05,.95,12).reshape(3,2,2).requires_grad_()
                x,_=pixels({},p,None);d=features(x[:,6:9]);s=siglip(x[:,9:12])
                clean=head(d,s).detach();teacher={'clean':clean,'target':clean+.2}
                direct=objective(head(d,s.detach() if method=='geometry_teacher' else s),teacher,method)
                expected=torch.autograd.grad(direct,p)[0]
                row={'observation':{},'language':'dummy','teacher':teacher}
                with patch.object(ref,'patch_pixels',pixels):
                    actual,metrics=Engine.staged_gradient(FakeEngine(),row,p,method)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
                self.assertGreater(metrics['gradient_l2'],0)

    def test_dino_only_keeps_actual_siglip_forward(self):
        d=torch.zeros(1,4,3);s=torch.ones_like(d,requires_grad=True)
        actual=head(d,s.detach());clean_siglip=head(d,torch.zeros_like(s))
        self.assertFalse(torch.equal(actual,clean_siglip))
        self.assertFalse(actual.requires_grad)

    def test_checkpoint_wrappers_keep_eval_and_exact_gradient(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2)).eval()
        for p in model.parameters():p.requires_grad_(False)
        x=torch.ones(1,3,requires_grad=True);before=model(x);g=torch.autograd.grad(before.sum(),x)[0]
        install_checkpointing(model)
        after=model(x);g2=torch.autograd.grad(after.sum(),x)[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        torch.testing.assert_close(g,g2,rtol=0,atol=0)
        self.assertFalse(any(m.training for m in model.modules()))

class TargetTests(unittest.TestCase):
    def test_far_boundary_ranges_and_upper_tie(self):
        c=torch.zeros(1,8,7);c[...,6]=.5;c[0,1,0]=.1;c[0,2,6]=.8
        t=far_boundary(c)
        self.assertEqual(t[0,0,0],1);self.assertEqual(t[0,1,0],-1)
        self.assertEqual(t[0,0,6],1);self.assertEqual(t[0,2,6],0)
        self.assertTrue(((t[...,6]>=0)&(t[...,6]<=1)).all())

    def test_teacher_uses_all_slots_only_translation(self):
        a=torch.zeros(1,8,7,requires_grad=True);target=torch.ones_like(a)
        loss=objective(a,{'target':target},'geometry_teacher');g=torch.autograd.grad(loss,a)[0]
        self.assertTrue((g[...,:3]!=0).all());self.assertEqual(g[...,3:].abs().sum(),0)

    def test_schedule_and_paired_budget_selection(self):
        self.assertAlmostEqual(learning_rate(1,24000,20,.002),.0001)
        self.assertEqual(learning_rate(20,24000,20,.002),.002)
        self.assertEqual(learning_rate(24000,24000,20,.002),0)
        self.assertEqual(paired_checkpoint([1000,2000,3000],[1000,2000]),2000)
        self.assertIsNone(paired_checkpoint([1000],[]))

class SchedulingTests(unittest.TestCase):
    def test_waves_barrier_and_cpu_interim(self):
        calls=[]
        def one(stage,indices,deps):calls.append((stage,indices,list(deps)));return stage+'_job'
        with patch.object(at_cli,'complete',return_value=False),patch.object(at_cli,'cache_complete',return_value=False),\
             patch.object(at_cli.time,'time',return_value=at_cli.cutoff()-3600):
            at_cli.stages(one)
        self.assertEqual([x[0] for x in calls],['cache','train','select','evaluate','report','interim'])
        self.assertEqual(calls[1][2],['cache_job']);self.assertEqual(calls[3][2],['select_job'])
        self.assertEqual(calls[0][1],[0,1]);self.assertEqual(calls[1][1],[0,1,2,3])
        self.assertEqual(calls[-1][2],[])

    def test_submit_commands_bound_arrays(self):
        with patch.object(at_cli,'signature',return_value='deadbeef'):
            c=at_cli.command('cache',[0,1]);t=at_cli.command('train',[0,1,2,3],['1'])
        self.assertIn('--array=0,1%2',c);self.assertIn('--array=0,1,2,3%4',t)
        self.assertIn('--dependency=afterany:1',t)

    def test_resume_never_changes_sealed_evaluation_selection(self):
        calls=[]
        def done(path):return Path(path).name=='SELECT_COMPLETE.json'
        with patch.object(at_cli,'complete',side_effect=done),patch.object(at_cli,'cache_complete') as cache,\
             patch.object(at_cli.time,'time',return_value=at_cli.cutoff()-3600):
            at_cli.stages(lambda stage,indices,deps:calls.append(stage) or stage,include_interim=False)
        self.assertEqual(calls,['evaluate','report']);cache.assert_not_called()

    def test_static_resources_and_no_submission_in_tests(self):
        root=Path(__file__).resolve().parents[1]
        for stage in ('cache','train','evaluate'):
            s=(root/f'slurm/{stage}.sbatch').read_text()
            self.assertIn('--gres=gpu:4090:1',s);self.assertIn('--cpus-per-task=4',s)
            self.assertIn('--mem='+('16G' if stage=='train' else '12G'),s)
        for stage in ('select','report','interim'):
            s=(root/f'slurm/{stage}.sbatch').read_text();self.assertNotIn('--gres',s)
            self.assertIn('--cpus-per-task=2',s);self.assertIn('--mem=4G',s)

class TrainingLoopTests(unittest.TestCase):
    def test_training_reaches_heartbeat_and_counts_skips_without_gpu(self):
        class SmallEngine:
            wrapper=SimpleNamespace(device=torch.device('cpu'))
            def gate(self,*args):pass
            def staged_gradient(self,row,p,method):return torch.ones_like(p)*.01,{'loss':.1,'gradient_l2':.01,'target_mse':.1}
        updates=[]
        monitor=SimpleNamespace(update=lambda **kw:updates.append(kw))
        pool=[{'state_id':str(i),'teacher':{'no_target':i%7==0}} for i in range(240)]
        cfg={**at_cli.config(),'updates':30,'checkpoint_interval':10}
        saved=[]
        def save(folder,p,opt,step,optimized,no_target,method,seed,terminal=False):
            saved.append((step,optimized,no_target))
            self.assertEqual(int(opt.state[p]['step']),optimized)
            return folder/f'checkpoint_{step}.json'
        with tempfile.TemporaryDirectory() as temp,patch.object(at_train,'ROOT',Path(temp)),\
             patch.object(at_train,'config',return_value=cfg),patch.object(at_train,'complete',return_value=False),\
             patch.object(at_train,'cutoff',return_value=10**12),patch.object(at_train,'records',return_value=iter(pool)),\
             patch.object(at_train,'make_engine',return_value=SmallEngine()),\
             patch.object(at_train.native,'require_gpu'),patch.object(at_train.native,'load_png',return_value=torch.ones(3,50,50)*.5),\
             patch.object(at_train,'save',side_effect=save),patch.object(at_train,'finish') as finished:
            at_train.run(0,monitor)
            self.assertTrue(any(v.get('completed')==25 for v in updates))
            step,opt,skip=saved[-1];self.assertEqual(step,30);self.assertEqual(opt+skip,30)
            self.assertTrue(finished.call_args.kwargs['budget_complete'])

class ReportingTests(unittest.TestCase):
    def test_missing_condition_does_not_become_attack_failure(self):
        original_config=at_common.config()
        keys=['clean','initial_s42','initial_s43','uada500','geometry_teacher_s42',
              'clean_far_boundary_s42','geometry_teacher_s43','clean_far_boundary_s43']
        with tempfile.TemporaryDirectory() as temp,patch.object(at_common,'ROOT',Path(temp)),\
             patch.object(at_report,'ROOT',Path(temp)),patch.object(at_evaluate,'conditions',return_value=dict.fromkeys(keys)),\
             patch.object(at_common,'signature',return_value='unit-test'):
            root=Path(temp)
            (root/'config.json').write_text(json.dumps(original_config))
            # One complete paired start, plus one clean-only incomplete start.
            for state,conditions in [(0,keys),(1,['clean'])]:
                for name in conditions:
                    folder=root/f'evaluation/episodes/test_t00_s{state}/{name}';folder.mkdir(parents=True)
                    meta={'suite':'libero_spatial','task_id':0,'state_index':state,'condition':name,
                          'success':int(name!='geometry_teacher_s42')}
                    p=folder/'EPISODE.json';p.write_text(json.dumps(meta))
                    at_common.finish(folder/'COMPLETE.json',[p])
            at_report.run(final=False)
            import pandas as pd
            status=json.loads((root/'results/STATUS.json').read_text())
            scores=pd.read_csv(root/'results/CASR.csv')
            self.assertEqual(status['closedloop_episodes'],9)
            self.assertEqual(status['fully_paired_starts'],1)
            row=scores[(scores.suite=='all')&(scores.condition=='geometry_teacher_s42')].iloc[0]
            self.assertEqual(row.clean_successes,1);self.assertEqual(row.failure_switches,1)
            self.assertEqual(row.failures_beyond_initial,1)
            self.assertEqual(status['status'],'PARTIAL_OR_FAILED')

if __name__=='__main__':unittest.main()
