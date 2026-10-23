"""CPU checks of targets, real staged-gradient code and submission contracts."""
import sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from sr_math import retreat_target,far_target,objective,learning_rate
from sr_engine import Engine,ref,install_checkpointing
from sr_geometry import spec,controller
from types import SimpleNamespace
import sr_cli,sr_train,sr_report

def pixels(obs,p,wrapper):return torch.cat([torch.zeros(1,6,2,2),p[None],p[None]],dim=1),p
def dino(x):return torch.sin(x).flatten(2).transpose(1,2)
def siglip(x):return torch.cos(x*.7).flatten(2).transpose(1,2)
def head(d,s):
    v=torch.tanh(d.mean(1)+.6*s.mean(1));return torch.cat([v,v.square(),v[...,:1].sigmoid()],-1)[:,None,:].expand(1,8,7)
class FakeEngine:
    wrapper=None
    spatial=lambda self,x:dino(x[:,6:9])
    siglip=staticmethod(siglip)
    command=lambda self,a:a[...,:3]*.05
    def action(self,obs,language,pixels,replacements=None):
        caps={'dino':self.spatial(pixels),'siglip':self.siglip(pixels[:,9:12])};caps.update(replacements or {})
        return head(caps['dino'],caps['siglip']),caps

class Targets(unittest.TestCase):
    def test_radial_target_changes_with_geometry_not_world_x(self):
        for u in np.eye(3):
            v=np.tile(u*.012,(1,8,1));t=retreat_target(v,u,np.full(3,-.05),np.full(3,.05),.005)
            np.testing.assert_allclose(t@u,-.005,atol=1e-8)
            np.testing.assert_allclose((t-v)*(1-u),0,atol=1e-8)
    def test_projection_is_feasible_and_locally_optimal(self):
        rng=np.random.default_rng(42)
        for _ in range(50):
            u=rng.normal(size=3);u/=np.linalg.norm(u);v=rng.uniform(-.08,.08,(8,3))
            target=retreat_target(v,u,np.full(3,-.05),np.full(3,.05),.005)
            self.assertTrue((target@u<=-.005+1e-8).all());self.assertTrue((abs(target)<=.05000001).all())
            trials=rng.uniform(-.05,.05,(1000,3));feasible=trials[trials@u<=-.005]
            self.assertTrue((np.sum((target-v)**2,axis=1)<=np.min(np.sum((feasible[:,None]-v[None])**2,axis=2),axis=0)+1e-8).all())
    def test_invalid_and_infeasible_direction_rejected(self):
        with self.assertRaises(ValueError):retreat_target(np.zeros(3),np.zeros(3),[-.05]*3,[.05]*3,.005)
        with self.assertRaises(ValueError):retreat_target(np.zeros(3),[1,0,0],[-.05]*3,[.05]*3,.1)
    def test_far_boundary_tie_upper_and_all_axes(self):
        np.testing.assert_array_equal(far_target(np.array([0,.01,-.02]),[-.05]*3,[.05]*3),np.array([.05,-.05,.05],np.float32))
    def test_loss_all_slots_three_coordinates(self):
        action=torch.zeros(1,8,7,requires_grad=True);teacher={'targets':{'radial_reverse':torch.ones(1,8,3)},'command_scale':torch.ones(3)}
        loss=objective(action[...,:3],teacher,'radial_reverse');g=torch.autograd.grad(loss,action)[0]
        self.assertTrue((g[...,:3]!=0).all());self.assertEqual(float(g[...,3:].abs().sum()),0)
    def test_schedule(self):
        self.assertAlmostEqual(learning_rate(1,4000,20,.002),.0001);self.assertEqual(learning_rate(4000,4000,20,.002),0)
    def test_entities_dont_fallback_to_plates(self):
        self.assertEqual(spec('libero_spatial',0)['entity'],'akita_black_bowl_1')
        self.assertEqual(spec('libero_goal',3)['kind'],'top_handle');self.assertEqual(spec('libero_goal',7)['kind'],'button')
    def test_controller_and_label_use_same_eef_site(self):
        c=SimpleNamespace(use_delta=True,control_dim=6,eef_name='grip',input_min=np.full(6,-1),input_max=np.ones(6),output_min=np.full(6,-.05),output_max=np.full(6,.05))
        robot=SimpleNamespace(controller=c,eef_site_id=4)
        env=SimpleNamespace(env=SimpleNamespace(robots=[robot]),sim=SimpleNamespace(model=SimpleNamespace(site_name2id=lambda n:3)))
        with self.assertRaises(RuntimeError):controller(env)
        robot.eef_site_id=3;self.assertEqual(controller(env)['eef_site_id'],3)

class Gradients(unittest.TestCase):
    def test_both_methods_staged_vjp_matches_full_nonlinear_gradient(self):
        for method in ('radial_reverse','translation_far_boundary'):
            p=torch.linspace(.05,.95,12).reshape(3,2,2).requires_grad_();x,_=pixels({},p,None)
            e=FakeEngine();a,_=e.action({},'',x);teacher={'targets':{method:torch.ones(1,8,3)*-.01},'command_scale':torch.ones(3)*.05,'unit':[1.,0.,0.]}
            expected=torch.autograd.grad(objective(e.command(a),teacher,method),p)[0]
            with patch.object(ref,'patch_pixels',pixels):actual,metrics=Engine.staged_gradient(e,{'observation':{},'language':'','teacher':teacher},p,method)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7);self.assertGreater(metrics['gradient_l2'],0)
    def test_checkpoint_preserves_eval_forward_and_gradient(self):
        m=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2)).eval()
        for p in m.parameters():p.requires_grad_(False)
        x=torch.ones(1,3,requires_grad=True);before=m(x);g=torch.autograd.grad(before.sum(),x)[0]
        install_checkpointing(m);after=m(x);g2=torch.autograd.grad(after.sum(),x)[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0);torch.testing.assert_close(g,g2,rtol=0,atol=0)
        self.assertFalse(any(p.training for p in m.modules()))

class Workflow(unittest.TestCase):
    def test_partial_report_does_not_claim_complete_or_count_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            samples=[('a','clean',1),('a','initial_s42',0),('b','clean',0)]
            for start,condition,success in samples:
                p=root/f'evaluation/episodes/{start}/{condition}';p.mkdir(parents=True)
                (p/'COMPLETE.json').write_text('{}')
                (p/'EPISODE.json').write_text(json.dumps({'suite':'libero_goal','task_id':0 if start=='a' else 1,
                    'state_index':0,'condition':condition,'success':success,'failure':1-success}))
            with patch.object(sr_report,'ROOT',root),patch.object(sr_report,'complete',side_effect=lambda p:Path(p).exists()):sr_report.run(final=True)
            value=json.loads((root/'results/STATUS.json').read_text())
            self.assertEqual(value['status'],'PARTIAL');self.assertEqual(value['clean_success_starts'],1)
            self.assertFalse((root/'results/COMPLETE.json').exists())
            import pandas as pd
            table=pd.read_csv(root/'results/CASR.csv');row=table[(table.suite=='all')&(table.condition=='initial_s42')].iloc[0]
            self.assertEqual(row.paired_completed,1);self.assertEqual(row.induced_failures,1)
    def test_optimizer_checkpoint_resumes_same_next_update(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);p=torch.nn.Parameter(torch.full((3,50,50),.5));opt=torch.optim.Adam([p],lr=.002)
            def update(parameter,optimizer):
                optimizer.zero_grad();parameter.square().mean().backward();optimizer.step()
            update(p,opt)
            with patch.object(sr_train,'finish',return_value=None):sr_train.save(folder,p,opt,1,1,0,'radial_reverse',42)
            v=torch.load(folder/'LAST.pt',map_location='cpu',weights_only=False)
            q=torch.nn.Parameter(v['patch'].clone());second=torch.optim.Adam([q],lr=.002);second.load_state_dict(v['optimizer'])
            update(p,opt);update(q,second);torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertEqual(v['optimized'],v['step']);self.assertEqual(int(second.state[q]['step']),2)
    def test_array_concurrency_and_dependencies(self):
        for stage,indices in [('cache',[0,1]),('train',[0,1,2,3]),('evaluate',[0,1])]:
            cmd=sr_cli.command(stage,indices,['123'])
            self.assertIn('--array='+','.join(map(str,indices))+'%2',cmd)
            self.assertIn('--dependency=afterok:123',cmd);self.assertIn('--kill-on-invalid-dep=yes',cmd)
        self.assertIn('--dependency=afterany:123',sr_cli.command('report',None,['123'],True))
    def test_no_submissions_in_dry_run_stages(self):
        seen=[]
        def record(stage,indices=None,deps=(),afterany=False):seen.append((stage,indices,deps,afterany));return stage
        with patch.object(sr_cli,'complete',return_value=False),patch.object(sr_cli,'cache_complete',return_value=False):sr_cli.stages(record)
        self.assertEqual([r[0] for r in seen],['cache','coverage','train','select','evaluate','report'])
        self.assertEqual(seen[2][2],['coverage']);self.assertEqual(seen[-1][-1],True)
    def test_same_seed_sampling_order_is_paired_and_resumable(self):
        import random
        def order(start):
            return [(i%4,random.Random(42*1000000+i).choice(list(range(8)))) for i in range(start,4000)]
        self.assertEqual(order(0)[1234:],order(1234));self.assertEqual(order(0),order(0))
    def test_slurm_host_memory_and_single_gpu(self):
        root=Path(__file__).resolve().parents[1]
        for stage in ('cache','train','evaluate'):
            s=(root/f'slurm/{stage}.sbatch').read_text();self.assertIn('--gres=gpu:4090:1',s);self.assertIn('--cpus-per-task=4',s)
            self.assertIn('--mem='+('16G' if stage=='train' else '12G'),s)
        for stage in ('coverage','select','report'):self.assertNotIn('--gres=',(root/f'slurm/{stage}.sbatch').read_text())

if __name__=='__main__':unittest.main()
