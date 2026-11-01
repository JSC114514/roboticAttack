# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU tests for the scientific differences and resumable execution contract."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from joint_repro.common import ROOT,read,sha
from joint_repro.sweep import study,resolve,batch_indices,validate
from joint_repro.methods import spec,loss,draw,seed_eot


class ScientificContract(unittest.TestCase):
    def test_frozen_sources_and_single_variable(self):validate()

    def test_exact_matrix_size(self):
        workers=study()['workers'];self.assertEqual(sum(w['gate']=='initial' for w in workers),20)
        self.assertEqual(len(workers),32)
        self.assertEqual(sum(len(w['eval_ids']) for w in workers if w['gate']=='initial'),24)

    def test_action_masks_and_auxiliary(self):
        self.assertEqual(spec('B01')['mask'],[0]);self.assertEqual(spec('B04')['mask'],list(range(7)))
        self.assertEqual(spec('O03')['mask'],[0]);self.assertEqual(spec('D07')['mask'],list(range(7)))
        self.assertEqual(spec('W025-D07'),spec('D07'))

    def test_batch_preserves_first_member_and_nested_prefix(self):
        rows=read(ROOT/'data/BANK_MANIFEST.json')['records']
        for outer in [1,2,24,25,500,1000,2000]:
            indices=batch_indices(rows,outer,8)
            self.assertEqual(indices[0],draw(rows,outer,20261026))
            self.assertEqual(len({rows[i]['trajectory_id'] for i in indices}),8)
            self.assertEqual(len({(rows[i]['suite'],rows[i]['task_id']) for i in indices}),1)
            for b in [1,2,4]:self.assertEqual(batch_indices(rows,outer,b),indices[:b])

    def test_sampling_does_not_mutate_transform_rng(self):
        import random,numpy as np
        rows=read(ROOT/'data/BANK_MANIFEST.json')['records'];seed_eot(42,1,3)
        before=(random.getstate(),torch.get_rng_state().clone(),np.random.get_state())
        batch_indices(rows,25,8)
        self.assertEqual(before[0],random.getstate());self.assertTrue(torch.equal(before[1],torch.get_rng_state()))
        self.assertTrue((before[2][1]==np.random.get_state()[1]).all())

    def test_gradient_accumulation_equals_mean(self):
        torch.manual_seed(42);x=torch.randn(4,3);p=torch.randn(3,requires_grad=True)
        direct=((x@p-1)**2).mean();g=torch.autograd.grad(direct,p)[0]
        for sample in x:((sample@p-1)**2/4).backward()
        torch.testing.assert_close(g,p.grad)

    def test_aux_scalar_only_and_zero_beta_equivalence(self):
        action=torch.randn(1,8,7,requires_grad=True);teacher=torch.randn_like(action)
        clean={'dino':[None,torch.randn(1,256,8)]};features={'dino':[None,torch.randn(1,256,8,requires_grad=True)]}
        base,_=loss(action,teacher,spec('B01'))
        aux,terms=loss(action,teacher,spec('O03'),features,clean,0.)
        torch.testing.assert_close(base,aux)
        value,_=loss(action,teacher,spec('O03'),features,clean,22.855948214853818*.25)
        torch.testing.assert_close(value,base+22.855948214853818*.25*terms['auxiliary'])

    def test_native_episode_has_not_changed(self):
        parent=Path(study()['parent_root'])/'joint_repro/evaluate.py'
        def body(p):return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='episode'),include_attributes=False)
        self.assertEqual(body(parent),body(ROOT/'joint_repro/evaluate.py'))

    def test_native_loss_eot_and_target_have_not_changed(self):
        parent=Path(study()['parent_root'])/'joint_repro/methods.py'
        a=ast.parse(parent.read_text());b=ast.parse((ROOT/'joint_repro/methods.py').read_text())
        for name in ['loss','opposite','target','draw','seed_eot']:
            fn=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
            self.assertEqual(fn(a),fn(b))

    def test_schedule_resume_does_not_restart_warmup(self):
        from transformers import AdamW,get_cosine_schedule_with_warmup
        def make():
            p=torch.tensor([.5],requires_grad=True);o=AdamW([p],lr=.002,weight_decay=0.,no_deprecation_warning=True)
            return p,o,get_cosine_schedule_with_warmup(o,20,2000)
        a,oa,sa=make()
        for _ in range(1000):a.square().sum().backward();oa.step();oa.zero_grad();sa.step()
        state=(a.detach().clone(),copy.deepcopy(oa.state_dict()),copy.deepcopy(sa.state_dict()))
        b,ob,sb=make();b.data.copy_(state[0]);ob.load_state_dict(state[1]);sb.load_state_dict(state[2])
        for _ in range(10):
            for p,o,s in [(a,oa,sa),(b,ob,sb)]:p.square().sum().backward();o.step();o.zero_grad();s.step()
        self.assertTrue(torch.equal(a,b));self.assertEqual(sa.state_dict(),sb.state_dict())

    def test_length_snapshot_identity(self):
        a=resolve('T0500-O03');b=resolve('T1000-O03')
        self.assertEqual(a['id'],b['id']);self.assertEqual(a['scheduler_outer'],b['scheduler_outer']);self.assertEqual(a['scheduler_outer'],2000)
        self.assertNotEqual(resolve('O03')['scheduler_outer'],a['scheduler_outer'])

    def test_snapshot_export_is_immutable(self):
        from joint_repro.train import export_checkpoint
        with patch('joint_repro.train.complete',return_value=True),patch('joint_repro.train.save_torch') as saver:
            export_checkpoint(resolve('LONG-B01'),42,500,{},{});saver.assert_not_called()

    def test_completed_snapshot_repairs_stage_without_loading_model(self):
        from joint_repro.train import train
        saved={'actual_optimizer_steps':25000,'sample_backprops':25000}
        with patch('joint_repro.train.validate'),patch('joint_repro.train.complete',side_effect=[False,True]),patch('joint_repro.train.read',return_value=saved),patch('joint_repro.train.finish') as finish,patch('joint_repro.train.load_wrapper') as load:
            train('W025-O03',42,500,None)
            finish.assert_called_once();load.assert_not_called()

    def test_holdout_is_new_and_paired(self):
        from joint_repro.evaluate import cases
        a=cases('holdout');b=cases('development')+cases('confirmation_extra')
        self.assertEqual(len(a),80);self.assertEqual({x['state'] for x in a},set(range(20,30)))
        self.assertFalse({x['case_id'] for x in a}&{x['case_id'] for x in b})

    def test_resource_cap_and_prohibition(self):
        from joint_repro.controller import active_jobs,gpu_job_limit,select_gpu
        self.assertEqual(gpu_job_limit(),4)
        sample={'probes':{'queue':{'returncode':0,'stdout':'1|JR04S-W025-O03|RUNNING|0:01|n|gres:1\n2|JR04A-D07|PENDING|0:0|Priority|gres:1'},'nodes':{'returncode':0,'stdout':'NodeName=n State=IDLE Gres=gpu:pro6000:2 CPUTot=64 CPUAlloc=0 RealMemory=220000 AllocMem=0 AllocTRES='}}}
        self.assertEqual(len(active_jobs(sample)),2);self.assertEqual(select_gpu(sample)['type'],'3090')

    def test_core_workers_precede_ready_extensions(self):
        from joint_repro.controller import ready_units
        workers=[dict(unit='extension',storage_id='extra',stop_outer=2000,eval_ids=['extra'],gate='length2000',seed=42),dict(unit='core',storage_id='core',stop_outer=500,eval_ids=['core'],gate='initial',seed=42)]
        with patch('joint_repro.controller.gates',return_value={'length2000':{'activate':True}}),patch('joint_repro.sweep.study',return_value={'workers':workers}),patch('joint_repro.sweep.resolve',return_value={'batch':1}),patch('joint_repro.controller.result_ready',return_value=False),patch('joint_repro.controller.screening_complete',return_value=False):
            ready=ready_units();self.assertTrue(ready[0][0].startswith('core-'))

    def test_session_counter_audit_rejects_wrong_batch_budget(self):
        from joint_repro.session_review import check_state
        state=dict(steps=100,outer=2,sample_backprops=200,patch=torch.ones(3,2,2)*.5,optimizer={'state':{0:{'step':100}}})
        check_state(state,2)
        with self.assertRaises(RuntimeError):check_state(state,1)
        state['optimizer']['state'][0]['step']=99
        with self.assertRaises(RuntimeError):check_state(state,2)

    def test_seed_mean_uses_macro_not_pooled_casr(self):
        from joint_repro.sweep_report import aggregate_seeds
        suites=['libero_spatial','libero_object','libero_goal','libero_10']
        rows=[dict(identity='trial',seed=seed,count=80,clean_success=71,success=success,macro_casr=macro,pooled_casr=.6,by_suite=[dict(suite=s,casr=macro) for s in suites]) for seed,success,macro in [(42,8,.7),(43,16,.8),(44,24,.9)]]
        value=aggregate_seeds(rows)[0]
        self.assertAlmostEqual(value['mean_macro_casr'],.8)
        self.assertAlmostEqual(value['mean_task_success_rate'],.2)
        self.assertAlmostEqual(value['std_macro_casr'],.1)
        self.assertEqual(value['starts_per_seed'],80)
        self.assertEqual(value['clean_success_starts'],71)

    def test_zero_aux_matching_ignores_only_beta(self):
        from joint_repro.controller import matched_reference
        self.assertEqual(matched_reference('W400-O03',['B01','B04']),'B01')
        self.assertEqual(matched_reference('T1000-D07',['T0500-B04','T1000-B04']),'T1000-B04')
        with self.assertRaises(RuntimeError):matched_reference('LR001-O03',['B01'])


if __name__=='__main__':unittest.main()
