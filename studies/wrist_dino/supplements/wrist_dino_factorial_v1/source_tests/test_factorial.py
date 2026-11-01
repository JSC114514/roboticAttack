# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU tests for the one-factor ablation, loss gradients and frozen reuse."""
import ast
import unittest
import torch
from unittest.mock import patch
from joint_repro.methods import BASE,spec,loss,target
from joint_repro.common import ROOT,read,sha
from joint_repro.supplement import validate,ready


class Factorial(unittest.TestCase):
    def test_only_missing_cell_changes(self):
        b,d,o=spec('B04'),spec('D07'),spec('O03')
        self.assertEqual([k for k in BASE if b[k]!=d[k]],['auxiliary'])
        self.assertEqual([k for k in BASE if o[k]!=d[k]],['mask'])
        self.assertEqual(d['mask'],list(range(7)))

    def test_exact_additive_loss_and_both_gradients(self):
        torch.manual_seed(42)
        action=torch.randn(1,8,7,requires_grad=True)
        feature=torch.randn(1,256,12,requires_grad=True)
        clean=torch.randn_like(feature,requires_grad=True)
        teacher=target(torch.randn_like(action),spec('D07'))
        features={'dino':[None,feature]};donor={'dino':[None,clean]}
        lam=read(ROOT/'configs/ABLATION.json')['auxiliary_lambda']
        main,_=loss(action,teacher,spec('B04'))
        total,_=loss(action,teacher,spec('D07'),features,donor,lam)
        expected=main+lam*(torch.nn.functional.cosine_similarity(feature,clean.detach(),dim=-1).mean()-1)
        torch.testing.assert_close(total,expected,atol=0,rtol=0)
        total.backward()
        self.assertTrue(torch.isfinite(action.grad).all());self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(float(action.grad.norm()),0);self.assertGreater(float(feature.grad.norm()),0)
        self.assertIsNone(clean.grad)

    def test_reused_parent_and_unchanged_training_pipeline(self):
        lock=validate();self.assertEqual(lock['new_training']['optimizer_steps_per_seed'],25000)
        self.assertEqual(lock['auxiliary_lambda'],22.855948214853818)

    def test_ready_has_three_seeds_and_one_diagnostic(self):
        with patch('joint_repro.supplement.complete',return_value=False):
            rows=ready()
        self.assertEqual([r[0] for r in rows],['D07-s42-train','D07-s43-train','D07-s44-train','M04-restore-O03'])

    def test_full_follows_development_without_result_gate(self):
        def done(p):return str(p).endswith(('outputs/training/D07-seed42/COMPLETE.json','D07-seed42-development_COMPLETE.json'))
        with patch('joint_repro.supplement.complete',side_effect=done):rows=ready()
        self.assertEqual(rows[0][0],'D07-s42-full')

    def test_no_fifth_gpu_and_no_pro6000(self):
        from joint_repro import controller
        with patch.object(controller,'audit',return_value={}),patch.object(controller,'active_jobs',return_value=[{}]*4),patch.object(controller,'command') as call:
            with self.assertRaisesRegex(RuntimeError,'limit'):controller.submit('bad',['train'])
            call.assert_not_called()
        sample={'probes':{'nodes':{'returncode':0,'stdout':'NodeName=bad State=IDLE Gres=gpu:pro6000:4 CPUTot=64 CPUAlloc=0 RealMemory=220000 AllocMem=0 AllocTRES='}}}
        self.assertEqual(controller.select_gpu(sample)['type'],'3090')


if __name__=='__main__':unittest.main()
