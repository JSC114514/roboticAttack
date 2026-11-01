# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU scientific-contract checks; no model, dataset rebuild or GPU allocation."""
import unittest
import numpy as np
import torch
from joint_repro.bank import anchors
from joint_repro.methods import spec,audit_differences,draw,target,loss
from joint_repro.common import config


class Contracts(unittest.TestCase):
    def test_all_pairs_change_only_declared_factor(self):
        self.assertEqual(len(audit_differences()),12)
    def test_common_target_changes_no_mask_or_reference(self):
        clean=torch.zeros(1,8,7);clean[...,1]=.8
        out=target(clean,spec('B03'))
        self.assertTrue(torch.equal(out[...,:1],torch.full((1,8,1),-1.)))
        self.assertTrue(torch.equal(out[...,6:],torch.ones(1,8,1)))
        self.assertTrue(torch.equal(target(clean,spec('B05')),torch.zeros_like(clean)))
    def test_worst_slot_penalizes_largest_not_best_slot(self):
        action=torch.zeros(1,8,7);action[0,0,0]=2;teacher=torch.zeros_like(action)
        mean,_=loss(action,teacher,spec('B01'));worst,_=loss(action,teacher,spec('O01'))
        self.assertEqual(float(mean),.5);self.assertEqual(float(worst),4.)
    def test_task_balance_independent_of_state_weights(self):
        rows=[]
        for suite in config()['suites']:
            for task in config()['splits'][suite]['train']:
                rows.extend([dict(suite=suite,task_id=task,j=j) for j in range(8)])
        plain=[rows[draw(rows,i,42)] for i in range(1,25)]
        weighted=[rows[draw(rows,i,42,[j+1 for j in range(len(rows))])] for i in range(1,25)]
        self.assertEqual(len({(r['suite'],r['task_id']) for r in plain}),24)
        self.assertEqual([(r['suite'],r['task_id']) for r in plain],[(r['suite'],r['task_id']) for r in weighted])
    def test_temporal_states_are_unique_and_have_real_future(self):
        for n in [15,16,100,300]:
            a=anchors(n);self.assertEqual(len(set(a)),8);self.assertLessEqual(int(a.max())+8,n)
    def test_target_reference_detached_but_prediction_has_gradient(self):
        clean=torch.zeros(1,8,7,requires_grad=True);prediction=torch.zeros(1,8,7,requires_grad=True)
        value,_=loss(prediction,target(clean,spec('B01')),spec('B01'));value.backward()
        self.assertIsNone(clean.grad);self.assertGreater(float(prediction.grad.norm()),0)


class SchedulerContracts(unittest.TestCase):
    def test_forbidden_gpu_is_never_selected(self):
        from joint_repro.controller import select_gpu
        text='NodeName=forbidden State=IDLE Gres=gpu:pro6000:8 CPUTot=64 CPUAlloc=0 RealMemory=250000 AllocMem=0 AllocTRES=\nNodeName=allowed State=IDLE Gres=gpu:3090:4 CPUTot=32 CPUAlloc=0 RealMemory=192000 AllocMem=0 AllocTRES='
        sample={'probes':{'nodes':{'returncode':0,'stdout':text}}}
        self.assertEqual(select_gpu(sample)['type'],'3090')
    def test_no_free_card_defaults_to_single_3090_queue(self):
        from joint_repro.controller import select_gpu
        sample={'probes':{'nodes':{'returncode':0,'stdout':'NodeName=busy State=MIXED Gres=gpu:4090:8 CPUTot=56 CPUAlloc=40 RealMemory=180000 AllocMem=100000 AllocTRES=cpu=40,gres/gpu=8'}}}
        chosen=select_gpu(sample);self.assertEqual(chosen['type'],'3090');self.assertEqual(chosen['free'],0)
    def test_affinity_source_function_keeps_current_feature_gradients(self):
        import sys,types
        from pathlib import Path
        from joint_repro.common import module,require
        previous=sys.modules.get('common');stub=types.ModuleType('common');stub.require=require;sys.modules['common']=stub
        try:source=module('jr04_test_affinity',Path(config()['project'])/'research_runs/wrist_dino_spatial_b12_diag500_v1/code/new_objectives.py')
        finally:
            if previous is None:sys.modules.pop('common',None)
            else:sys.modules['common']=previous
        torch.manual_seed(42);clean=torch.randn(1,256,32);current=torch.randn(1,256,32,requires_grad=True)
        value,_=source.affinity_objective(clean,current,torch.zeros(256),topk=8);value.backward()
        self.assertTrue(torch.isfinite(current.grad).all());self.assertGreater(float(current.grad.norm()),0)


if __name__=='__main__':unittest.main()
