# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""CPU tests of scientific identity, branch scope, video and scheduling rules."""
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from contracts import paste_wrist,paired_noise,action_chunk
from common import ROOT

class Contracts(unittest.TestCase):
    def obs(self):return dict(full_image=np.zeros((224,224,3),np.uint8),wrist_image=np.ones((224,224,3),np.uint8),state=np.zeros(8))
    def test_paste_only_wrist(self):
        o=self.obs();p=np.full((50,50,3),200,np.uint8);a=paste_wrist(o,p)
        self.assertIs(a['full_image'],o['full_image']);self.assertIs(a['state'],o['state']);self.assertEqual(int((a['wrist_image']!=o['wrist_image']).sum()),50*50*3);self.assertEqual(int(o['wrist_image'].max()),1)
    def test_zero_identity(self):
        o=self.obs();a=paste_wrist(o,None)
        for k in o:self.assertTrue(np.array_equal(o[k],a[k]))
    def test_no_placement_search(self):
        with self.assertRaises(RuntimeError):paste_wrist(self.obs(),np.zeros((50,50,3),np.uint8),(0,0))
    def test_noise_pairs_conditions(self):
        cfg=dict(noise_seed=20261031,state_indices=list(range(20,30)),models={'pi05':dict(action_horizon=10,model_action_dim=32)})
        _,a=paired_noise(1,20,0,'libero_goal',cfg);_,b=paired_noise(1,20,0,'libero_goal',cfg);_,c=paired_noise(1,20,1,'libero_goal',cfg)
        self.assertTrue(np.array_equal(a,b));self.assertFalse(np.array_equal(a,c));self.assertEqual(a.shape,(10,32))
    def test_native_shape(self):
        self.assertEqual(action_chunk(np.zeros((8,7)),8).shape,(8,7))
        with self.assertRaises(RuntimeError):action_chunk(np.zeros((10,7)),8)
    def test_frozen_selection_and_grid(self):
        from common import frozen
        a=frozen();self.assertEqual(len(a['config']['patches']),15);self.assertEqual(len(a['cases']),80);self.assertEqual(len(a['replays']),312)
        self.assertEqual({p['seed'] for p in a['config']['patches']},{42,43,44})
        self.assertEqual({p['method'] for p in a['config']['patches']},{'INITIAL','B04','LR004-B01','LR004-O03','W025-D07'})
    def test_native_policy_predict_unchanged(self):
        old=ROOT.parents[2]/'cross_model_patch_interface_20261013_v4/code/policies.py'
        def trees(p):
            t=ast.parse(p.read_text());return {n.name+'.'+m.name:ast.dump(m) for n in t.body if isinstance(n,ast.ClassDef) for m in n.body if isinstance(m,ast.FunctionDef)}
        a,b=trees(old),trees(ROOT/'code/policies.py')
        for name in ['VLAAdapterPolicy.prepare','VLAAdapterPolicy.predict','VLAAdapterPolicy.environment_actions','Pi05Policy.prepare','Pi05Policy.predict','Pi05Policy.environment_actions']:self.assertEqual(a[name],b[name])
    def test_four_cap_and_no_pro6000(self):
        from controller import card
        data=dict(probes={'nodes':dict(returncode=0,stdout='NodeName=x State=IDLE Gres=gpu:pro6000:2 CPUTot=64 CPUAlloc=0 RealMemory=220000 AllocMem=0 AllocTRES=')})
        self.assertEqual(card(data,12)['type'],'3090')
    def test_resource_sampler_during_module_import(self):
        import common,types
        from unittest.mock import patch
        with patch.dict(sys.modules,{'torch':types.ModuleType('torch'),'jax':types.ModuleType('jax')}):
            self.assertIn('rss_hwm_bytes',common.resources())
    def test_unit_receipt_keeps_suite_identity(self):
        """The final condition label must never replace the Slurm unit name."""
        import worker, types
        from unittest.mock import patch, MagicMock
        graphics=types.ModuleType('graphics_runtime')
        graphics.prepare_graphics_backend=lambda:None
        graphics.make_env=lambda *args:(object(),object(),object())
        graphics.close_environment=lambda env:None
        policies=types.ModuleType('policies');policies.load_policy=lambda *args:object()
        config=dict(libero_root='/unused',patches=[dict(id='last-png')])
        sample=dict(config=config,assets={},cases=[dict(suite='libero_10',task_id=9,seed=42,case_id='case')])
        with patch.dict(sys.modules,graphics_runtime=graphics,policies=policies),patch.dict(worker.os.environ,SLURM_JOB_ID='test'),patch.object(worker,'frozen',return_value=sample),patch.object(worker,'complete',return_value=False),patch.object(worker,'Monitor',return_value=MagicMock()),patch.object(worker,'run_episode'),patch.object(worker,'done') as receipt:
            worker.run('test_model','libero_10','transfer')
        self.assertEqual(receipt.call_args.args[0],ROOT/'reports/units/test_model-libero_10-transfer.json')

if __name__=='__main__':unittest.main()
