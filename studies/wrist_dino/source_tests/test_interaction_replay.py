# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""Saved-command metadata repair: alignment and preserved original evidence."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from joint_repro import interaction_replay as repair
from joint_repro import common


class SavedCommandReplay(unittest.TestCase):
    def attempt(self,mismatch):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'configs').mkdir();(root/'configs/PROTOCOL.json').write_text('{}')
            folder=root/'outputs/evaluation/development/B01-seed42/case';folder.mkdir(parents=True)
            original=dict(seed=42,suite='libero_spatial',task_id=3,state=0,case_id='case',condition='B01-seed42')
            (folder/'RESULT.json').write_text(json.dumps(original))
            steps=[dict(step=i+1,decision=0,slot=i,command=[1.]*7,eef_before=[float(i),0.,0.],eef_after=[float(i+1),0.,0.],task_success=i==1) for i in range(2)]
            if mismatch:steps[0]['eef_after'][0]=100.
            (folder/'STEPS.json').write_text(json.dumps(steps));(folder/'DECISIONS.json').write_text('[{"step_start":0,"decision":0}]')
            reference=root/'data/paired_starts/development/case';reference.mkdir(parents=True);(reference/'START.json').write_text('{}')
            originals={p.name:common.sha(p) for p in folder.iterdir()};counter=types.SimpleNamespace(step=0,closed=False)
            def raw():return dict(robot0_eef_pos=np.asarray([float(counter.step),0.,0.]))
            def step(command):
                self.assertEqual(command,[1.]*7);counter.step+=1
                return raw(),0.,counter.step==2,{}
            env=types.SimpleNamespace(step=step,sim=types.SimpleNamespace(forward=lambda:None))
            runtime=types.SimpleNamespace(make_env=lambda *a:(env,None,[None]),reset_wait=lambda *a:raw(),close_environment=lambda e:setattr(counter,'closed',True))
            monitor=types.SimpleNamespace(update=lambda **kw:None)
            with patch.object(repair,'ROOT',root),patch.object(common,'ROOT',root),patch.object(common,'source_hashes',lambda:{}),patch.object(repair,'runtime_config',lambda:{}),patch.object(repair,'fingerprint',lambda e:{}),patch.object(repair,'target_geometry',lambda *a:lambda:dict(step=counter.step,contact=counter.step==1,grasp=False)):
                if mismatch:
                    with self.assertRaisesRegex(RuntimeError,'EEF-after mismatch'):repair.repair_one(runtime,'development',folder,monitor)
                    self.assertFalse((folder/'INTERACTION_REPLAY_COMPLETE.json').exists())
                    self.assertEqual(len(list(folder.glob('INTERACTION_REPLAY_FAILURE-*.json'))),1)
                else:
                    result=repair.repair_one(runtime,'development',folder,monitor)
                    self.assertEqual(result['status'],'MATCHED_REPLAY');self.assertEqual(result['model_calls'],0)
                    self.assertEqual(result['decisions'][0]['truth']['step'],0)
                    self.assertEqual([r['truth']['step'] for r in result['steps']],[1,2])
                    self.assertTrue(common.complete(folder/'INTERACTION_REPLAY_COMPLETE.json'))
            self.assertTrue(counter.closed)
            for name,value in originals.items():self.assertEqual(common.sha(folder/name),value)

    def test_matched_commands_record_correct_times_without_rewriting_outcomes(self):self.attempt(False)
    def test_divergent_physics_refuses_completed_labels_and_preserves_failure(self):self.attempt(True)


if __name__=='__main__':unittest.main()
