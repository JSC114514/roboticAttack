import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import rgb_common as c
import rgb_cache as cache
import rgb_train as train
import rgb_evaluate as evaluate
import rgb_cli as cli

class RGBTests(unittest.TestCase):
    def test_split_and_budget(self):
        pools=c.selections()
        self.assertEqual([len(pools[k]) for k in ('fit','verify')],[240,240])
        self.assertFalse({r['episode_id'] for r in pools['fit']}&{r['episode_id'] for r in pools['verify']})
        for rows in pools.values():
            self.assertEqual(len({r['episode_id'] for r in rows}),48)
            self.assertEqual({r['frame_rank'] for r in rows},set(range(5)))
            self.assertEqual(len({(r['suite'],r['task_id']) for r in rows}),24)
        self.assertEqual(c.config()['visits']*240,4800)
        self.assertFalse(c.config()['eot']);self.assertFalse(c.config()['action_loss'])

    def test_cache_workers_disjoint_and_complete(self):
        tasks=c.tasks();self.assertEqual(len(tasks),24)
        self.assertFalse(set(tasks[::2])&set(tasks[1::2]))
        self.assertEqual(set(tasks[::2])|set(tasks[1::2]),set(tasks))

    def test_full_loss_decomposition_and_finite_gradient(self):
        generator=torch.Generator().manual_seed(42)
        minus=torch.randn(256,8,generator=generator)
        plus=minus+torch.randn(256,8,generator=generator)
        feature=torch.randn(256,8,generator=generator,requires_grad=True)
        full,metrics=train.objective(feature,minus,plus,'full')
        projection,_=train.objective(feature,minus,plus,'projection')
        self.assertAlmostEqual(float(full),float(projection)+metrics['orthogonal'],places=5)
        direction=plus-minus;energy=direction.square().mean(-1);weights=energy/energy.sum()
        exact=(weights*(feature-plus).square().mean(-1)).sum()/(weights*energy).sum()
        self.assertTrue(torch.allclose(full,exact,atol=1e-5))
        full.backward();self.assertTrue(torch.isfinite(feature.grad).all());self.assertGreater(float(feature.grad.norm()),0)

    def test_zero_feature_target_explicit(self):
        x=torch.zeros(256,4)
        self.assertIsNone(train.objective(x,x,x,'full'))

    def test_alignment_exposes_pixel_and_state_errors(self):
        obs={'full_image':np.zeros((2,2,3),np.uint8),'wrist_image':np.zeros((2,2,3),np.uint8),'state':np.zeros(8)}
        other=copy.deepcopy(obs);other['wrist_image'][0,0,0]=255;other['state'][0]=.01
        values=cache.alignment(obs,other)
        self.assertEqual(values['full_image'],0);self.assertGreater(values['wrist_image'],1e-4)
        self.assertAlmostEqual(values['proprio_max_abs'],.01)

    def test_submission_graph_is_bounded_and_sequential(self):
        calls=[]
        def one(stage,indices,deps):calls.append((stage,indices,list(deps)));return stage+'_ID'
        with patch.object(cli,'cache_complete',return_value=False),patch.object(cli,'complete',return_value=False):cli.stages(one)
        self.assertEqual([v[0] for v in calls],['cache','train','evaluate','report'])
        self.assertEqual(calls[0][1],[0,1]);self.assertEqual(calls[1][1],[0,1])
        self.assertEqual(calls[1][2],['cache_ID']);self.assertEqual(calls[2][2],['train_ID'])
        self.assertIn('--array=0,1%2',cli.command('train',[0,1]))

    def test_complete_wave_skips_resources(self):
        with patch.object(cli,'cache_complete',return_value=True),patch.object(cli,'complete',return_value=True):
            calls=[];cli.stages(lambda *args:calls.append(args))
        self.assertEqual(calls,[])

    def test_worker_refuses_local_execution(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError):cli.worker('train')

    def test_resources_separate_by_stage(self):
        for stage,mem in [('cache','12G'),('train','16G'),('evaluate','12G')]:
            text=(c.ROOT/f'slurm/{stage}.sbatch').read_text()
            for value in ('--partition=rtx4090','--gres=gpu:4090:1','--cpus-per-task=4','--mem='+mem):self.assertIn(value,text)
        text=(c.ROOT/'slurm/report.sbatch').read_text();self.assertNotIn('--gres',text);self.assertIn('--mem=4G',text)

    def test_saved_action_replay_stops_at_last_selected_observation(self):
        self.mock_replay(False)

    def test_early_success_is_engineering_failure(self):
        self.mock_replay(True)

    def mock_replay(self,early):
        obs={'full_image':np.zeros((224,224,3),np.uint8),'wrist_image':np.zeros((224,224,3),np.uint8),'state':np.zeros(8)}
        selected=[{'state_id':f'frame_{i}','episode_id':'episode','suite':'suite','task_id':0,
            'state_index':40,'rollout_seed':17,'cohort':'fit','decision_block':i,'decision_count':6,'frame_rank':i-1,'observation':obs} for i in range(1,6)]
        traces=pd.DataFrame([{'suite':'suite','task_id':0,'state_index':40,'rollout_seed':17,'decision_block':i,
            'executed_slots':8,'step_end':(i+1)*8,'predicted_environment_action_chunk':np.ones((8,7)).tolist()} for i in range(6)])
        class Env:
            steps=0;closed=False
            def step(self,action):
                self.steps+=1
                return {},0,(early and self.steps==12),{}
        env=Env()
        runtime=SimpleNamespace(make_env=lambda *a,**k:(env,SimpleNamespace(name='task',language='test'),[None]*50),
            reset_wait=lambda *a:{},observation=lambda *a:copy.deepcopy(obs),close_environment=lambda e:setattr(e,'closed',True))
        class Camera:
            def __init__(self,env):self.key=None
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def arm(self,key):self.key=key
            def take(self,raw,key):
                if key!=self.key:raise AssertionError('stale sample')
                return {'images':{'plus':np.zeros((256,256,3),np.uint8),'minus':np.zeros((256,256,3),np.uint8)},'no_target':False}
        engine=SimpleNamespace(spatial_observation=lambda obs:torch.ones((256,4)),verify_patch_path=lambda row:None)
        dummy=SimpleNamespace(process_libero_action=lambda x:x)
        manifest={'trace_path':'unused','initial_states':{'suite/0':{'task_name':'task'}}}
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'VLAAttacker.multiview.openvla_oft_experiments':dummy}), \
             patch.object(cache,'ROOT',Path(folder)),patch.object(cache,'selections',return_value={'fit':selected,'verify':[]}), \
             patch.object(cache,'tasks',return_value=[('suite',0)]),patch.object(cache,'verify_inputs',return_value=manifest), \
             patch.object(cache,'complete',return_value=False),patch.object(cache,'cache_receipt',side_effect=lambda eid:Path(folder)/(eid+'.json')), \
             patch.object(cache,'make_engine',return_value=engine),patch.object(cache.native,'require_gpu'), \
             patch.object(cache.native,'runtime_module',return_value=runtime),patch.object(cache,'Camera',Camera), \
             patch.object(cache.pd,'read_parquet',return_value=traces),patch.object(cache.torch,'load',return_value={'records':selected}), \
             patch.object(cache,'finish'):
            if early:
                with self.assertRaises(RuntimeError):cache.run(0,SimpleNamespace(update=lambda **kw:None))
                self.assertEqual(env.steps,12)
            else:
                cache.run(0,SimpleNamespace(update=lambda **kw:None))
                self.assertEqual(env.steps,40)
                self.assertTrue((Path(folder)/'episode.pt').exists())
        self.assertTrue(env.closed)

    def test_native_png_episode_executes_only_patch_actions(self):
        self.mock_png_episode(None,10,[8,2])

    def test_native_png_episode_stops_mid_chunk(self):
        self.mock_png_episode(3,16,[3])

    def mock_png_episode(self,success_at,limit,expected):
        obs={'full_image':np.zeros((224,224,3),np.uint8),'wrist_image':np.zeros((224,224,3),np.uint8),'state':np.zeros(8)}
        class Env:
            def __init__(self):self.actions=[]
            def raw(self):return {'robot0_eef_pos':np.zeros(3),'robot0_gripper_qpos':np.zeros(2)}
            def step(self,action):
                self.actions.append(action);return self.raw(),0,len(self.actions)==success_at,{}
        env=Env();runtime=SimpleNamespace(reset_wait=lambda *a:env.raw(),observation=lambda *a:copy.deepcopy(obs))
        class Wrapper:
            def predict_action(self,current,language):return np.ones((8,7)) if current['wrist_image'].any() else np.zeros((8,7))
        dummy=SimpleNamespace(process_libero_action=lambda x:x)
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'VLAAttacker.multiview.openvla_oft_experiments':dummy}), \
             patch.object(evaluate,'ROOT',Path(folder)),patch.object(evaluate,'pairing_gate'), \
             patch.object(evaluate,'config',return_value={**c.config(),'max_steps':limit}):
            result,trace,images=evaluate.episode(env,SimpleNamespace(language='test'),None,17,np.ones((50,50,3),np.uint8),
                Wrapper(),runtime,SimpleNamespace(task_id=0,state_index=44,update=lambda **kw:None),Path(folder))
            self.assertEqual([x['executed_slots'] for x in trace],expected)
            self.assertTrue(all(np.asarray(action).all() for action in env.actions))
            self.assertEqual(result['success'],int(success_at is not None))
            self.assertTrue(all(np.asarray(r['same_observation_clean_actions']).sum()==0 for r in trace))

if __name__=='__main__':unittest.main()
