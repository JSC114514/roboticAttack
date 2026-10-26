import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from wr_data import pair_anchors, native_expert_actions, trajectory_id, match_task, camera_members, select_episodes,_decode_chunk
from wr_camera import local_shift
from wr_camera import restore_sample_visuals
from wr_data import replay_alignment
from types import SimpleNamespace
import copy


class DataContracts(unittest.TestCase):
    def test_anchors_keep_both_future_chunks(self):
        a = pair_anchors(64)
        self.assertEqual(a.shape, (8, 2))
        self.assertTrue(np.all(a[:, 1]-a[:, 0] == 8))
        self.assertLessEqual(a[-1, 1]+8, 64)
        self.assertEqual(len(set(a[:, 0])), 8)
        with self.assertRaises(RuntimeError): pair_anchors(22)

    def test_native_gripper_and_normalization(self):
        raw = np.zeros((3, 7), np.float32); raw[:, 6] = [-1, 0, 1]
        stats = dict(q01=[-1]*6+[0], q99=[1]*7, mask=[True]*6+[False])
        native, normalized = native_expert_actions(raw, stats)
        np.testing.assert_array_equal(native[:, 6], [1, 1, 0])
        np.testing.assert_array_equal(normalized[:, 6], [1, 1, 0])
        np.testing.assert_array_equal(raw[:, 6], [-1, 0, 1])

    def test_id_uses_values_and_source(self):
        a, s = np.zeros((40, 7)), np.zeros((40, 8))
        h = trajectory_id('one', a, s)
        self.assertNotEqual(h, trajectory_id('two', a, s))
        s[0, 0] = 1
        self.assertNotEqual(h, trajectory_id('one', a, s))

    def test_task_source_exact(self):
        self.assertEqual(match_task('/a/pick_bowl_demo.hdf5', ['pick_bowl', 'open_drawer']), 0)
        with self.assertRaises(RuntimeError): match_task('unknown.hdf5', ['pick_bowl'])

    def test_camera_translation_local_coordinates(self):
        r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_allclose(local_shift([1, 2, 3], r, 'x', .01), [1, 2.01, 3])
        np.testing.assert_allclose(local_shift([1, 2, 3], r, 'z', .01), [1, 2, 2.99])

    def test_trajectory_split_no_overlap(self):
        rows = [dict(suite='s', task_id=0, episode_id=str(i), file_path=str(i), eligible=True) for i in range(14)]
        chosen, shortages = select_episodes(rows, {'s': [0]}, 8, 5)
        self.assertFalse(shortages)
        fit = {r['episode_id'] for r in chosen if r['split'] == 'train'}
        val = {r['episode_id'] for r in chosen if r['split'] == 'validation'}
        self.assertEqual((len(fit), len(val)), (8, 5)); self.assertFalse(fit & val)

    def test_real_index_membership(self):
        rows = camera_members('pilot')
        self.assertEqual((len(rows['train']), len(rows['validation'])), (80, 80))
        main = camera_members('main')
        self.assertEqual((len(main['train']), len(main['validation'])), (720, 240))

    def test_parquet_nested_action_chunk(self):
        nested=np.empty(8,dtype=object)
        for i in range(8):nested[i]=np.arange(7,dtype=np.float32)+i
        a=_decode_chunk(nested)
        self.assertEqual(a.shape,(8,7));np.testing.assert_array_equal(a[7],np.arange(7)+7)

    def replay_fixture(self):
        obs=dict(full_image=np.zeros((224,224,3),np.uint8),wrist_image=np.zeros((224,224,3),np.uint8),state=np.zeros(8,np.float32))
        fingerprint=dict(arrays={'sim_state':'same','cam_xpos':'same'},time=1.,observable_clocks={'wrist':{'time':1.}})
        original=dict(observation=copy.deepcopy(obs),sample_fingerprint=copy.deepcopy(fingerprint))
        sample=dict(fingerprint=copy.deepcopy(fingerprint),sample_time_s=1.)
        return obs,original,sample

    def test_cached_rgb_difference_audited_not_used_as_pair(self):
        obs,original,sample=self.replay_fixture();obs['wrist_image'][0,0,0]=1
        out=replay_alignment(obs,original,sample)
        self.assertFalse(out['cached_rgb']['wrist_image']['exact'])
        self.assertEqual(out['cached_rgb']['wrist_image']['changed_pixels'],1)
        self.assertEqual(out['cached_rgb']['wrist_image']['max_abs_gray'],1)
        self.assertEqual(out['pair_clean_source'],'current_native_replay')
        self.assertFalse(out['cached_rgb_used_in_pair'])
        self.assertEqual(original['observation']['wrist_image'][0,0,0],0)

    def test_exact_replay_state_timing_and_proprio_required(self):
        for field in ('arrays','time','clock','proprio'):
            obs,original,sample=self.replay_fixture()
            if field=='arrays':sample['fingerprint']['arrays']['cam_xpos']='different'
            elif field=='time':sample['sample_time_s']=2.
            elif field=='clock':sample['fingerprint']['observable_clocks']['wrist']['time']=2.
            else:obs['state'][0]=1.
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                replay_alignment(obs,original,sample)

    def test_visual_snapshot_is_copied_into_shadow_only(self):
        live=SimpleNamespace(site_rgba=np.array([[1.,0.,0.,0.]]))
        sample={'site_rgba':live.site_rgba.copy()}
        shadow=SimpleNamespace(site_rgba=live.site_rgba.copy())
        live.site_rgba[0,3]=1.
        restore_sample_visuals(shadow,sample)
        self.assertEqual(shadow.site_rgba[0,3],0.)
        self.assertEqual(live.site_rgba[0,3],1.)


if __name__ == '__main__': unittest.main()
