from pathlib import Path
import json
import numpy as np
import pytest
from atlas_common import config,parts_selection
from atlas_math import (angular_error,balanced_token_indices,cluster_interval,crop_map,grouped_folds,
                        patch_majority,permuted_targets,rgb_error,rgb_to_lab,binary_intervals,rotation_error)


def rows(tasks=3,episodes=6,states=5):
    return [{'task_id':t,'episode_id':f't{t}e{e}','step_id':s,'split':'discovery'}
            for t in range(tasks) for e in range(episodes) for s in range(states)]


def test_trajectory_folds_never_leak():
    data=rows(); folds=grouped_folds(data)
    assert set(folds)==set(range(5))
    for name in {r['episode_id'] for r in data}:
        assert len(set(folds[[i for i,r in enumerate(data) if r['episode_id']==name]]))==1


def test_small_manual_panel_still_has_five_episode_folds():
    assert set(grouped_folds(rows(10,2,1)))==set(range(5))


def test_permutation_preserves_within_episode_sequence():
    data=rows(); y=np.asarray([[int(r['episode_id'].split('e')[1])*100+r['step_id']] for r in data])
    p=permuted_targets(y,data,42)
    assert not np.array_equal(p,y)
    for start in range(0,len(data),5): assert np.array_equal(np.diff(p[start:start+5,0]),np.ones(4))


def test_ci_weights_tasks_and_episodes_not_frames():
    data=[{'task_id':0,'episode_id':'a'}]*100+[{'task_id':1,'episode_id':'b'}]
    ci=cluster_interval(np.r_[np.zeros(100),10.],data,300,42)
    assert ci['mean']==5 and ci['episodes']==2
    assert ci['ci95_low']<=5<=ci['ci95_high']


def test_ci_missing_is_not_zero_evidence():
    r=cluster_interval([np.nan],[{'task_id':0,'episode_id':'a'}],20)
    assert r['mean'] is None and r['tasks']==0


def test_native_crop_is_centered_and_label_interpolation_is_nearest():
    grid=np.tile(np.arange(224),(224,1)).astype(float)
    c=crop_map(grid)
    assert c[0,0]==pytest.approx((1-np.sqrt(.9))*223/2)
    labels=(grid>111).astype(np.int16)
    assert set(np.unique(crop_map(labels,True)))=={0,1}


def test_patch_labels_keep_boundary_confidence():
    a=np.zeros((224,224),int); a[:14,:7]=1
    label,confidence=patch_majority(a,2)
    assert confidence[0]==.5 and confidence[1]==1


def test_rgb_rms_allows_tiny_hash_difference_but_rejects_large_drift():
    a=np.zeros((224,224,3),np.uint8); b=a.copy(); b[0,0,0]=1
    assert 0<rgb_error(a,b)['normalized_rms']<1e-4
    b[0,0,0]=255
    assert rgb_error(a,b)['normalized_rms']>1e-4


def test_lab_d65_white_black():
    lab=rgb_to_lab(np.array([[1.,1.,1.],[0.,0.,0.]]))
    assert lab[0,0]==pytest.approx(100,abs=1e-4)
    assert np.allclose(lab[0,1:],0,atol=1e-3)
    assert np.allclose(lab[1],0,atol=1e-8)


def test_symmetric_normal_angle():
    assert angular_error([[0,0,1]],[[0,0,5]])[0]==0
    assert angular_error([[0,0,1]],[[1,0,0]])[0]==90
    assert np.isnan(angular_error([[0,0,1]],[[0,0,0]])[0])


def test_complete_end_effector_orientation_error():
    z=np.array([[0,0,1.]]); x=np.array([[1.,0,0]]); yp=np.array([[0,1.,0]])
    assert rotation_error(z,x,z,x)[0]==0
    assert rotation_error(z,x,z,yp)[0]==90


def test_binary_intervals_respect_ties_and_cluster_sampling():
    data=rows(2,4,5); y=np.arange(len(data))%2
    for p,expected in [(y,1.),(np.full(len(y),.5),.5)]:
        r=binary_intervals(y,p,data,200)
        assert r['auroc_cluster_mean']==pytest.approx(expected)
        assert r['auroc_ci95_low']==pytest.approx(expected)
        assert r['auroc_ci95_high']==pytest.approx(expected)


def test_balanced_tokens_dont_turn_unknown_into_background():
    y=np.r_[np.zeros(200),np.ones(30),np.full(26,-1)]
    ids=balanced_token_indices(y,32,42)
    assert len(ids)==32 and (y[ids]>=0).all() and (y[ids]==1).sum()==16


def test_parts_selection_fixed_80_no_visibility_or_probe_selection():
    data=[]
    for t in range(10):
        for split in ('discovery','validation'):
            for e in range(12):
                for s in range(5):
                    data.append({'task_id':t,'split':split,'episode_id':f'{split}_{t}_{e:02d}',
                                 'state_id':f'{split}_{t}_{e}_{s}','step_id':s,'normalized_progress':s/4})
    chosen=parts_selection(data)
    assert len(chosen)==80 and len({(r['state_id'],r['view']) for r in chosen})==80
    assert all(r['state_id'].endswith('_2') for r in chosen)


def test_resource_requests_match_config():
    root=Path(__file__).parents[1]; c=config()['resources']
    for stage,script in [('capture','01_capture.sbatch'),('analyze','02_analyze.sbatch')]:
        text=(root/'slurm'/script).read_text()
        assert f"--mem={c[stage]['host_memory_gib']}G" in text
        assert f"--cpus-per-task={c[stage]['cpus']}" in text
        assert ('--gres=gpu:4090:1' in text)==bool(c[stage]['gpus'])


def test_workspace_table_and_gripper_are_not_background():
    from types import SimpleNamespace
    from atlas_labels import entity_map
    class Model:
        names=['bowl_geom','plate_geom','table_visual','table_collision','table_leg1_visual','arm_geom','finger_geom']
        ngeom=len(names)
        def geom_id2name(self,i): return self.names[i]
        def geom_name2id(self,n): return self.names.index(n)
        def body_name2id(self,n): return {'bowl':1,'plate':2,'table':3}[n]
    def obj(n,body=None): return SimpleNamespace(contact_geoms=[n],visual_geoms=[n],root_body=body)
    inner=SimpleNamespace(objects_dict={'akita_black_bowl_1':obj('bowl_geom','bowl'),'plate_1':obj('plate_geom','plate')},
                          fixtures_dict={},robots=[SimpleNamespace(robot_model=obj('arm_geom'),gripper=obj('finger_geom'))])
    entities=entity_map(SimpleNamespace(env=inner,sim=SimpleNamespace(model=Model())))
    table=next(e for e in entities if e['name']=='main_table')
    assert table['geom_ids']==[2,3,4] and table['source']=='arena_workspace'
    assert next(e for e in entities if e['name']=='gripper')['geom_ids']==[6]
