import copy
import numpy as np
import pytest
from pilot_common import config
from pilot_geometry import action_metrics,candidates,cohort,conditions


def truth(relative):
    return {**dict(zip(('object_eef_relative_x','object_eef_relative_y','object_eef_relative_z'),relative)),
            'object_eef_distance':float(np.linalg.norm(relative)),'contact':0.,'grasp':0.}


@pytest.mark.parametrize('relative',[(.03,.04,-.1),(.1,-.001,-.02),(-.008,-.009,.2),(.001,.002,0.)])
def test_controls_separate_range_from_bearing_and_respect_cap(relative):
    relative=np.array(relative); rows=conditions(truth(relative),config())
    assert len(rows)==8
    for row in rows:
        shifted=np.array(row['relative_position_m'])
        assert shifted[2]==relative[2]
        assert np.linalg.norm(shifted-relative)<=.03+1e-12
        if row['target'].startswith('direction'):
            assert np.linalg.norm(shifted)==pytest.approx(np.linalg.norm(relative),abs=1e-12)
            assert row['angle_degrees']*row['sign']>0
        else:
            assert abs(np.cross(shifted[:2],relative[:2]))<1e-12
            assert np.sign(np.linalg.norm(shifted)-np.linalg.norm(relative))==row['sign']


def test_zero_radius_is_recordable_engineering_failure():
    with pytest.raises(ValueError,match='Undefined'): conditions(truth((0,0,.1)),config())


def records():
    rows=[]
    for task in range(10):
        for split in ('discovery','validation'):
            for ep in range(2):
                for step,distance,contact in [(1,.1,0),(2,.05,1),(3,.08,0),(4,.08,0)]:
                    rows.append({'task_id':task,'split':split,'episode_id':f'{task}_{split}_{ep}',
                                 'step_id':step,'state_id':f'{task}_{split}_{ep}_{step}',
                                 'truth':{'contact':contact,'grasp':0,'object_eef_distance':distance},
                                 'visibility':{'wrist':{'subject_visible':True}},'attack_score':999-step})
    return rows


def test_cohort_uses_fixed_episodes_and_clean_eligibility_only():
    rows=records(); selected=cohort(rows)
    assert len(selected)==40 and len({r['episode_id'] for r in selected})==40
    assert all(r['step_id']==3 for r in selected)
    for row in rows: row['attack_score']=-row['attack_score']
    assert [r['state_id'] for r in cohort(rows)]==[r['state_id'] for r in selected]
    assert not {r['episode_id'] for r in selected if r['split']=='discovery'} & {r['episode_id'] for r in selected if r['split']=='validation'}


def test_missing_eligible_episode_does_not_silently_replace_it():
    rows=records()
    for row in rows:
        if row['episode_id']=='0_discovery_0': row['visibility']['wrist']['subject_visible']=False
    with pytest.raises(ValueError,match='No eligible'): cohort(rows)


def test_fixed_grid_is_16_shared_candidates_not_per_image_patches():
    grid=candidates(config())
    assert len(grid)==16 and len({r['candidate_id'] for r in grid})==16
    assert {r['seed'] for r in grid}=={42,43}
    assert not any('state_id' in row for row in grid)


def test_metric_is_next_action_and_point_five_is_not_open():
    clean=np.zeros((8,7)); clean[:,6]=.5
    action=clean.copy(); action[1:,:3]=999; action[0,:3]=[2,0,0]
    condition={'delta_unit':[1,0,0],'canonical_axis_unit':[1,0,0]}
    value=action_metrics(action,clean,condition)
    assert value['first_projection']==2 and value['first_open_flip']==0
    action[0,6]=.50001
    assert action_metrics(action,clean,condition)['first_open_flip']==1
