from types import SimpleNamespace
import numpy as np
import pytest
from pilot_render import SceneTranslation,clocks


@pytest.fixture
def scene():
    return SimpleNamespace(ngeom=3,geoms=[
        SimpleNamespace(objtype=5,objid=8,pos=np.array([1.,2.,3.])),
        SimpleNamespace(objtype=5,objid=9,pos=np.array([4.,5.,6.])),
        SimpleNamespace(objtype=4,objid=8,pos=np.array([7.,8.,9.]))])


@pytest.mark.parametrize('raises',[False,True])
def test_scene_translation_restores_target_and_renderer_on_all_exits(scene,raises):
    before=[g.pos.copy() for g in scene.geoms]
    def original(**kwargs):
        assert np.allclose(scene.geoms[0].pos,before[0]+[.01,0,0])
        assert np.array_equal(scene.geoms[1].pos,before[1])
        assert np.array_equal(scene.geoms[2].pos,before[2])
        if raises: raise RuntimeError('renderer failure')
        return 'rendered'
    module=SimpleNamespace(mjr_render=original)
    if raises:
        with pytest.raises(RuntimeError,match='renderer failure'):
            with SceneTranslation(scene,[8],[.01,0,0],module): module.mjr_render(scn=scene)
    else:
        with SceneTranslation(scene,[8],[.01,0,0],module) as hook:
            assert module.mjr_render(scn=scene)=='rendered'
        assert hook.calls==1 and hook.matched==[1]
    assert module.mjr_render is original
    assert all(np.array_equal(g.pos,p) for g,p in zip(scene.geoms,before))


def test_unowned_scene_is_never_changed(scene):
    before=[g.pos.copy() for g in scene.geoms]
    other=SimpleNamespace(ngeom=0,geoms=[])
    module=SimpleNamespace(mjr_render=lambda **kwargs:'other')
    with SceneTranslation(scene,[8],[.01,0,0],module) as hook:
        assert module.mjr_render(scn=other)=='other'
    assert hook.calls==0
    assert all(np.array_equal(g.pos,p) for g,p in zip(scene.geoms,before))


def test_missing_render_geom_is_a_failure_not_a_silent_null(scene):
    module=SimpleNamespace(mjr_render=lambda **kwargs:None)
    with pytest.raises(RuntimeError,match='Target geom'):
        with SceneTranslation(scene,[999],[.01,0,0],module): module.mjr_render(scn=scene)


def test_real_mujoco_scene_translation_needs_no_renderer_or_physics_update():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    scene=mujoco.MjvScene(model,maxgeom=20)
    mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),mujoco.MjvPerturb(),mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
    state=data.qpos.copy(); positions=data.xpos.copy(); original=[scene.geoms[i].pos.copy() for i in range(scene.ngeom)]
    module=SimpleNamespace(mjr_render=lambda **kwargs:None)
    with SceneTranslation(scene,[0],[.02,0,0],module): module.mjr_render(scn=scene)
    assert np.array_equal(data.qpos,state) and np.array_equal(data.xpos,positions)
    assert all(np.array_equal(scene.geoms[i].pos,p) for i,p in enumerate(original))


def test_clock_snapshot_does_not_invoke_sensors():
    obs=SimpleNamespace(_sensor=lambda cache:(_ for _ in ()).throw(AssertionError('called')),
                        _time_since_last_sample=.002,_sampled=True,_obs=np.zeros((224,224,3)))
    value=clocks(SimpleNamespace(_observables={'wrist':obs}))
    assert value['wrist']['_time_since_last_sample']==.002 and '_obs' not in value['wrist']
