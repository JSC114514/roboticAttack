"""Installed camera scheduler, synthetic physics: reproduce the original 2ms drift."""
import importlib.util
from pathlib import Path
import sysconfig
from types import SimpleNamespace
import numpy as np
import pytest
import atlas_labels
import atlas_sampling
import pilot_render as render


def setup(monkeypatch):
    path=Path(sysconfig.get_path('purelib'))/'robosuite/utils/observables.py'
    spec=importlib.util.spec_from_file_location('geometry_native_observable_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    env=SimpleNamespace(state=0,sim=SimpleNamespace(data=SimpleNamespace(time=0.,body_xpos=np.zeros((1,3)))))
    env.env=SimpleNamespace(_observables={})
    sensors={}
    for name in ['agentview_image','robot0_eye_in_hand_image']+list(atlas_sampling.PROPRIO):
        image=name.endswith('_image')
        @module.sensor(modality='image' if image else 'proprio')
        def sensor(cache,image=image):
            return np.full((224,224,3),env.state,np.uint8) if image else np.array([env.state],dtype=float)
        sensors[name]=sensor
        env.env._observables[name]=module.Observable(name=name,sensor=sensor,sampling_rate=20)
    truth={key:0. for key in ('object_eef_relative_x','object_eef_relative_y','object_eef_relative_z','object_eef_distance')}
    monkeypatch.setattr(atlas_labels,'entity_map',lambda e:[{'name':'bowl','geom_ids':[0],'body_id':0}])
    monkeypatch.setattr(atlas_labels,'physical_labels',lambda *a:truth)
    monkeypatch.setattr(atlas_sampling,'sampled_proprio',lambda raw:raw['robot0_eef_pos'])
    monkeypatch.setattr(render,'fingerprint',lambda e:{'state':e.state,'time':e.sim.data.time,'clocks':render.clocks(e.env)})
    def fake_render(*a):
        return {'rgb':np.full((224,224,3),env.state,np.uint8)}, {'subject_visible':True}
    monkeypatch.setattr(render,'render_view',fake_render)
    def step():
        cache={}
        for _ in range(25):
            env.state+=1; env.sim.data.time+=.002
            for observable in env.env._observables.values(): observable.update(.002,cache)
        return {name:observable.obs for name,observable in env.env._observables.items()}
    row={'truth':truth,'conditions':[{'condition_id':'shift','translation_m':[.01,0,0]}]}
    return env,step,sensors,row


def test_counterfactual_capture_follows_cached_sample_and_leaves_all_observables_unchanged(monkeypatch):
    reference,step,_,_=setup(monkeypatch); expected=[step(),step()]
    env,step,sensors,row=setup(monkeypatch)
    with render.CounterfactualCamera(env,{'manipulated_object':'bowl'}) as sampler:
        for expected_step in expected:
            sampler.arm(row); actual=step()
            assert all(np.array_equal(actual[k],expected_step[k]) for k in actual)
            assert np.array_equal(sampler.latest['pixels'],actual['robot0_eye_in_hand_image'][::-1,::-1])
            assert int(sampler.latest['pixels'][0,0,0])!=env.state
            assert sampler.latest['sample_time_s']<env.sim.data.time
    assert all(env.env._observables[name]._sensor is sensor for name,sensor in sensors.items())
    assert env.state==reference.state and env.sim.data.time==reference.sim.data.time


def test_counterfactual_camera_restores_sensor_after_render_error(monkeypatch):
    env,step,sensors,row=setup(monkeypatch)
    def fail(*a): raise RuntimeError('render failed')
    monkeypatch.setattr(render,'render_view',fail)
    with pytest.raises(RuntimeError,match='render failed'):
        with render.CounterfactualCamera(env,{'manipulated_object':'bowl'}) as sampler:
            sampler.arm(row); step()
    assert all(env.env._observables[name]._sensor is sensor for name,sensor in sensors.items())


def test_counterfactual_capture_rejects_unexpected_physics_mutation(monkeypatch):
    env,step,sensors,row=setup(monkeypatch)
    def mutating(*a):
        env.state+=1
        return {'rgb':np.full((224,224,3),env.state-1,np.uint8)},{}
    monkeypatch.setattr(render,'render_view',mutating)
    with pytest.raises(RuntimeError,match='changed physics'):
        with render.CounterfactualCamera(env,{'manipulated_object':'bowl'}) as sampler:
            sampler.arm(row); step()
    assert env.env._observables['robot0_eye_in_hand_image']._sensor is sensors['robot0_eye_in_hand_image']
