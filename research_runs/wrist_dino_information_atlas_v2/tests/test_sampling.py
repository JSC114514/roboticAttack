"""Reproduce cached RGB vs end-of-step drift using the installed Observable scheduler.

The physics/rendering data is a deterministic surrogate. No EGL context, VLA,
CUDA allocation or Slurm call is made by these tests.
"""
import importlib.util
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace
import numpy as np
import pytest
import atlas_sampling as sampling


def native_observable():
    path=Path(sysconfig.get_path('purelib'))/'robosuite/utils/observables.py'
    spec=importlib.util.spec_from_file_location('atlas_native_observables_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def setup(monkeypatch,initial_updates=0):
    module=native_observable()
    env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=0.)),state=0)
    env.env=SimpleNamespace(_observables={}); env.get_sim_state=lambda:np.array([env.state])
    image=lambda:np.full((224,224,3),env.state,np.uint8)
    sensors={}
    for view,camera in sampling.VIEWS.items():
        @module.sensor(modality='image')
        def sensor(cache): return image()
        sensors[camera+'_image']=sensor
    for name in sampling.PROPRIO:
        @module.sensor(modality='proprio')
        def sensor(cache): return np.array([env.state],dtype=float)
        sensors[name]=sensor
    for name,sensor in sensors.items():
        env.env._observables[name]=module.Observable(name=name,sensor=sensor,sampling_rate=20)
    # Native reset and set_init_state can force observations before warmup,
    # shifting the sampling phase without shifting the physical time.
    for _ in range(initial_updates):
        cache={}
        for ob in env.env._observables.values(): ob.update(timestep=.002,obs_cache=cache,force=True)
    monkeypatch.setattr(sampling,'entity_map',lambda e:[])
    monkeypatch.setattr(sampling,'pose_bundle',lambda e:{'state':e.state})
    monkeypatch.setattr(sampling,'physical_labels',lambda *a:{'position':float(env.state)})
    monkeypatch.setattr(sampling,'sampled_proprio',lambda raw:raw['robot0_eef_pos'])
    def render(env,raw,entities,task,views):
        assert len(views)==1
        view=views[0]; assert np.array_equal(raw[sampling.VIEWS[view]+'_image'],image())
        return {view:{'state':env.state,'direct_rgb_error':{'normalized_rms':0.}}}
    monkeypatch.setattr(sampling,'render_labels',render)
    def step():
        cache={}
        for _ in range(25):
            env.state+=1; env.sim.data.time+=.002
            for ob in env.env._observables.values(): ob.update(timestep=.002,obs_cache=cache)
        return {name:ob.obs for name,ob in env.env._observables.items()}
    return env,step,sensors


def test_callback_labels_follow_original_cached_frame_and_keep_clocks(monkeypatch):
    env,step,sensors=setup(monkeypatch)
    with sampling.CameraLabels(env,{}) as sampler:
        sampler.arm('checkpoint_0'); raw=step()
        # The native scheduler returns an earlier image while the control step progressed.
        assert raw['agentview_image'][0,0,0]!=env.state
        clocks={k:(o._time_since_last_sample,o._sampled) for k,o in env.env._observables.items()}
        labels,truth,bundle,timing=sampler.take(raw)
        sampled=int(raw['agentview_image'][0,0,0])
        assert labels['front']['state']==labels['wrist']['state']==truth['position']==sampled
        assert sampled!=env.state and bundle=={'state':sampled}
        assert timing['front']['cache_age_s']>0 and timing['front']['returned_rgb_exact']
        assert clocks=={k:(o._time_since_last_sample,o._sampled) for k,o in env.env._observables.items()}
        sampler.arm(None); step()
        assert sampler.latest=={}
    assert all(env.env._observables[n]._sensor is s for n,s in sensors.items())


def test_tampered_cached_frame_is_rejected_and_hooks_restored(monkeypatch):
    env,step,sensors=setup(monkeypatch)
    with pytest.raises(RuntimeError,match='cached RGB'):
        with sampling.CameraLabels(env,{}) as sampler:
            sampler.arm('one'); raw=step(); raw['agentview_image']=raw['agentview_image'].copy()
            raw['agentview_image'][0,0,0]+=1; sampler.take(raw)
    assert all(env.env._observables[n]._sensor is s for n,s in sensors.items())


@pytest.mark.parametrize('initial_updates',[0,1,2])
def test_repeated_checkpoints_and_no_clock_changes_to_unhooked_run(monkeypatch,initial_updates):
    reference,step,_=setup(monkeypatch,initial_updates)
    expected=[step(),step(),step()]
    env,step,_=setup(monkeypatch,initial_updates)
    with sampling.CameraLabels(env,{}) as sampler:
        for i in range(3):
            sampler.arm(str(i)); actual=step(); sampler.take(actual)
            assert all(np.array_equal(actual[k],expected[i][k]) for k in actual)
    assert env.state==reference.state and env.sim.data.time==reference.sim.data.time


def test_callback_render_failure_restores_original_sensors(monkeypatch):
    env,step,sensors=setup(monkeypatch)
    def fail(*a,**k): raise RuntimeError('synthetic render failure')
    monkeypatch.setattr(sampling,'render_labels',fail)
    with pytest.raises(RuntimeError,match='synthetic render failure'):
        with sampling.CameraLabels(env,{}) as sampler:
            sampler.arm('one'); step()
    assert all(env.env._observables[n]._sensor is s for n,s in sensors.items())


def test_sampled_proprio_must_match_returned_native_proprio(monkeypatch):
    env,step,_=setup(monkeypatch)
    with sampling.CameraLabels(env,{}) as sampler:
        sampler.arm('one'); raw=step(); raw['robot0_eef_pos']=raw['robot0_eef_pos']+1
        with pytest.raises(RuntimeError,match='proprio sample timing'): sampler.take(raw)
