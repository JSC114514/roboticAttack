"""Read labels at the native camera sensor call, without refreshing its observable.

Only the sensor callable is temporarily wrapped. Sampling clocks, caches, physics,
camera settings, raw return values and the original executed actions are retained.
"""
from __future__ import annotations
from functools import wraps
import numpy as np
from atlas_common import config,require
from atlas_labels import VIEWS,array_hash,entity_map,physical_labels,pose_bundle,render_labels

PROPRIO=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def sampled_proprio(raw):
    from atlas_runtime import quat2axisangle
    return np.concatenate([np.asarray(raw[PROPRIO[0]],np.float32),quat2axisangle(raw[PROPRIO[1]]),
                           np.asarray(raw[PROPRIO[2]],np.float32)])


def scalar_truth(truth):
    return {k:float(v) if np.isfinite(v) else None for k,v in truth.items()}


class CameraLabels:
    def __init__(self,env,task):
        self.env=env; self.inner=env.env; self.task=task; self.originals={}; self.latest={}; self.tag=None
        self.entities=entity_map(env); self.sample_counts={view:0 for view in VIEWS}

    def __enter__(self):
        try:
            for view,camera in VIEWS.items():
                observable=self.inner._observables[camera+'_image']
                require(observable.is_enabled() and observable.is_active(),'Native camera disabled: '+view)
                original=observable._sensor; self.originals[view]=(observable,original)
                @wraps(original)
                def sensor(cache,original=original,view=view,camera=camera):
                    pixels=original(cache)
                    self.sample_counts[view]+=1
                    if self.tag is not None:
                        before=pose_bundle(self.env)
                        # Native proprio sensors are pure direct data reads. Do not call
                        # Observable.update/set_sensor: those alter clocks or probe sensors.
                        raw={name:np.array(self.inner._observables[name]._sensor(cache),copy=True) for name in PROPRIO}
                        raw[camera+'_image']=pixels
                        truth=scalar_truth(physical_labels(self.env,raw,self.entities,self.task))
                        labels=render_labels(self.env,raw,self.entities,self.task,views=[view])[view]
                        require(before==pose_bundle(self.env),'Camera label callback changed simulator data')
                        self.latest[view]={'tag':self.tag,'pixels':np.array(pixels,copy=True),'labels':labels,
                                           'truth':truth,'proprio':sampled_proprio(raw),'pose_bundle':before,
                                           'sample_time_s':float(self.env.sim.data.time),
                                           'sample_state_sha256':array_hash(np.asarray(self.env.get_sim_state())),
                                           'sensor_calls':self.sample_counts[view]}
                    return pixels
                observable._sensor=sensor
            return self
        except BaseException:
            self.__exit__(None,None,None); raise

    def arm(self,tag):
        self.tag=tag; self.latest={}

    def take(self,raw):
        require(self.tag is not None and set(self.latest)==set(VIEWS),'Missing camera sample for target checkpoint')
        labels={}; timing={}; truth=None; bundle=None
        for view,camera in VIEWS.items():
            record=self.latest[view]
            require(record['tag']==self.tag,'Stale camera labels')
            require(np.array_equal(record['pixels'],raw[camera+'_image']),'Labels do not belong to returned cached RGB: '+view)
            error=float(abs(record['proprio']-sampled_proprio(raw)).max())
            require(error<=config()['replay']['proprio_atol'],'Camera and native proprio sample timing differs: '+view)
            require(bundle is None or bundle==record['pose_bundle'],'Front/wrist sampled different realized states')
            require(truth is None or truth==record['truth'],'Front/wrist physical truth differs')
            truth=record['truth']; bundle=record['pose_bundle']; labels[view]=record['labels']
            timing[view]={'sample_time_s':record['sample_time_s'],'checkpoint_time_s':float(self.env.sim.data.time),
                          'cache_age_s':float(self.env.sim.data.time)-record['sample_time_s'],
                          'sample_state_sha256':record['sample_state_sha256'],'sensor_calls':record['sensor_calls'],
                          'returned_rgb_exact':True,'cached_proprio_max_abs':error}
        self.tag=None
        return labels,truth,bundle,timing

    def __exit__(self,*_):
        for observable,original in self.originals.values(): observable._sensor=original
        self.originals.clear()
