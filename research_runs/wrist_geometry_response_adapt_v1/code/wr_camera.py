"""Same-state wrist camera translations captured at the native sensor timestamp.

The native simulator and its camera cache are never edited. Rendering uses a
copy of the already-realized model and an MjData snapshot taken by the sensor.
"""
from __future__ import annotations
import copy
from functools import wraps
import os
import numpy as np


def local_shift(position, rotation, axis, distance):
    """Translate in camera coordinates; MuJoCo optical forward is local -Z."""
    if axis not in ('x', 'z'):
        raise ValueError('Camera axis must be x or z')
    delta = np.zeros(3, dtype=float)
    delta[0 if axis == 'x' else 2] = float(distance) * (1 if axis == 'x' else -1)
    return np.asarray(position, float) + np.asarray(rotation, float).reshape(3, 3) @ delta


def nrmse(a, b):
    return float(np.sqrt(np.mean(((np.asarray(a, float)-np.asarray(b, float))/255.)**2)))


def restore_sample_visuals(model, sampled):
    # LIBERO stove visualization changes site_rgba in MjModel after sensor
    # sampling. Copy its sampled value into the shadow model, not the live one.
    np.copyto(model.site_rgba, sampled['site_rgba'])


class ShadowRenderer:
    def __init__(self, env):
        import mujoco
        from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
        from wr_common import native_runtime
        self.env, self.mujoco = env, mujoco
        self.sim = MjSim(copy.copy(env.sim.model._model))
        self.context = MjRenderContextOffscreen(self.sim, device_id=int(os.environ['MUJOCO_EGL_DEVICE_ID']))
        self.context.vopt = copy.copy(env.sim._render_context_offscreen.vopt)
        self.context.scn.flags[:] = env.sim._render_context_offscreen.scn.flags
        native_runtime().bind_environment_context(env)

    def render(self, snapshot, position=None, rotation=None, segmentation=False, visuals=None):
        from wr_common import native_runtime, native_helpers, require
        before = native_helpers().fingerprint(self.env)
        try:
            if visuals is not None:
                restore_sample_visuals(self.sim.model, visuals)
            self.mujoco.mj_copyData(self.sim.data._data, self.sim.model._model, snapshot)
            camera = self.sim.model.camera_name2id('robot0_eye_in_hand')
            if position is not None:
                self.sim.data.cam_xpos[camera] = position
                self.sim.data.cam_xmat[camera] = np.asarray(rotation).reshape(9)
            self.context.gl_ctx.make_current()
            result = self.sim.render(camera_name='robot0_eye_in_hand', width=256,
                                     height=256, segmentation=segmentation)
            from robosuite import macros
            sign = 1 if macros.IMAGE_CONVENTION == 'opengl' else -1
            return np.asarray(result)[::sign].copy()
        finally:
            native_runtime().bind_environment_context(self.env)
            require(native_helpers().fingerprint(self.env) == before,
                    'Shadow render changed original physics, camera or sensor clocks')

    def close(self):
        from wr_common import native_runtime
        if self.sim is not None:
            self.context = None
            self.sim.free()
            self.sim = None
            native_runtime().bind_environment_context(self.env)


class CameraSamples:
    """Sensor wrapper; take() validates pixels against the returned native frame."""
    def __init__(self, env, tolerance=1e-4, target_ids=None):
        self.env, self.tolerance = env, tolerance
        self.target_ids = target_ids
        self.key, self.latest, self.shadow = None, None, None

    def __enter__(self):
        self.observable = self.env.env._observables['robot0_eye_in_hand_image']
        self.original = self.observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels = self.original(cache)
            if self.key is not None:
                from wr_common import native_helpers
                camera = self.env.sim.model.camera_name2id('robot0_eye_in_hand')
                self.latest = dict(key=self.key, pixels=np.array(pixels, copy=True),
                    snapshot=copy.copy(self.env.sim.data._data),
                    visuals=dict(site_rgba=np.array(self.env.sim.model.site_rgba,copy=True)),
                    position=np.array(self.env.sim.data.cam_xpos[camera], copy=True),
                    rotation=np.array(self.env.sim.data.cam_xmat[camera], copy=True).reshape(3, 3),
                    fingerprint=native_helpers().fingerprint(self.env),
                    sample_time_s=float(self.env.sim.data.time))
            return pixels
        self.observable._sensor = sensor
        return self

    def arm(self, key):
        self.key, self.latest = key, None

    def take(self, raw, key=None):
        from wr_common import require
        expected = self.key if key is None else key
        require(self.latest is not None and self.latest['key'] == expected,
                'Missing native camera sensor snapshot')
        require(np.array_equal(raw['robot0_eye_in_hand_image'], self.latest['pixels']),
                'Native observation differs from sampled camera pixels')
        return self.latest

    def shifted_raw(self, raw, axis, distance, *, audit_visibility=True, check_zero=True):
        from wr_common import require
        w = self.take(raw)
        if self.shadow is None:
            self.shadow = ShadowRenderer(self.env)
        error = None
        if check_zero:
            zero = self.shadow.render(w['snapshot'], visuals=w['visuals'])
            error = nrmse(zero, w['pixels'])
            require(error <= self.tolerance, 'Zero camera rendering differs from native RGB')
        p = local_shift(w['position'], w['rotation'], axis, distance)
        image = self.shadow.render(w['snapshot'], p, w['rotation'], visuals=w['visuals'])
        altered = dict(raw, robot0_eye_in_hand_image=image)
        audit = dict(axis=axis, shift_m=float(distance), zero_nrmse=error,
                     sampled_position=w['position'].tolist(), rendered_position=p.tolist(),
                     sampled_rotation=w['rotation'].tolist(), sample_time_s=w['sample_time_s'],
                     camera_frame='x_right_z_optical_forward', physical_state_modified=False)
        if self.target_ids is not None and audit_visibility:
            def count(position):
                seg = self.shadow.render(w['snapshot'], position, w['rotation'], True, visuals=w['visuals'])
                # Segmentation and RGB undergo the same final native 180-degree
                # rotation and center crop; count is invariant to that rotation.
                n = int(np.sqrt(.9)*256); k = (256-n)//2
                mask = (seg[:, :, 0] == 5) & np.isin(seg[:, :, 1], self.target_ids)
                return int(mask[k:k+n, k:k+n].sum())
            audit.update(native_target_pixels=count(w['position']), shifted_target_pixels=count(p))
            audit['target_lost'] = audit['native_target_pixels'] > 0 and audit['shifted_target_pixels'] == 0
        return altered, audit

    def __exit__(self, *_):
        self.observable._sensor = self.original
        if self.shadow is not None:
            self.shadow.close()

