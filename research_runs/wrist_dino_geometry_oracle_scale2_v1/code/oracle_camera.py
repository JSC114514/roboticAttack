"""Same sensor-time scene translation as the source cache; translation only."""
from __future__ import annotations
from functools import wraps
import hashlib
import time
import numpy as np
from oracle_common import geometry, require, config

def scaled_deltas(target, eef):
    base, relation = geometry().paired_deltas(target, eef)
    scale = float(config()['displacement_scale'])
    # Scale AFTER the source cap; merely increasing the cap would leave small
    # near-target displacements unchanged and would not implement a 2x dose.
    return {k: v * scale for k, v in base.items()}, {
        **relation, 'base_displacement_m': relation['matched_displacement_m'],
        'matched_displacement_m': relation['matched_displacement_m'] * scale,
        'displacement_scale': scale}

def fingerprint(env):
    arrays = {'sim_state': np.asarray(env.get_sim_state())}
    for name in ('body_xpos', 'body_xquat', 'geom_xpos', 'geom_xmat', 'site_xpos', 'site_xmat', 'cam_xpos', 'cam_xmat'):
        arrays[name] = np.asarray(getattr(env.sim.data, name))
    clocks = {name: {key: value.item() if isinstance(value, np.generic) else value
                     for key, value in vars(obs).items() if key != '_sensor' and
                     isinstance(value, (bool, int, float, str, np.generic))}
              for name, obs in env.env._observables.items()}
    return {'arrays': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()},
            'time': float(env.sim.data.time), 'observable_clocks': clocks}

def capture(env, pixels):
    g = geometry()
    upright = np.asarray(pixels)[::-1, ::-1].copy()
    entities, failures = g.task_candidates(env)
    attempts = []
    entity = None
    for candidate in entities:
        zero, mask = g.render(env, candidate, np.zeros(3))
        error = float(np.sqrt(np.mean(((upright.astype(float) - zero.astype(float)) / 255.) ** 2)))
        require(error <= 1e-4, 'Zero translation failed native RGB reproduction')
        attempts.append({'name': candidate['name'], 'visible_pixels': int(mask.sum()), 'zero_rgb_nrmse': error})
        if mask.any():
            entity = candidate
            break
    if entity is None:
        return {'images': {s: upright.copy() for s in ('plus', 'minus')}, 'no_target': True,
                'reason': 'no_visible_task_relevant_entity', 'entity': None, 'attempts': attempts,
                'resolution_failures': failures, 'displacement_m': 0., 'visible_pixels': {'plus': 0, 'minus': 0}}
    eef = np.asarray(env.sim.data.site_xpos[env.env.robots[0].eef_site_id], dtype=float).copy()
    target = np.asarray(env.sim.data.body_xpos[entity['root_body_id']] if entity['anchor'] == 'joint_body'
        else env.env.object_states_dict[entity['name']].get_geom_state()['pos'], dtype=float).reshape(3)
    if np.linalg.norm((target - eef)[:2]) <= 1e-9:
        return {'images': {s: upright.copy() for s in ('plus', 'minus')}, 'no_target': True,
                'reason': 'degenerate_horizontal_relation', 'entity': entity['name'], 'attempts': attempts,
                'resolution_failures': failures, 'displacement_m': 0., 'visible_pixels': {'plus': 0, 'minus': 0}}
    # Exact original paired_deltas: world X +/- displacement matched to capped
    # 30-degree bearing chord; NOT a newly imposed fixed 30mm shift.
    deltas, relation = scaled_deltas(target, eef)
    images, visibility = {}, {}
    for sign in ('plus', 'minus'):
        images[sign], mask = g.render(env, entity, deltas['translation_' + sign])
        visibility[sign] = int(mask.sum())
    return {'images': images, 'no_target': False, 'reason': None, 'entity': entity['name'],
            'attempts': attempts, 'resolution_failures': failures, 'eef_xyz': eef.tolist(),
            'target_xyz': target.tolist(), 'displacement_m': relation['matched_displacement_m'],
            'base_displacement_m': relation['base_displacement_m'], 'displacement_scale': relation['displacement_scale'],
            'deltas': {s: deltas['translation_' + s].tolist() for s in ('plus', 'minus')},
            'visible_pixels': visibility}

class Camera:
    def __init__(self, env):
        self.env, self.latest, self.key = env, None, None
        self.captures, self.elapsed_s = 0, 0.
    def __enter__(self):
        self.observable = self.env.env._observables['robot0_eye_in_hand_image']
        self.original = self.observable._sensor
        @wraps(self.original)
        def sensor(cache):
            pixels = self.original(cache)
            if self.key is not None:
                started = time.monotonic()
                before = fingerprint(self.env)
                payload = capture(self.env, pixels)
                require(fingerprint(self.env) == before, 'Rendering changed physics, poses, cameras or sensor clocks')
                self.latest = {**payload, 'native_pixels': np.array(pixels, copy=True), 'key': self.key,
                               'sample_time_s': float(self.env.sim.data.time)}
                self.captures += 1
                self.elapsed_s += time.monotonic() - started
            return pixels
        self.observable._sensor = sensor
        return self
    def arm(self, key):
        self.key, self.latest = key, None
    def take(self, raw, key):
        require(self.latest is not None and self.latest['key'] == key, 'Missing current camera sample')
        require(np.array_equal(self.latest['native_pixels'], raw['robot0_eye_in_hand_image']),
                'Camera callback does not match returned native RGB')
        return self.latest
    def __exit__(self, *args):
        self.observable._sensor = self.original
