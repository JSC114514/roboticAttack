"""Table-plane controls: angle changes preserve range; range changes preserve bearing."""
from __future__ import annotations
import math
import numpy as np

TARGETS = ('direction_pos', 'direction_neg', 'distance_near', 'distance_far')


def candidates(config):
    return [{'candidate_id': f'{target}_b{layer}_s{seed}', 'target': target,
             'layer': layer, 'seed': seed}
            for target in TARGETS for layer in config['layers'] for seed in config['seeds']]


def conditions(truth, config):
    relative = np.asarray([truth['object_eef_relative_'+a] for a in 'xyz'], dtype=float)
    radius = float(np.linalg.norm(relative[:2]))
    if not np.isfinite(relative).all() or radius < 1e-9:
        raise ValueError('Undefined table-plane bearing; retain this state as an engineering failure')
    radial = relative[:2] / radius
    tangent = np.array([-radial[1], radial[0]])
    cap = config['max_displacement_m']
    output = []
    for target in TARGETS:
        sign = -1 if target in ('direction_neg', 'distance_near') else 1
        for strength, index in [('small', 0), ('large', 1)]:
            new = relative.copy()
            if target.startswith('direction'):
                desired = math.radians(config['direction_degrees'][index])
                angle = sign * min(desired, 2*math.asin(min(1., cap/(2*radius))))
                new[:2] = radius*(math.cos(angle)*radial + math.sin(angle)*tangent)
                actual_angle = math.degrees(angle)
                axis = np.r_[tangent, 0.]
            else:
                length = min(radius*config['distance_fractions'][index], cap)
                new[:2] += sign*length*radial
                actual_angle = 0.
                axis = np.r_[radial, 0.]
            delta = new-relative
            norm = float(np.linalg.norm(delta))
            if norm > cap+1e-12 or new[2] != relative[2]:
                raise AssertionError('Geometry control violated displacement/height invariant')
            output.append({'condition_id':target+'_'+strength, 'target':target, 'strength':strength,
                           'sign':sign, 'translation_m':delta.tolist(), 'delta_unit':(delta/norm).tolist(),
                           'canonical_axis_unit':axis.tolist(), 'angle_degrees':actual_angle,
                           'horizontal_radius_m':float(np.linalg.norm(new[:2])),
                           'relative_position_m':new.tolist(), 'distance_m':float(np.linalg.norm(new)),
                           'displacement_m':norm, 'physics_state_modified':False,
                           'source':'renderer_scene_only_counterfactual'})
    return output


def cohort(records, count=2):
    chosen=[]
    for task in range(10):
        for split in ('discovery', 'validation'):
            episodes=sorted({r['episode_id'] for r in records if r['task_id']==task and r['split']==split})[:count]
            if len(episodes) != count:
                raise ValueError(f'Missing task/split episodes: {task}/{split}')
            for episode in episodes:
                eligible=[r for r in records if r['episode_id']==episode and r['truth']['contact']==0
                          and r['truth']['grasp']==0 and r['visibility']['wrist']['subject_visible']]
                if not eligible:
                    raise ValueError('No eligible frame; do not silently substitute a trajectory: '+episode)
                chosen.append(min(eligible,key=lambda r:(r['truth']['object_eef_distance'],r['step_id'])))
    return chosen


def action_metrics(action, clean, condition):
    action=np.asarray(action,dtype=float).reshape(8,7)
    clean=np.asarray(clean,dtype=float).reshape(8,7)
    difference=action-clean
    return {'first_projection':float(difference[0,:3]@np.asarray(condition['delta_unit'])),
            'first_axis_projection':float(difference[0,:3]@np.asarray(condition['canonical_axis_unit'])),
            'first_translation_l2':float(np.linalg.norm(difference[0,:3])),
            'first_rotation_l2':float(np.linalg.norm(difference[0,3:6])),
            'first_gripper_delta':float(difference[0,6]),
            'chunk_translation_l2_mean':float(np.linalg.norm(difference[:,:3],axis=1).mean()),
            'first_open_flip':int(clean[0,6]<=.5 and action[0,6]>.5),
            'action_units':'native_LIBERO_controller_commands_not_realized_displacement'}
