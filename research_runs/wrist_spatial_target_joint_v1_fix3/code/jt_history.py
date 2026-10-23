"""Reproduce historical reset order without patching installed LIBERO or its RNG."""
import gc
import hashlib
import numpy as np
from jt_common import require


def source_histories(episodes):
    """Mirror sr_cache.run: original worker parity, then task grouping/order.

    Use all original episodes, including those with no selected visible frame.
    Selection of 94/96 observations must not change reset history.
    """
    result={}
    for worker in (0,1):
        assigned=episodes[worker::2]
        for suite,task in sorted({(e['suite'],e['task_id']) for e in assigned}):
            prior=[]
            for entry in assigned:
                if (entry['suite'],entry['task_id'])!=(suite,task):continue
                result[entry['episode_id']]=list(prior)
                prior.append({k:entry[k] for k in ('episode_id','suite','task_id','rollout_seed')})
    return result


def sampler_state(env):
    return [{'name':str(s.name),'type':type(s).__name__,
             'range':np.asarray(getattr(s,'joint_ranges',[]),dtype=float).tolist()}
            for s in env.env.object_property_initializers]


def replay_previous_resets(env,entry,runtime):
    """Hard-reset previous episodes with their own seeds, before target seed.

    Hard reset reconstructs model/controllers, so previous episode physics need
    not be executed. Its reset/sampler history does need to be reconstructed.
    Strict native source fingerprint/RGB gates remain the final acceptance test.
    """
    require(bool(env.env.hard_reset) and not env.env.deterministic_reset,
            'Historical reset replay requires hard-reset model reconstruction')
    proof={'before':sampler_state(env),'prior_resets':[]}
    for previous in entry.get('source_reset_history',[]):
        require((previous['suite'],previous['task_id'])==(entry['suite'],entry['task_id']),
                'Historical reset crosses task boundary')
        env.seed(previous['rollout_seed']);runtime.bind_environment_context(env)
        runtime.reset_environment(env);gc.collect();runtime.renderer_metadata(env)
        proof['prior_resets'].append({**previous,'samplers':sampler_state(env)})
    return proof


def numerical_snapshot(env):
    """Diagnostic arrays, copied without advancing sim or refreshing sensors."""
    result={}
    for owner,prefix,names in (
        (env.sim.data,'data',('qpos','qvel','qacc','qacc_warmstart','ctrl','act','body_xpos',
                            'body_xquat','site_xpos','site_xmat','cam_xpos','cam_xmat')),
        (env.sim.model,'model',('body_pos','body_quat','geom_pos','geom_quat','site_pos','site_quat',
                              'cam_pos','cam_quat')),
        (env.env.robots[0].controller,'controller',('initial_joint','initial_ee_pos',
          'initial_ee_ori_mat','goal_pos','goal_ori','ee_pos','ee_ori_mat'))):
        for name in names:
            if hasattr(owner,name) and getattr(owner,name) is not None:
                result[prefix+'_'+name]=np.array(getattr(owner,name),copy=True)
    return result


def model_hashes(env):
    return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in numerical_snapshot(env).items()
            if k.startswith('model_')}
