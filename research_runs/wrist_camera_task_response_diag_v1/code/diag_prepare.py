# 发布注释：只依据可见性、接触、抓持和距离选状态，不依据新模型响应选样。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Replay saved actions only; select nearest eligible sample before policy use."""
from contextlib import contextmanager
from functools import lru_cache
import copy,time
import numpy as np
from diag_common import (ROOT,PROJECT,SOURCE,config,read,atomic,require,source_rows,choose,
    conditions,signature,complete,finish,native_runtime,runtime_config,require_gpu,native_helpers)
from diag_geometry import Samples,Renderer,anchor_for,vector,set_translation,contacts,validity,crop_count
from wr_data import _decode_chunk,replay_alignment,geometry_module

@lru_cache(maxsize=1)
def trace_for(episode):
    import pandas as pd
    row=next(r for r in source_rows() if r['episode_id']==episode)
    t=pd.read_parquet(PROJECT/config()['source_traces'],filters=[('condition','==','clean'),
        ('suite','==',row['suite']),('task_id','==',row['task_id']),('state_index','==',row['state_index']),
        ('rollout_seed','==',row['rollout_seed'])]).sort_values('decision_block')
    require(list(t.decision_block)==list(range(row['decision_count'])),'Saved action trace incomplete')
    return t

@contextmanager
def replay(episode,stop_decision,visit=None):
    """Each branch gets an exact saved-action reset, not a new clean policy path."""
    rt=native_runtime();rc=runtime_config();rt.prepare_graphics_backend();rt.add_paths(rc)
    rows={r['decision_block']:r for r in source_rows() if r['episode_id']==episode}
    first=next(iter(rows.values()));env,task,initial=rt.make_env(rc,first['suite'],first['task_id'])
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    try:
        native_helpers().seed_all(first['rollout_seed'])
        raw=rt.reset_wait(env,initial[first['state_index']],first['rollout_seed'],rc['num_steps_wait'])
        anchor=anchor_for(env,first['suite'],first['task_id'])
        audit=read(SOURCE/f'data/pilot/camera/{episode}_AUDIT.json')
        audits={a['state_id']:a for a in audit['samples']}
        with np.load(SOURCE/f'data/pilot/camera/{episode}.npz',allow_pickle=False) as f:
            historical={s:dict(observation={'state':f['proprio'][i], 'full_image':f['front_rgb'][i],
                'wrist_image':f['wrist_rgb'][i]},sample_fingerprint=audits[s]['sample_fingerprint'])
                for i,s in enumerate(f['state_ids'].tolist())}
        with Samples(env,anchor) as samples:
            rendered=Renderer(env)
            try:
                executed=0
                for saved in trace_for(episode).itertuples(index=False):
                    decision=int(saved.decision_block)
                    if decision in rows:
                        row=rows[decision];sample=samples.take(raw,row['state_id'])
                        align=replay_alignment(rt.observation(raw,rc),historical[row['state_id']],sample)
                        gate=rendered.zero_gate(raw,samples,anchor)
                        if visit:
                            seg=rendered.image(sample,'robot0_eye_in_hand',anchor,segmentation=True)
                            visit(dict(row=row,truth=sample['truth'],visible_pixels=crop_count(seg,anchor['render_ids']),
                                       replay_alignment=align,zero_rgb_nrmse=gate))
                    if decision==stop_decision:
                        require(decision in rows,'Chosen decision not in fixed cache')
                        yield env,task,raw,samples,rendered,anchor,executed
                        return
                    actions=_decode_chunk(saved.predicted_environment_action_chunk)
                    slots=int(saved.executed_slots);require(1<=slots<=8,'Invalid replay slot count')
                    for slot in range(slots):
                        nxt=rows.get(decision+1) if slot==slots-1 else None
                        samples.arm(nxt['state_id'] if nxt else None)
                        raw,_,done,_=env.step(process_libero_action(actions[slot].copy()).tolist());executed+=1
                        require(not done or decision+1>=stop_decision,'Saved replay ended before selected state')
                    require(executed==int(saved.step_end),'Replay control-step boundary changed')
                raise RuntimeError('Chosen replay state missing')
            finally:rendered.close()
    finally:rt.close_environment(env)

def serial_anchor(a):return {k:v for k,v in a.items() if k!='gripper'}

def prepare(monitor):
    require_gpu();rows=source_rows();episodes=sorted({r['episode_id'] for r in rows});out=[]
    monitor.update(total=len(episodes),completed=0,resumed=0,phase='prepare_geometry')
    for i,episode in enumerate(episodes):
        folder=ROOT/'data/anchors'/episode;receipt=folder/'COMPLETE.json'
        if complete(receipt):
            out.append(read(folder/'ANCHOR.json'));monitor.update(completed=i+1);continue
        folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();replay_steps=0
        samples_list=[];group=[r for r in rows if r['episode_id']==episode]
        try:
            with replay(episode,max(r['decision_block'] for r in group),visit=samples_list.append) as values:
                replay_steps+=values[-1]
            require(len(samples_list)==5,'Did not inspect all five fixed observations')
            selected=choose(samples_list)
            meta=dict(signature=signature(),episode_id=episode,suite=group[0]['suite'],task_id=group[0]['task_id'],
                      state_index=group[0]['state_index'],selected=selected['row'] if selected else None,
                      selection_rule='nearest visible uncontacted ungrasped; earlier decision tie',
                      inspected=samples_list,conditions=[],policy_called=False)
            files=[]
            if selected is not None:
                with replay(episode,selected['row']['decision_block']) as (env,task,raw,samples,renderer,a,count):
                    replay_steps+=count;meta['language']=task.language;meta['anchor']=serial_anchor(a)
                    original=copy.copy(env.sim.data._data);qpos=np.array(env.sim.data.qpos,copy=True)
                    # Standard forward aligns derived geometry at the unchanged decision-time qpos.
                    # Observable clocks/policy proprio remain the native sampled values.
                    env.sim.forward();baseline=contacts(env,a)
                    baseline_truth=geometry_module().truth(env,a)
                    frames=[];meta['baseline_truth']=baseline_truth;meta['base_replay_sample_fingerprint']=samples.latest['fingerprint']
                    native=rt_observation(raw);clock_before=native_helpers().fingerprint(env)['observable_clocks']
                    for c in conditions():
                        renderer.mujoco.mj_copyData(env.sim.data._data,env.sim.model._model,original)
                        delta=vector(env,c['target_shift_m']);physics=set_translation(env,a,delta)
                        changed,visibility=renderer.paired_raw(raw,samples,a,delta,c['camera_shift_m'])
                        observation=rt_observation(changed);truth=geometry_module().truth(env,a)
                        require(np.array_equal(observation['state'],native['state']),'Intervention changed policy proprio')
                        require(np.max(abs(np.asarray(truth['target_xyz'])-baseline_truth['target_xyz']-delta))<1e-9,
                                'Real target geometry did not translate as specified')
                        require(native_helpers().fingerprint(env)['observable_clocks']==clock_before,'Physical intervention refreshed sensor clocks')
                        reasons=validity(baseline,contacts(env,a))
                        if visibility['visible_pixels']==0:reasons.append('target_not_visible')
                        if c['target_shift_m']==0:
                            require(np.array_equal(observation['full_image'],native['full_image']),
                                    'Camera-only intervention changed front image')
                        if c['target_shift_m']==0 and c['camera_shift_m']==0:
                            require(np.array_equal(observation['wrist_image'],native['wrist_image']),
                                    'Zero intervention differs from native wrist RGB')
                        meta['conditions'].append(dict(**c,valid=not reasons,reasons=reasons,physics=physics,
                            visibility=visibility,initial_truth=truth,physical_fingerprint=native_helpers().fingerprint(env)))
                        frames.append(observation)
                    renderer.mujoco.mj_copyData(env.sim.data._data,env.sim.model._model,original)
                    require(np.array_equal(env.sim.data.qpos,qpos),'Preparation failed to restore original qpos')
                    np.savez_compressed(folder/'CONDITIONS.npz',front=np.stack([f['full_image'] for f in frames]),
                        wrist=np.stack([f['wrist_image'] for f in frames]),proprio=np.stack([f['state'] for f in frames]))
                    files.append(folder/'CONDITIONS.npz')
            meta.update(replay_steps=replay_steps,elapsed_s=time.monotonic()-started)
            atomic(meta,folder/'ANCHOR.json');finish(receipt,[folder/'ANCHOR.json',*files]);out.append(meta)
        except BaseException as e:
            from diag_common import resources
            atomic(dict(status='FAILED',error=repr(e),elapsed_s=time.monotonic()-started,replay_steps=replay_steps,**resources()),
                   folder/('FAILURE_'+str(time.time_ns())+'.json'));raise
        monitor.update(completed=i+1,episode_id=episode)
    covered=set();camera_covered=set()
    for a in out:
        valid={(c['target_shift_m'],c['camera_shift_m']) for c in a['conditions'] if c['valid']}
        if any(t!=0 and (0.,c) in valid for t,c in valid):covered.add(a['suite'])
        if any(c!=0 and (t,0.) in valid for t,c in valid):camera_covered.add(a['suite'])
    status='COMPLETE' if covered==camera_covered==set(config()['tasks']) else 'INCOMPLETE_DATA'
    atomic(dict(signature=signature(),status=status,anchors=out,total_candidate_trajectories=12,
                selected_trajectories=sum(a['selected'] is not None for a in out),
                valid_conditions=sum(c['valid'] for a in out for c in a['conditions']),
                physical_pair_covered_suites=sorted(covered),camera_pair_covered_suites=sorted(camera_covered),all_invalid_and_missing_retained=True),ROOT/'data/MANIFEST.json')
    require(status=='COMPLETE','Valid physical variants do not cover all four suites; inspect manifest, no inference queued to execute')
    finish(ROOT/'reports/PREPARE_COMPLETE.json',[ROOT/'data/MANIFEST.json',*[ROOT/'data/anchors'/a['episode_id']/'COMPLETE.json' for a in out]])

def rt_observation(raw):return native_runtime().observation(raw,runtime_config())
