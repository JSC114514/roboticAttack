"""One GPU, no VLA load: exact historical source and baseline reset checks."""
import numpy as np
import torch
from jt_common import (ROOT,BASE,native,verify,selected,source_row,require,complete,finish,atomic)


def run(monitor):
    receipt=ROOT/'reports/REPLAY_CHECK_COMPLETE.json'
    if complete(receipt):return
    from jt_cache import prefix,constructor_rng,restore_constructor_rng
    from jt_evaluate import rollout_functions
    import pandas as pd
    native.require_gpu('historical_reset_replay_check')
    value=verify();runtime=native.runtime_module();runtime.prepare_graphics_backend()
    entries=[e for e in selected() if e['suite']=='libero_spatial' and e['task_id']==4]
    require(sorted(e['state_index'] for e in entries)==[40,41,42,43],'Missing four-state regression set')
    # Every Goal task is checked against its already-completed clean start before
    # spending any training budget. No policy action or new scientific trial here.
    baseline=[c for c in value['cases'] if c['state_index']==2]
    require(len(baseline)==10,'Goal reset verification coverage changed')
    paired_start=rollout_functions().__globals__['paired_start']
    proofs=[];done=0;total=len(entries)*2+len(baseline)
    monitor.update(phase='verify_historical_reset_context_no_model',completed=0,total=total)
    for entry in entries:
        row=source_row(entry)
        initial=torch.load(value['initial_states'][f"{entry['suite']}/{entry['task_id']}"]['path'],
                           map_location='cpu',weights_only=False)[entry['state_index']]
        traces=pd.read_parquet(value['trace_path'],filters=[('condition','==','clean'),
            ('suite','==',entry['suite']),('task_id','==',entry['task_id']),
            ('state_index','==',entry['state_index'])]).sort_values('decision_block')
        traces=traces[traces.rollout_seed==entry['rollout_seed']]
        require(list(traces.decision_block)==list(range(entry['decision_count'])),'Saved gate trace incomplete')
        construction=constructor_rng();expected=None
        for repeat in range(2):
            env=None
            try:
                restore_constructor_rng(construction)
                env,task,states=runtime.make_env(native.runtime_config(),entry['suite'],entry['task_id'])
                require(np.array_equal(initial,states[entry['state_index']]),'Gate initial differs')
                raw,target,capture,decision=prefix(env,initial,entry,row,traces,True)
                if expected is not None:require(decision==expected,'Fresh repeated gate state differs')
                expected=decision
                proofs.append({'state_id':entry['state_id'],'repeat':repeat,
                    'source_sample_fingerprint':capture['fingerprint'],'decision_fingerprint':decision,
                    'reset_context':capture['reset_context'],'zero_rgb_nrmse':capture['zero_rgb_nrmse'],
                    'source_observation_exact':True})
                done+=1;monitor.update(completed=done,total=total,state_id=entry['state_id'])
            finally:
                if env is not None:runtime.close_environment(env)
    for case in baseline:
        env=None
        try:
            env,task,states=runtime.make_env(native.runtime_config(),case['suite'],case['task_id'])
            initial=torch.load(case['initial_file'],map_location='cpu',weights_only=False)[case['state_index']]
            require(task.name==case['task_name'] and np.array_equal(initial,states[case['state_index']]),
                    'Gate baseline task or initial differs')
            raw=runtime.reset_wait(env,initial,case['rollout_seed'],native.runtime_config()['num_steps_wait'])
            paired_start(env,raw,case['rollout_seed'],ROOT/'reports/GATES'/case['start'],
                         BASE/'paired_starts'/case['start'])
            proofs.append({'baseline_start':case['start'],'exact_reset_fingerprint_and_RGB':True})
            done+=1;monitor.update(phase='verify_all_Goal_task_clean_reset_starts',completed=done,total=total)
        finally:
            if env is not None:runtime.close_environment(env)
    path=ROOT/'reports/GATES/RESET_CONTEXT.json'
    atomic({'status':'PASS','source_replays':8,'Goal_baseline_resets':10,'proofs':proofs,
            'model_loaded':False,'scientific_training_or_rollouts':False,'tolerance_changed':False},path)
    finish(receipt,[path],source_replays=8,baseline_resets=10)
