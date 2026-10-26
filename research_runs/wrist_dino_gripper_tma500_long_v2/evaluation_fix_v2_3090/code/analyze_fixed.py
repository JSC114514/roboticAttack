#!/usr/bin/env python3
"""Task -> episode paired summaries. No step-level pseudoreplication."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval_common import (ROOT, REPAIR_ID, CONFIG, METHODS, CONDITIONS, DENOMINATORS, atomic_json, atomic_parquet,
                            config, condition_metadata, digest, load_json, preparation, require, sha256_file)
from evaluate_fixed import KEY, frozen_clean, load_patches, validate_frame, require_smoke

ATTACKS = CONDITIONS[1:]
PAIRS = ((METHODS[1],METHODS[0]),(METHODS[1],METHODS[2]),(METHODS[0],"reference_uada500"),
         (METHODS[1],"reference_uada500"),(METHODS[2],"reference_uada500"))


def bootstrap_suite(eligible,draws=10000,seed=20260907):
    wide = eligible.pivot(index=["task_id","state_index"],columns="condition",values="failure").reindex(columns=ATTACKS)
    require(not wide.isna().any().any() and set(wide.index.get_level_values("task_id"))==set(range(10)),"Unpaired eligible grid")
    arrays = [wide.xs(t,level="task_id").to_numpy(float) for t in range(10)]
    points = np.mean([a.mean(axis=0) for a in arrays],axis=0)
    samples = np.zeros((draws,len(ATTACKS)))
    rng = np.random.default_rng(seed)
    tasks = rng.integers(0,10,size=(draws,10))
    for position in range(10):
        for task,values in enumerate(arrays):
            rows = np.flatnonzero(tasks[:,position]==task)
            if len(rows):
                states = rng.integers(0,len(values),size=(len(rows),len(values)))
                samples[rows] += values[states].mean(axis=1)/10
    return points,samples


def analyze():
    cfg,prep = config(),preparation()
    _,hashes = load_patches(cfg)
    require_smoke(cfg, prep, hashes)
    rows, traces, shards = [],[],{}
    for suite in DENOMINATORS:
        folder = ROOT/"evaluation"/suite
        for task in range(10):
            receipt = load_json(folder/f"task_{task:02d}_STATUS.json")
            rp,tp = folder/f"task_{task:02d}_rollouts.parquet",folder/f"task_{task:02d}_traces.parquet"
            require(receipt["status"]=="COMPLETE" and receipt["repair_id"]==REPAIR_ID
                    and receipt["preparation_signature"]==digest(prep) and receipt["config_signature"]==digest(cfg)
                    and receipt["patches"]==hashes and receipt["evaluation_training_hooks"] is False
                    and sha256_file(rp)==receipt["rollouts_sha256"] and sha256_file(tp)==receipt["traces_sha256"],"Incomplete/changed evaluation shard")
            require(all(sha256_file(p)==h for p,h in receipt["source"].items()),"Loaded evaluation source changed")
            r,t = pd.read_parquet(rp),pd.read_parquet(tp)
            validate_frame(r,t,cfg,task,suite,hashes)
            clean = frozen_clean(cfg,suite).loc[lambda f:f.task_id.astype(int).eq(task)]
            match = r.merge(clean[["state_index","success","rollout_seed"]],on="state_index",suffixes=("","_frozen"),validate="many_to_one")
            require(match.frozen_clean_success.eq(match.success_frozen).all()
                    and match.rollout_seed.eq(match.rollout_seed_frozen).all(),"Frozen eligibility/seeds changed")
            rows.append(r); traces.append(t)
            shards[str(rp)] = sha256_file(rp)
            shards[str(tp)] = sha256_file(tp)
    raw,trace = pd.concat(rows,ignore_index=True),pd.concat(traces,ignore_index=True)
    require(len(raw)==2000 and not raw.duplicated(KEY).any() and raw.groupby(["suite","condition"]).size().eq(100).all(),"Final rollout grid incomplete")
    fresh = raw.loc[raw.condition.eq("clean"),["suite","task_id","state_index","success"]].rename(columns={"success":"fresh_clean_success"})
    raw = raw.merge(fresh,on=["suite","task_id","state_index"],validate="many_to_one")
    points,samples,summaries,contrasts = {},{},{},[]
    records = []
    for index,suite in enumerate(DENOMINATORS):
        eligible = raw.loc[raw.suite.eq(suite)&raw.frozen_clean_success.eq(1)&~raw.condition.eq("clean")]
        require(len(eligible)==DENOMINATORS[suite]*len(ATTACKS),"Eligible rollout count changed")
        points[suite],samples[suite] = bootstrap_suite(eligible,10000,20260907+index)
        for j,method in enumerate(ATTACKS):
            frame = eligible.loc[eligible.condition.eq(method)]
            current = raw.loc[raw.suite.eq(suite)&raw.condition.eq(method)&raw.fresh_clean_success.eq(1)]
            task_current = current.groupby("task_id").failure.mean()
            event = frame.loc[frame.event_metrics_applicable]
            grasp_pairs = frame.merge(raw.loc[raw.suite.eq(suite)&raw.condition.eq("clean"),
                                             ["task_id","state_index","stable_grasp_events"]],
                                      on=["task_id","state_index"],suffixes=("","_clean"),validate="one_to_one")
            grasp_eligible = grasp_pairs.loc[grasp_pairs.stable_grasp_events_clean.gt(0)&grasp_pairs.event_metrics_applicable]
            low,high = np.percentile(samples[suite][:,j],[2.5,97.5])
            record = {"suite_or_endpoint":suite,"condition":method,"task_equal_casr":float(points[suite][j]),
                      **condition_metadata(method,cfg),"evaluation_checkpoint_suite":suite,
                      "ci95_low":float(low),"ci95_high":float(high),"frozen_clean_success_states":DENOMINATORS[suite],
                      "fresh_clean_success_states":int(fresh.loc[fresh.suite.eq(suite)].fresh_clean_success.sum()),
                      "fresh_clean_task_equal_casr":float(task_current.mean()) if len(task_current) else None,
                      "fresh_clean_tasks_with_eligible_states":len(task_current),
                      "episode_equal_casr":float(frame.failure.mean()),
                      "wrong_open_episode_mean":float(frame.groupby("task_id").wrong_open_rate.mean().mean())
                          if frame.wrong_open_rate.notna().any() else None,
                      "wrong_open_executed_slots":int(frame.wrong_open_slots.sum()),
                      "eligible_executed_close_slots":int(frame.eligible_close_slots.sum()),
                      "event_applicable_episodes":len(event),
                      "never_held_proxy_episode_rate":float(event.never_held_proxy.mean()) if len(event) else None,
                      "clean_grasp_proxy_eligible_episodes":len(grasp_eligible),
                      "lost_grasp_establishment_proxy_rate":float(grasp_eligible.stable_grasp_events.eq(0).mean()) if len(grasp_eligible) else None,
                      "contact_loss_proxy_events":int(event.contact_loss_events.sum()) if len(event) else None,
                      "suspected_drop_proxy_events":int(event.suspected_drop_events.sum()) if len(event) else None,
                      "evaluation_episode_gpu_time_s":float(raw.loc[raw.suite.eq(suite)&raw.condition.eq(method)].elapsed_s.sum())}
            records.append(record)
    groups = {s:[s] for s in DENOMINATORS}
    groups.update(spatial_goal_macro=["libero_spatial","libero_goal"],
                  main_three_suite_macro=["libero_object","libero_spatial","libero_goal"],
                  four_suite_macro=list(DENOMINATORS))
    for group,suites in groups.items():
        point,sample = np.mean([points[s] for s in suites],axis=0),np.mean([samples[s] for s in suites],axis=0)
        if len(suites)>1:
            for j,method in enumerate(ATTACKS):
                low,high = np.percentile(sample[:,j],[2.5,97.5])
                records.append({"suite_or_endpoint":group,"condition":method,"task_equal_casr":float(point[j]),
                                **condition_metadata(method,cfg),
                                "ci95_low":float(low),"ci95_high":float(high),
                                "frozen_clean_success_states":sum(DENOMINATORS[s] for s in suites)})
        for left,right in PAIRS:
            li,ri = ATTACKS.index(left),ATTACKS.index(right)
            low,high = np.percentile(sample[:,li]-sample[:,ri],[2.5,97.5])
            contrasts.append({"suite_or_endpoint":group,"left":left,"right":right,
                              "left_patch_training_suite":condition_metadata(left,cfg)["patch_training_suite"],
                              "right_patch_training_suite":condition_metadata(right,cfg)["patch_training_suite"],
                              "difference_left_minus_right":float(point[li]-point[ri]),
                              "ci95_low":float(low),"ci95_high":float(high)})
    out = ROOT/"evaluation"
    atomic_parquet(raw,out/"GRIPPER_TMA500_ROLLOUTS.parquet")
    atomic_parquet(trace,out/"GRIPPER_TMA500_TRACES.parquet")
    pd.DataFrame(records).to_csv(out/"GRIPPER_TMA500_SUMMARY.csv",index=False)
    pd.DataFrame(contrasts).to_csv(out/"GRIPPER_TMA500_CONTRASTS.csv",index=False)
    counts = fresh.groupby("suite").fresh_clean_success.sum().astype(int).to_dict()
    atomic_json({"status":"COMPLETE","repair_id":REPAIR_ID,"experiment_id":cfg["experiment_id"],"config_sha256":sha256_file(CONFIG),
                 "episodes":2000,"eligible_attack_episodes":sum(DENOMINATORS.values())*4,
                 "patches":hashes,"shards":shards,"fresh_clean_success_counts":counts,
                 "bootstrap_draws":10000,"bootstrap_seed":20260907,"resampling_unit":"task_then_episode_paired",
                 "grid_type":"previously_inspected_development_grid","training_seed_count":1,
                 "training_suite":cfg["training_suite"],"long":"mandatory_source_domain_secondary",
                 "primary_transfer_suites":["libero_object","libero_spatial","libero_goal"],
                 "evaluation_models":"suite_specific_oft_finetuned_weights",
                 "reference_training_suite":cfg["reference_training_suite"],
                 "event_metrics":"simulation_proxies_not_true_drop_causality",
                 "stop":"END_500_PILOT_NO_AUTOMATIC_EXTENSION"},out/"STATUS.json")
    print(pd.DataFrame(records)[["suite_or_endpoint","condition","task_equal_casr"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    analyze()
