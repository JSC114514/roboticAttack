"""Offline native-forward diagnostics and normal PNG closed-loop rollouts."""
from __future__ import annotations
import argparse, gc, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, PROJECT, PYTHON, atomic_json, atomic_parquet, atomic_torch, cfg, complete,
                    finish, load_png, load_wrapper, paste_numpy, progress, read_json, require,
                    require_gpu, resources, runtime_config, runtime_module, sha, signature, cost_attempt, cache_shard_origin_matches)
from engine import Engine

def png_array(path):
    with Image.open(path) as image:
        require(image.mode=="RGB" and image.size==(50,50),"Deployment patch must be RGB 50x50")
        return np.array(image,copy=True)

def patch_set(split):
    c=cfg(); output={"clean":None,"initial":png_array(ROOT/"training/translation/epoch_00.png"),
                     "uada":png_array(PROJECT/c["uada_png"])}
    if split=="validation":
        for method in c["methods"]: output[method]=png_array(ROOT/f"training/{method}/epoch_50.png")
    else:
        winner=read_json(ROOT/"results/WINNER.json")["winner"]
        output["winner"]=png_array(ROOT/f"training/{winner}/epoch_50.png")
    return output

def relation_metrics(value, negative, positive):
    value=np.asarray(value,dtype=np.float64).reshape(256,-1)
    negative=np.asarray(negative,dtype=np.float64).reshape(256,-1); positive=np.asarray(positive,dtype=np.float64).reshape(256,-1)
    direction=positive-negative; energy=np.mean(direction**2,axis=-1)
    if energy.sum()<=1e-12:
        return {"q":None,"orthogonal":None,"target_distance_normalized":None}
    weight=energy/energy.sum(); denominator=float(np.sum(weight*energy))
    denominator=max(denominator,1e-12)
    q=float(np.sum(weight*np.mean((value-negative)*direction,axis=-1))/denominator)
    residual=value-(negative+q*direction)
    return {"q":q,"orthogonal":float(np.sum(weight*np.mean(residual**2,axis=-1))/denominator),
            "target_distance_normalized":float(np.sum(weight*np.mean((value-positive)**2,axis=-1))/denominator)}

def weighted_q(value,negative,positive):
    return relation_metrics(value,negative,positive)["q"]

def action_metrics(action, clean):
    delta=np.asarray(action)-np.asarray(clean)
    return {"first_translation_l2":float(np.linalg.norm(delta[0,:3])),
            "first_rotation_l2":float(np.linalg.norm(delta[0,3:6])),
            "first_gripper_delta":float(delta[0,6]),
            "chunk_translation_l2_mean":float(np.linalg.norm(delta[:,:3],axis=1).mean())}

def offline_suite(split,suite,engine):
    tasks=cfg()["task_splits"][suite][split]; rows=[]; started=time.monotonic(); patches=patch_set(split)
    methods=cfg()["methods"] if split=="validation" else [read_json(ROOT/"results/WINNER.json")["winner"]]
    conditions=cfg()["validation_conditions"] if split=="validation" else cfg()["heldout_conditions"]
    for task_index,task in enumerate(tasks):
        require(complete(ROOT/f"data/cache/{split}/{suite}/task_{task:02d}.json"),"Offline cache shard is not sealed")
        shard=torch.load(ROOT/f"data/cache/{split}/{suite}/task_{task:02d}.pt",map_location="cpu",weights_only=False)
        require(cache_shard_origin_matches(shard,ROOT/f"data/cache/{split}/{suite}/task_{task:02d}.json"),
                "Offline cache signature changed")
        for record in shard["records"]:
            obs,language=record["observation"],record["language"]
            clean_action,clean_features=engine.forward(obs,language)
            reference={}
            for method in methods:
                values={}
                for sign in ("plus","minus"):
                    name=f"{method}_{sign}"; cf_obs={**obs,"wrist_image":record["counterfactual_wrist"][name]}
                    cf_action,cf_features=engine.forward(cf_obs,language)
                    dino_action,_=engine.forward(obs,language,{"dino":cf_features["dino"]})
                    values[sign]={"action":cf_action,"dino_action":dino_action,"features":cf_features}
                b=values["plus"]["dino_action"][0,:3]-values["minus"]["dino_action"][0,:3]
                reference[method]={"plus":values["plus"]["features"]["b22"],
                                   "minus":values["minus"]["features"]["b22"],
                                   "dino_action_direction":b,
                                   "native_cf_action_plus":values["plus"]["action"].tolist(),
                                   "native_cf_action_minus":values["minus"]["action"].tolist()}
            for condition in conditions:
                patch=patches[condition]
                current=obs if patch is None else paste_numpy(obs,patch)
                action,features=engine.forward(current,language)
                row={k:record[k] for k in ("state_id","episode_id","suite","task_id","state_index","split","frame_rank")}
                row.update(condition=condition,actions=action.tolist(),clean_actions=clean_action.tolist(),
                           method=None,dino_signed_q=None,native_action_direction_dot=None,native_dino_cf_direction_l2=None,
                           dino_orthogonal=None,dino_target_distance_normalized=None,clean_dino_signed_q=None,
                           clean_dino_orthogonal=None,dino_signed_q_delta=None,
                           no_target=bool(record["geometry"].get("no_target",False)),
                           target_entity=record["binding"]["entity"],
                           native_cf_action_plus=None,native_cf_action_minus=None,restore_dino_actions=None,
                           restore_siglip_actions=None,restore_dino_first_translation_l2=None,
                           restore_siglip_first_translation_l2=None,**action_metrics(action,clean_action))
                method=condition if condition in cfg()["methods"] else None
                if split=="heldout" and condition=="winner": method=methods[0]
                if method is not None:
                    ref=reference[method]; delta=action[0,:3]-clean_action[0,:3]; b=ref["dino_action_direction"]
                    metric=relation_metrics(features["b22"],ref["minus"],ref["plus"])
                    clean_metric=relation_metrics(clean_features["b22"],ref["minus"],ref["plus"])
                    row.update(method=method,dino_signed_q=metric["q"],dino_orthogonal=metric["orthogonal"],
                               dino_target_distance_normalized=metric["target_distance_normalized"],
                               clean_dino_signed_q=clean_metric["q"],clean_dino_orthogonal=clean_metric["orthogonal"],
                               dino_signed_q_delta=None if metric["q"] is None else metric["q"]-clean_metric["q"],
                               native_action_direction_dot=float(delta@b),native_dino_cf_direction_l2=float(np.linalg.norm(b)),
                               native_cf_action_plus=ref["native_cf_action_plus"],native_cf_action_minus=ref["native_cf_action_minus"])
                    restore_dino,_=engine.forward(current,language,{"dino":clean_features["dino"]})
                    restore_siglip,_=engine.forward(current,language,{"siglip":clean_features["siglip"]})
                    row["restore_dino_actions"]=restore_dino.tolist(); row["restore_siglip_actions"]=restore_siglip.tolist()
                    row["restore_dino_first_translation_l2"]=action_metrics(restore_dino,clean_action)["first_translation_l2"]
                    row["restore_siglip_first_translation_l2"]=action_metrics(restore_siglip,clean_action)["first_translation_l2"]
                rows.append(row)
        progress("offline_"+split,suite=suite,tasks_complete=task_index+1,tasks_total=len(tasks),rows=len(rows),elapsed_s=time.monotonic()-started)
    expected=160*len(conditions)//4
    require(len(rows)==expected,"Offline evaluation row count mismatch")
    path=ROOT/f"evaluation/{split}/{suite}_offline.parquet"; atomic_parquet(pd.DataFrame(rows),path)
    return path

def episode(env,wrapper,task,initial,seed,patch,runtime):
    from VLAAttacker.multiview.openvla_oft_experiments import process_libero_action
    raw=runtime.reset_wait(env,initial,seed,runtime_config()["num_steps_wait"])
    step,decision,success,traces=0,0,False,[]; limit=runtime_config()["max_steps_by_suite"][wrapper.gen_cfg.task_suite_name]
    while step<limit and not success:
        obs=runtime.observation(raw,runtime_config()); obs=obs if patch is None else paste_numpy(obs,patch)
        with torch.no_grad(): actions=np.asarray(wrapper.predict_action(obs,task.language),dtype=np.float32).reshape(8,7)
        require(np.isfinite(actions).all(),"Non-finite rollout action")
        used=0
        for slot in range(min(8,limit-step)):
            raw,_reward,done,_info=env.step(process_libero_action(actions[slot]).tolist()); step+=1; used+=1
            if done: success=True; break
        traces.append({"decision_block":decision,"executed_slots":used,"step_end":step,
                       "predicted_environment_action_chunk":actions.tolist()}); decision+=1
    return {"success":int(success),"failure":int(not success),"executed_steps":step,"decision_blocks":decision},traces

def closedloop_suite(split,suite,wrapper):
    c=cfg(); runtime=runtime_module(); patches=patch_set(split); tasks=c["task_splits"][suite][split]
    states=c["validation_closedloop_states"] if split=="validation" else c["heldout_closedloop_states"]
    conditions=c["validation_conditions"] if split=="validation" else c["heldout_conditions"]
    source=pd.read_parquet(PROJECT/c["source_run"]/"results/LONG4_ROLLOUTS.parquet")
    source=source.loc[(source.condition=="clean") & (source.suite==suite)]
    rows,traces=[],[]; started=time.monotonic()
    for task_index,task_id in enumerate(tasks):
        env,task,initial=runtime.make_env(runtime_config(),suite,task_id)
        try:
            for state in states:
                old=source.loc[(source.task_id==task_id)&(source.state_index==state)]
                require(len(old)==1,"Missing reusable clean rollout")
                seed=int(old.iloc[0].rollout_seed)
                for condition in conditions:
                    metadata={"split":split,"suite":suite,"task_id":task_id,"state_index":state,
                              "condition":condition,"rollout_seed":seed,"signature":signature()}
                    if condition=="clean":
                        base={k:old.iloc[0][k] for k in ("success","failure","executed_steps","decision_blocks")}
                        rows.append({**base,**metadata,"reused_clean":True,"deployed_method":None}); continue
                    result,trace=episode(env,wrapper,task,initial[state],seed,patches[condition],runtime)
                    method=condition if condition in c["methods"] else None
                    if split=="heldout" and condition=="winner": method=read_json(ROOT/"results/WINNER.json")["winner"]
                    rows.append({**result,**metadata,"reused_clean":False,"deployed_method":method})
                    traces.extend({**x,**metadata,"deployed_method":method} for x in trace)
        finally: runtime.close_environment(env)
        progress("closedloop_"+split,suite=suite,tasks_complete=task_index+1,tasks_total=len(tasks),episodes=len(rows),elapsed_s=time.monotonic()-started)
    expected=len(tasks)*len(states)*len(conditions)
    require(len(rows)==expected,"Closed-loop grid incomplete")
    rollout=ROOT/f"evaluation/{split}/{suite}_rollouts.parquet"; trace=ROOT/f"evaluation/{split}/{suite}_traces.parquet"
    atomic_parquet(pd.DataFrame(rows),rollout); atomic_parquet(pd.DataFrame(traces),trace)
    return rollout,trace

def suite_worker(split,suite):
    require_gpu(split); require(complete(ROOT/"reports/TRAIN_COMPLETE.json"),"Training incomplete")
    cache_receipts=[]
    if split=="heldout":
        require(complete(ROOT/"reports/SELECT_COMPLETE.json"),"Winner must be sealed before held-out access")
        from cache import cache_heldout_suite
        cache_receipts=cache_heldout_suite(suite)
    wrapper=load_wrapper(suite); engine=Engine(wrapper)
    offline=offline_suite(split,suite,engine); rollout,trace=closedloop_suite(split,suite,wrapper)
    receipt=ROOT/f"evaluation/{split}/{suite}_COMPLETE.json"
    finish(receipt,[offline,rollout,trace,*cache_receipts],split=split,suite=suite,**resources())

def run_split(split):
    require(split in ("validation","heldout"),"Unknown evaluation split")
    artifacts=[]
    for suite in cfg()["suites"]:
        receipt=ROOT/f"evaluation/{split}/{suite}_COMPLETE.json"
        if not complete(receipt):
            command=[PYTHON,"-B",str(Path(__file__).resolve()),"--split",split,"--suite",suite]
            cost=ROOT/f"reports/COSTS/{split}_{suite}_{time.time_ns()}.json"
            environment={**os.environ,"RELATION_SUITE_COST_PATH":str(cost)}
            started=time.monotonic()
            atomic_json({"status":"LAUNCHING","stage":split,"suite":suite,"signature":signature(),
                         "measurement_scope":"child_process","command":command},cost)
            response=subprocess.run(command,check=False,env=environment)
            if response.returncode!=0:
                child=read_json(cost)
                if child.get("status")!="FAILED":
                    atomic_json({**child,"status":"PROCESS_EXITED_WITHOUT_FINAL_COST",
                                 "exit_code":response.returncode,"parent_observed_elapsed_s":time.monotonic()-started,
                                 "metrics_unavailable_reason":"Child terminated before writing final process metrics",
                                 "process_rss_hwm_bytes":None,"cuda_peak_allocated_bytes":None,
                                 "cuda_peak_reserved_bytes":None},cost)
            require(response.returncode==0,f"{split}/{suite} child failed with {response.returncode}")
        artifacts.append(receipt)
    finish(ROOT/f"reports/{split.upper()}_COMPLETE.json",artifacts,split=split,
           offline_frames=160,closedloop_records=40 if split=="validation" else 64,
           expected_new_rollouts=32 if split=="validation" else 48)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--split",choices=("validation","heldout"),required=True)
    parser.add_argument("--suite",choices=tuple(cfg()["suites"]),required=True); args=parser.parse_args()
    with cost_attempt(args.split,suite=args.suite,path=os.environ.get("RELATION_SUITE_COST_PATH")):
        suite_worker(args.split,args.suite)

if __name__=="__main__":main()
