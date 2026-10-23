"""Same-observation diagnosis after formal rollouts; never steps the simulator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from common import atomic_json, atomic_parquet, require
from deployment import paste_png
from engine import OFTEngine
from monitor import observed_forward, pair_summary, preserve_rng

def diagnose(wrapper,cases,folder,task_id):
    engine=OFTEngine(wrapper)
    rows=[]
    with preserve_rng(),torch.no_grad():
        for case in cases:
            record={"observation":case["observation"],"task_name":case["task_name"]}
            inputs,proprio=engine.base(record)
            ref={"inputs":inputs,"proprio":proprio}
            clean_action,clean=observed_forward(engine,ref)
            rgb=case["patch_rgb"]
            attacked_obs=record["observation"] if rgb is None else paste_png(record["observation"],rgb)
            patched=wrapper.prepare_hf_inputs(attacked_obs,record["task_name"])["pixel_values"]
            action,current=observed_forward(engine,ref,patched)
            env_action=wrapper.vla._unnormalize_actions(action.cpu().numpy()[0],wrapper.gen_cfg.unnorm_key)
            parity=float(np.abs(env_action-np.asarray(case["formal_action_chunk"])).max())
            require(parity<=.02,"Replay differs from the actual formal deployment action")
            restored,_,_=engine.forward(inputs,proprio,patched,replacement=clean["dino"])
            identity,_,_=engine.forward(inputs,proprio,replacement=clean["dino"])
            identity_error=float((identity-clean_action).abs().max())
            require(identity_error<=1e-4,"Clean DINO identity replay failed")
            gap=float((action-clean_action).norm())
            residual=float((restored-clean_action).norm())
            row={k:case[k] for k in ("suite","task_id","state_index","condition","decision_block")}
            row.update(replay_environment_action_max_error=parity,identity_action_max_error=identity_error,
                       same_state_action_gap_l2=gap,dino_restored_action_gap_l2=residual,
                       dino_restore_gap_reduction=gap-residual,
                       restore_conclusion_scope="same observation; off-manifold branch intervention; no rollout success claim")
            for name in clean:
                if name in current:row.update({name+"_"+k:v for k,v in pair_summary(clean[name],current[name]).items()})
            row["action_axis_rms"]=(action-clean_action).float().square().mean((0,1)).sqrt().cpu().tolist()
            row["action_slot_rms"]=(action-clean_action).float().square().mean((0,2)).sqrt().cpu().tolist()
            rows.append(row)
            del clean,current,action,restored,identity
    atomic_parquet(pd.DataFrame(rows),folder/f"task_{task_id:02d}_REPLAY.parquet")
    atomic_json({"status":"COMPLETE","cases":len(rows),"formal_rollouts_modified":False},
                folder/f"task_{task_id:02d}_REPLAY_STATUS.json")
    print(f"INTERNAL_REPLAY_COMPLETE task={task_id} cases={len(rows)}",flush=True)
