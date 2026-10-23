"""Read-only, hash-locked reuse of v1 RAW data, never reuse its clean anchors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gripper_common import ROOT, digest, load_json, require, sha256_file
from long_source import decision_indices


def recovery_inventory(cfg, current_assets=None):
    manifest = load_json(ROOT/"RECOVERY_SOURCE.json")
    old_root = Path(cfg["recovery_source_root"]).resolve()
    require(old_root == ROOT.with_name("wrist_dino_gripper_tma500_long_v1").resolve()
            and Path(manifest["source_root"]).resolve()==old_root,
            "Recovery must be the specific failed Long v1, not another experiment")
    require(manifest["status"]=="FROZEN_V1_RAW_RECOVERY_SOURCE"
            and manifest["discard_old_clean_action_and_visual_anchors"] is True
            and manifest["total_saved_records"]==228, "Wrong frozen raw recovery manifest")
    old_cfg = load_json(manifest["config_path"])
    old_prep = load_json(manifest["preparation_path"])
    require(sha256_file(manifest["config_path"])==manifest["config_sha256"]==old_prep["config_sha256"]
            and sha256_file(manifest["preparation_path"])==manifest["preparation_sha256"],
            "Old recovery config/preparation bytes changed")
    require(old_cfg["experiment_id"]=="OFT-WRIST-DINO-GRIPPER-TMA500-LONG-V1"
            and old_cfg["training_suite"]==cfg["training_suite"]=="libero_10"
            and old_cfg["training_source"]==cfg["training_source"]
            and old_cfg["evaluation"]==cfg["evaluation"], "Raw recovery source/model/grid differs")
    require(all(sha256_file(p)==h for p,h in old_prep["source"].items()),
            "Frozen v1 source bytes changed; preserve evidence rather than bypassing identity")
    require(set(manifest["tasks"])==set(map(str,range(10))), "Raw recovery tasks incomplete")
    for task, entry in manifest["tasks"].items():
        path = Path(entry["path"]).resolve()
        require(path.is_relative_to(old_root/"data/labels") and sha256_file(path)==entry["sha256"],
                f"Frozen raw recovery shard changed: {task}")
    if current_assets is not None:
        require(all(current_assets[k]==v for k,v in old_prep["assets"].items()),
                "Model/statistics/environment/clean eligibility changed since v1 collection")
    return manifest


def load_recovery_task(cfg, prep, task_id):
    manifest = prep["assets"]["recovery_source"]
    entry = manifest["tasks"][str(task_id)]
    require(sha256_file(entry["path"])==entry["sha256"], "Raw shard changed after preparation")
    old = torch.load(entry["path"],map_location="cpu",weights_only=False)
    old_prep = load_json(manifest["preparation_path"])
    require(old["preparation_signature"]==digest(old_prep) and old["training_suite"]=="libero_10"
            and old["task_id"]==task_id and len(old["records"])==len(old["items"])==entry["records"]
            and old["successful_trajectories"]==entry["successful_trajectories"]
            and len(old["attempts"])==entry["attempts"], "Raw shard identity/counts invalid")
    require(all(sha256_file(p)==h for p,h in old["source"].items()), "Actual v1 runtime source changed")
    require([a["state_index"] for a in old["attempts"]]==cfg["training_source"]["state_indices"][:len(old["attempts"])],
            "Old attempted-state prefix changed")
    successes = [a for a in old["attempts"] if a["accepted"]]
    require(len(successes)==old["successful_trajectories"] and len(old["records"])==6*len(successes),
            "Old accepted trajectories incomplete")
    for rank, attempt in enumerate(successes):
        group = old["records"][rank*6:(rank+1)*6]
        require([r["decision_index"] for r in group]==decision_indices(attempt["decisions"],6),
                "Old selected frames are not full-episode uniform")
        for position, record in enumerate(group):
            require(record["task_id"]==task_id and record["trajectory_rank"]==rank
                    and record["bank_index"]==task_id*30+rank*6+position
                    and record["state_index"]==attempt["state_index"]
                    and record["source_suite"]=="libero_10" and record["episode_success"] is True
                    and record["state_index"] not in cfg["evaluation"]["state_indices"]
                    and record["rollout_seed"]==cfg["training_source"]["seed"]+task_id*100+record["state_index"],
                    "Old raw trajectory metadata changed or leaks evaluation states")
            item = old["items"][rank*6+position]
            require(item["bank_index"]==record["bank_index"], "Raw masks are not paired to observations")
            for name in ("target","gripper","robot"):
                mask = np.asarray(item["pixel_masks"][name])
                require(mask.shape==(256,256) and np.isin(mask,[0,1]).all(), "Old raw geom mask invalid")
    # Old clean_* tensors and close/region labels are deliberately not returned.
    return {"attempts":old["attempts"],"records":[{k:v for k,v in r.items() if k!="clean_action"} for r in old["records"]],
            "pixel_masks":[i["pixel_masks"] for i in old["items"]],"sha256":entry["sha256"]}
