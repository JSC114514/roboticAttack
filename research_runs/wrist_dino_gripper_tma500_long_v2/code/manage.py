#!/usr/bin/env python3
"""CPU stages. Preparation does not interpret a mock PASS as a model result."""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gripper_common import (CONFIG, ROOT, METHODS, atomic_json, atomic_parquet, atomic_torch, check_rgb,
                            config, digest, load_json, preflight, preparation, read_training_pool,
                            reference_patch, require, sha256_file)
from TMA_OFT import validate_resume
from objectives import patch_statistics
from long_source import validate_records, records_digest


def prepare():
    result = preflight()
    path = ROOT/"reports/PREPARATION.json"
    if path.exists():
        require(load_json(path)==result,"Frozen preparation changed; preserve results and inspect")
    else:
        atomic_json(result,path)
    print("PREPARATION_COMPLETE; model/renderer/scientific result NOT_TESTED",flush=True)


def build_pool():
    cfg, prep = config(), preparation()
    items, exclusions, shards, records = [], [], {}, []
    source = None
    for task in range(10):
        path = ROOT/"data/labels"/f"task_{task:02d}.pt"
        receipt = load_json(path.with_suffix(".json"))
        require(receipt["status"]=="COMPLETE" and receipt["records"]==30
                and receipt["training_suite"]==cfg["training_suite"] and receipt["successful_trajectories"]==5
                and receipt["preparation_signature"]==digest(prep) and sha256_file(path)==receipt["sha256"],"Incomplete/changed label shard")
        payload = torch.load(path,map_location="cpu",weights_only=False)
        require(payload["preparation_signature"]==digest(prep) and payload["training_suite"]==cfg["training_suite"]
                and len(payload["records"])==len(payload["items"])==30,"Long label preparation/source changed")
        require([r["bank_index"] for r in payload["records"]]==[i["bank_index"] for i in payload["items"]],
                "Long labels are not paired to collected source observations")
        records.extend(payload["records"])
        if source is None:
            source = payload["source"]
        require(source==payload["source"],"Different model source across label tasks")
        shards[str(task)] = receipt["sha256"]
        for item in payload["items"]:
            require(item["source_suite"]==cfg["training_suite"] and item["task_id"]==task,"Non-Long label in pool")
            require(item["preprocessing"]=="native_uint8_crop_processor_v2" and item["native_close_slots_match"]
                    and item["native_alignment_max_abs"]<=.02, "Old or nonaligned clean labels in repaired pool")
            require(sorted(item["task_permutation"])==sorted(item["background_permutation"])==list(range(256)),
                    "Region permutations changed or incomplete")
            reasons = []
            for key in ("clean_action","clean_dino","clean_siglip"):
                require(torch.isfinite(item[key]).all(),f"Nonfinite clean {key}")
            require(item["close_slots"].shape==(8,) and item["close_slots"].dtype==torch.bool,"Close slot shape/type")
            require(torch.equal(item["close_slots"],item["clean_action"][0,:,6]<.5),"Close target labels differ from fresh prediction")
            task_n, background_n = int(item["task_tokens"].sum()), int(item["background_tokens"].sum())
            if not item["close_slots"].any():
                reasons.append("no_clean_close_slot")
            if task_n==0:
                reasons.append("task_region_not_visible")
            if background_n==0:
                reasons.append("background_region_not_visible")
            if reasons:
                exclusions.append({"bank_index":item["bank_index"],"task_id":task,"reasons":reasons})
            else:
                items.append({k:v for k,v in item.items() if k!="pixel_masks"})
    items.sort(key=lambda r:r["bank_index"])
    require(items and len(items)+len(exclusions)==300 and len({i["bank_index"] for i in items})==len(items),
            "Empty/duplicate/incomplete common pool")
    records.sort(key=lambda r:r["bank_index"])
    validate_records(records,cfg)
    record_signature = records_digest(records)
    bank_path = ROOT/"data/LONG_TRAIN_BANK.pt"
    if bank_path.exists():
        saved_bank = torch.load(bank_path,map_location="cpu",weights_only=False)
        require(saved_bank["training_suite"]==cfg["training_suite"]
                and saved_bank["preparation_signature"]==digest(prep) and saved_bank["label_shards"]==shards
                and saved_bank["actual_runtime_source"]==source and records_digest(saved_bank["records"])==record_signature,
                "Existing Long bank source/records changed; preserve and inspect")
        validate_records(saved_bank["records"],cfg)
    else:
        atomic_torch({"experiment_id":cfg["experiment_id"],"training_suite":cfg["training_suite"],
                      "records":records,"records_signature":record_signature,"label_shards":shards,
                      "actual_runtime_source":source,"preparation_signature":digest(prep)},bank_path)
    bank_sha = sha256_file(bank_path)
    order = []
    epoch = 0
    while len(order)<cfg["outer_iterations"]:
        indices = list(range(len(items)))
        random.Random(cfg["data_seed"]+epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    pool = {"items":items,"training_order":order[:cfg["outer_iterations"]],"label_shards":shards,
            "preprocessing":"native_uint8_crop_processor_v2",
            "training_suite":cfg["training_suite"],"source_bank_sha256":bank_sha,
            "preparation_signature":digest(prep),"actual_runtime_source":source}
    path = ROOT/"data/TRAINING_POOL.pt"
    receipt_path = path.with_name("POOL.json")
    if path.exists() and receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["preparation_signature"]==digest(prep) and receipt["label_shards"]==shards
                and receipt["source_bank_sha256"]==bank_sha and receipt["training_suite"]==cfg["training_suite"]
                and sha256_file(path)==receipt["pool_sha256"],"Existing common pool changed")
        return
    atomic_torch(pool,path)
    coverage = {str(t):sum(i["task_id"]==t for i in items) for t in range(10)}
    result = {"status":"COMPLETE","pool_sha256":sha256_file(path),"preparation_signature":digest(prep),
              "preprocessing":"native_uint8_crop_processor_v2",
              "training_suite":cfg["training_suite"],"source_bank_sha256":bank_sha,"source_records":300,
              "successful_source_trajectories":50,"records_signature":record_signature,
              "label_shards":shards,"accepted_records":len(items),"excluded_records":len(exclusions),
              "exclusions":exclusions,"accepted_by_task":coverage,"methods_share_all_records_and_slots":True,
              "accepted_by_phase":{str(p):sum(i["phase_bucket"]==p for i in items) for p in range(3)},
              "region_pairing":"fixed_permutation_equal_min_count_after_patch_exclusion",
              "task_region_subset_records":sum(int(i["background_tokens"].sum())<int(i["task_tokens"].sum()) for i in items),
              "coverage_warnings":[f"no_accepted_record_for_task_{t}" for t in range(10) if coverage[str(t)]==0],
              "training_order_sha256":digest(pool["training_order"]),
              "region_type":"whole_goal_subject_objects_and_fixtures_plus_gripper"}
    atomic_json(result,receipt_path)
    print(json.dumps({k:result[k] for k in ("status","accepted_records","excluded_records","accepted_by_task")}),flush=True)


def validated_training(write=True):
    cfg, prep = config(), preparation()
    pool, receipt = read_training_pool()
    checkpoints, manifests, initial_patches, metrics = {}, {}, [], []
    for method in METHODS:
        folder = ROOT/"training"/method
        status, manifest = load_json(folder/"status.json"), load_json(folder/"RUN_MANIFEST.json")
        require(status["status"]=="COMPLETE" and status["outer_iterations"]==500
                and status["optimizer_updates"]==25000 and status["signature"]==manifest["signature"],"Training not complete")
        require(manifest["config"]==cfg and manifest["identity"]["pool_sha256"]==receipt["pool_sha256"]
                and manifest["identity"]["source_bank_sha256"]==receipt["source_bank_sha256"]
                and manifest["identity"]["training_suite"]==cfg["training_suite"]
                and manifest["identity"]["preparation_signature"]==digest(prep),"Training input/source changed")
        actual = manifest["identity"]["actual_runtime_source"]
        require(all(sha256_file(path)==checksum for path,checksum in actual.items()),"Loaded training source bytes changed")
        saved = torch.load(folder/"resume.pt",map_location="cpu",weights_only=False)
        validate_resume(saved,manifest["signature"],cfg,method)
        require(saved["outer"]==500 and load_json(folder/"GPU_INTERFACE.json")["status"]=="GPU_INTERFACE_PASS", "GPU interface/update evidence missing")
        history = pd.read_parquet(folder/"TRAIN_HISTORY.parquet")
        require(len(history)==500 and history.outer_iteration.tolist()==list(range(1,501))
                and history.optimizer_updates.tolist()==[50*i for i in range(1,501)],"Training history incomplete")
        require(np.isfinite(history.select_dtypes(include=[np.number]).drop(columns=["cf_loss","cf_wrong_open_rate"],errors="ignore")).all().all(),
                "Nonfinite history; missing plain CF is allowed only in the plain method")
        if method != METHODS[0]:
            require(np.isfinite(history[["cf_loss","cf_wrong_open_rate"]]).all().all(),"Missing/nonfinite CF statistics")
        require(history.bank_index.tolist()==[pool["items"][idx]["bank_index"] for idx in pool["training_order"]],"Method used a different state order")
        initial_patches.append(check_rgb(saved["initial"]))
        require(not torch.equal(saved["patch"],saved["initial"]) and int(history.effective_rgb_updates.sum())>0,
                "Patch never changed; do not run automatic evaluation")
        checkpoints[method] = {}
        for cp in cfg["checkpoint_outer_iterations"]:
            pt, png = folder/f"outer_{cp:04d}.pt", folder/f"outer_{cp:04d}.png"
            payload = torch.load(pt,map_location="cpu",weights_only=False)
            require(payload["signature"]==manifest["signature"] and payload["method"]==method
                    and payload["outer_iterations"]==cp and payload["optimizer_updates"]==cp*50,"Checkpoint budget/source changed")
            patch = check_rgb(payload["patch"])
            expected = patch.permute(1,2,0).mul(255).round().byte().numpy()
            with Image.open(png) as img:
                require(img.mode=="RGB" and np.array_equal(np.asarray(img),expected),"PNG is not exact unsmoothed quantized patch")
            checkpoints[method][str(cp)] = {"pt":sha256_file(pt),"png":sha256_file(png)}
            metrics.append({"method":method,"outer_iteration":cp,**patch_statistics(patch,saved["initial"])})
        manifests[method] = sha256_file(folder/"RUN_MANIFEST.json")
    expected = torch.rand((3,50,50),generator=torch.Generator().manual_seed(cfg["seed"]))
    require(all(torch.equal(p,expected) for p in initial_patches),"Methods did not start from the same random RGB")
    _, ref = reference_patch(cfg)
    result = {"status":"COMPLETE","config_sha256":sha256_file(CONFIG),"pool_sha256":receipt["pool_sha256"],
              "training_suite":cfg["training_suite"],"source_bank_sha256":receipt["source_bank_sha256"],
              "checkpoints":checkpoints,"manifests":manifests,"reference":ref,"from_random_seed":42,
              "optimizer_updates_per_method":25000,"evaluation_model_hooks":False,
              "scientific_result":"AWAITING_NORMAL_RGB_CLOSED_LOOP"}
    if write:
        atomic_json(result,ROOT/"reports/AUTO_VALIDATION.json")
        atomic_parquet(pd.DataFrame(metrics),ROOT/"reports/PATCH_APPEARANCE.parquet")
        print("TRAINING_VALIDATED; starting ordinary RGB evaluation; no visual efficacy assumed",flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","prepare","pool","validate"))
    args = parser.parse_args()
    if args.stage=="preflight":
        result = preflight()
        print(json.dumps({k:v for k,v in result.items() if k not in {"assets","source"}},indent=2))
    else:
        {"prepare":prepare,"pool":build_pool,"validate":validated_training}[args.stage]()
