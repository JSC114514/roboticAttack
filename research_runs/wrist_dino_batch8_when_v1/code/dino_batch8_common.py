"""Independent batch8 workflow; old trainers and completed artifacts stay intact."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, random, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/"scripts"), str(ROOT/"VLAAttacker")]
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file
from dino_state_axis_common import source_records, replay_records, frozen_digest, check_rgb, digest_json
CONFIG = RUN/"config.json"
METHODS = ("original_uada_b8", "uniform_replay_b8", "dino_sampling_b8", "when_dino_b8")
TRAIN_ORDER = METHODS
REFERENCES = ("original_uada_b1", "uniform_replay_b1", "dino_sampling_b1")
CHECKPOINTS = (1000,2000)

def config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    require(cfg["experiment_id"] == "OFT-BATCH8-WHEN-DINO2000-V1"
            and tuple(cfg["methods"]) == METHODS and cfg["seed"] == 42
            and cfg["original_data_seed"] == 20260920 and cfg["data_seed"] == 20260929
            and cfg["outer_iterations"] == cfg["scheduler_total_outer"] == 2000
            and cfg["inner_loop"] == 50 and cfg["batch_size"] == 8
            and cfg["microbatch_size"] == 1 and cfg["gradient_accumulation_steps"] == 8
            and cfg["base_outer_iterations"] == 0 and cfg["bootstrap_outer"] == 250
            and cfg["patch_shape"] == [3,50,50] and cfg["patch_xy"] == [26,137]
            and cfg["learning_rate"] == .002 and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20
            and cfg["refresh_after_outer"] == [250,500,1000,1500]
            and cfg["checkpoint_outer_iterations"] == [0,250,500,1000,1500,2000]
            and cfg["eval_checkpoint_outer_iterations"] == [1000,2000]
            and cfg["when_lambda_con"] == 10 and cfg["when_temperature"] == .07
            and cfg["training_gpu_limit"] == 4 and cfg["evaluation_gpu_limit"] == 8
            and cfg["suite"] == "libero_object" and cfg["train_axes"] == [0,1,2]
            and cfg["appearance_regularization"] is False and cfg["hard_projection"] is False,
            "Batch8 identity/budget/threat drift")
    require(Path(cfg["output_root"]).resolve() == RUN.resolve(), "Output scope drift")
    require((ROOT/"VLAAttacker/white_patch/UADA.py").stat().st_mode & 0o222 == 0,
            "Exploring source must remain read-only")
    for key in ("training_bank","replay_bank","eval_reference"):
        require(frozen_digest(cfg,cfg[key]) == cfg[key+"_sha256"], "Frozen data/reference changed")
    ids=json.loads((Path(cfg["checkpoint"])/"config.json").read_text())["timm_model_ids"]
    require("dinov2" in ids[0].lower() and "siglip" in ids[1].lower(), "DINO branch order drift")
    return cfg

def draw_batch(clean, replay, method, outer, cfg):
    """Expand existing shuffled record order into eight distinct records."""
    from source_original_oft import ExploringOFTOpenVLAAttacker as Base
    require(method in METHODS and 1 <= outer <= cfg["outer_iterations"], "Invalid sampler call")
    selected=[]; used=set()
    seed=cfg["original_data_seed"] if method=="original_uada_b8" else cfg["data_seed"]+cfg["seed"]
    local=outer-cfg["bootstrap_outer"]
    origin="clean" if method=="original_uada_b8" or local<=0 or local%2 else "replay"
    phase=("early","middle","late")[(local//2-1)%3] if origin=="replay" else None
    for slot in range(8):
        virtual=(outer-1)*8+slot+1
        base_index=Base._training_index(virtual,len(clean),seed)
        if origin=="clean":
            # A batch may straddle two shuffled epochs; reject cross-epoch repeats.
            while base_index in used:
                virtual+=1
                base_index=Base._training_index(virtual,len(clean),seed)
        record,index,axis=clean[base_index],base_index,0
        if origin=="replay":
            candidates=[(i,j) for i,r in enumerate(replay)
                        if int(r["task_id"])==int(record["task_id"]) and r["phase_proxy"]==phase
                        and i not in used
                        for j in (cfg["train_axes"] if method=="dino_sampling_b8" else [0])]
            require(candidates, "Insufficient unique records in task/phase bucket")
            weights=None
            if method=="dino_sampling_b8":
                lookup={ (i,int(s["axis"])):float(s["dino_remaining_score"])
                         for i,r in enumerate(replay) for s in r.get("axis_scores",[]) }
                scores=np.asarray([lookup.get(k,0.) for k in candidates])
                require(np.isfinite(scores).all() and (scores>=0).all(), "Invalid DINO score")
                if scores.sum()>0:
                    weights=.5/len(scores)+.5*scores/scores.sum()
            rng=random.Random(seed*1000000+outer*100+slot)
            index,axis=candidates[rng.choices(range(len(candidates)),weights=weights,k=1)[0]]
            record=replay[index]
        require(index not in used, "Repeated record in effective batch")
        used.add(index)
        selected.append({"record":record, "origin":origin, "index":index,
                         "base_index":base_index, "axis":int(axis), "slot":slot})
    return selected

def data_identity(cfg):
    return {"clean_bank":frozen_digest(cfg,cfg["training_bank"]),
            "window_bank":frozen_digest(cfg,cfg["replay_bank"]), "config":sha256_file(CONFIG)}

def source_manifest(cfg, wrapper=None):
    from white_patch.UADA_OFT_DINO_STATE_AXIS import source_manifest as previous
    sources=previous(cfg,wrapper)
    paths=list(HERE.glob("*.py"))+[CONFIG,RUN/"PLAN.md",
        ROOT/"submit_exploring_oft_batch8_when2000_v1.sh"]
    paths+=list((ROOT/"slurm/exploring_oft_dino").glob("*batch8_when*.sbatch"))
    for p in paths:
        sources[str(p.resolve())]=sha256_file(p)
    return sources

def source_preflight(cfg):
    sources=source_manifest(cfg)
    for table in ("frozen_inputs","frozen_eval_assets"):
        for path in cfg[table]:
            frozen_digest(cfg,path)
    for name in ("source_exploring.py","source_when.py"):
        require((HERE/name).stat().st_mode & 0o222 == 0, "Author source snapshot must be read-only")
    require(sha256_file(HERE/"source_exploring.py")==sha256_file(ROOT/"VLAAttacker/white_patch/UADA.py")
            =="808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
            "Exploring source copy changed")
    require(sha256_file(HERE/"source_when.py")==sha256_file(cfg["author_when_source"])
            =="6042500f2c8c9a55a06de0d4029d46b63d849148c39852c5c5978cdf9cb676ff",
            "WHEN author source copy changed")
    return {"status":"PASS","source":sources,"data":data_identity(cfg),
            "model_loaded":False,"cuda_initialized":False,"submitted":False}

def external_baselines(cfg):
    require(set(cfg["external_baselines"])=={f"{m}_{cp}" for m in REFERENCES for cp in CHECKPOINTS},
            "B1 reference coverage changed")
    results={}
    for condition,item in cfg["external_baselines"].items():
        path=Path(item["pt"]); cp=item["checkpoint_outer"]
        for file,key in ((path,"pt_sha256"),(path.with_suffix(".png"),"png_sha256"),
                         (Path(item["manifest"]),"manifest_sha256"),(Path(item["history"]),"history_sha256")):
            require(sha256_file(file)==item[key], "Historical baseline changed: "+str(file))
        manifest=json.loads(Path(item["manifest"]).read_text())
        require(manifest["config"]["batch_size"]==1 and manifest["config"]["seed"]==42
                and manifest["config"]["outer_iterations"]==2000, "Baseline lineage drift")
        value=torch.load(path,map_location="cpu",weights_only=False)
        require(value["method"]==item["method"], "Baseline method drift")
        if item["method"]=="exploring_oft_uada":
            require(value["outer_iterations"]==2000 and value["inner_loop"]==50, "Original budget drift")
        else:
            require(value["total_outer"]==cp and value["total_optimizer_updates"]==cp*50
                    and value["signature"]==manifest["signature"], "Sampling budget drift")
        row=pd.read_parquet(item["history"]).loc[lambda f:f.outer_iteration.eq(cp)]
        require(len(row)==1 and int(row.iloc[0].optimizer_updates)==cp*50, "Baseline checkpoint cost missing")
        results[condition]={"patch":check_rgb(path,path.with_suffix(".png")), "pt":item["pt_sha256"],
                            "png":item["png_sha256"],"checkpoint_outer":cp,
                            "recorded_training_elapsed_s":float(row.iloc[0].elapsed_s)}
    return results

def pipeline_preflight(cfg):
    clean=source_records(cfg); replay,_=replay_records(cfg)
    for outer in range(1,2001):
        for method in METHODS:
            draw_batch(clean,replay,method,outer,cfg)
    refs=external_baselines(cfg)
    from evaluate import load_config, frozen_clean, DENOMINATORS
    ev=load_config(cfg)
    require({s:int(frozen_clean(ev,s).success.sum()) for s in DENOMINATORS}==DENOMINATORS,
            "Clean qualification changed")
    return {**source_preflight(cfg),"clean_records":len(clean),"replay_records":len(replay),
        "methods":list(METHODS),"effective_batch_size":8,"optimizer_updates_per_method":100000,
        "sample_backprops_per_method":800000,"maximum_attack_episodes":5600,
        "eligible_rows_expected":5348,"training_gpu_limit":4,"evaluation_gpu_limit":8,
        "baseline_count":len(refs),"fresh_start_outer":0}

def compact_preflight(report):
    result=dict(report); sources=result.pop("source",{})
    result.update(source_files=len(sources),source_manifest_digest=digest_json(sources))
    return result

def prepare(cfg):
    report=pipeline_preflight(cfg)
    result={"status":"COMPLETE","signature":digest_json({"config":cfg,"source":report["source"],
            "data":report["data"]}),"identity":{"config":cfg,"source":report["source"],"data":report["data"]}}
    path=RUN/"reports/PREPARATION.json"
    if path.exists():
        require(json.loads(path.read_text())==result, "Preparation changed")
    else: atomic_json(result,path)
    return {"status":"COMPLETE","signature":result["signature"],"model_loaded":False,"submitted":False}

def require_preparation(cfg, report=None):
    report=report or source_preflight(cfg)
    saved=json.loads((RUN/"reports/PREPARATION.json").read_text())
    require(saved["status"]=="COMPLETE" and saved["signature"]==digest_json(saved["identity"])
            and saved["identity"]=={"config":cfg,"source":report["source"],"data":report["data"]},
            "Preparation/source drift")
    return saved

def validate_training(cfg, methods=METHODS, write_report=True):
    from train import validate_resume
    prepared=require_preparation(cfg)
    clean=source_records(cfg); replay,_=replay_records(cfg)
    hashes={};movement={};initial_hashes={}
    for method in methods:
        folder=RUN/"training"/method
        status=json.loads((folder/"status.json").read_text())
        manifest=json.loads((folder/"RUN_MANIFEST.json").read_text())
        require(status["status"]=="COMPLETE" and status["outer"]==2000
                and status["optimizer_updates"]==100000 and manifest["config"]==cfg
                and status["sample_backprops"]==800000, "Training budget incomplete")
        require(manifest["fresh_start_outer"]==0 and manifest["initial_optimizer_state_empty"]
                and manifest["feature_loss"]==(method=="when_dino_b8")
                and not manifest["appearance_regularization"], "Training mode drift")
        saved=torch.load(folder/"resume.pt",map_location="cpu",weights_only=False)
        validate_resume(saved,cfg)
        require(saved["source"]==manifest["source"] and saved["data"]==data_identity(cfg)
                and saved["signature"]==status["signature"]==manifest["signature"], "Training identity changed")
        require(all(sha256_file(p)==h for p,h in manifest["source"].items()), "Runtime source changed")
        for file,key in (("outer_2000.pt","final_patch_sha256"),("outer_2000.png","final_png_sha256"),
                         ("resume.pt","resume_sha256"),("source_snapshot.zip","source_snapshot_sha256")):
            require(sha256_file(folder/file)==status[key],"Final artifact changed")
        alignment=json.loads((folder/"RUNTIME_ALIGNMENT.json").read_text())
        require(alignment["status"]=="PASS" and alignment["full_forward_calls"]==2
                and len(alignment["records"])==2 and all(
                    r["pixel_exact"] and r["feature_close"] for r in alignment["records"]),
                "Actual OFT/direct Wrist-DINO alignment missing or failed")
        expected=torch.rand((3,50,50),generator=torch.Generator().manual_seed(42))
        require(torch.equal(saved["initial"],expected), "Training used a learned initial patch")
        initial_hashes[method]=hashlib.sha256(expected.numpy().tobytes()).hexdigest()
        rows=pd.read_parquet(folder/"training.parquet")
        require(len(rows)==2000 and np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Training log incomplete/nonfinite")
        live=[dict(r) for r in replay];score_history=[]
        for row in rows.itertuples():
            outer=int(row.outer_iteration)
            if method=="dino_sampling_b8" and outer-1 in cfg["refresh_after_outer"]:
                score=json.loads((folder/f"scores_after_{outer-1:04d}.json").read_text())
                require(len(score["records"])==258 and score["full_forward_calls"]==258*5
                        and score["outer"]==outer-1, "DINO refresh incomplete")
                live=[dict(r,**s) for r,s in zip(replay,score["records"],strict=True)]
                score_history.append(outer-1)
            actual=draw_batch(clean,live,method,outer,cfg)
            spec=[{k:b[k] for k in ("origin","index","base_index","axis","slot")} for b in actual]
            require(json.loads(row.batch_json)==spec, "Batch sequence changed")
            require(row.optimizer_updates==outer*50 and row.sample_backprops==outer*400,
                    "Update/sample accounting changed")
        require(score_history==status["score_refresh_outer"], "Scoring history drift")
        hashes[method]={};movement[method]={};previous=expected
        for cp in cfg["checkpoint_outer_iterations"]:
            path=folder/f"outer_{cp:04d}.pt";value=check_rgb(path,path.with_suffix(".png"))
            payload=torch.load(path,map_location="cpu",weights_only=False)
            require(payload["total_outer"]==cp and payload["total_optimizer_updates"]==cp*50
                    and payload["sample_backprops"]==cp*400 and payload["method"]==method
                    and payload["signature"]==status["signature"], "Checkpoint budget/identity drift")
            if cp==0:
                require(torch.equal(value,expected), "Initial checkpoint differs from seed42 RGB")
            hashes[method][str(cp)]={"pt":sha256_file(path),"png":sha256_file(path.with_suffix(".png"))}
            movement[method][str(cp)]={"mae_from_initial":float((value-expected).abs().mean()),
                                      "mae_from_previous_checkpoint":float((value-previous).abs().mean())}
            previous=value
        require(torch.equal(saved["patch"],previous), "Final checkpoint/resume RGB mismatch")
        require(movement[method]["2000"]["mae_from_initial"]>1e-7, "No real patch update")
    result={"status":"COMPLETE","config_sha256":sha256_file(CONFIG),
            "preparation_signature":prepared["signature"],"checkpoints":hashes,
            "movement_from_initial":movement,"initial_rgb_sha256":initial_hashes,"evaluation_hooks":False,
            "fresh_start_outer":0,"optimizer_updates":100000,"sample_backprops":800000}
    if write_report:
        require(tuple(methods)==METHODS,"Final validation requires all methods")
        atomic_json(result,RUN/"AUTO_VALIDATION.json")
    return result

def submit_chain(cfg):
    (RUN/"logs").mkdir(exist_ok=True);(RUN/"reports").mkdir(exist_ok=True)
    with (RUN/"reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queue=subprocess.run(["squeue","-h","-u",os.environ.get("USER",""),"-o","%i|%j|%b|%R"],
                             capture_output=True,text=True,timeout=30)
        require(queue.returncode==0,"Queue query failed; nothing submitted: "+queue.stderr.strip())
        names={"oft_b8_"+s for s in ("prepare","train","validate","eval","stats")}
        external=set()
        for line in queue.stdout.splitlines():
            fields=line.strip().split("|")
            require(len(fields)==4,"Unknown queue format")
            job,name,resources,reason=fields
            require(name not in names,"Batch8 chain is already queued")
            if name.startswith("oft_") or "gpu" in resources:
                require("DependencyNeverSatisfied" not in reason,"Resolve invalid old dependency: "+job)
                match=re.match(r"^([0-9]+)",job)
                require(match is not None,"Unknown array identity")
                external.add(match[1])
        final=RUN/"evaluation/four_suite/STATUS.json"
        if final.exists() and json.loads(final.read_text()).get("status")=="COMPLETE":
            require(json.loads(final.read_text())["config_sha256"]==sha256_file(CONFIG),"Final config drift")
            return {"status":"ALREADY_COMPLETE","jobs":[]}
        done=[m for m in METHODS if (RUN/"training"/m/"status.json").exists()
              and json.loads((RUN/"training"/m/"status.json").read_text()).get("status")=="COMPLETE"]
        if done: validate_training(cfg,tuple(done),write_report=False)
        pending=[i for i,m in enumerate(TRAIN_ORDER) if m not in done]
        stages=[("prepare","142_batch8_when_prepare_cpu.sbatch",None)]
        if pending: stages.append(("train","143_batch8_when_train_4090.sbatch",
                                  ",".join(map(str,pending))+"%4"))
        stages += [("validate","144_batch8_when_validate_cpu.sbatch",None),
                   ("eval","145_batch8_when_eval_3090.sbatch","0-9%8"),
                   ("stats","146_batch8_when_stats_cpu.sbatch",None)]
        receipt={"status":"SUBMITTING","config_sha256":sha256_file(CONFIG),"jobs":[],
                 "external_wait_jobs":sorted(external,key=int),"training_gpu_limit":4,"evaluation_gpu_limit":8}
        target=RUN/"reports/SUBMISSION_RECEIPT.json"
        if target.exists(): receipt["previous_receipt"]=json.loads(target.read_text())
        atomic_json(receipt,target);previous=None
        for stage,script,array in stages:
            cmd=["sbatch","--parsable","--kill-on-invalid-dep=yes"]
            if previous: cmd.append("--dependency=afterok:"+previous)
            elif external: cmd.append("--dependency=afterany:"+":".join(sorted(external,key=int)))
            if array: cmd.append("--array="+array)
            cmd.append(str(ROOT/"slurm/exploring_oft_dino"/script))
            answer=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            job=answer.stdout.strip().split(";")[0]
            if answer.returncode or not re.fullmatch(r"[0-9]+",job):
                receipt.update(status="SUBMISSION_FAILED",failed_stage=stage,
                               error=answer.stderr.strip() or answer.stdout.strip())
                atomic_json(receipt,target);raise RuntimeError(str(receipt))
            receipt["jobs"].append({"stage":stage,"job_id":job,"command":cmd})
            atomic_json(receipt,target);previous=job
        receipt["status"]="SUBMITTED";atomic_json(receipt,target)
        return receipt

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    for key in ("preflight","prepare","validate","submit"): group.add_argument("--"+key,action="store_true")
    args=parser.parse_args();cfg=config()
    result=(compact_preflight(pipeline_preflight(cfg)) if args.preflight else prepare(cfg) if args.prepare
            else validate_training(cfg) if args.validate else submit_chain(cfg))
    print(json.dumps(result,indent=2))
