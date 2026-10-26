"""Source-copy OFT-UADA, replay/DINO sampling, and WHEN-DINO, global batch eight."""
from __future__ import annotations
import argparse, fcntl, json, os, random, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/"scripts"),str(ROOT/"VLAAttacker")]
from source_axis import ExploringOFTDINOStateAxisAttacker, _rng_state, _restore_rng, _save_checkpoint
from source_original_oft import ExploringOFTOpenVLAAttacker as Original
from source_optimizer import init_uada_patch, build_uada_adamw, step_uada_patch
from white_patch.UADA_OFT_DINO import atomic_json,atomic_torch,atomic_parquet,require,patch_metrics,sha256_file
from white_patch.UADA_OFT_ILA import seed_transform
from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import (
    canonical_quantized_patch, patch_tensor_sha256, _terminal_float32_rounding_evidence)
from objectives import feature_objective,mean_gradient

def validate_resume(saved,cfg):
    outer=saved["outer"]
    require(saved["config"]==cfg and 0<=outer<=cfg["outer_iterations"] and len(saved["rows"])==outer,
            "Resume config/outer/rows drift")
    expected=outer*cfg["inner_loop"]
    require(saved["scheduler"]["last_epoch"]==outer,"Scheduler reset")
    states=saved["optimizer"]["state"]
    require((not states and expected==0) or (len(states)==1 and all(
        int(s["step"])==expected and bool(torch.isfinite(s["exp_avg"]).all())
        and bool(torch.isfinite(s["exp_avg_sq"]).all()) for s in states.values())),
        "Adam state/step reset")
    require(all(r["outer_iteration"]==i and r["optimizer_updates"]==i*cfg["inner_loop"]
                and r["sample_backprops"]==i*cfg["inner_loop"]*8
                for i,r in enumerate(saved["rows"],1)),"Resume accounting drift")
    for key in ("patch","initial"):
        p=saved[key]
        require(tuple(p.shape)==tuple(cfg["patch_shape"]) and bool(torch.isfinite(p).all())
                and bool(((p>=0)&(p<=1)).all()),"Invalid resume RGB")
    require(saved.keys()>={"torch_rng","numpy_rng","python_rng","cuda_rng","source","data",
                          "score_history","scores"},"Incomplete resume")
    refresh=[x for x in cfg["refresh_after_outer"] if x<outer] if saved["method"]=="dino_sampling_b8" else []
    require([h["outer"] for h in saved["score_history"]]==refresh,"Resume scoring drift")

class Batch8Attacker(ExploringOFTDINOStateAxisAttacker):
    def _patched_pixels(self,observation,patch,geometry):
        if self.active_method=="original_uada_b8":
            return Original._patched_pixels(self,observation,patch,geometry)
        return super()._patched_pixels(observation,patch,geometry)

    def _wrist_pixels(self,observation,patch,geometry):
        if self.active_method=="original_uada_b8":
            full=self._patched_pixels(observation,patch,geometry)
            return full[:,6:9],torch.zeros(256,device=full.device)
        return super()._wrist_pixels(observation,patch,geometry)

    def train(self,clean,replay,cfg,method,source,data,model_load_s=0.):
        from dino_batch8_common import draw_batch
        self.active_method=method
        self.transfer_diverse=False
        out=Path(cfg["output_root"])/"training"/method
        out.mkdir(parents=True,exist_ok=True)
        signature=__import__("hashlib").sha256(json.dumps(
            {"cfg":cfg,"method":method,"source":source,"data":data},sort_keys=True).encode()).hexdigest()
        status_path,resume_path=out/"status.json",out/"resume.pt"
        previous=json.loads(status_path.read_text()) if status_path.exists() else {}
        final_path=out/f"outer_{cfg['outer_iterations']:04d}.pt"
        require(not previous or previous["signature"]==signature,"Existing run identity changed")
        if previous.get("status")=="COMPLETE":
            require(previous["final_patch_sha256"]==sha256_file(final_path)
                    and previous["resume_sha256"]==sha256_file(resume_path),"Complete run changed")
            return
        seed_transform(cfg["seed"])
        initial=init_uada_patch(tuple(cfg["patch_shape"]),"cpu").detach()
        patch=initial.to(self.wrapper.device).clone().requires_grad_(True)
        optimizer,scheduler=build_uada_adamw(patch,cfg["learning_rate"],cfg["warmup_outer_iterations"],
                                             cfg["outer_iterations"],cfg["weight_decay"])
        manifest={"config":cfg,"method":method,"signature":signature,"source":source,"data":data,
                  "fresh_start_outer":0,"initial_optimizer_state_empty":True,"parameter_count":7500,
                  "feature_loss":method=="when_dino_b8","appearance_regularization":False,
                  "feature_replacement_during_updates":False,"batch_size":8,"microbatch_size":1,
                  "gradient_accumulation_steps":8,"initial_patch_sha256":patch_tensor_sha256(initial),
                  "evaluation":"normal OFT; one Wrist RGB patch"}
        mpath=out/"RUN_MANIFEST.json"
        if mpath.exists(): require(json.loads(mpath.read_text())==manifest,"Manifest drift")
        else: atomic_json(manifest,mpath)
        snap=out/"source_snapshot.zip"
        if not snap.exists():
            pending=snap.with_suffix(".pending.zip")
            with zipfile.ZipFile(pending,"w",compression=zipfile.ZIP_DEFLATED) as archive:
                for path,h in source.items():
                    require(sha256_file(path)==h,"Snapshot source changed")
                    archive.write(path,arcname=path.lstrip("/"))
            os.replace(pending,snap)
        rows=[];history=[];start=1;prior_elapsed=0.;costs={}
        live=[dict(r) for r in replay]
        if resume_path.exists():
            saved=torch.load(resume_path,map_location="cpu",weights_only=False)
            validate_resume(saved,cfg)
            require(saved["signature"]==signature and saved["source"]==source and saved["data"]==data
                    and saved["method"]==method and torch.equal(saved["initial"],initial),"Resume identity drift")
            with torch.no_grad(): patch.copy_(saved["patch"].to(patch.device))
            optimizer.load_state_dict(saved["optimizer"]);scheduler.load_state_dict(saved["scheduler"])
            rows=saved["rows"];history=saved["score_history"];start=saved["outer"]+1
            live=[dict(r,**s) for r,s in zip(replay,saved["scores"],strict=True)]
            prior_elapsed=max(saved["elapsed_s"],previous.get("elapsed_s",0.))
            costs=dict(saved["costs"])
            for k,v in previous.get("costs",{}).items(): costs[k]=max(costs.get(k,0),v)
            _restore_rng(saved,patch)
        else:
            require(not previous and not optimizer.state,"Status without recoverable state")
        costs["model_load_s"]=costs.get("model_load_s",0.)+model_load_s
        started=time.monotonic()
        completed=start-1
        def elapsed(): return prior_elapsed+model_load_s+time.monotonic()-started
        score_fields=("axis_scores","score_views","score_view_count","score_full_forward_calls",
                      "score_route","dino_score","patched_target_mse","restored_target_mse","dino_delta_rms")
        def scores(): return [{k:r[k] for k in score_fields if k in r} for r in live]
        def save_resume():
            payload={"config":cfg,"method":method,"signature":signature,"source":source,"data":data,
                     "outer":completed,"patch":patch.detach().cpu(),"initial":initial,
                     "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                     "rows":rows,"elapsed_s":elapsed(),"costs":dict(costs),
                     "score_history":history,"scores":scores(),**_rng_state(patch)}
            validate_resume(payload,cfg)
            atomic_torch(payload,resume_path)
        def checkpoint(cp):
            _save_checkpoint(out/f"outer_{cp:04d}.pt",patch,{
                "method":method,"signature":signature,"total_outer":cp,
                "total_optimizer_updates":cp*cfg["inner_loop"],"sample_backprops":cp*cfg["inner_loop"]*8,
                "initial_patch_sha256":patch_tensor_sha256(initial),**patch_metrics(patch,initial.to(patch.device))})
        def status(kind,**extra):
            return {"status":kind,"method":method,"signature":signature,"outer":completed,
                    "optimizer_updates":completed*cfg["inner_loop"],
                    "sample_backprops":completed*cfg["inner_loop"]*8,"elapsed_s":elapsed(),"costs":costs,
                    "score_refresh_outer":[h["outer"] for h in history],**extra}
        low=torch.tensor(cfg["action_head_q01"],device=patch.device).reshape(1,1,7)
        high=torch.tensor(cfg["action_head_q99"],device=patch.device).reshape(1,1,7)
        try:
            if not resume_path.exists():
                checkpoint(0);save_resume()
            state=_rng_state(patch)
            try: alignment=self._validate_runtime(clean[0],patch.detach(),cfg)
            finally:
                _restore_rng(state,patch);self.transfer_diverse=False
            atomic_json(alignment,out/"RUNTIME_ALIGNMENT.json")
            for outer in range(start,cfg["outer_iterations"]+1):
                if method=="dino_sampling_b8" and outer-1 in cfg["refresh_after_outer"]:
                    tick=time.monotonic()
                    live=self._refresh_axis_scores(live,patch,cfg,"dino_remaining",outer-1)
                    item={"outer":outer-1,"records":scores(),"full_forward_calls":len(live)*5,
                          "quantized_patch_sha256":patch_tensor_sha256(canonical_quantized_patch(patch)),
                          "elapsed_s":time.monotonic()-tick}
                    atomic_json(item,out/f"scores_after_{outer-1:04d}.json")
                    history.append({k:v for k,v in item.items() if k!="records"})
                    costs["score_full_forward_calls"]=costs.get("score_full_forward_calls",0)+len(live)*5
                    costs["score_refresh_s"]=costs.get("score_refresh_s",0.)+item["elapsed_s"]
                batch=draw_batch(clean,live,method,outer,cfg)
                prepare_tick=time.monotonic()
                prepared=[]
                for b in batch:
                    record=b["record"]
                    observations=record["observations"] if b["origin"]=="replay" else [record["observation"]]
                    frames=[]
                    for obs in observations:
                        if method=="when_dino_b8":
                            with torch.no_grad(): feat,_=self._features(obs,None,False)
                            costs["clean_dino_forward_calls"]=costs.get("clean_dino_forward_calls",0)+1
                            frames.append((obs,None,None,feat.detach()))
                        else:
                            inputs,proprio=self._base_inputs({"observation":obs,"task_name":record["task_name"]})
                            with torch.no_grad(): action,_,_=self._forward(inputs,proprio,obs,None,False)
                            costs["training_clean_forward_calls"]=costs.get("training_clean_forward_calls",0)+1
                            target=torch.where(action>=(low+high)/2,low,high).detach()
                            frames.append((obs,inputs,proprio,target))
                    prepared.append(frames)
                costs["clean_prepare_s"]=costs.get("clean_prepare_s",0.)+time.monotonic()-prepare_tick
                before_outer=patch.detach().clone();lr=float(optimizer.param_groups[0]["lr"])
                sums={"main_loss":0.,"patch_gradient_norm":0.,"effective_rgb_updates":0,
                      "rgb_step_mean_abs_sum":0.,"rgb_step_max_abs":0.,"feature_l1":0.,
                      "feature_contrast":0.,"feature_valid_tokens":0.}
                terminal=[]
                for inner in range(cfg["inner_loop"]):
                    optimizer.zero_grad(set_to_none=True)
                    def losses():
                        for slot,(b,frames) in enumerate(zip(batch,prepared,strict=True)):
                            seed_transform(cfg["seed"]*100000000+outer*1000+inner*8+slot)
                            obs,inputs,proprio,target=frames[inner%len(frames)]
                            geometry=bool(cfg["geometry"]) and (
                                method=="original_uada_b8" or inner>=cfg["inner_loop"]//2)
                            self.transfer_diverse=bool(method!="original_uada_b8"
                                                      and outer>cfg["bootstrap_outer"] and geometry)
                            try:
                                if method=="when_dino_b8":
                                    adv,coverage=self._features(obs,patch,geometry)
                                    loss,detail=feature_objective(target,adv,coverage,cfg["when_temperature"],
                                                                 cfg["when_lambda_con"])
                                    for k,v in detail.items(): sums[k]+=v/(8*cfg["inner_loop"])
                                else:
                                    action,_,_=self._forward(inputs,proprio,obs,patch,geometry)
                                    axis=b["axis"]
                                    loss=F.mse_loss(action[...,axis],target[...,axis])
                            finally: self.transfer_diverse=False
                            costs["training_patched_forward_calls"]=costs.get("training_patched_forward_calls",0)+1
                            yield loss
                    grad,mean_loss=mean_gradient(patch,losses(),8)
                    if lr!=0. and outer<=cfg["warmup_outer_iterations"]+1:
                        require(float(grad.norm())>1e-12,"Zero early batch gradient")
                    patch.grad=grad
                    before_step=patch.detach().clone()
                    step_uada_patch(patch,optimizer,self.wrapper.vla)
                    costs["actual_optimizer_updates"]=costs.get("actual_optimizer_updates",0)+1
                    delta=(patch.detach()-before_step).abs()
                    sums["effective_rgb_updates"]+=int(float(delta.max())>0.)
                    sums["rgb_step_mean_abs_sum"]+=float(delta.mean())
                    sums["rgb_step_max_abs"]=max(sums["rgb_step_max_abs"],float(delta.max()))
                    sums["main_loss"]+=mean_loss/cfg["inner_loop"]
                    sums["patch_gradient_norm"]+=float(grad.norm())/cfg["inner_loop"]
                    if float(delta.max())==0. and lr!=0. and cfg["outer_iterations"]-outer<2:
                        # Adam increments refer to the moments AFTER this step.
                        evidence=_terminal_float32_rounding_evidence(before_step,optimizer,grad,grad)
                        if evidence is not None: terminal.append(evidence)
                scheduler.step()
                delta=(patch.detach()-before_outer).abs()
                if lr!=0. and float(delta.max())==0.:
                    require(cfg["outer_iterations"]-outer<2 and len(terminal)==cfg["inner_loop"],
                            "RGB did not update despite nonzero LR")
                row={"outer_iteration":outer,"optimizer_updates":outer*cfg["inner_loop"],
                     "sample_backprops":outer*cfg["inner_loop"]*8,"elapsed_s":elapsed(),
                     "learning_rate":lr,"rgb_outer_mean_abs":float(delta.mean()),
                     "rgb_outer_max_abs":float(delta.max()),"terminal_rounding_count":len(terminal),
                     "batch_json":json.dumps([{k:b[k] for k in ("origin","index","base_index","axis","slot")}
                                              for b in batch],sort_keys=True),
                     **sums,**patch_metrics(patch,initial.to(patch.device))}
                rows.append(row);completed=outer
                if outer in cfg["checkpoint_outer_iterations"]: checkpoint(outer)
                save_resume()
                if outer%20==0 or outer==cfg["outer_iterations"]:
                    atomic_parquet(pd.DataFrame(rows),out/"training.parquet")
                    atomic_json(status("RUNNING"),status_path)
                    print(f"BATCH8 method={method} outer={outer}/{cfg['outer_iterations']} "
                          f"loss={sums['main_loss']:.6g} RGBstep={sums['rgb_step_max_abs']:.6g} "
                          f"elapsed_h={elapsed()/3600:.2f}",flush=True)
                if outer==20 and patch.is_cuda:
                    atomic_json({"outer":20,"elapsed_s":elapsed(),
                        "estimated_remaining_s":(elapsed()-model_load_s)/20*(cfg["outer_iterations"]-20),
                        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                        "effective_batch_size":8,"microbatch_size":1},out/"TIMING.json")
            require(all(sha256_file(p)==h for p,h in source.items()),"Training source changed during run")
            atomic_parquet(pd.DataFrame(rows),out/"training.parquet")
            atomic_json(status("COMPLETE",final_patch_sha256=sha256_file(final_path),
                final_png_sha256=sha256_file(final_path.with_suffix(".png")),resume_sha256=sha256_file(resume_path),
                source_snapshot_sha256=sha256_file(snap)),status_path)
        except BaseException as error:
            atomic_json(status("FAILED",attempted_outer=locals().get("outer",start),
                               error=f"{type(error).__name__}: {error}"),status_path)
            raise

def main():
    from dino_batch8_common import (
        METHODS,config,pipeline_preflight,compact_preflight,require_preparation,
        source_preflight,source_records,replay_records,source_manifest)
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--preflight","--dry-run",dest="preflight",action="store_true")
    args=parser.parse_args();cfg=config()
    if args.preflight:
        print(json.dumps(compact_preflight(pipeline_preflight(cfg)),indent=2));return
    require(args.method and os.environ.get("SLURM_JOB_ID"),"Explicit user-submitted Slurm job required")
    preflight=source_preflight(cfg);require_preparation(cfg,preflight)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,"One Slurm-mapped GPU required")
    from UADA_OFT_DINO_wrapper import load_wrapper
    out=Path(cfg["output_root"])/"training"/args.method;out.mkdir(parents=True,exist_ok=True)
    with (out/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tick=time.monotonic();wrapper=load_wrapper(cfg)
        for module in (wrapper.vla,wrapper.action_head,wrapper.proprio_projector,wrapper.noisy_action_projector):
            if module is not None:
                module.eval()
                for p in module.parameters(): p.requires_grad_(False)
        torch.cuda.synchronize();load_s=time.monotonic()-tick
        actual=source_manifest(cfg,wrapper)
        require(all(actual.get(k)==v for k,v in preflight["source"].items()),"Loaded runtime source drift")
        trainer=Batch8Attacker(wrapper,cfg["output_root"],cfg["patch_xy"])
        trainer.train(source_records(cfg),replay_records(cfg)[0],cfg,args.method,actual,preflight["data"],load_s)

if __name__=="__main__": main()
