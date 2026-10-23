"""Two serial shared patches, trained only on signed DINO relation directions."""
from __future__ import annotations
import json, math, os, random, time
import torch
from common import (ROOT, atomic_json, atomic_torch, cfg, complete, finish, initial_patch,
                    load_wrapper, progress, read_json, require, require_gpu, resources, save_png, signature, cache_shard_origin_matches)
from engine import Engine, signed_projection

def learning_rate(step, total):
    c = cfg(); warmup = c["warmup_updates"]
    if step <= warmup: return c["learning_rate"]*step/warmup
    fraction = (step-warmup)/(total-warmup)
    return c["learning_rate"]*0.5*(1+math.cos(math.pi*fraction))

def task_order(epoch):
    c=cfg(); tasks=[(suite,task) for suite in c["suites"] for task in c["task_splits"][suite]["train"]]
    random.Random(c["seed"]*1000003+epoch).shuffle(tasks)
    return tasks

def rows_for_epoch(shard, epoch, suite, task):
    rows=list(shard["records"])
    random.Random(cfg()["seed"]*1000003+epoch*1009+cfg()["suites"].index(suite)*100+task).shuffle(rows)
    return rows

def train_method(method, engine):
    c=cfg(); folder=ROOT/f"training/{method}"; receipt=folder/"COMPLETE.json"
    if complete(receipt): return
    folder.mkdir(parents=True,exist_ok=True)
    total=c["epochs"]*c["updates_per_epoch"]
    initial=initial_patch(engine.wrapper.device)
    patch=torch.nn.Parameter(initial.clone()); optimizer=torch.optim.Adam([patch],lr=c["learning_rate"])
    start_epoch=0; step=0; optimized=0; no_target=0
    resume=folder/"LAST.pt"
    if resume.exists():
        saved=torch.load(resume,map_location="cpu",weights_only=False)
        require(saved["signature"]==signature() and saved["method"]==method,"Training resume identity changed")
        start_epoch=int(saved["epoch"]); step=int(saved["step"]); optimized=int(saved["optimized"]); no_target=int(saved["no_target"])
        require(0<=start_epoch<=c["epochs"] and step==start_epoch*c["updates_per_epoch"],
                "Resume checkpoint epoch/update count mismatch")
        require(optimized+no_target==step,"Resume optimization counters mismatch")
        with torch.no_grad(): patch.copy_(saved["patch"].to(patch.device))
        optimizer.load_state_dict(saved["optimizer"])
    else:
        save_png(patch,folder/"epoch_00.png")
    log=folder/f"ATTEMPT_{time.time_ns()}.jsonl"; started=time.monotonic(); initial_step=step
    with log.open("x") as stream:
        stream.write(json.dumps({"event":"attempt_started","method":method,"resumed_epoch":start_epoch,
                                 "resumed_step":step})+"\n")
    if start_epoch==c["epochs"]:
        # LAST.pt is atomic and already contains the final patch and Adam state.
        save_png(patch,folder/f"epoch_{c['epochs']:02d}.png")
    positive,negative=f"{method}_plus",f"{method}_minus"
    for epoch in range(start_epoch,c["epochs"]):
        epoch_updates=0
        for suite,task in task_order(epoch):
            require(complete(ROOT/f"data/cache/train/{suite}/task_{task:02d}.json"),"Training cache shard is not sealed")
            shard=torch.load(ROOT/f"data/cache/train/{suite}/task_{task:02d}.pt",map_location="cpu",weights_only=False)
            require(cache_shard_origin_matches(shard,ROOT/f"data/cache/train/{suite}/task_{task:02d}.json")
                    and len(shard["records"])==40,"Training cache changed")
            for row in rows_for_epoch(shard,epoch,suite,task):
                step += 1; epoch_updates += 1
                rate=learning_rate(step,total)
                for group in optimizer.param_groups: group["lr"]=rate
                optimizer.zero_grad(set_to_none=True); before=patch.detach().clone()
                spatial,_wrist=engine.spatial_patch(row["observation"],patch)
                result=signed_projection(spatial,row["target_features"][negative],row["target_features"][positive])
                if result is None:
                    no_target+=1; values={"loss":None,"q":None,"orthogonal":None,"target_energy":0.,"gradient_l2":0.}
                else:
                    loss,q,orthogonal,energy=result
                    require(torch.isfinite(loss) and torch.isfinite(q) and torch.isfinite(orthogonal),"Non-finite DINO relation loss")
                    loss.backward(); require(patch.grad is not None and torch.isfinite(patch.grad).all(),"Patch gradient failed")
                    grad=float(patch.grad.norm()); optimizer.step()
                    with torch.no_grad(): patch.clamp_(0,1)
                    optimized+=1; values={"loss":float(loss.detach()),"q":float(q.detach()),
                                          "orthogonal":float(orthogonal.detach()),"target_energy":energy,"gradient_l2":grad}
                require(all(p.grad is None for p in engine.wrapper.vla.parameters()),"Frozen VLA accumulated gradients")
                entry={"method":method,"epoch":epoch+1,"step":step,"state_id":row["state_id"],"suite":suite,
                       "task_id":task,"lr":rate,"pixel_step_l2":float((patch.detach()-before).norm()),
                       "pixel_from_initial_l2":float((patch.detach()-initial).norm()),
                       "optimized":optimized,"no_target":no_target,"elapsed_s":time.monotonic()-started,**values}
                with log.open("a") as stream: stream.write(json.dumps(entry,allow_nan=False)+"\n")
                del spatial,result
            del shard
        require(epoch_updates==c["updates_per_epoch"],"Epoch did not contain exactly 960 candidate updates")
        completed=epoch+1
        if completed in c["checkpoint_epochs"]: save_png(patch,folder/f"epoch_{completed:02d}.png")
        atomic_torch({"signature":signature(),"method":method,"epoch":completed,"step":step,"patch":patch.detach().cpu(),
                      "optimizer":optimizer.state_dict(),"optimized":optimized,"no_target":no_target},resume)
        progress("train",method=method,epoch=completed,epochs=c["epochs"],step=step,total=total,optimized=optimized,
                 no_target=no_target,elapsed_s=time.monotonic()-started,
                 estimated_method_remaining_s=(time.monotonic()-started)/max(completed-start_epoch,1)*(c["epochs"]-completed))
    cost=folder/f"COST_{time.time_ns()}.json"
    atomic_json({"method":method,"elapsed_s":time.monotonic()-started,"candidate_updates":step,
                 "executed_candidate_updates_this_attempt":step-initial_step,
                 "optimized_updates":optimized,"no_target_updates":no_target,"native_action_objective_used":False,
                 "eot_used":False,"validation_used":False,**resources()},cost)
    artifacts=([folder/f"epoch_{e:02d}.png" for e in c["checkpoint_epochs"]]+[resume]
               +sorted(folder.glob("ATTEMPT_*.jsonl"))+sorted(folder.glob("COST_*.json")))
    finish(receipt,artifacts,method=method,candidate_updates=step,optimized=optimized,no_target=no_target)

def run_train():
    require(complete(ROOT/"reports/CACHE_COMPLETE.json"),"Counterfactual cache is incomplete")
    require_gpu("train")
    wrapper=load_wrapper(cfg()["training_checkpoint_suite"]); engine=Engine(wrapper)
    for method in cfg()["methods"]: train_method(method,engine)
    artifacts=[ROOT/f"training/{method}/COMPLETE.json" for method in cfg()["methods"]]
    finish(ROOT/"reports/TRAIN_COMPLETE.json",artifacts,methods=cfg()["methods"],total_candidate_updates=96000,**resources())
