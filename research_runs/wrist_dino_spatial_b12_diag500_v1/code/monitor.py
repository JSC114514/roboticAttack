"""Observers only: never merge a diagnostic gradient into the optimizer."""
from __future__ import annotations
import random
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, atomic_torch, config, require
from objectives import action_loss, target_action

@contextmanager
def preserve_rng():
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])
        if saved[3] is not None:torch.cuda.set_rng_state_all(saved[3])

def selected(outer,inner):
    cfg=config()["monitor"]
    return outer in cfg["diagnostic_outer_iterations"] and inner in cfg["diagnostic_inner_iterations"]

def cosine(a,b):
    a,b=a.detach().float().flatten(),b.detach().float().flatten()
    denom=a.norm()*b.norm()
    return float((a*b).sum()/denom) if float(denom)>1e-20 else 0.

def gradient_summary(g,delta):
    g,delta=g.detach().float(),delta.detach().float()
    return {"gradient_rgb_l2":float(g.norm()),"actual_step_l2":float(delta.norm()),
            "gradient_dot_actual_step":float((g*delta).sum()),
            "descent_actual_step_cosine":cosine(-g,delta)}

def pair_summary(clean,current):
    a,b=clean.detach().float(),current.detach().float()
    require(a.shape==b.shape,"Diagnostic feature shape changed")
    return {"relative_l2":float((b-a).norm()/a.norm().clamp_min(1e-12)),
            "cosine":cosine(a,b)}

def batch_cancellation(parts):
    if not parts:return 0.
    denom=sum(float(p.detach().float().norm()) for p in parts)
    return float(torch.stack(parts).sum(0).float().norm())/denom if denom>1e-20 else 0.

@contextmanager
def tap(engine):
    values,handles,counts={},[],{}
    old=getattr(engine,"_diagnostic_observer",None)
    engine._diagnostic_observer=values
    def capture(name,wrist=False):
        def hook(_m,_args,out):
            counts[name]=counts.get(name,0)+1
            if wrist and counts[name]!=2:return
            value=out[0] if isinstance(out,tuple) else out
            values[name]=value[:,int(engine.dino.num_prefix_tokens):] if wrist else value
            # No return: a read-only observer must not replace the activation.
        return hook
    try:
        handles.append(engine.wrapper.vla.projector.register_forward_hook(capture("projector_all_visual_tokens")))
        for i in (8,16,22):
            handles.append(engine.dino.blocks[i].register_forward_hook(capture(f"dino_block{i}_wrist",True)))
        layers=engine.wrapper.vla.language_model.model.layers
        for i in sorted({0,len(layers)//2,len(layers)-1}):
            handles.append(layers[i].register_forward_hook(capture(f"llm_layer{i}_all_tokens")))
        yield values
    finally:
        for h in handles:h.remove()
        engine._diagnostic_observer=old

def observed_forward(engine,ref,pixels=None,replacement=None):
    with tap(engine) as features:
        action,captures,_=engine.forward(ref["inputs"],ref["proprio"],pixels,replacement=replacement)
    return action,{**captures,**features}

def scalar_row(outer,inner,loss,actual_gradient,before,proposal,after,detail):
    step=after.detach()-before
    row={"outer_iteration":outer,"inner_iteration":inner,"main_loss":loss,
         **gradient_summary(actual_gradient,step),
         "rgb_update_rms":float(step.square().mean().sqrt()),
         "rgb_update_max_abs":float(step.abs().max()),
         "quantized_changed_fraction":float((after.mul(255).round()!=before.mul(255).round()).float().mean()),
         "saturated_pixel_fraction":float(((after<=0)|(after>=1)).float().mean()),**detail}
    if proposal is not None:
        removed=proposal-after
        row.update(clipped_channel_fraction=float((removed!=0).float().mean()),
                   clipping_removed_step_l2=float(removed.norm()),
                   unclipped_step_l2=float((proposal-before).norm()))
    return row

def before_probe(engine,records,refs,before,seeds,family,cfg,center,actual_gradient,parts):
    """Two action targets on the exact training input, independent RGB leaf."""
    from batch_runtime import pixel_batch_backward,generator_batch_backward
    result={"rows":[],"grad_uada":torch.zeros_like(before),"grad_tma":torch.zeros_like(before),
            "family":family,"center":center}
    with preserve_rng():
        for record,ref,seed in zip(records,refs,seeds):
            with torch.no_grad():
                clean_action,clean=observed_forward(engine,ref)
                clean={k:v.detach() for k,v in clean.items()}
            probe=before.detach().clone().requires_grad_(True)
            pixels,footprint=engine.pixels(record,probe,seed,diverse=True)
            action,current=observed_forward(engine,ref,pixels)
            targets={f:target_action(record["expert_actions"].to(probe.device),f,cfg) for f in ("uada","tma")}
            losses={f:action_loss(action,targets[f],f,cfg) for f in targets}
            row={"uid":record["uid"],"task_name":record.get("task_name",""),
                 "seed":seed,"footprint_tokens":int((footprint>0).sum())}
            for name in clean:
                if name in current:
                    row.update({name+"_"+k:v for k,v in pair_summary(clean[name],current[name]).items()})
            row["action_clean_distance_l2"]=float((action.detach()-clean_action).norm())
            row["action_axis_l2"]=(action.detach()-clean_action).float().square().mean((0,1)).sqrt().cpu().tolist()
            row["action_slot_l2"]=(action.detach()-clean_action).float().square().mean((0,2)).sqrt().cpu().tolist()
            for f in targets:
                gp,gz=torch.autograd.grad(losses[f],(probe,current["dino"]),retain_graph=f=="uada")
                result["grad_"+f]+=gp.detach()/len(records)
                row[f+"_loss_before"]=float(losses[f].detach())
                row[f+"_clean_loss"]=float(action_loss(clean_action,targets[f],f,cfg))
                row[f+"_dino_local_directional_derivative"]=float((gz.detach().float()*(current["dino"].detach().float()-clean["dino"].float())).sum())
                row[f+"_gradient_vs_training_gradient_cosine"]=cosine(gp,actual_gradient)
            result["rows"].append(row)
            del probe,pixels,action,current,clean,losses
        # A second reproducible EOT draw diagnoses gradient disagreement, not a
        # new training transform distribution. Never touches optimizer leaves.
        alternate=before.detach().clone().requires_grad_(True)
        alternate_seeds=[s+500000000000000 for s in seeds]
        old=getattr(engine,"_training_sample_gradients",None)
        engine._training_sample_gradients=None
        try:
            backward=generator_batch_backward if family.startswith("advencoder_") else pixel_batch_backward
            if family.startswith("advencoder_"):
                backward(engine,records,refs,alternate,alternate_seeds,family,cfg)
            else:
                backward(engine,records,refs,alternate,alternate_seeds,family,cfg,center)
        finally:
            engine._training_sample_gradients=old
        require(alternate.grad is not None,"Diagnostic alternate EOT gradient disconnected")
        result["eot_gradient_cosine"]=cosine(actual_gradient,alternate.grad)
        result["sample_gradient_cancellation_ratio"]=batch_cancellation(parts)
        if family.startswith("advencoder_"):
            from remaining_objectives import rgb_terms
            rgb_regularizer_gradient=torch.zeros_like(before)
            for record,ref,seed in zip(records,refs,seeds):
                leaf=before.detach().clone().requires_grad_(True)
                _,_,rendered=engine.pixels(record,leaf,seed,diverse=True,return_rgb=True)
                penalty,_,_=rgb_terms(ref["clean_rgb"],rendered,cfg["generator"])
                rgb_regularizer_gradient+=torch.autograd.grad(penalty/len(records),leaf)[0].detach()
            feature_gradient=actual_gradient-rgb_regularizer_gradient
            result["generator_terms"]={
                "generator_feature_rgb_gradient_l2":float(feature_gradient.norm()),
                "generator_regularizer_rgb_gradient_l2":float(rgb_regularizer_gradient.norm()),
                "generator_feature_regularizer_gradient_cosine":cosine(feature_gradient,rgb_regularizer_gradient),
                "generator_feature_uada_gradient_cosine":cosine(feature_gradient,result["grad_uada"]),
                "generator_feature_tma_gradient_cosine":cosine(feature_gradient,result["grad_tma"])}
    return result

def after_probe(engine,records,refs,before,after,seeds,cfg,result,method,outer,inner,actual_gradient):
    rows=[]
    with preserve_rng(),torch.no_grad():
        method_loss_change={}
        if "actual_loss_before" in result:
            from batch_runtime import pixel_batch_backward,generator_batch_backward
            with torch.enable_grad():
                probe=after.detach().clone().requires_grad_(True)
                if result["family"].startswith("advencoder_"):
                    loss_after,_=generator_batch_backward(engine,records,refs,probe,seeds,result["family"],cfg)
                else:
                    loss_after,_=pixel_batch_backward(engine,records,refs,probe,seeds,result["family"],cfg,result["center"])
            method_loss_change={"method_same_input_loss_before":result["actual_loss_before"],
                                "method_same_input_loss_after":loss_after,
                                "method_same_input_loss_change":loss_after-result["actual_loss_before"]}
        for record,ref,seed,row in zip(records,refs,seeds,result["rows"]):
            pixels,_=engine.pixels(record,after,seed,diverse=True)
            action,_,_=engine.forward(ref["inputs"],ref["proprio"],pixels)
            for f in ("uada","tma"):
                target=target_action(record["expert_actions"].to(after.device),f,cfg)
                row[f+"_loss_after"]=float(action_loss(action,target,f,cfg))
                row[f+"_same_input_loss_change"]=row[f+"_loss_after"]-row[f+"_loss_before"]
                g=result["grad_"+f]
                row[f+"_batch_gradient_dot_actual_step"]=float((g*(after-before)).sum())
                row[f+"_batch_descent_step_cosine"]=cosine(-g,after-before)
            rows.append({"method":method,"outer_iteration":outer,"inner_iteration":inner,
                         "eot_gradient_cosine":result["eot_gradient_cosine"],
                         "sample_gradient_cancellation_ratio":result["sample_gradient_cancellation_ratio"],
                         **method_loss_change,**result.get("generator_terms",{}),**row})
        if inner==0 and outer in (1,250,500):
            atomic_torch({"method":method,"outer":outer,"inner":inner,"uids":[r["uid"] for r in records],
                          "seeds":seeds,"before_rgb":before.cpu(),"after_rgb":after.cpu(),
                          "actual_training_rgb_gradient":actual_gradient.cpu(),
                          "uada_rgb_gradient":result["grad_uada"].cpu(),
                          "tma_rgb_gradient":result["grad_tma"].cpu()},
                         ROOT/"training"/method/f"diagnostic_vectors_{outer:04d}_{inner:02d}.pt")
    return rows
