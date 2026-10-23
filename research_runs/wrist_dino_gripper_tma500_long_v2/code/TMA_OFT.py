"""Independent OFT adaptation of Exploring source/TMA.py's patchattack loop.

Kept: random full RGB, AdamW, outer iterator, innerLoop EOT, per-inner
step/clamp/zero_grad, and per-outer cosine scheduler. Replaced: discrete-token
target CE/validation and dataloader with continuous gripper targets/region CF
and a frozen success-state iterator. Full pre-adaptation file is in source/.
There is no arbitrary feature-distance auxiliary or appearance constraint.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from PIL import Image

from gripper_common import ROOT, METHODS, atomic_json, atomic_parquet, atomic_torch, digest, require, sha256_file
from objectives import open_loss, patch_statistics, region_value


def save_patch(folder, patch, initial, method, outer, cfg, signature):
    path = Path(folder)/f"outer_{outer:04d}.pt"
    payload = {"experiment_id":cfg["experiment_id"],"method":method,"outer_iterations":outer,
               "optimizer_updates":outer*cfg["inner_loop"],"patch":patch.detach().cpu(),
               "initial_patch":initial.cpu(),"signature":signature,"config":cfg}
    atomic_torch(payload,path)
    rgb = patch.detach().cpu().clamp(0,1).permute(1,2,0).mul(255).round().byte().numpy()
    png = path.with_suffix(".png")
    temporary = png.with_name("."+png.name+".tmp")
    Image.fromarray(rgb).save(temporary,format="PNG")
    temporary.replace(png)
    return {"pt":sha256_file(path),"png":sha256_file(png),**patch_statistics(patch,initial)}


def validate_resume(saved, signature, cfg, method):
    require(saved["signature"]==signature and saved["method"]==method and saved["config"]==cfg,
            "Resume config/source/pool identity changed")
    outer = saved["outer"]
    require(0 <= outer <= cfg["outer_iterations"] and len(saved["rows"])==outer,
            "Resume outer/history incomplete")
    expected = outer * cfg["inner_loop"]
    state = saved["optimizer"]["state"]
    require((not state and expected==0) or (len(state)==1 and all(int(s["step"])==expected
            and torch.isfinite(s["exp_avg"]).all() and torch.isfinite(s["exp_avg_sq"]).all() for s in state.values())),
            "Resume Adam step/moments reset or nonfinite")
    require(saved["scheduler"]["last_epoch"]==outer, "Resume scheduler is not per-outer")
    require(saved.keys()>={"torch_rng","numpy_rng","python_rng","cuda_rng"},"Incomplete resume RNG state")
    for key in ("patch","initial"):
        value = saved[key]
        require(tuple(value.shape)==tuple(cfg["patch_shape"]) and torch.isfinite(value).all()
                and ((value>=0)&(value<=1)).all(),"Resume RGB invalid")


class ExploringOFTGripperTMA:
    def __init__(self, engine):
        self.engine = engine
        self.vla = engine.wrapper.vla

    def objective_step(self, inputs, proprio, record, item, patch, seed, cfg, method, clean):
        engine = self.engine
        pixels, pair, used_seed, retry = engine.eot(record,item,patch,seed,cfg)
        action, caps = engine.forward(inputs,proprio,pixels)
        slots = item["close_slots"].to(patch.device)
        full_loss = open_loss(action,slots,cfg["gripper_target"])
        full_value = float(full_loss.detach())
        full_open = float((action[0,slots,6].detach()>.5).float().mean())
        require(torch.isfinite(full_loss),"Normal opening loss nonfinite")
        full_grad = torch.autograd.grad(full_loss,patch)[0]
        # Do not hold two language model backward graphs at once.
        del pixels, action, caps, full_loss
        cf_value, cf_open, choose_cf = None, None, False
        chosen_grad = full_grad
        if method != METHODS[0]:
            pixels, pair2, seed2, retry2 = engine.eot(record,item,patch,seed,cfg)
            require(seed2==used_seed and retry2==retry and all(torch.equal(a,b) for a,b in zip(pair,pair2)),
                    "Recomputed actual donor uses different EOT/region")
            donor = engine.direct_dino(pixels)
            leaf = donor.detach().clone().requires_grad_(True)
            selected = pair2[0 if method==METHODS[1] else 1]
            cf = region_value(clean["dino"],leaf,selected)
            cf_action, cf_caps = engine.forward(inputs,proprio,replacement=cf)
            require(torch.equal(cf_caps["siglip"],clean["siglip"]),"Region CF changed clean Wrist SigLIP")
            cf_loss = open_loss(cf_action,slots,cfg["gripper_target"])
            require(torch.isfinite(cf_loss),"Region opening loss nonfinite")
            cf_value = float(cf_loss.detach())
            cf_open = float((cf_action[0,slots,6].detach()>.5).float().mean())
            # Exact gradient of max, choosing full on ties. No gradient *2,
            # auxiliary ratio, arbitrary feature target, or extra optimizer step.
            if cf_value > full_value:
                upstream = torch.autograd.grad(cf_loss,leaf)[0]
                del cf_loss, cf_action, cf_caps, cf
                chosen_grad = torch.autograd.grad(donor,patch,grad_outputs=upstream)[0]
                choose_cf = True
        require(torch.isfinite(chosen_grad).all(),"Nonfinite selected RGB gradient")
        return chosen_grad, {"full_loss":full_value,"cf_loss":cf_value,
                              "objective_loss":max(full_value,cf_value) if cf_value is not None else full_value,
                              "full_wrong_open_rate":full_open,"cf_wrong_open_rate":cf_open,
                              "cf_gradient_selected":int(choose_cf),"region_tokens":int(pair[0].sum()),
                              "task_available_tokens":int(item["task_tokens"].sum()),
                              "selected_task_fraction":float(pair[0].sum()/item["task_tokens"].sum()),
                              "eot_retries":retry,"gradient_l2":float(chosen_grad.norm())}

    def patchattack_unconstrained(self, records, pool, cfg, method, identity, folder=None, session_started=None):
        require(method in METHODS,"Unknown TMA method")
        folder = Path(folder) if folder is not None else ROOT/"training"/method
        folder.mkdir(parents=True,exist_ok=True)
        signature = digest({"identity":identity,"config":cfg,"method":method})
        manifest = {"experiment_id":cfg["experiment_id"],"method":method,"signature":signature,
                    "identity":identity,"config":cfg,"parameter_count":7500,"from_random":True,
                    "trainable":"RGB_patch_only","objective":"open_only" if method==METHODS[0] else "max(full_open,region_cf_open)",
                    "base_algorithm":"Exploring TMA adapted from complete source/TMA.py",
                    "same_optimizer_updates_not_equal_compute":True}
        manifest.update(native_rgb_forward="official_uint8_crop_processor",rgb_backward="torch_straight_through",
                        region_pairing="fixed_permutation_equal_min_count_after_patch_exclusion")
        manifest_path = folder/"RUN_MANIFEST.json"
        if manifest_path.exists():
            require(__import__("json").loads(manifest_path.read_text())==manifest,"Existing training provenance changed")
        else:
            atomic_json(manifest,manifest_path)
        status_path, resume_path = folder/"status.json", folder/"resume.pt"
        status = __import__("json").loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status")=="COMPLETE":
            require(status["signature"]==signature and (folder/"outer_0500.pt").is_file(),"Complete training changed")
            print(f"TMA_ALREADY_COMPLETE {method}",flush=True)
            return
        patch = torch.rand(cfg["patch_shape"],generator=torch.Generator().manual_seed(cfg["seed"]))
        initial = patch.clone()
        patch = patch.to(self.engine.wrapper.device).requires_grad_(True)
        # Same Exploring optimizer and scheduler family/clock, not a surrogate
        # low-dimensional image or a learned initialization from UADA.
        optimizer = transformers.AdamW([patch],lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"],no_deprecation_warning=True)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=cfg["warmup_outer_iterations"],
                                                                num_training_steps=cfg["outer_iterations"],num_cycles=.5)
        outer, rows, previous_elapsed = 0, [], 0.
        if resume_path.exists():
            saved = torch.load(resume_path,map_location="cpu",weights_only=False)
            validate_resume(saved,signature,cfg,method)
            patch.data.copy_(saved["patch"].to(patch.device))
            initial = saved["initial"]
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value,torch.Tensor):
                        state[key] = value.to(patch.device)
            outer, rows, previous_elapsed = saved["outer"],saved["rows"],saved["elapsed_s"]
        else:
            save_patch(folder,patch,initial,method,0,cfg,signature)
        started = session_started if session_started is not None else time.monotonic()
        first_item = pool["items"][pool["training_order"][min(outer,cfg["outer_iterations"]-1)]]
        try:
            alignment = self.engine.validate(records[first_item["bank_index"]],first_item,cfg,patch)
            atomic_json(alignment,folder/"GPU_INTERFACE.json")
            if resume_path.exists():
                torch.set_rng_state(saved["torch_rng"])
                np.random.set_state(saved["numpy_rng"])
                random.setstate(saved["python_rng"])
                if patch.is_cuda:
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
            for i in range(outer,cfg["outer_iterations"]):
                item = pool["items"][pool["training_order"][i]]
                record = records[item["bank_index"]]
                inputs, proprio = self.engine.base(record)
                with torch.no_grad():
                    fresh_action, clean = self.engine.forward(inputs,proprio)
                require(torch.equal(fresh_action[0,:,6].cpu()<.5,item["close_slots"]),"Shared close target slots changed")
                clean = {key:value.detach() for key,value in clean.items()}
                before = patch.detach().clone()
                learning_rate = float(optimizer.param_groups[0]["lr"])
                inner_stats, moved = [], 0
                for inner_loop in range(cfg["inner_loop"]):
                    seed = cfg["data_seed"]*1000000+i*cfg["inner_loop"]*cfg["max_eot_attempts"]+inner_loop*cfg["max_eot_attempts"]
                    gradient, metrics = self.objective_step(inputs,proprio,record,item,patch,seed,cfg,method,clean)
                    patch.grad = gradient
                    before_update = patch.detach().clone()
                    optimizer.step()
                    with torch.no_grad():
                        patch.clamp_(0,1)
                    optimizer.zero_grad(set_to_none=True)
                    moved += int(not torch.equal(before_update,patch.detach()))
                    inner_stats.append(metrics)
                scheduler.step()
                summary = {name:float(np.mean([r[name] for r in inner_stats if r[name] is not None]))
                           if any(r[name] is not None for r in inner_stats) else None for name in inner_stats[0]}
                outer = i+1
                elapsed = previous_elapsed+time.monotonic()-started
                rows.append({"outer_iteration":outer,"optimizer_updates":outer*cfg["inner_loop"],
                             "bank_index":item["bank_index"],"task_id":record["task_id"],
                             "close_slots":int(item["close_slots"].sum()),"learning_rate":learning_rate,
                             "effective_rgb_updates":moved,"rgb_outer_mae":float((patch.detach()-before).abs().mean()),
                             "elapsed_s":elapsed,**summary,**patch_statistics(patch,initial)})
                saved = {"signature":signature,"method":method,"config":cfg,"outer":outer,
                         "patch":patch.detach().cpu(),"initial":initial.cpu(),"rows":rows,"elapsed_s":elapsed,
                         "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                         "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),
                         "python_rng":random.getstate(),"cuda_rng":torch.cuda.get_rng_state_all() if patch.is_cuda else []}
                validate_resume(saved,signature,cfg,method)
                atomic_torch(saved,resume_path)
                atomic_parquet(pd.DataFrame(rows),folder/"TRAIN_HISTORY.parquet")
                if outer in cfg["checkpoint_outer_iterations"]:
                    save_patch(folder,patch,initial,method,outer,cfg,signature)
                atomic_json({"status":"RUNNING","signature":signature,"method":method,"outer_iterations":outer,
                             "optimizer_updates":outer*cfg["inner_loop"],"elapsed_s":elapsed,
                             "latest":rows[-1]},status_path)
                if outer%5==0 or outer==1:
                    recent = rows[-min(20,len(rows)):]
                    rate = (recent[-1]["elapsed_s"]-recent[0]["elapsed_s"])/max(1,len(recent)-1)
                    print(f"GRIPPER_TMA {method} {outer}/{cfg['outer_iterations']} full={summary['full_loss']:.5f} "
                          f"cf={summary['cf_loss']} open={summary['full_wrong_open_rate']:.3f} "
                          f"rgb_mae={rows[-1]['rgb_mae_from_initial']:.5f} eta_h={(cfg['outer_iterations']-outer)*rate/3600:.2f}",flush=True)
            atomic_json({"status":"COMPLETE","signature":signature,"method":method,"outer_iterations":outer,
                         "optimizer_updates":outer*cfg["inner_loop"],"elapsed_s":rows[-1]["elapsed_s"],
                         "latest":rows[-1],"scientific_result":"AWAITING_NORMAL_RGB_CLOSED_LOOP"},status_path)
        except BaseException as error:
            atomic_json({"status":"FAILED","signature":signature,"method":method,"last_committed_outer":outer,
                         "error":repr(error),"resume_preserved":resume_path.is_file()},status_path)
            raise
