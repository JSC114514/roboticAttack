# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""P03: source EOT identity, native policy and natural full-branch gradients.

The GPU witness checks both branches independently without changing training
routing. The two optimizer updates are a dedicated smoke asset, not B results.
"""
import copy
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,config,read,require,atomic,sha,finish,Monitor,resources
from .bank import load_record
from .methods import spec,target,loss,seed_eot,audit_differences
from .runtime import Engine,load_wrapper,switch_suite,policy_observation


def run(args):
    with Monitor('P03_preflight') as monitor:
        require((ROOT/'reports/P01_COMPLETE.json').exists(),'P01 bank not completed')
        c=config();rows=read(ROOT/'data/BANK_MANIFEST.json')['records'];method=spec('B01');w,r=load_wrapper(monitor);engine=Engine(w)
        torch.manual_seed(42)
        from white_patch.UADA_OFT_SOURCE_COPY import init_uada_patch,build_uada_adamw,step_uada_patch
        patch=init_uada_patch((3,50,50),w.device);initial=patch.detach().clone();gates=[]
        # Same patch/seed via the source adapter method and our selected-view
        # path must produce identical normalized12channel pixels. No secondEOT.
        from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker
        reference=ExploringOFTOpenVLAAttacker(w,ROOT/'outputs/preflight')
        for suite in c['suites']:
            row=next(x for x in rows if x['suite']==suite);raw,language=load_record(row);obs=policy_observation(raw);switch_suite(w,suite)
            seed_eot(42,1,0);ours=engine.pixels(obs,patch,'wrist',True)
            seed_eot(42,1,0);old=reference._patched_pixels(obs,patch,True)
            eot_error=float((ours-old).abs().max());require(eot_error==0,'Source EOT pixels changed')
            with torch.no_grad():
                native=engine.native(obs,language);a0,clean=engine.action(obs,language)
                unnorm=w.vla._unnormalize_actions(a0[0].cpu().numpy(),w.gen_cfg.unnorm_key)
            native_error=float(np.max(abs(native-unnorm)));require(native_error<=1e-5,f'Native action interface mismatch: {native_error}')
            seed_eot(42,1,0);a,features=engine.action(obs,language,patch,capture=True)
            objective,_=loss(a,target(a0,method),method)
            tensors=[features['dino'][1],features['siglip'][1],patch]
            gradients=torch.autograd.grad(objective,tensors)
            norms=[float(g.float().norm()) for g in gradients]
            require(all(torch.isfinite(g).all() and float(g.norm())>0 for g in gradients),'Disconnected/nonfinite visual or patch gradient')
            gates.append(dict(suite=suite,eot_max_abs=eot_error,native_max_abs=native_error,dino_grad_norm=norms[0],siglip_grad_norm=norms[1],patch_grad_norm=norms[2]))
            del gradients,features,a,objective,tensors,ours,old
            engine.assert_frozen();monitor.update(completed=len(gates),total=4)
        # Direct vs recomputed language gradients. Exact same seed,input,patch.
        obs,language=load_record(rows[0]);obs=policy_observation(obs);switch_suite(w,rows[0]['suite'])
        with torch.no_grad():a0,_=engine.action(obs,language)
        teacher=target(a0,method)
        seed_eot(42,2,0);a,_=engine.action(obs,language,patch);value,_=loss(a,teacher,method);g_checkpoint=torch.autograd.grad(value,patch)[0].detach();del a,value
        torch.cuda.empty_cache()
        seed_eot(42,2,0)
        with engine.direct():
            a,_=engine.action(obs,language,patch);value,_=loss(a,teacher,method);g_direct=torch.autograd.grad(value,patch)[0].detach()
        relative=float((g_direct-g_checkpoint).norm()/g_direct.norm().clamp_min(1e-12))
        require(relative<=.02,'Language recomputation changed patch gradient')
        del a,value,g_direct,g_checkpoint;torch.cuda.empty_cache()
        optimizer,scheduler=build_uada_adamw(patch,.002,20,500,weight_decay=0.)
        actual=0
        for inner in range(2):
            seed_eot(42,1,inner);optimizer.zero_grad(set_to_none=True)
            a,_=engine.action(obs,language,patch);value,_=loss(a,teacher,method);value.backward()
            require(patch.grad is not None and torch.isfinite(patch.grad).all(),'Invalid smoke gradient')
            step_uada_patch(patch,optimizer,w.vla);actual+=1
        scheduler.step();engine.assert_frozen()
        from .train import save_png
        (ROOT/'outputs/preflight').mkdir(parents=True,exist_ok=True)
        save_png(ROOT/'outputs/preflight/initial.png',initial)
        # Native scheduler initializes at0LR: the firstouter has gradients but
        # zeroRGBchange. Log this fact; do not silently "fix" its source semantics.
        output=ROOT/'reports/P03_VALIDATION.json'
        atomic(dict(status='PASS',gates=gates,checkpoint_gradient_relative_error=relative,smoke_optimizer_steps=actual,smoke_pixel_rms_change=float((patch-initial).square().mean().sqrt()),scheduler_initial_zero_lr=True,source_eot_sha256=sha(Path(c['project'])/'VLAAttacker/white_patch/appply_random_transform.py'),single_factor_diff=audit_differences(),**resources()),output)
        (ROOT/'EOT_GRADIENT_VALIDATION.md').write_text('# EOT与完整梯度预检\n\nPASS。四套件原生动作误差≤1e-5；Exploring→OFT源EOT像素一致；腕部DINO、SigLIP和RGB梯度有限且非零。语言重计算与直接梯度已配对核对。\n\n原scheduler从lr0开始，第一outer的50次更新不会改变像素；本轮保留此源码行为，不改学习率语义。详细日志见reports/P03_VALIDATION.json。\n')
        finish(ROOT/'reports/P03_COMPLETE.json',[output,ROOT/'EOT_GRADIENT_VALIDATION.md'],optimizer_steps=actual)
