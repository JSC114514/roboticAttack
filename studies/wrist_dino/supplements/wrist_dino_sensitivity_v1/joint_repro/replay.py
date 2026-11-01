# 发布阅读说明：保留本次完成实验的科学计算与来源检查。函数内已有输入、输出及约束说明；本模块不负责集群提交。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""S03: collect one common B01 on-policy pool, then freeze restoration scores.

24trainingtasks×initial20/21, at most6time-spread native decision observations
per episode. No per-method collection. Source DINO restoration scoring is a
separate no-grad diagnostic, never part of the patch training backward graph.
"""
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import ROOT,config,read,atomic,require,finish,complete,save_npz,Monitor,sha
from .runtime import load_wrapper,Engine,runtime_config,switch_suite
from .evaluate import paste
from .methods import opposite


def record_score(engine,obs,language,patch,view='wrist'):
    """Original score=max(0,L_DINOrestored−L_PNG), using one frozen PNG."""
    with torch.no_grad():
        clean,f0=engine.action(obs,language)
        attacked,fp=engine.action(paste(obs,patch,view),language)
        restored,_=engine.action(paste(obs,patch,view),language,replacements={('dino',1):f0['dino'][1]})
        teacher=opposite(clean);lp=(attacked[...,:1]-teacher[...,:1]).square().mean();lr=(restored[...,:1]-teacher[...,:1]).square().mean()
        return dict(dino_score=max(0.,float(lr-lp)),patched_target_mse=float(lp),restored_target_mse=float(lr),dino_delta_rms=float((fp['dino'][1].float()-f0['dino'][1].float()).square().mean().sqrt()),model_calls=3)


def score_weights(rows,seed=20261026):
    """Source top-half-positive weights2/1; shuffle within task and phase."""
    out=[dict(r) for r in rows]
    for group in sorted({(r['suite'],r['task_id']) for r in out}):
        ids=[i for i,r in enumerate(out) if (r['suite'],r['task_id'])==group]
        ranked=sorted(ids,key=lambda i:(-out[i]['dino_score'],out[i]['record_id']))
        positive=[i for i in ranked if out[i]['dino_score']>0];strong=set(positive[:(len(ranked)+1)//2])
        for i in ids:out[i]['dino_weight']=2. if i in strong else 1.
        for phase in ['early','middle','late']:
            subset=[i for i in ids if out[i]['phase']==phase];weights=[out[i]['dino_weight'] for i in subset]
            random.Random(seed+config()['suites'].index(group[0])*10000+group[1]*100+['early','middle','late'].index(phase)).shuffle(weights)
            for i,w in zip(subset,weights):out[i]['shuffled_weight']=w
    return out


def collect(monitor):
    c=config();require(complete(ROOT/'outputs/training/B01-seed42/COMPLETE.json'),'Freeze completedB01before replay')
    if complete(ROOT/'reports/S03_COMPLETE.json'):return
    png=ROOT/'outputs/training/B01-seed42/final.png';patch=np.asarray(Image.open(png).convert('RGB'),np.uint8)
    w,r=load_wrapper(monitor);engine=Engine(w,checkpoint=False);rows=[];files=[];completed=0
    from experiments.robot.robot_utils import normalize_gripper_action,invert_gripper_action
    for si,suite in enumerate(c['suites']):
        for tid in c['splits'][suite]['train']:
            for state in [20,21]:
                identity=f'{suite}-task{tid:02d}-init{state:02d}';path=ROOT/'data/replay'/identity;path.mkdir(parents=True,exist_ok=True)
                sealed=path/'COMPLETE.json'
                if complete(sealed):rows.extend(read(path/'RECORDS.json'));files.extend([path/'RECORDS.json',path/'OBSERVATIONS.npz']);completed+=1;continue
                env=None
                try:
                    seed=c['seed_base']+si*10000+tid*100+state
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);switch_suite(w,suite)
                    env,task,initial=r.make_env(runtime_config(),suite,tid);require(state<len(initial),'Replay init unavailable')
                    raw=r.reset_wait(env,initial[state],seed,10);obs=r.observation(raw,runtime_config());buffer=[];step=0;success=False
                    while step<c['max_steps'][suite] and not success:
                        buffer.append((step,obs));action=engine.native(paste(obs,patch),task.language)
                        for a in action[:min(8,c['max_steps'][suite]-step)]:
                            command=invert_gripper_action(normalize_gripper_action(a,binarize=True));raw,_,done,_=env.step(command.tolist());env.sim.forward();step+=1;obs=r.observation(raw,runtime_config())
                            if done:success=True;break
                    count=len(buffer);chosen=list(range(count)) if count<=6 else [round(i*(count-1)/5) for i in range(6)]
                    selected=[buffer[i] for i in chosen]
                    shard=path/'OBSERVATIONS.npz';save_npz(shard,front_rgb=np.stack([o['full_image'] for _,o in selected]),wrist_rgb=np.stack([o['wrist_image'] for _,o in selected]),proprio=np.stack([o['state'] for _,o in selected]),language=np.asarray(task.language),anchors=np.asarray([t for t,_ in selected]))
                    saved=[]
                    for j,(t,observation) in enumerate(selected):
                        score=record_score(engine,observation,task.language,patch)
                        saved.append(dict(record_id=identity+f'-decision{chosen[j]:03d}',suite=suite,task_id=tid,trajectory_id=identity,source_timestep=t,phase=('early','middle','late')[min(2,t*3//max(1,step))],path=str(shard.relative_to(ROOT)),row=j,language=task.language,attacked_episode_success=success,**score))
                    atomic(saved,path/'RECORDS.json');finish(sealed,[path/'RECORDS.json',shard],episode_success=success,actual_steps=step)
                    rows.extend(saved);files.extend([path/'RECORDS.json',shard]);completed+=1;monitor.update(completed=completed,total=48)
                finally:
                    if env is not None:r.close_environment(env)
    # Freeze the same survivor-preferred source subset for EVERY replay arm.
    selected=[]
    for suite in c['suites']:
        for tid in c['splits'][suite]['train']:
            group=[r for r in rows if r['suite']==suite and r['task_id']==tid];survivors=[r for r in group if r['attacked_episode_success']]
            require(group,'No replay observations for training task');selected.extend(survivors if survivors else group)
    weighted=score_weights(selected);out=ROOT/'data/REPLAY_MANIFEST.json';atomic(dict(records=weighted,all_collected_records=len(rows),selected_records=len(weighted),frozen_baseline_png_sha256=sha(png),source='fixedB01,all24trainingtasks,init20/21,max6decisions',model_calls_scoring=3*len(rows)),out)
    finish(ROOT/'reports/S03_COMPLETE.json',[out,*files],replay_records=len(weighted))


def run(args):
    with Monitor('S03-replay') as m:collect(m)
