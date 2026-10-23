"""Validation-only method selection and final held-out reporting."""
from __future__ import annotations
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from common import ROOT, atomic_json, atomic_parquet, cfg, complete, finish, read_json, require, resources, signature

def frames(split,kind):
    paths=[ROOT/f"evaluation/{split}/{suite}_{kind}.parquet" for suite in cfg()["suites"]]
    require(all(p.is_file() for p in paths),f"Missing {split} {kind} files")
    return pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)

def casr_table(rollouts):
    output=[]
    for condition in sorted(set(rollouts.condition)-{"clean"}):
        values=[]
        for (suite,task,state),group in rollouts.groupby(["suite","task_id","state_index"]):
            clean=group.loc[group.condition.eq("clean")]; attacked=group.loc[group.condition.eq(condition)]
            require(len(clean)==len(attacked)==1,"Paired rollout grid changed")
            if int(clean.iloc[0].success)==1:
                values.append({"suite":suite,"task_id":int(task),"state_index":int(state),
                               "switch":int(attacked.iloc[0].success)==0})
        task_means=[]; task_states={}
        if values:
            for (_suite,_task),rows in pd.DataFrame(values).groupby(["suite","task_id"]):
                task_means.append(float(rows.switch.mean()))
                task_states[(_suite,int(_task))]=rows.switch.to_numpy(dtype=float)
        interval=(None,None)
        if task_states:
            rng=np.random.default_rng(42); keys=list(task_states); draws=[]
            for _ in range(cfg()["bootstrap_draws"]):
                selected=rng.choice(len(keys),len(keys),replace=True); means=[]
                for index in selected:
                    states=task_states[keys[index]]
                    means.append(float(rng.choice(states,len(states),replace=True).mean()))
                draws.append(float(np.mean(means)))
            interval=(float(np.quantile(draws,.025)),float(np.quantile(draws,.975)))
        output.append({"condition":condition,"task_equal_casr":float(np.mean(task_means)) if task_means else None,
                       "casr_ci95_low":interval[0],"casr_ci95_high":interval[1],
                       "eligible_clean_starts":len(values),"task_groups":len(task_means),
                       "paired_failure_switches":int(sum(v["switch"] for v in values))})
    return output

def offline_consistency(frame,method):
    rows=frame.loc[frame.condition.eq(method) if method in set(frame.condition) else frame.method.eq(method)].copy()
    require(len(rows)==160,"Expected 160 offline method observations")
    rows["consistent"]=[0.0 if float(norm)==0 else float(float(dot)>0)
                        for dot,norm in zip(rows.native_action_direction_dot,rows.native_dino_cf_direction_l2)]
    trajectories=rows.groupby(["suite","task_id","episode_id"],as_index=False).consistent.mean()
    tasks=trajectories.groupby(["suite","task_id"],as_index=False).consistent.mean()
    return {"task_equal_directional_consistency":float(tasks.consistent.mean()),
            "zero_native_cf_directions":int((rows.native_dino_cf_direction_l2==0).sum()),
            "observations":len(rows),"trajectories":len(trajectories),"tasks":len(tasks)}

def select_winner():
    require(complete(ROOT/"reports/VALIDATION_COMPLETE.json"),"Validation stage incomplete")
    rollouts=frames("validation","rollouts"); offline=frames("validation","offline")
    casr={r["condition"]:r for r in casr_table(rollouts)}; details={}
    for method in cfg()["methods"]:
        require(method in casr,"Candidate missing from validation rollouts")
        details[method]={**casr[method],**offline_consistency(offline,method)}
    a,b=cfg()["methods"]
    casr_a=-1.0 if details[a]["task_equal_casr"] is None else details[a]["task_equal_casr"]
    casr_b=-1.0 if details[b]["task_equal_casr"] is None else details[b]["task_equal_casr"]
    if casr_a>casr_b: winner,rule=a,"task_equal_validation_casr"
    elif casr_b>casr_a: winner,rule=b,"task_equal_validation_casr"
    elif details[a]["task_equal_directional_consistency"]>details[b]["task_equal_directional_consistency"]: winner,rule=a,"offline_directional_consistency_tiebreak"
    elif details[b]["task_equal_directional_consistency"]>details[a]["task_equal_directional_consistency"]: winner,rule=b,"offline_directional_consistency_tiebreak"
    else: winner,rule="bearing","predeclared_bearing_final_tiebreak"
    path=ROOT/"results/WINNER.json"
    atomic_json({"status":"SEALED_BEFORE_HELDOUT","signature":signature(),"winner":winner,"selection_rule":rule,
                 "validation":details,"heldout_used_for_selection":False,
                 "scientific_note":"A ranking is produced even if both candidates have zero CASR; ranking alone is not an efficacy claim."},path)
    finish(ROOT/"reports/SELECT_COMPLETE.json",[path],winner=winner,selection_rule=rule)

def bootstrap_task(rows,value,draws=2000,seed=42):
    rng=np.random.default_rng(seed); groups={k:g for k,g in rows.groupby(["suite","task_id"])}; keys=list(groups)
    if not keys:return {"mean":None,"ci95_low":None,"ci95_high":None}
    def task_value(frame,rng=None):
        trajectories=[g[value].dropna().to_numpy(dtype=float) for _,g in frame.groupby("episode_id")]
        trajectories=[v for v in trajectories if len(v)]
        if not trajectories:return float("nan")
        if rng is None:return float(np.mean([v.mean() for v in trajectories]))
        selected=rng.choice(len(trajectories),len(trajectories),replace=True)
        return float(np.mean([rng.choice(trajectories[i],len(trajectories[i]),replace=True).mean() for i in selected]))
    points=[task_value(groups[k]) for k in keys]; points=[v for v in points if np.isfinite(v)]
    if not points:return {"mean":None,"ci95_low":None,"ci95_high":None,"undefined":int(rows[value].isna().sum())}
    point=float(np.mean(points)); samples=[]
    for _ in range(draws):
        chosen=rng.choice(len(keys),len(keys),replace=True)
        values=[task_value(groups[keys[i]],rng) for i in chosen]; values=[v for v in values if np.isfinite(v)]
        if values:samples.append(float(np.mean(values)))
    return {"mean":point,"ci95_low":float(np.quantile(samples,.025)),"ci95_high":float(np.quantile(samples,.975)),
            "undefined":int(rows[value].isna().sum())}

def analyze():
    require(complete(ROOT/"reports/HELDOUT_COMPLETE.json"),"Held-out confirmation incomplete")
    winner=read_json(ROOT/"results/WINNER.json"); output=ROOT/"results"; output.mkdir(parents=True,exist_ok=True)
    validation_roll=frames("validation","rollouts"); heldout_roll=frames("heldout","rollouts")
    validation_off=frames("validation","offline"); heldout_off=frames("heldout","offline")
    roll_summary=pd.DataFrame([{**r,"split":"validation"} for r in casr_table(validation_roll)]+[{**r,"split":"heldout"} for r in casr_table(heldout_roll)])
    offline_rows=[]
    for split,frame in (("validation",validation_off),("heldout",heldout_off)):
        methods=cfg()["methods"] if split=="validation" else [winner["winner"]]
        for method in methods:
            condition=method if split=="validation" else "winner"
            selected=frame.loc[frame.condition.eq(condition)].copy()
            for metric in ("dino_signed_q","clean_dino_signed_q","dino_signed_q_delta","dino_orthogonal",
                           "clean_dino_orthogonal","dino_target_distance_normalized","first_translation_l2",
                           "native_action_direction_dot","restore_dino_first_translation_l2",
                           "restore_siglip_first_translation_l2"):
                offline_rows.append({"split":split,"method":method,"metric":metric,**bootstrap_task(selected,metric,cfg()["bootstrap_draws"])})
    offline_summary=pd.DataFrame(offline_rows)
    roll_path=output/"ROLLOUT_SUMMARY.csv"; offline_path=output/"OFFLINE_SUMMARY.csv"
    roll_summary.to_csv(roll_path,index=False); offline_summary.to_csv(offline_path,index=False)
    atomic_parquet(validation_roll,output/"VALIDATION_ROLLOUTS.parquet"); atomic_parquet(heldout_roll,output/"HELDOUT_ROLLOUTS.parquet")
    held=next(r for r in casr_table(heldout_roll) if r["condition"]=="winner")
    val=winner["validation"][winner["winner"]]
    def metric(value):
        return "NA" if value is None else f"{value:.4f}"
    report=output/"RESULTS.md"
    report.write_text("\n".join([
        "# Spatial DINO 空间关系补丁结果","",
        f"验证集锁定的方案为 `{winner['winner']}`，选择规则为 `{winner['selection_rule']}`。留出任务没有参与选择。","",
        "## 闭环结果","",
        f"- 胜者验证集 task-equal CASR：{metric(val['task_equal_casr'])} [{metric(val['casr_ci95_low'])}, {metric(val['casr_ci95_high'])}]（干净成功起点 {val['eligible_clean_starts']} 个）。",
        f"- 胜者留出集 task-equal CASR：{metric(held['task_equal_casr'])} [{metric(held['casr_ci95_low'])}, {metric(held['casr_ci95_high'])}]（干净成功起点 {held['eligible_clean_starts']} 个）。",
        "- `ROLLOUT_SUMMARY.csv` 同时给出初始随机补丁、UADA 和候选补丁的配对结果。","",
        "## 解释边界","",
        "训练损失只使用冻结 DINO block 22 的有符号反事实空间方向。闭环行只部署 50x50 PNG，并经过完整 DINO、SigLIP 和原生动作头。",
        "渲染反事实、DINO-only 替换和分支恢复只用于机制诊断，不计作真实补丁成功。A/B 与 UADA 的训练池和更新预算不同，UADA 是部署参考。",
        "离线 `dino_signed_q`、动作方向一致性和分支恢复结果见 `OFFLINE_SUMMARY.csv` 及逐行 parquet；表示变化、动作变化和任务失败应分别解读。","",
        "q 是单个几何方向的投影坐标，q 接近 1 不证明完整空间关系被复现。另报正交残差、到正向目标的归一化距离以及相对干净特征的变化；这些监测不参与补丁优化或胜者选择。","",
        "## 成本","",
        "各 worker 的进程 RSS HWM、CUDA allocated/reserved 峰值和耗时保存在 `reports/COSTS`；Slurm MaxRSS 由状态入口另行显示。",""
    ]))
    artifacts=[roll_path,offline_path,output/"VALIDATION_ROLLOUTS.parquet",output/"HELDOUT_ROLLOUTS.parquet",report,output/"WINNER.json"]
    finish(output/"COMPLETE.json",artifacts,winner=winner["winner"],validation_casr=val["task_equal_casr"],heldout_casr=held["task_equal_casr"])
    atomic_json({"status":"COMPLETE","signature":signature(),"winner":winner["winner"],
                 "closed_loop_casr_measured":True,"heldout_used_for_selection":False},output/"STATUS.json")
