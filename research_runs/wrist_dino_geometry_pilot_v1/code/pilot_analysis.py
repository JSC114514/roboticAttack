"""Episode-paired exploratory statistics; renderer/probe effects are not CASR."""
from __future__ import annotations
from collections import defaultdict
import json
import os
import time
import numpy as np
import pandas as pd
from pilot_common import (ROOT,atlas_path,atomic_json,commit,complete,config,load_json,require)
from pilot_geometry import action_metrics


def interval(values,rows):
    atlas_path()
    from atlas_math import cluster_interval
    return cluster_interval(values,rows,draws=config()['bootstrap_draws'],seed=42)


def table(groups,metrics):
    output=[]
    for group,rows in groups.items():
        out=dict(group)
        for metric in metrics:
            values=[float(r[metric]) if r.get(metric) is not None else float('nan') for r in rows]
            result=interval(values,rows)
            out.update({metric+'_'+key:value for key,value in result.items()})
        out.update(records=len(rows),episodes=len({r['episode_id'] for r in rows}),tasks=len({r['task_id'] for r in rows}))
        output.append(out)
    return output


def grouped(rows,keys):
    result=defaultdict(list)
    for row in rows: result[tuple((key,row[key]) for key in keys)].append(row)
    return result


def failure_report(prep,missing):
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    atomic_json({'status':'INCOMPLETE','signature':prep['signature'],'missing':missing,
                 'scientific_conclusion':'No complete pilot conclusion; inspect failure receipts and stage logs'},ROOT/'results/STATUS.json')
    (ROOT/'results/RESULTS.md').write_text('# Spatial DINO 几何验证\n\n实验未完整完成，暂不报告总体效果。\n\n'+
        '\n'.join('- 缺少：'+str(p) for p in missing)+'\n\n失败和已完成部分均保留；修复后同一提交入口可续跑。\n')


def analyze(prep):
    require(prep['signature'],'Missing preparation signature')
    missing=[str(p.relative_to(ROOT)) for p in [ROOT/'reports/MECHANISM_COMPLETE.json',ROOT/'reports/PILOT_COMPLETE.json']
             if not complete(p,prep['signature'])]
    if missing:
        failure_report(prep,missing)
        raise RuntimeError('Incomplete upstream work; no scientific success claim was generated')
    output=ROOT/'results'; output.mkdir(parents=True,exist_ok=True)
    mechanism=[]; evaluations=[]; baselines=[]
    bystate={r['state_id']:r for r in prep['states']}
    for row in prep['states']:
        mechanism += load_json(ROOT/'data/mechanism'/row['state_id']/'ROWS.json')
        if row['split']=='validation': baselines += load_json(ROOT/'evaluation/baselines'/row['state_id']/'ROWS.json')
    lookup={(r['state_id'],r['baseline'],r['draw']):r for r in baselines}
    for candidate in prep['candidates']:
        for row in prep['states']:
            if row['split']!='validation': continue
            entries=load_json(ROOT/'evaluation'/candidate['candidate_id']/row['state_id']/'ROWS.json')
            condition=next(c for c in row['conditions'] if c['condition_id']==candidate['target']+'_large')
            meta=load_json(ROOT/'data/features'/row['state_id']/'METADATA.json')
            clean=np.asarray(meta['actions']['clean']); site=str(candidate['layer'])
            clean_geometry=np.asarray(meta['monitor_predictions_m']['clean'][site])
            for entry in entries:
                cf_geometry=np.asarray(meta['monitor_predictions_m'][condition['condition_id']][site])
                patched_geometry=np.asarray(entry['monitor_predictions_m'][site])
                entry['geometry_projection_m']=float((patched_geometry[:3]-clean_geometry[:3])@condition['delta_unit'])
                entry['cf_geometry_projection_m']=float((cf_geometry[:3]-clean_geometry[:3])@condition['delta_unit'])
                entry['geometry_fraction']=entry['geometry_projection_m']/condition['displacement_m']
                entry['readout_distance_change_m']=float(patched_geometry[3]-clean_geometry[3])
                entry['true_cf_distance_change_m']=condition['distance_m']-row['truth']['object_eef_distance']
                entry['feature_target_improvement_vs_clean']=None if entry['feature_target_distance'] is None else 1-entry['feature_target_distance']
                for label,baseline in [('uada','uada500_b1'),('initial',f'initial_s{candidate["seed"]}')]:
                    reference=lookup[(row['state_id'],baseline,entry['draw'])]
                    response=action_metrics(reference['actions'],clean,condition)
                    entry[f'projection_minus_{label}']=entry['first_projection']-response['first_projection']
                    entry[f'translation_l2_minus_{label}']=entry['first_translation_l2']-response['first_translation_l2']
                    initial_distance=reference['target_feature_distances'][f'{candidate["target"]}|{candidate["layer"]}']
                    entry[f'feature_distance_improvement_vs_{label}']=(None if initial_distance is None or entry['feature_target_distance'] is None
                                                                     else initial_distance-entry['feature_target_distance'])
                for branch,value in entry.get('branch_restorations',{}).items():
                    entry[branch+'_projection']=value['first_projection']
                    entry[branch+'_projection_drop']=entry['first_projection']-value['first_projection']
                evaluations.append(entry)
    files=[]
    for name,rows in [('MECHANISM_ROWS.json',mechanism),('PNG_ROWS.json',evaluations),('BASELINE_ROWS.json',baselines)]:
        path=output/name; atomic_json(rows,path); files.append(path)
    mechanism_summary=[]
    for visible in (False,True):
        rows=[r for r in mechanism if not visible or (r['clean_subject_visible'] and r['cf_subject_visible'])]
        values=table(grouped(rows,['split','branch','layer','target','strength']),
                     ['first_projection','first_translation_l2','first_rotation_l2','first_gripper_delta'])
        mechanism_summary += [{**r,'visibility_subset':'both_visible' if visible else 'all'} for r in values]
    summary=[]
    metrics=['first_projection','first_translation_l2','feature_target_improvement_vs_clean',
             'feature_distance_improvement_vs_initial','feature_distance_improvement_vs_uada','geometry_projection_m',
             'geometry_fraction','projection_minus_initial','projection_minus_uada','translation_l2_minus_uada',
             'restore_dino_projection_drop','restore_siglip_projection_drop']
    for draw_group in ('fixed','eot'):
        selected=[r for r in evaluations if (r['draw']=='fixed')==(draw_group=='fixed')]
        for visible in (False,True):
            rows=[r for r in selected if not visible or r['cf_subject_visible']]
            values=table(grouped(rows,['candidate_id','target','layer','seed']),metrics)
            summary += [{**r,'draw_group':draw_group,'visibility_subset':'both_visible' if visible else 'all'} for r in values]
    for name,rows in [('MECHANISM_SUMMARY.csv',mechanism_summary),('PNG_SUMMARY.csv',summary),
                      ('PNG_SEED_AGGREGATE.csv',table(grouped(evaluations,['target','layer']),metrics))]:
        path=output/name; pd.DataFrame(rows).to_csv(path,index=False); files.append(path)
    contrasts=[]; pairs=defaultdict(dict)
    for row in mechanism:
        family='direction' if row['target'].startswith('direction') else 'distance'
        key=(row['state_id'],row['branch'],row['layer'],family)
        pairs[key][(row['sign'],row['strength'])]=row
    for key,pair in pairs.items():
        base=next(iter(pair.values()))
        for strength in ('small','large'):
            a,b=pair[(1,strength)],pair[(-1,strength)]
            contrasts.append({**{k:base[k] for k in ('state_id','episode_id','task_id','split','branch','layer')},
                              'family':key[-1],'contrast':'positive_minus_negative_'+strength,
                              'value':a['first_axis_projection']-b['first_axis_projection']})
        for sign in (-1,1):
            contrasts.append({**{k:base[k] for k in ('state_id','episode_id','task_id','split','branch','layer')},
                              'family':key[-1],'contrast':f'large_minus_small_sign{sign}',
                              'value':sign*(pair[(sign,'large')]['first_axis_projection']-pair[(sign,'small')]['first_axis_projection'])})
    path=output/'MECHANISM_CONTRASTS.csv'
    pd.DataFrame(table(grouped(contrasts,['split','branch','layer','family','contrast']),['value'])).to_csv(path,index=False); files.append(path)
    figures=make_figures(prep,summary); files.extend(figures)
    cost_paths=sorted((ROOT/'reports/COSTS').glob('*.json'))
    costs=[load_json(p) for p in cost_paths]
    path=output/'COST_SUMMARY.json'
    atomic_json({'attempts':costs,'failed_attempts':sum(r['status']=='FAILED' for r in costs),
                 'unfinished_or_current_attempts':sum(r['status']=='STARTED' for r in costs),
                 'recorded_elapsed_s':sum(r.get('elapsed_s',0.) for r in costs),
                 'scope':'process HWM and CUDA peaks; sacct sampled MaxRSS is reported separately',
                 'analysis_current_attempt_cost':'written by worker on exit'},path); files.append(path)
    report=output/'RESULTS.md'; write_report(report,summary,mechanism_summary,prep); files.append(report)
    commit(output/'COMPLETE.json',files,prep['signature'],candidates=16,validation_episodes=20)
    atomic_json({'status':'COMPLETE','signature':prep['signature'],'candidates':16,'validation_episodes':20,
                 'closed_loop_casr_measured':False,'parts_annotation_required':False},output/'STATUS.json')


def make_figures(prep,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    output=ROOT/'results/figures'; output.mkdir(parents=True,exist_ok=True); files=[]
    row=prep['states'][0]
    with np.load(ROOT/'data/renders'/row['state_id']/'FRAMES.npz',allow_pickle=False) as z:
        fig,axes=plt.subplots(3,3,figsize=(9,9))
        names=['clean']+[c['condition_id'] for c in row['conditions']]
        for ax,name in zip(axes.flat,names): ax.imshow(z[name+'__rgb']); ax.set_title(name,fontsize=9); ax.axis('off')
        fig.suptitle('Renderer counterfactuals; physics unchanged'); fig.tight_layout()
        path=output/'counterfactuals.png'; fig.savefig(path,dpi=140); plt.close(fig); files.append(path)
    fig,axes=plt.subplots(4,4,figsize=(8,8))
    for ax,c in zip(axes.flat,prep['candidates']):
        with Image.open(ROOT/'training'/c['candidate_id']/f'step_{config()["updates"]:04d}.png') as image: ax.imshow(image)
        ax.set_title(c['candidate_id'],fontsize=7); ax.axis('off')
    fig.tight_layout(); path=output/'final_patches.png'; fig.savefig(path,dpi=140); plt.close(fig); files.append(path)
    rows=[r for r in summary if r['draw_group']=='fixed' and r['visibility_subset']=='all']
    fig,ax=plt.subplots(figsize=(10,5)); values=[]; lower=[]; upper=[]
    for r in rows:
        point=r['first_projection_mean']; values.append(point)
        lower.append(max(0,point-r['first_projection_ci95_low'])); upper.append(max(0,r['first_projection_ci95_high']-point))
    ax.errorbar(np.arange(len(rows)),values,yerr=[lower,upper],fmt='o',capsize=3)
    ax.axhline(0,color='gray',linewidth=.8); ax.set_ylabel('Next-action projection (controller command units)')
    ax.set_xticks(np.arange(len(rows)),[r['candidate_id'] for r in rows],rotation=65,ha='right',fontsize=7)
    ax.set_title('Final PNG, fixed deployment; task -> episode 95% intervals')
    fig.tight_layout(); path=output/'native_action_projection.png'; fig.savefig(path,dpi=150); plt.close(fig); files.append(path)
    # These are weakest directional responses, not claimed closed-loop failures.
    raw=load_json(ROOT/'results/PNG_ROWS.json')
    selected=sorted([r for r in raw if r['draw']=='fixed'],key=lambda r:r['first_projection'])[:6]
    if selected:
        from pilot_pixels import compose,load_png
        from pilot_preprocessing import deployment_uint8
        from pilot_common import observation
        lookup={r['state_id']:r for r in prep['states']}
        fig,axes=plt.subplots(len(selected),3,figsize=(9,3*len(selected)),squeeze=False)
        cases=[]
        for axes_row,entry in zip(axes,selected):
            row=lookup[entry['state_id']]; obs,_=observation(row)
            png=ROOT/'training'/entry['candidate_id']/f'step_{config()["updates"]:04d}.png'
            patched,_=compose(obs['wrist_image'],load_png(png),0,False)
            with np.load(ROOT/'data/renders'/row['state_id']/'FRAMES.npz',allow_pickle=False) as z:
                cf=z[entry['target']+'_large__rgb']
            images=[obs['wrist_image'],cf,deployment_uint8(patched)]
            titles=[f'Clean: {entry["state_id"]}','Render counterfactual',
                    f'{entry["candidate_id"]}\nprojection={entry["first_projection"]:.5f}']
            for ax,value,title in zip(axes_row,images,titles):
                ax.imshow(value); ax.axis('off'); ax.set_title(title,fontsize=9)
            cases.append({k:entry[k] for k in ('state_id','episode_id','task_id','candidate_id','first_projection','feature_target_distance','cf_subject_visible')})
        fig.suptitle('Weakest directional responses; not closed-loop task failures'); fig.tight_layout(rect=(0,0,1,.97))
        path=output/'weak_response_cases.png'; fig.savefig(path,dpi=140); plt.close(fig); files.append(path)
        path=ROOT/'results/WEAK_RESPONSE_CASES.json'
        atomic_json({'selection':'six lowest fixed-PNG projections; may still be positive','cases':cases},path); files.append(path)
    return files


def write_report(path,summary,mechanism_summary,prep):
    fixed=[r for r in summary if r['draw_group']=='fixed' and r['visibility_subset']=='all']
    fixed=sorted(fixed,key=lambda r:r['first_projection_mean'],reverse=True)
    lines=['# Spatial DINO 几何与共享 RGB 验证','',
           '本轮是固定 40 条轨迹上的探索性验证。训练使用 Discovery20，最终 PNG 使用 Validation20；验证数据曾用于旧研究。',
           '渲染反事实、内部表示替换、真实 PNG 动作响应分别报告；本轮没有测量闭环任务失败率或 CASR。','',
           '## 最终 PNG 的原生动作响应','',
           '下表为固定位置的下一条待执行平移动作，在该观测几何目标变化方向上的投影。正值表示朝目标方向变化；单位是控制命令。区间按任务→轨迹重采样，种子与 EOT 不当独立轨迹。','',
           '|候选|方向响应及95%区间|较初始补丁增加|DINO目标距离改善（干净=0）|',
           '|---|---:|---:|---:|']
    for r in fixed:
        def f(key): return '未定义' if r.get(key) is None else f'{r[key]:.5f}'
        lines.append(f'|{r["candidate_id"]}|{f("first_projection_mean")} [{f("first_projection_ci95_low")}, {f("first_projection_ci95_high")}]|{f("projection_minus_initial_mean")}|{f("feature_target_improvement_vs_clean_mean")}|')
    if fixed:
        best=fixed[0]
        lines += ['',f'固定部署下，正向投影均值最高的是 `{best["candidate_id"]}`（{best["first_projection_mean"]:.5f}）。这是查看全部候选后的描述性排序，不能把其区间解释为筛选后仍有保证的确认性结论。']
    lines += ['', '## DINO 内部几何变化是否传到动作','',
              '下面只列 Validation 的 large 条件；内部替换与真实 PNG 分开判断。随机对照逐 token 匹配变化范数。','',
              '|目标/层|DINO-only投影及95%区间|随机DINO投影|原生CF RGB投影|',
              '|---|---:|---:|---:|']
    def select(branch,target,layer):
        return next((r for r in mechanism_summary if r['split']=='validation' and r['strength']=='large'
                     and r['visibility_subset']=='all' and r['branch']==branch and r['target']==target and r['layer']==layer),None)
    for r in mechanism_summary:
        if not (r['split']=='validation' and r['strength']=='large' and r['visibility_subset']=='all' and r['branch']=='dino_only'): continue
        random=select('random_dino',r['target'],r['layer']); native=select('native_cf_rgb',r['target'],22)
        lines.append(f'|{r["target"]}/b{r["layer"]}|{r["first_projection_mean"]:.5f} [{r["first_projection_ci95_low"]:.5f}, {r["first_projection_ci95_high"]:.5f}]|{random["first_projection_mean"]:.5f}|{native["first_projection_mean"]:.5f}|')
    lines += ['', '## 如何理解结果','',
              '- 特征匹配改善而原生动作不响应：当前 RGB 变化未形成所需的控制效应，不能认定目标有效。',
              '- 内部干预响应而 PNG 不响应：主要瓶颈在共享局部 RGB 的可实现性，不能把内部干预效果当补丁结果。',
              '- PNG 响应且恢复干净 DINO 后减弱：提供分支依赖证据；恢复干预可能产生分布外组合，不作唯一因果路径结论。',
              '- UADA 只作相同观测与 EOT 下的动作对照；定向投影高于 UADA 不等于闭环攻击超过 UADA。',
              '- 本轮不按效果门槛删候选；表格排序仅供探索，不是新的确认性检验。','',
              '## 完整结果与边界','',
              '- `MECHANISM_SUMMARY.csv`：两层、DINO/SigLIP/联合/随机/零变化，以及共同可见条件。',
              '- `MECHANISM_CONTRASTS.csv`：正负方向与强弱条件的配对差异。',
              '- `PNG_SUMMARY.csv`：每候选固定部署和 EOT、可见条件、几何读出、初始/UADA配对差异、分支恢复。',
              '- `PNG_SEED_AGGREGATE.csv`：跨种子和 EOT 的轨迹级汇总，重复次数不增加独立样本量。',
              '- `MECHANISM_ROWS.json`、`PNG_ROWS.json`、`BASELINE_ROWS.json`：完整 8×7 输出与状态关联。',
              '- `COST_SUMMARY.json`、`reports/COSTS` 与 `reports/FAILURES`：包含失败成本；Slurm MaxRSS 与进程 HWM 分开解释。','',
              '方位条件保持三维物体—末端距离，距离条件保持水平朝向；两者均保持物体高度。反事实只改变渲染场景，接触/碰撞没有重新模拟；遮挡代理不是完整遮挡真值。',
              '几何变化也会改变边界、显露背景和遮挡；对应特征差异不声称是完全解耦的纯几何坐标。',
              '当前结果支持的最强结论是观测上的几何/分支响应和真实 PNG 动作变化；持续控制和任务失败需后续闭环验证。','',
              '![反事实](figures/counterfactuals.png)','![补丁](figures/final_patches.png)',
              '![动作响应](figures/native_action_projection.png)','![弱响应案例](figures/weak_response_cases.png)','']
    path.write_text('\n'.join(lines))
