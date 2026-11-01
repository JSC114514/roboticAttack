# 发布阅读说明：按回合、任务与套件分析动作和交互代理。帧与动作槽位不是独立统计样本。
# 本文件为原运行源码的注释副本；训练目标、更新规则和原生前向的可执行AST未改变。
"""M03 CPU analysis of observed action deviations, interaction and task failure.

This is paired descriptive analysis, not a new attack objective. First-object
contact/grasp proxies are distinguished from complete task stages in Long.
"""
import numpy as np
from .common import ROOT,config,read,atomic,finish,complete,require


def run():
    if complete(ROOT/'reports/M03_COMPLETE.json'):return
    require(all(complete(ROOT/'reports'/f'{key}_COMPLETE.json') for key in ['M01','M02','M03_INTERACTION_REPLAY']),'M03 diagnostics/interaction labels incomplete')
    summaries=[];stats=read(__import__('pathlib').Path(config()['model_dir'])/'dataset_statistics.json')
    for base in ['development','mechanism_development']:
        directory=ROOT/'outputs/evaluation'/base
        if not directory.exists():continue
        for p in sorted(directory.glob('*/*/RESULT.json')):
            if not (p.parent/'COMPLETE.json').exists():continue
            result=read(p);decisions=read(p.parent/'DECISIONS.json');steps=read(p.parent/'STEPS.json')
            if result['condition'].startswith('M02-'):continue
            # Metadata repairs replay only original controller commands. Read
            # their sidecars without rewriting native outcomes/action receipts.
            repair=p.parent/'INTERACTION_REPLAY_COMPLETE.json';geometry_repaired=False
            if complete(repair):
                labels=read(p.parent/'INTERACTION_REPLAY.json')
                if labels['status']=='MATCHED_REPLAY':
                    before={v['decision']:v['truth'] for v in labels['decisions']};after={v['step']:v['truth'] for v in labels['steps']}
                    for row in decisions:row['truth_before']=before[row['decision']]
                    for row in steps:row['truth']=after[row['step']]
                    geometry_repaired=True
            s=stats[result['suite']+'_no_noops']['action'];lo=np.asarray(s['q01']);hi=np.asarray(s['q99']);mask=np.asarray(s.get('mask',[True]*7));scale=np.where(mask,2/(hi-lo+1e-8),1.)
            dimensions=[];slot=[];phase={};seen_grasp=False;contact_time=None;grasp_time=None
            for row in decisions:
                delta=(np.asarray(row['actions'])-np.asarray(row['same_state_clean']))*scale
                dimensions.append((delta**2).mean(0));slot.append((delta**2).mean(1))
                truth=row.get('truth_before') or{};grasp=bool(truth.get('grasp',False));contact=bool(truth.get('contact',False))
                label='unavailable' if not (result['geometry_available'] or geometry_repaired) else 'grasped' if grasp else 'post_grasp' if seen_grasp else 'contact' if contact else 'pre_contact'
                seen_grasp|=grasp;phase.setdefault(label,[]).append(float((delta[:row['executed_slots']]**2).mean()))
            for step in steps:
                truth=step.get('truth') or{}
                if contact_time is None and truth.get('contact'):contact_time=step['step']
                if grasp_time is None and truth.get('grasp'):grasp_time=step['step']
            movement=sum(float(np.linalg.norm(np.asarray(x['eef_after'])-np.asarray(x['eef_before']))) for x in steps)
            summaries.append(dict(suite=result['suite'],task_id=result['task_id'],state=result['state'],condition=result['condition'],success=result['success'],native_png=result['native_png'],intervention=result.get('intervention'),geometry_available=result['geometry_available'] or geometry_repaired,geometry_from_saved_command_replay=geometry_repaired,action_mse_by_dim=np.mean(dimensions,axis=0).tolist() if dimensions else [],action_mse_by_slot=np.mean(slot,axis=0).tolist() if slot else [],executed_action_mse_by_interaction={k:float(np.mean(v)) for k,v in phase.items()},first_contact_step=contact_time,first_grasp_step=grasp_time,actual_eef_path_m=movement,executed_steps=result['executed_steps']))
    output=ROOT/'reports/M03_SUMMARY.json';atomic(dict(records=summaries,unit='M03',first_entity_interaction_proxy=True,action_slots_not_independent_samples=True),output)
    (ROOT/'M03_RESULTS.md').write_text('# 动作偏差、交互与任务失败\n\n逐起点7维／8槽动作误差、实际末端路程、首次接触／抓持和任务结果保存在reports/M03_SUMMARY.json。接触／抓持阶段指固定首个交互实体，不代表Long任务的所有子目标。内部恢复与原生PNG分开。\n')
    finish(ROOT/'reports/M03_COMPLETE.json',[output,ROOT/'M03_RESULTS.md'],records=len(summaries))
