"""Task-clustered short-window evidence, explicit partials, and single-view exports."""
import numpy as np
import pandas as pd
from va_common import ROOT,PROJECT,config,read,atomic,finish,complete,require,manifest,cases,branch_receipt,branch_files,lock,signature
from va_math import all_conditions,FAULTS,residual,bootstrap,auc

def branch_scores(files,models,calibration,detail,clean):
    with np.load(files['FEATURES.npz'],allow_pickle=False) as z:
        features=z['features'];proprio=z['proprio'];previous=z['previous_features'];lastp=z['previous_proprio']
    decisions=pd.read_parquet(files['DECISIONS.parquet']);rows=[]
    for i,decision in enumerate(decisions.itertuples()):
        row={'start_id':detail['start_id'],'condition':detail['condition'],'task_id':detail['task_id'],'step':int(decision.step)}
        for vi,view in enumerate(('front','wrist')):
            model={k.split('__',1)[1]:v for k,v in models.items() if k.startswith(view+'__')}
            row[view+'_residual']=residual(previous[vi],features[i,vi],proprio[i]-lastp,model)
            row[view+'_drift']=float(np.linalg.norm(features[i,vi]-previous[vi]))
            row[view+'_sharpness']=float(getattr(decision,view+'_sharpness'))
        previous=features[i];lastp=proprio[i];rows.append(row)
    return rows

def summarize(rows,expected):
    frame=pd.DataFrame(rows);summary=[];contrasts=[]
    for name in expected:
        values=[r for r in rows if r['condition']==name]
        summary.append(dict(condition=name,completed=len(values),expected=12,
            contacts=sum(r['first_contact_step'] is not None for r in values),
            grasps=sum(r['first_grasp_step'] is not None for r in values),
            task_successes_within_window=sum(r['task_success_within_window'] for r in values),
            mean_progress_m=float(np.mean([r['actual_progress_to_start_target_m'] for r in values])) if values else None))
    lookup={(r['start_id'],r['condition']):r for r in rows}
    def compare(a,b,label,matched_prefix=False):
        pairs=[]
        for case in cases():
            left=lookup.get((case['episode_id'],a));right=lookup.get((case['episode_id'],b))
            if not left or not right:continue
            h=min(left['executed_steps'],right['executed_steps']) if matched_prefix else min(left['planned_horizon'],right['planned_horizon'])
            if matched_prefix:h=min(h,left['random_matched_prefix_steps'],right['random_matched_prefix_steps'])
            la=left['first_contact_step'];lb=right['first_contact_step']
            pairs.append(dict(task_id=case['task_id'],difference=int(la is not None and la<=h)-int(lb is not None and lb<=h)))
        contrasts.append(dict(label=label,a=a,b=b,pairs=len(pairs),contact_rate_difference=float(np.mean([r['difference'] for r in pairs])) if pairs else None,
                              task_trajectory_95ci=bootstrap(pairs,draws=config()['bootstrap_draws'])))
    for fault in FAULTS:
        raw='a_'+fault+'_raw'
        compare(raw,'clean',fault+' information loss')
        for gate in ('attenuate_damaged','attenuate_other'):compare('a_'+fault+'_'+gate,raw,fault+' '+gate)
    for p in ('relation','uada7'):
        compare('b_'+p+'_first_chunk','b_'+p+'_native',p+' removal after first block')
        compare('b_'+p+'_replay','b_'+p+'_native',p+' clean-camera path',True)
        compare('b_'+p+'_random','b_'+p+'_native',p+' matched random-camera path',True)
        # Primary feedback contrast subtracts each camera path's own no-PNG control.
        differences=[]
        for case in cases():
            names=['b_'+p+'_replay','b_'+p+'_native','b_'+p+'_random','b_'+p+'_random_clean','clean','b_clean_replay']
            vals=[lookup.get((case['episode_id'],n)) for n in names]
            if not all(vals):continue
            h=min(v['executed_steps'] for v in vals);h=min(h,*(v['random_matched_prefix_steps'] for v in vals))
            contacted=lambda v:int(v['first_contact_step'] is not None and v['first_contact_step']<=h)
            rep,nat,rnd,rndclean,cl,clrep=map(contacted,vals)
            # Higher: clean-camera replay repairs patch harm more than a matched random change.
            diff=((rep-clrep)-(nat-cl))-((rnd-rndclean)-(nat-cl))
            differences.append(dict(task_id=case['task_id'],difference=diff))
        contrasts.append(dict(label=p+' camera-specific feedback difference-in-differences',a=p,b='matched camera controls',pairs=len(differences),
            contact_rate_difference=float(np.mean([x['difference'] for x in differences])) if differences else None,
            task_trajectory_95ci=bootstrap(differences,draws=config()['bootstrap_draws'])))
    return summary,contrasts

def warning_statistics(scores,lookup,calibration):
    """AUC units are start/condition; never pool decision frames as replicates."""
    result=[]
    if not len(scores):return result
    for view in ('front','wrist'):
        for metric in ('residual','drift','sharpness'):
            units=[];alarms=[]
            for (start,name),data in scores.groupby(['start_id','condition']):
                if name not in ('clean','a_'+view+'_blur_raw','a_'+view+'_lag_raw'):continue
                d=lookup[(start,name)];clean=lookup.get((start,'clean'))
                if not clean or clean['first_contact_step'] is None:continue
                deadline=clean['first_contact_step'];data=data[data.step<deadline].sort_values('step')
                if not len(data):continue
                series=data[view+'_'+metric].to_numpy();series=-series if metric=='sharpness' else series
                # Primary branch risk score is the first decision, before the policy acts.
                missed=d['first_contact_step'] is None
                units.append(dict(task_id=d['task_id'],start=start,condition=name,label=int(missed),score=float(series[0])))
                threshold= -calibration[view]['sharpness_lower_05'] if metric=='sharpness' else calibration[view][metric+'_threshold_95']
                hits=data.step.to_numpy()[series>threshold]
                alarms.append(dict(task_id=d['task_id'],start_id=start,condition=name,metric=metric,view=view,
                    first_alarm_step=int(hits[0]) if len(hits) else None,
                    lead_controls_relative_to_clean_contact=int(deadline-hits[0]) if len(hits) else None,
                    label='missed first contact within the common short horizon; not eventual task failure'))
            labels=[u['label'] for u in units];auc_value=auc(labels,[u['score'] for u in units]) if units else None
            # Cluster bootstrap duplicates whole starts, retaining their conditions.
            ci=None
            if auc_value is not None:
                tasks=sorted({u['task_id'] for u in units});rng=np.random.default_rng(42);values=[]
                for _ in range(config()['bootstrap_draws']):
                    sample=[]
                    for task in rng.choice(tasks,len(tasks),replace=True):
                        starts=sorted({u['start'] for u in units if u['task_id']==task})
                        for start in rng.choice(starts,len(starts),replace=True):sample.extend(u for u in units if u['task_id']==task and u['start']==start)
                    y=[u['label'] for u in sample]
                    if len(set(y))==2:values.append(auc(y,[u['score'] for u in sample]))
                if values:ci=np.quantile(values,[.025,.975]).tolist()
            clean_alarms=[a['first_alarm_step'] is not None for a in alarms if a['condition']=='clean']
            result.append(dict(view=view,metric=metric,units=len(units),positive_units=sum(labels),AUROC=auc_value,task_start_95ci=ci,
                clean_precontact_window_false_alarm_rate=float(np.mean(clean_alarms)) if clean_alarms else None,
                alarms=alarms,interpretation='AUC undefined when all short-window outcomes are identical'))
    return result

def run(monitor,export=False):
    with lock('report'):
        (ROOT/'results').mkdir(parents=True,exist_ok=True)
        if complete(ROOT/'results/COMPLETE.json'):
            if export:export_videos(monitor)
            return
        value=manifest() if (ROOT/'data/MANIFEST.json').exists() else None
        rows=[];missing=[];receipts=[];all_scores=[];errors=[]
        model_exists=complete(ROOT/'reports/CALIBRATION_COMPLETE.json')
        models=dict(np.load(ROOT/'data/CALIBRATION.npz',allow_pickle=False)) if model_exists else None
        calibration=read(ROOT/'data/CALIBRATION.json') if model_exists else None
        for case in value['cases'] if value else []:
            clean_receipt=branch_receipt(case,'clean');clean=read(branch_files(clean_receipt)['DETAIL.json']) if complete(clean_receipt) else None
            for name in all_conditions():
                receipt=branch_receipt(case,name)
                try:
                    if not complete(receipt):missing.append(dict(start=case['episode_id'],condition=name));continue
                    files=branch_files(receipt);detail=read(files['DETAIL.json']);receipts.append(receipt);rows.append(detail)
                    if model_exists and clean:all_scores.extend(branch_scores(files,models,calibration,detail,clean))
                except Exception as error:
                    errors.append(dict(start=case['episode_id'],condition=name,error=repr(error)))
            monitor.update(phase='short_window_paired_statistics',completed=len(rows),total=288)
        summary,contrasts=summarize(rows,all_conditions()) if value else ([],[])
        flat=[{k:v for k,v in r.items() if k not in ('camera_path','camera_amplitudes')} for r in rows]
        pd.DataFrame(flat).to_csv(ROOT/'results/BRANCHES.csv',index=False)
        pd.DataFrame(summary).to_csv(ROOT/'results/SUMMARY.csv',index=False)
        per_task=[]
        for name in all_conditions():
            for task in config()['tasks']:
                group=[r for r in rows if r['condition']==name and r['task_id']==task]
                per_task.append(dict(condition=name,task_id=task,completed=len(group),expected=2,
                                     first_contacts=sum(r['first_contact_step'] is not None for r in group),
                                     grasps=sum(r['first_grasp_step'] is not None for r in group)))
        pd.DataFrame(per_task).to_csv(ROOT/'results/PER_TASK.csv',index=False)
        pd.DataFrame(contrasts).to_csv(ROOT/'results/CONTRASTS.csv',index=False)
        scores=pd.DataFrame(all_scores);scores.to_csv(ROOT/'results/DECISION_SCORES.csv',index=False)
        lookup={(r['start_id'],r['condition']):r for r in rows}
        warning=warning_statistics(scores,lookup,calibration) if model_exists else []
        atomic(warning,ROOT/'results/WARNING.json')
        # These are descriptive screening leads, not automatic acceptance gates.
        leads=[]
        for view in ('front','wrist'):
            ws={w['metric']:w for w in warning if w['view']==view}
            if ws and all(ws[m]['AUROC'] is not None for m in ('residual','drift','sharpness')):
                leads.append(dict(view=view,residual_auc_advantage=ws['residual']['AUROC']-max(ws['drift']['AUROC'],ws['sharpness']['AUROC'])))
        recovery=[r for r in contrasts if any(g in r['label'] for g in ('attenuate_damaged','attenuate_other'))]
        feedback=[r for r in contrasts if 'difference-in-differences' in r['label']]
        a_lead=any(r['residual_auc_advantage']>0 for r in leads) and any((r['contact_rate_difference'] or 0)>0 for r in recovery)
        b_lead=any((r['contact_rate_difference'] or 0)>0 for r in feedback)
        direction='优先非攻击小模块设计' if a_lead else '优先讨论补丁反馈机制' if b_lead else '本轮尚无明确优先方向；报告弱效应、区间与证据缺口，不自动追加实验'
        recommendation=dict(exploratory_only=True,direction=direction,warning_leads=leads,recovery_contrasts=recovery,feedback_contrasts=feedback,
            explanation='方向标签仅按观察效应给讨论顺序；不代表统计显著、投稿创新成立或新方法验证成功。需结合跨任务重复性和区间解读。')
        atomic(recommendation,ROOT/'results/RECOMMENDATION.json')
        costs=[read(p) for p in (ROOT/'reports/COSTS').glob('*.json') if read(p).get('signature')==signature()]
        pd.DataFrame([{k:v.get(k) for k in ('stage','status','job_id','elapsed_s','rss_hwm_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','device','error')} for v in costs]).to_csv(ROOT/'results/COSTS.csv',index=False)
        stage_ok=all(complete(ROOT/f'reports/{s}_COMPLETE.json') for s in ('SHARED','CALIBRATION','NONATTACK','PATCH'))
        final=len(rows)==288 and not missing and not errors and stage_ok
        if not final:
            direction='等待完整配对，暂不作路线优先结论'
            recommendation['direction']=direction
            atomic(recommendation,ROOT/'results/RECOMMENDATION.json')
        status=dict(status='COMPLETE' if final else 'PARTIAL',completed=len(rows),planned=288,missing=missing,invalid=errors,
                    summary=summary,contrasts=contrasts,short_window_only=True,no_full_CASR=True,
                    failed_costs=[read(p) for p in (ROOT/'reports/COSTS').glob('*.json') if read(p).get('status')=='FAILED'],
                    incomplete_GPU_costs=[v for v in costs if v.get('status')=='STARTED' and v.get('stage') in ('shared','nonattack','patch')])
        atomic(status,ROOT/'results/STATUS.json')
        lines=['# 腕部不对称性：两路线机制筛查',f"状态：{status['status']}；完成{len(rows)}/288，缺失{len(missing)}，无效{len(errors)}。",
               'Goal任务0/1/3/5/6/7，初态42/43；最多64控制步。接触率不是完整任务CASR。',
               '|条件|完整分支|首次接触|抓持|窗口内任务完成|','|---|---:|---:|---:|---:|']
        lines += [f"|{r['condition']}|{r['completed']}/12|{r['contacts']}|{r['grasps']}|{r['task_successes_within_window']}|" for r in summary]
        lines += ['','非攻击路线：参见WARNING.json中的预警分数、AUROC、按任务→起点区间和误报；不能将退化类型识别当作任务风险识别。',
                  '减弱方案按起点事后取最好，仅是固定候选集内的潜力，不是可部署恢复器，也不约束所有恢复方法的上限。',
                  '补丁路线：主要解释清洁相机回放相对随机相机变化、扣除各自无PNG条件后的差异。只有持续覆盖更强，不能证明腕部运动反馈。',
                  '相机供体提前终止时，两随机条件持有终止幅度；相机因果比较只使用有真实匹配幅度的共同前缀。',
                  '未训练新补丁、适配器或VLA。未预设高分门槛；结合效应、区间、任务重复性讨论继续方向，不自动追加任务。',
                  '探索性讨论建议：'+direction+'。依据和区间见RECOMMENDATION.json；部分结果不能作为路线最终结论。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(lines)+'\n')
        if final:finish(ROOT/'results/COMPLETE.json',[*receipts,*[ROOT/'results'/p for p in ('BRANCHES.csv','SUMMARY.csv','PER_TASK.csv','CONTRASTS.csv','DECISION_SCORES.csv','WARNING.json','STATUS.json','RESULTS.md','RECOMMENDATION.json','COSTS.csv')]])
        if export:export_videos(monitor)

def export_videos(monitor):
    if not (ROOT/'data/MANIFEST.json').exists() or not list((ROOT/'episodes').glob('*/*/COMPLETE.json')):
        monitor.update(phase='no_completed_video_sources_yet');return
    import sys
    sys.path.insert(1,str(PROJECT/'research_tools'))
    from export_singleview_videos import export_run
    monitor.update(phase='CPU_singleview_MP4_WebM_export')
    output=PROJECT/'research_videos'/(ROOT.name+'_singleview')
    export_run(ROOT,output)
    (output/'README.txt').write_text('本目录为最多64控制步的机制诊断片段；前视、腕部独立文件，无拼接或叠字。\n'
        '直接在IDE打开IDE_WebM/条件/front或wrist/下的WebM。MP4用于普通播放器。\n'
        '文件名task_success_0仅表示这个短窗口内没有完成任务，不能解释为完整任务失败或攻击成功。\n'
        '相机和特征干预属于内部诊断；特征减弱在RGB中不可见。来源与全解码检查见MANIFEST.json。\n')
