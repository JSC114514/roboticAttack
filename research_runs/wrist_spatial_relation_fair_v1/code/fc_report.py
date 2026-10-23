"""Fair matched-data comparisons; original/added starts separately reported."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from fc_common import ROOT,PROJECT,config,read,sha,atomic,complete,finish,lock,require,verify,cases,condition_names,episode_receipt,episode_files,signature
from audit_math import paired_interval


def process_summary(files):
    frame=pd.read_parquet(files['STEPS.parquet']);values=[];ever_contact=False
    for row in frame.itertuples():
        before=row.geometry_before
        ever_contact|=bool(before['contact'] or before['grasp'])
        if ever_contact or row.step>32:continue
        direction=np.asarray(before['target_xyz'])-np.asarray(row.eef_before)
        norm=float(np.linalg.norm(direction))
        if norm<=1e-9:continue
        unit=direction/norm;actual=np.asarray(row.eef_after)-row.eef_before
        target_motion=np.asarray(row.geometry_after['target_xyz'])-before['target_xyz']
        command=np.clip(np.asarray(row.deployed_action)[:3],-1,1)*.05
        values.append({'actual_radial_m':float(actual@unit),'command_radial_m':float(command@unit),
            'target_movement_m':float(np.linalg.norm(target_motion)),
            'tracking_error_peak_m':float(row.controller['tracking_error_peak_m']),
            'peak_robot_contact_force_n':float(row.controller['peak_robot_contact_force_n'])})
    if not values:return {'precontact32_valid_steps':0,'actual_radial_progress_m':None,'radial_command_mean_m':None,'tracking_error_max_m':None,'contact_force_max_n':None,'target_movement_sum_m':None}
    return {'precontact32_valid_steps':len(values),'actual_radial_progress_m':sum(v['actual_radial_m'] for v in values),
        'radial_command_mean_m':float(np.mean([v['command_radial_m'] for v in values])),
        'tracking_error_max_m':max(v['tracking_error_peak_m'] for v in values),'contact_force_max_n':max(v['peak_robot_contact_force_n'] for v in values),
        'target_movement_sum_m':sum(v['target_movement_m'] for v in values)}


def statistics(frame):
    summary=[];contrasts=[]
    for cohort in ('original40','expanded40','all80'):
        scope=frame if cohort=='all80' else frame[frame.cohort==cohort] if len(frame) else frame
        for name in condition_names():
            data=scope[scope.condition==name] if len(scope) else pd.DataFrame()
            eligible=data[data.clean_success==1] if len(data) else data
            pairs=[dict(suite=r.suite,task_id=r.task_id,difference=r.induced_failure) for r in eligible.itertuples()]
            summary.append({'cohort':cohort,'condition':name,'episodes_completed':len(data),
                'robot_successes':int(data.success.sum()) if len(data) else 0,'robot_task_SR':float(data.success.mean()) if len(data) else None,
                'clean_success_pairs':len(eligible),'induced_failures':int(eligible.induced_failure.sum()) if len(eligible) else 0,
                'CASR':float(eligible.induced_failure.mean()) if len(eligible) else None,
                'task_start_95ci':paired_interval(pairs,config()['bootstrap_draws'],42)})
        for seed in (42,43):
            main=f'matched_relation_s{seed}'
            for base in ('uada_clean_xyz','uada_clean_7dof','translation_far_boundary','shuffled_relation'):
                other=f'{base}_s{seed}';paired=pd.DataFrame()
                if len(scope):
                    usable=scope[scope.clean_success==1]
                    paired=usable[usable.condition==main].merge(usable[usable.condition==other],on=['start','suite','task_id'],suffixes=('_a','_b'),validate='one_to_one')
                pairs=[dict(suite=r.suite,task_id=r.task_id,difference=int(r.induced_failure_a-r.induced_failure_b)) for r in paired.itertuples()]
                contrasts.append({'cohort':cohort,'seed':seed,'a':main,'b':other,'paired':len(pairs),
                    'CASR_difference':float(np.mean([p['difference'] for p in pairs])) if pairs else None,
                    'matched_only_failure':int(((paired.induced_failure_a==1)&(paired.induced_failure_b==0)).sum()) if len(paired) else 0,
                    'comparator_only_failure':int(((paired.induced_failure_a==0)&(paired.induced_failure_b==1)).sum()) if len(paired) else 0,
                    'task_start_95ci':paired_interval(pairs,config()['bootstrap_draws'],42)})
    return summary,contrasts


def reuse_video_exports(destination):
    """Reuse fully decoded old exports; rebind provenance to new read-only links."""
    from export_singleview_videos import codec_args
    for run in ('wrist_spatial_relation_shuffle_eval_v1','wrist_spatial_relation_expand_v1'):
        previous=PROJECT/'research_videos'/(run+'_singleview')
        if not (previous/'MANIFEST.json').exists():continue
        for row in read(previous/'MANIFEST.json')['files']:
            relative=Path(row['file']);name=relative.parts[1];start=relative.stem.split('__task_success_')[0]
            receipt=ROOT/'episodes'/start/name/'COMPLETE.json'
            if not receipt.exists() or not read(receipt).get('reused'):continue
            files=episode_files(receipt);view=relative.parts[2];raw='front.mp4' in files
            source=files[view+'.mp4'] if raw else files['video.mp4']
            crop=None if raw else 'crop=224:224:'+('0' if view=='front' else '224')+':64'
            require(row['identity']['source_sha256']==sha(source) and row['identity']['crop']==crop,'Video source identity differs')
            target=destination/relative;proof=target.with_suffix(target.suffix+'.json')
            if proof.exists():continue
            original=previous/relative;require(row['decoded_complete'] and sha(original)==row['sha256'],'Old video changed')
            require(read(original.with_suffix(original.suffix+'.json'))['sha256']==row['sha256'],'Old video proof changed')
            identity=dict(source=str(source),source_sha256=sha(source),source_receipt=str(receipt),source_signature=signature(),
                crop=crop,view=view,task_success=row['identity']['task_success'],encoded_legacy_derivative=not raw,codec_args=codec_args(target.suffix))
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(sha(target)==row['sha256'],'Video collision')
            else:shutil.copy2(original,target)
            atomic({'identity':identity,'sha256':row['sha256'],'decoded_complete':True,'metadata':row['metadata'],
                    'reused_verified_export':str(original),'original_source_signature':row['identity']['source_signature']},proof)


def export_with_attested_links(output,value):
    from fc_common import module
    exporter=module('fc_private_video_exporter',PROJECT/'research_tools/export_singleview_videos.py')
    allowed={Path(p).resolve():h for p,h in value['inputs'].items()}
    def verify_receipt(root,path,expected_signature=None):
        require(complete(path),'Video source incomplete')
        saved=read(path);require(expected_signature in (None,saved['signature']),'Video source revision differs')
        for relative,expected in saved['artifacts'].items():
            source=(Path(root)/relative).resolve()
            require(source.is_relative_to(ROOT) or allowed.get(source)==expected,'Video symlink is not an attested historical asset')
        return saved
    exporter.verify_receipt=verify_receipt
    return exporter.export_run(ROOT,output)


def run(monitor,export=False):
    with lock('report'):
        if complete(ROOT/'results/COMPLETE.json'):return
        value=verify() if (ROOT/'data/MANIFEST.json').exists() else None
        rows=[];missing=[];errors=[];receipts=[]
        for case in value['cases'] if value else []:
            clean_receipt=episode_receipt(case,'clean')
            if complete(clean_receipt):clean=read(episode_files(clean_receipt)['EPISODE.json'])
            else:
                missing.extend(dict(start=case['start'],condition=name,reason='clean_missing') for name in condition_names());continue
            for name in condition_names():
                receipt=episode_receipt(case,name)
                try:
                    if not complete(receipt):missing.append(dict(start=case['start'],condition=name));continue
                    files=episode_files(receipt);meta=read(files['EPISODE.json'])
                    require(meta['checkpoint']=='libero_goal' and meta['condition']==name and meta['strict_paired_reset'],'Wrong model/condition')
                    require(all(meta[k]==case[k] for k in ('start','task_id','suite','state_index','rollout_seed')),'Wrong episode identity')
                    path=value['conditions'][name]
                    require(meta['png_sha256']==(sha(path) if path else None),'Wrong evaluated PNG')
                    cache=ROOT/'results/process_cache'/case['start']/(name+'.json')
                    identity=sha(files['STEPS.parquet'])
                    if cache.exists() and read(cache).get('source_sha256')==identity:process=read(cache)['summary']
                    else:
                        process=process_summary(files);atomic({'source_sha256':identity,'summary':process},cache)
                    rows.append({**{k:case[k] for k in ('start','cohort','task_id','suite','state_index')},'condition':name,
                        'clean_success':int(clean['success']),'success':int(meta['success']),
                        'induced_failure':int(clean['success'] and not meta['success']),**meta.get('events',{}),**process})
                    receipts.append(receipt)
                except Exception as error:errors.append(dict(start=case['start'],condition=name,error=repr(error)))
        frame=pd.DataFrame(rows);summary,contrasts=statistics(frame)
        original_ready=value is not None and sum(r['cohort']=='original40' for r in rows)==520
        done=value is not None and len(receipts)==1040 and not missing and not errors
        status='COMPLETE' if done else 'ORIGINAL40_COMPLETE' if original_ready else 'PARTIAL' if value else 'WAITING_PREPARATION'
        costs=[{'source':str(p),**read(p)} for p in sorted((ROOT/'reports/COSTS').glob('*.json'))]
        result={'status':status,'episodes_completed':len(receipts),'planned_episodes':1040,'reused_episodes':value['reused_episodes'] if value else 0,
            'new_episodes_completed':sum(not read(p).get('reused',False) for p in receipts),'missing_episodes':missing,'invalid_episodes':errors,
            'training_complete':{name:complete(ROOT/'training'/name/'COMPLETE.json') for name,_,_ in __import__('fc_common').candidates()},
            'summary':summary,'contrasts':contrasts,'costs':costs,'reference':'clean native model; expert-reference UADA not tested',
            'interpretation':'Goal-only; original and added development starts separately reported; 7DoF comparator has broader loss dimensions'}
        atomic(result,ROOT/'results/STATUS.json')
        frame.to_csv(ROOT/'results/EPISODES.csv',index=False)
        pd.DataFrame(summary).to_csv(ROOT/'results/CASR.csv',index=False)
        pd.DataFrame(contrasts).to_csv(ROOT/'results/PAIRED_CONTRASTS.csv',index=False)
        if len(frame):
            frame.groupby(['cohort','condition','task_id']).agg(completed=('success','size'),robot_successes=('success','sum'),
                clean_success_pairs=('clean_success','sum'),induced_failures=('induced_failure','sum')).reset_index().to_csv(ROOT/'results/TASK_RESULTS.csv',index=False)
        texts=['# 同条件UADA适配对照',f'状态：{status}；完成{len(receipts)}/1040；缺失{len(missing)}，无效{len(errors)}。',
            '统一Goal模型、217条混合观测、相同初始化／采样、4,000更新、Adam、无EOT、50×50固定腕部PNG。',
            'UADA对照为OFT连续动作头、clean参考适配版；不冒称专家参考原始UADA。XYZ主对照与关系方法维度匹配，7DoF补充完整报告。',
            '物理命令空间远边界同池对照保留；历史Spatial UADA500只作条件不一致的背景，见原报告，不参加主比较。',
            '原40、增加40及合并80分别报告，不按测试选种子。seed42为关系方法主结果，seed43检查重复性。']
        for cohort in ('original40','expanded40','all80'):
            texts+=['',f'## {cohort}','|条件|完成回合|任务成功率|诱发失败/clean成功配对|CASR|','|---|---:|---:|---:|---:|']
            for row in [r for r in summary if r['cohort']==cohort]:
                sr='待完成' if row['robot_task_SR'] is None else f"{row['robot_task_SR']:.2%}"
                casr='待完成' if row['CASR'] is None else f"{row['CASR']:.2%}"
                texts.append(f"|{row['condition']}|{row['episodes_completed']}|{sr}|{row['induced_failures']}/{row['clean_success_pairs']}|{casr}|")
        texts+=['','配对差异及任务→起点重采样区间见PAIRED_CONTRASTS.csv；多个槽位／视频帧不当独立样本。',
            '过程指标使用前32控制步且首次接触前的有效步，分别记录实际末端径向推进、命令径向分量、目标移动、跟踪误差及接触力；它们是描述性证据，不等于因果恢复实验。',
            '中心距离不是表面间距或精确抓取误差。可见性、接触、抓持和失败过程见EPISODES.csv及原始STEPS/EVENTS。',
            '视频位于research_videos/'+ROOT.name+'_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/。']
        (ROOT/'results/RESULTS.md').write_text('\n'.join(texts)+'\n')
        if export and receipts:
            import sys
            sys.path.insert(0,str(PROJECT/'research_tools'))
            output=PROJECT/'research_videos'/(ROOT.name+'_singleview');output.mkdir(parents=True,exist_ok=True)
            monitor.update(phase='CPU_video_export',completed=len(receipts),total=1040)
            reuse_video_exports(output);export_with_attested_links(output,value)
        if original_ready:
            snapshot=ROOT/'results/original40/SUMMARY.json'
            atomic({'summary':[r for r in summary if r['cohort']=='original40'],
                    'contrasts':[r for r in contrasts if r['cohort']=='original40'],'completed_episodes':520},snapshot)
            finish(ROOT/'reports/ORIGINAL40_REPORT_COMPLETE.json',[snapshot])
        if done:
            finish(ROOT/'results/COMPLETE.json',[*receipts,*[ROOT/'results'/p for p in ('STATUS.json','CASR.csv','PAIRED_CONTRASTS.csv','TASK_RESULTS.csv','EPISODES.csv','RESULTS.md')]])
        monitor.update(phase=status,completed=len(receipts),total=1040)
