"""Within-seed matched/shuffled contrasts; absent episodes remain absent."""
from pathlib import Path
import numpy as np
import pandas as pd
from se_common import ROOT,PROJECT,config,read,atomic,complete,finish,lock,require,verify
from audit_math import paired_interval

BASELINES=('clean','initial_s42','initial_s43','uada500')


def statistics(frame):
    summary=[];contrasts=[]
    for name in (*BASELINES,*config()['methods']):
        data=frame[frame.condition==name] if len(frame) else pd.DataFrame()
        eligible=data[data.clean_success==1] if len(data) else data
        pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':r.induced_failure} for r in eligible.itertuples()]
        summary.append({'condition':name,'episodes_completed':len(data),'robot_successes':int(data.success.sum()) if len(data) else 0,
            'robot_task_SR':float(data.success.mean()) if len(data) else None,'clean_success_pairs':len(eligible),
            'induced_failures':int(eligible.induced_failure.sum()) if len(eligible) else 0,
            'CASR':float(eligible.induced_failure.mean()) if len(eligible) else None,
            'task_start_95ci':paired_interval(pairs,config()['bootstrap_draws'],config()['bootstrap_seed'])})
    for seed in (42,43):
        a=f'matched_relation_s{seed}';b=f'shuffled_relation_s{seed}'
        paired=pd.DataFrame()
        if len(frame):
            usable=frame[frame.clean_success==1]
            paired=usable[usable.condition==a].merge(usable[usable.condition==b],on=['start','suite','task_id'],suffixes=('_a','_b'),validate='one_to_one')
        pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':int(r.induced_failure_a-r.induced_failure_b)} for r in paired.itertuples()]
        contrasts.append({'seed':seed,'a':a,'b':b,'paired':len(pairs),
            'matched_only_failure':int(((paired.induced_failure_a==1)&(paired.induced_failure_b==0)).sum()) if len(paired) else 0,
            'shuffled_only_failure':int(((paired.induced_failure_a==0)&(paired.induced_failure_b==1)).sum()) if len(paired) else 0,
            'CASR_difference':float(np.mean([r['difference'] for r in pairs])) if pairs else None,
            'task_start_95ci':paired_interval(pairs,config()['bootstrap_draws'],config()['bootstrap_seed'])})
    return summary,contrasts


def baseline_videos(destination,manifest):
    import shutil
    from se_common import sha
    source=PROJECT/'research_videos/wrist_spatial_relation_expand_v1_singleview'
    if not (source/'MANIFEST.json').exists():return
    saved=read(source/'MANIFEST.json');current=read(destination/'MANIFEST.json')
    files={r['file']:r for r in current['files']};permitted={(r['start'],r['condition']) for r in manifest['baselines']}
    for row in saved['files']:
        relative=Path(row['file']);condition=relative.parts[1];start=relative.stem.split('__task_success_')[0]
        if (start,condition) not in permitted:continue
        original=source/relative;target=destination/relative;proof=original.with_suffix(original.suffix+'.json')
        require(sha(original)==row['sha256'] and read(proof)['sha256']==row['sha256'], 'Baseline video derivative changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(sha(target)==row['sha256'], 'Video output collision')
        else:shutil.copy2(original,target)
        shutil.copy2(proof,target.with_suffix(target.suffix+'.json'));files[row['file']]=row
    current.update(files=list(files.values()),baseline_manifest=str(source/'MANIFEST.json'),baseline_manifest_sha256=sha(source/'MANIFEST.json'))
    atomic(current,destination/'MANIFEST.json')


def run(monitor,export=True):
    with lock('report'):
        if complete(ROOT/'results/COMPLETE.json'):return
        value=verify() if (ROOT/'data/MANIFEST.json').exists() else None
        records=[];missing=[];errors=[];receipts=[]
        baseline={(r['start'],r['condition']):read(r['episode']) for r in value['baselines']} if value else {}
        for case in value['cases'] if value else []:
            clean=baseline[(case['start'],'clean')]
            for condition in (*BASELINES,*config()['methods']):
                if condition in BASELINES:
                    meta=baseline.get((case['start'],condition))
                    if meta is None:continue
                else:
                    receipt=ROOT/'episodes'/case['start']/condition/'COMPLETE.json'
                    try:
                        if not complete(receipt):missing.append({'start':case['start'],'condition':condition});continue
                        paths=[ROOT/p for p in read(receipt)['artifacts'] if p.endswith('/EPISODE.json')]
                        require(len(paths)==1, 'Ambiguous episode receipt')
                        meta=read(paths[0]);asset=next(a for a in value['assets'] if a['condition']==condition)
                        require(meta['condition']==condition and meta['png_sha256']==asset['png_sha256'], 'Wrong evaluated PNG')
                        require(all(meta[k]==case[k] for k in ('start','suite','task_id','state_index','rollout_seed')), 'Wrong evaluated start')
                        require(meta['success'] in (0,1) and meta['checkpoint']=='libero_goal' and meta['strict_paired_reset'], 'Invalid native evaluation')
                        receipts.append(receipt)
                    except Exception as error:
                        errors.append({'start':case['start'],'condition':condition,'error':repr(error)});continue
                records.append({'start':case['start'],'suite':case['suite'],'task_id':case['task_id'],'state_index':case['state_index'],
                    'condition':condition,'clean_success':int(clean['success']),'success':int(meta['success']),
                    'induced_failure':int(clean['success'] and not meta['success']),**meta.get('events',{})})
        frame=pd.DataFrame(records);summary,contrasts=statistics(frame)
        tasks=[]
        if len(frame):
            for (condition,task),data in frame.groupby(['condition','task_id']):
                eligible=data[data.clean_success==1]
                tasks.append({'condition':condition,'task_id':int(task),'completed':len(data),'robot_successes':int(data.success.sum()),
                    'robot_task_SR':float(data.success.mean()),'clean_success_pairs':len(eligible),
                    'induced_failures':int(eligible.induced_failure.sum()),'CASR':float(eligible.induced_failure.mean()) if len(eligible) else None})
        costs=[{'source':str(p),**read(p)} for p in sorted((ROOT/'reports/COSTS').glob('*.json'))]
        done=value is not None and len(receipts)==160 and not missing and not errors and all(complete(ROOT/f'reports/EVALUATE_{i}_COMPLETE.json') for i in (0,1))
        status='COMPLETE' if done else 'PARTIAL' if value else 'WAITING_PREPARATION'
        result={'status':status,'native_new_episodes':len(receipts),'expected_new_episodes':160,'missing_episodes':missing,
            'invalid_episodes':errors,'summary':summary,'contrasts':contrasts,'costs':costs,'new_training_performed':False,
            'scope':'Fixed Goal checkpoint; Goal development starts 2-5; matched/shuffled within seed; CASR conditional on clean success'}
        atomic(result,ROOT/'results/STATUS.json')
        files=['EPISODES.csv','CASR.csv','PAIRED_CONTRASTS.csv','TASK_RESULTS.csv','STATUS.json','RESULTS.md']
        for name,data in [('EPISODES.csv',frame),('CASR.csv',pd.DataFrame(summary)),('PAIRED_CONTRASTS.csv',pd.DataFrame(contrasts)),('TASK_RESULTS.csv',pd.DataFrame(tasks))]:
            data.to_csv(ROOT/'results'/name,index=False)
        text=['# 正确关系／置乱关系：配对闭环评测',f'状态：{status}；新回合{len(receipts)}/160。缺失、无效不计成功。',
            '固定Goal模型、Goal10任务初态2—5，四张4,000次最终PNG；复用原217条有效混合训练观测和目标，没有新训练。',
            '任务成功率按各组已完成回合计算；CASR按该组已完成且clean成功的起点计算。历史初始补丁/UADA仅39回合，不能写成40回合的无条件成功率。',
            '|条件|完成回合|任务成功率|诱发失败/clean成功配对|CASR|','|---|---:|---:|---:|---:|']
        for row in summary:
            sr='待完成' if row['robot_task_SR'] is None else f"{row['robot_task_SR']:.2%}"
            casr='待完成' if row['CASR'] is None else f"{row['CASR']:.2%}"
            text.append(f"|{row['condition']}|{row['episodes_completed']}|{sr}|{row['induced_failures']}/{row['clean_success_pairs']}|{casr}|")
        text+=['','正确减置乱的CASR差异（正值表示正确关系诱发失败更多）；区间按任务→起点重采样，不跨种子混作独立起点：']
        for row in contrasts:text.append(f"- seed{row['seed']}：完整配对{row['paired']}；差异{row['CASR_difference']}；95%区间{row['task_start_95ci']}")
        text+=['','本轮正确关系PNG与此前扩大评测不同，不借用74.36%/76.92%成绩。本结果只回答错误方向与当前关系对应是否有帮助，不自动证明DINO独占机制或跨环境通用。',
            '视频：research_videos/'+ROOT.name+'_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/；无拼接。',
            f'缺失{len(missing)}回合；无效{len(errors)}回合；见STATUS.json。']
        (ROOT/'results/RESULTS.md').write_text('\n'.join(text)+'\n')
        if export and receipts:
            import sys
            sys.path.insert(0,str(PROJECT/'research_tools'))
            from export_singleview_videos import export_run
            monitor.update(phase='CPU_singleview_video_export',completed=len(receipts),total=160)
            output=PROJECT/'research_videos'/(ROOT.name+'_singleview')
            export_run(ROOT,output);baseline_videos(output,value)
        if done:finish(ROOT/'results/COMPLETE.json',[*receipts,*[ROOT/'results'/p for p in files]])
        monitor.update(phase=status,completed=len(receipts),total=160)
