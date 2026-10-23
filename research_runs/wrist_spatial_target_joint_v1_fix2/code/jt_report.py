"""Partial-safe statistics; missing jobs never become attack successes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from jt_common import ROOT,PROJECT,config,read,atomic,complete,finish,lock,resources

def baseline_videos(destination,manifest):
    """Reuse verified baseline-style derivatives without simulator work or re-encoding."""
    import shutil
    from jt_common import sha,require
    source=PROJECT/'research_videos/wrist_spatial_relation_expand_v1_singleview'
    if not (source/'MANIFEST.json').exists():return
    saved=read(source/'MANIFEST.json');current=read(destination/'MANIFEST.json')
    files={r['file']:r for r in current['files']}
    permitted={(r['start'],r['condition']) for r in manifest['baselines']}
    for row in saved['files']:
        relative=Path(row['file']);condition=relative.parts[1];start=relative.stem.split('__task_success_')[0]
        if (start,condition) not in permitted:continue
        original=source/relative;target=destination/relative
        require(sha(original)==row['sha256'],'Baseline video derivative changed')
        proof=original.with_suffix(original.suffix+'.json');require(read(proof)['sha256']==row['sha256'],'Baseline video receipt changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(sha(target)==row['sha256'],'Video output collision')
        else:shutil.copy2(original,target)
        shutil.copy2(proof,target.with_suffix(target.suffix+'.json'));files[row['file']]=row
    current.update(files=list(files.values()),baseline_manifest=str(source/'MANIFEST.json'),baseline_manifest_sha256=sha(source/'MANIFEST.json'))
    atomic(current,destination/'MANIFEST.json')

def run(monitor,export=True):
    with lock('report'):
        if complete(ROOT/'results/COMPLETE.json'):return
        manifest=read(ROOT/'data/MANIFEST.json') if (ROOT/'data/MANIFEST.json').exists() else None
        stats=read(ROOT/'data/TARGET_STATS.json') if (ROOT/'data/TARGET_STATS.json').exists() else None
        training={method:complete(ROOT/'training'/method/'COMPLETE.json') for method in config()['methods']}
        episodes=[];receipts=[];missing=[];baseline={}
        if manifest:
            for entry in manifest['baselines']:
                meta=read(entry['episode']);baseline[(entry['start'],entry['condition'])]=meta
            for case in manifest['cases']:
                clean=baseline[(case['start'],'clean')]
                for condition in ('clean','initial_s42','uada500',*config()['methods']):
                    if condition in config()['methods']:
                        receipt=ROOT/'episodes'/case['start']/condition/'COMPLETE.json'
                        if not complete(receipt):missing.append({'start':case['start'],'condition':condition});continue
                        saved=read(receipt);paths=[ROOT/p for p in saved['artifacts'] if p.endswith('/EPISODE.json')]
                        meta=read(paths[0]);receipts.append(receipt)
                    else:
                        meta=baseline.get((case['start'],condition))
                        if meta is None:continue
                    episodes.append({'start':case['start'],'suite':case['suite'],'task_id':case['task_id'],'state_index':case['state_index'],
                                     'condition':condition,'clean_success':int(clean['success']),'success':int(meta['success']),
                                     'induced_failure':int(clean['success'] and not meta['success']),**meta.get('events',{})})
        from audit_math import paired_interval
        frame=pd.DataFrame(episodes);summary=[];contrasts=[]
        for name in ('clean','initial_s42','uada500',*config()['methods']):
            data=frame[frame.condition==name] if len(frame) else pd.DataFrame()
            eligible=data[data.clean_success==1] if len(data) else data
            pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':r.induced_failure} for r in eligible.itertuples()]
            summary.append({'condition':name,'episodes_completed':len(data),'robot_successes':int(data.success.sum()) if len(data) else 0,
                'robot_task_SR':float(data.success.mean()) if len(data) else None,'clean_success_pairs':len(eligible),
                'induced_failures':int(eligible.induced_failure.sum()) if len(eligible) else 0,
                'CASR':float(eligible.induced_failure.mean()) if len(eligible) else None,'task_start_95ci':paired_interval(pairs,2000,42)})
        if len(frame):
            for other in ('fixed_harm','realizability_only','translation_boundary','uada500'):
                a=frame[(frame.condition=='joint')&(frame.clean_success==1)]
                b=frame[(frame.condition==other)&(frame.clean_success==1)]
                paired=a.merge(b,on=['start','suite','task_id'],suffixes=('_a','_b'))
                pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':float(r.induced_failure_a-r.induced_failure_b)} for r in paired.itertuples()]
                contrasts.append({'a':'joint','b':other,'paired':len(paired),'CASR_difference':float(np.mean([v['difference'] for v in pairs])) if pairs else None,
                                  'task_start_95ci':paired_interval(pairs,2000,42)})
        offline=[]
        for p in (ROOT/'evaluation/offline').glob('*.json'):
            if complete(p):
                for line in p.with_suffix('.jsonl').read_text().splitlines():offline.append(json.loads(line))
        cost_files=sorted((ROOT/'reports/COSTS').glob('*.json'));costs=[read(p) for p in cost_files]
        done=all(training.values()) and all(complete(ROOT/f'reports/EVALUATE_{i}_COMPLETE.json') for i in (0,1)) and not missing
        status='COMPLETE' if done else stats['status'] if stats and not stats['proceed'] else 'PARTIAL'
        value={'status':status,'training_complete':training,'target_statistics':stats,'missing_episodes':missing,
               'native_new_episodes':len(receipts),'expected_new_episodes':160,'offline_records':len(offline),
               'summary':summary,'contrasts':contrasts,'costs':costs,'parent_costs':manifest.get('parent_costs',[]) if manifest else [],
               'reusable_visual_cache_count':len(manifest.get('reusable_visual_cache',{})) if manifest else 0,
               'interpretation':'Goal development evaluation; all 40 starts; CASR conditional on clean success; internal teachers not attack results'}
        atomic(value,ROOT/'results/STATUS.json')
        frame.to_csv(ROOT/'results/EPISODES.csv',index=False);pd.DataFrame(summary).to_csv(ROOT/'results/CASR.csv',index=False)
        pd.DataFrame(contrasts).to_csv(ROOT/'results/PAIRED_CONTRASTS.csv',index=False)
        atomic({'records':offline,'selection_uses_validation':False},ROOT/'results/OFFLINE.json')
        histories={m:read(ROOT/'training'/m/'SELECTION_HISTORY.json') for m in config()['methods'] if (ROOT/'training'/m/'SELECTION_HISTORY.json').exists()}
        atomic(histories,ROOT/'results/TARGET_SELECTION.json')
        text=['# 空间目标与共享补丁联合选择',f'状态：{status}。已完成原生PNG闭环 {len(receipts)}/160；缺失不计成功。',
              '固定Goal模型；四套件混合47观测训练、47跨轨迹验证；共有无效目标见TARGET_STATS。',
              '反传预算4000包含丢弃探测；它不是相同GPU计算成本。成本包括额外前向、缓存与失败。',
              '|条件|完成回合|机械臂任务成功率|诱发失败/clean成功配对|CASR|','|---|---:|---:|---:|---:|']
        for v in summary:
            sr='未完成' if v['robot_task_SR'] is None else f"{v['robot_task_SR']:.2%}"
            casr='未完成' if v['CASR'] is None else f"{v['CASR']:.2%}"
            text.append(f"|{v['condition']}|{v['episodes_completed']}|{sr}|{v['induced_failures']}/{v['clean_success_pairs']}|{casr}|")
        text+=['','先看目标表示是否实现，再看原生动作和实际接近过程，最后判断闭环失败。完整特征匹配不自动等于模型空间判断被特定方式改变。',
               '历史UADA为Spatial训练资产；同条件动作参照是本轮47观测重训的translation_boundary。',
               '视频前视、腕部分开，无拼接，统一导出MP4与IDE_WebM。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        if export and receipts:
            import sys
            sys.path.insert(0,str(PROJECT/'research_tools'))
            from export_singleview_videos import export_run
            output=PROJECT/'research_videos'/(ROOT.name+'_singleview')
            export_run(ROOT,output);baseline_videos(output,manifest)
        if done:
            finish(ROOT/'results/COMPLETE.json',[*receipts,ROOT/'results/STATUS.json',ROOT/'results/CASR.csv',
                   ROOT/'results/EPISODES.csv',ROOT/'results/PAIRED_CONTRASTS.csv',ROOT/'results/OFFLINE.json',ROOT/'results/TARGET_SELECTION.json',ROOT/'results/RESULTS.md'])
        monitor.update(phase=status,completed=len(receipts),total=160)
