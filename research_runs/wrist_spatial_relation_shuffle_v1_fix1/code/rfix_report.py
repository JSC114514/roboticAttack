"""Missing training/selection produces an explicit partial report, not a crash."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from rfix_common import ROOT,PROJECT,BASE,config,candidates,read,sha,atomic,require,complete,finish,external_complete,lock
from audit_math import paired_interval


def episode_path(root,receipt):
    value=read(receipt);paths=[root/k for k in value['artifacts'] if k.endswith('/EPISODE.json')]
    require(len(paths)==1,'Ambiguous native episode');return paths[0].parent


def restore_baseline_videos(destination):
    source=PROJECT/'research_videos/wrist_spatial_relation_expand_v1_singleview'
    require((source/'MANIFEST.json').exists(),'Existing single-camera baseline conversion must finish first')
    old=read(source/'MANIFEST.json');new=read(destination/'MANIFEST.json')
    files={r['file']:r for r in new['files']}
    for row in old['files']:
        relative=Path(row['file'])
        if relative.parts[1] not in ('clean','initial_s42','initial_s43','uada500'):continue
        src=source/relative;target=destination/relative
        require(sha(src)==row['sha256'],'Baseline converted video changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(sha(target)==row['sha256'],'Video destination collision')
        else:shutil.copy2(src,target)
        proof=src.with_suffix(src.suffix+'.json')
        require(read(proof)['sha256']==row['sha256'],'Baseline video proof changed')
        shutil.copy2(proof,target.with_suffix(target.suffix+'.json'));files[row['file']]=row
    new['files']=list(files.values());new['baseline_video_source']=str(source)
    new['baseline_manifest_sha256']=sha(source/'MANIFEST.json')
    new['episodes']=len({(Path(p).parts[1],Path(p).stem) for p in files})
    atomic(new,destination/'MANIFEST.json')


def run(monitor):
    with lock('report'):
        manifest=read(ROOT/'data/MANIFEST.json');names=[n for n,_,_ in candidates()]
        expected=['clean','initial_s42','initial_s43','uada500',*names]
        rows=[];own_receipts=[];missing=[]
        training={name:complete(ROOT/'training'/name/'COMPLETE.json') for name in names}
        selected=complete(ROOT/'reports/SELECT_COMPLETE.json')
        final=selected and all(training.values())
        for i,case in enumerate(manifest['cases']):
            lookup={}
            for name in expected:
                root=BASE if name in expected[:4] else ROOT
                receipt=root/'episodes'/case['start']/name/'COMPLETE.json'
                valid=external_complete(root,receipt) if root==BASE else complete(receipt)
                if valid:
                    lookup[name]=episode_path(root,receipt)
                    if root==ROOT:own_receipts.append(receipt)
            require('clean' in lookup,'Matched expansion clean result is missing')
            clean=read(lookup['clean']/'EPISODE.json');eligible=bool(clean['success'])
            if eligible:
                for name in expected:
                    if name not in lookup:missing.append({'start':case['start'],'condition':name});final=False
            for name,folder in lookup.items():
                value=read(folder/'EPISODE.json')
                require(value['rollout_seed']==case['rollout_seed'] and value['checkpoint']=='libero_goal','Reuse identity changed')
                rows.append({'start':case['start'],'suite':case['suite'],'task_id':case['task_id'],'state_index':case['state_index'],
                             'condition':name,'clean_success':int(eligible),'success':value['success'],
                             'induced_failure':int(eligible and not value['success']),'executed_steps':value['executed_steps'],**value['events']})
            monitor.update(phase='paired_native_statistics',completed=i+1,total=len(manifest['cases']))
        frame=pd.DataFrame(rows);summary=[];contrasts=[]
        clean_count=int(frame[frame.condition=='clean'].success.sum())
        for name in expected:
            data=frame[(frame.condition==name)&(frame.clean_success==1)]
            pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':float(r.induced_failure)} for r in data.itertuples()]
            summary.append({'condition':name,'clean_successes':clean_count,'paired_completed':len(data),
                            'induced_failures':int(data.induced_failure.sum()),'CASR':float(data.induced_failure.mean()) if len(data) else None,
                            'task_start_bootstrap_95ci':paired_interval(pairs,2000,42)})
        for seed in config()['seeds']:
            a,b=f'matched_relation_s{seed}',f'shuffled_relation_s{seed}'
            left=frame[(frame.condition==a)&(frame.clean_success==1)];right=frame[(frame.condition==b)&(frame.clean_success==1)]
            paired=left.merge(right,on=['start','suite','task_id'],suffixes=('_a','_b'))
            pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':float(r.induced_failure_a-r.induced_failure_b)} for r in paired.itertuples()]
            contrasts.append({'a':a,'b':b,'paired_completed':len(paired),
                              'CASR_difference':float(np.mean([r['difference'] for r in pairs])) if pairs else None,
                              'task_start_bootstrap_95ci':paired_interval(pairs,2000,42)})
        if not all(complete(ROOT/f'reports/EVALUATE_{i}_COMPLETE.json') for i in (0,1)):final=False
        status='COMPLETE' if final else 'WAITING_TRAINING' if not selected else 'PARTIAL'
        payload={'status':status,'training_complete':training,'selection_complete':selected,'missing_pairs':missing,
                 'clean_starts_complete':len(manifest['cases']),'clean_successes':clean_count,'native_new_episodes':len(own_receipts),
                 'expected_new_episodes':clean_count*4,'summary':summary,'contrasts':contrasts,
                 'parent_failed_costs':manifest.get('parent_failed_costs',[]),
                 'failed_costs':[read(p) for p in (ROOT/'reports/COSTS').glob('*.json') if read(p).get('status')=='FAILED']}
        atomic(payload,ROOT/'results/STATUS.json')
        frame.to_csv(ROOT/'results/EPISODES.csv',index=False);pd.DataFrame(summary).to_csv(ROOT/'results/CASR.csv',index=False)
        pd.DataFrame(contrasts).to_csv(ROOT/'results/PAIRED_CONTRASTS.csv',index=False)
        text=['# 关系对应对照修复结果',f'状态：{status}。缺失不计成功；最终选择文件存在：{selected}。',
              '复用两张正确关系4000次结果；只从封存2000次检查点恢复置乱关系，固定4000次最终PNG。Goal十任务、初态2–5；四套件混合训练池保持原样。',
              '|条件|完整配对|诱发失败|CASR|','|---|---:|---:|---:|']
        for r in summary:text.append(f"|{r['condition']}|{r['paired_completed']}|{r['induced_failures']}|{r['CASR'] if r['CASR'] is not None else '未完成'}|")
        text+=['','按任务→起点重采样；槽位和帧不当独立样本。正确关系对置乱关系是匹配幅度的对应对照，不是对所有非关系方法的优越性证明，也不证明DINO专属机制。',
               '视频为前视和腕部独立单回合文件，IDE_WebM优先用于IDE；无组合、无叠字或长合集。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        if final:
            finish(ROOT/'results/COMPLETE.json',[*own_receipts,*[ROOT/'results'/f for f in ('STATUS.json','CASR.csv','EPISODES.csv','PAIRED_CONTRASTS.csv','RESULTS.md')]])
            import sys
            sys.path.insert(0,str(PROJECT/'research_tools'))
            from export_singleview_videos import export_run
            output=PROJECT/'research_videos'/(ROOT.name+'_singleview')
            export_run(ROOT,output);restore_baseline_videos(output)
