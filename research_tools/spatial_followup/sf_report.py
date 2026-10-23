"""Paired task/start estimates, native video comparisons, centralized export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sf_common import ROOT, PROJECT, config, read, atomic, require, finish, complete, lock, sha, external_complete
from sf_evaluate import conditions, folder, base_root
from audit_math import paired_interval


def external_artifacts(receipt):
    base=base_root();value=read(receipt)
    require(external_complete(base,receipt),'Expansion episode incomplete')
    paths=[base/p for p in value['artifacts'] if p.endswith('/EPISODE.json')]
    require(len(paths)==1,'Ambiguous base episode')
    return paths[0].parent


def lookup(case):
    out={}
    for name in conditions():
        receipt=ROOT/'episodes'/case['start']/name/'COMPLETE.json'
        if complete(receipt):out[name]=folder(receipt)
    if config()['scope']=='control':
        for name in ('clean','initial_s42','initial_s43','uada500'):
            receipt=base_root()/'episodes'/case['start']/name/'COMPLETE.json'
            if receipt.exists():out[name]=external_artifacts(receipt)
    return out


def compose(case, paths, destination):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    names=list(paths); results={n:read(p/'EPISODE.json') for n,p in paths.items()}
    fps=results[names[0]]['fps'];require(all(r['fps']==fps for r in results.values()),'Control rates differ')
    lengths=[results[n]['executed_steps']+1 for n in names]
    readers=[];last=[None]*len(names);columns=4;rows=(len(names)+3)//4
    try:
        readers=[imageio.get_reader(paths[n]/'video.mp4',ffmpeg_params=['-threads','1']) for n in names]
        with imageio.get_writer(destination/'comparison.mp4',fps=fps,codec='libx264',quality=7,macro_block_size=16,
                                ffmpeg_params=['-threads','2']) as writer:
            for k in range(max(lengths)):
                image=Image.new('RGB',(448*columns,288*rows+32),(24,24,24))
                for i,n in enumerate(names):
                    if k<lengths[i]:
                        try:last[i]=readers[i].get_next_data()
                        except (StopIteration,IndexError):raise RuntimeError('Truncated video: '+str(paths[n]))
                    require(last[i] is not None,'Empty video')
                    panel=Image.fromarray(last[i]).copy()
                    if k>=lengths[i]:
                        draw=ImageDraw.Draw(panel);draw.rectangle((0,238,448,288),fill='black')
                        draw.text((5,243),f'ENDED at {(lengths[i]-1)/fps:.2f}s; last frame held',fill='yellow')
                        draw.text((5,261),f'Success={results[n]["success"]}',fill='yellow')
                    image.paste(panel,((i%columns)*448,(i//columns)*288))
                ImageDraw.Draw(image).text((5,288*rows+5),case['start']+f' | native PNG | common clock {k/fps:.2f}s',fill='white')
                writer.append_data(np.asarray(image))
    finally:
        for reader in readers:reader.close()


def run(monitor):
    with lock('report'):
        v=read(ROOT/'data/MANIFEST.json');rows=[];all_complete=True;receipts=[];comparisons=[];pairs_by_name={}
        own=list(conditions());expected=['clean',*own] if 'clean' not in own else own
        if config()['scope']=='control':expected=['clean','initial_s42','initial_s43','uada500',*own]
        for n,case in enumerate(v['cases']):
            paths=lookup(case)
            if 'clean' not in paths:all_complete=False;continue
            clean=read(paths['clean']/'EPISODE.json');eligible=bool(clean['success'])
            needed=expected if eligible else ['clean']
            if not all(k in paths for k in needed):all_complete=False
            for condition,p in paths.items():
                result=read(p/'EPISODE.json')
                rows.append({'start':case['start'],'suite':case['suite'],'task_id':case['task_id'],'state_index':case['state_index'],
                             'condition':condition,'clean_success':int(eligible),'success':result['success'],
                             'induced_failure':int(eligible and not result['success']),
                             'executed_steps':result['executed_steps'],**result['events']})
            if eligible and all(k in paths for k in expected):
                destination=ROOT/'comparisons'/case['start'];destination.mkdir(parents=True,exist_ok=True)
                receipt=destination/'COMPLETE.json'
                if not complete(receipt):
                    compose(case,{k:paths[k] for k in expected},destination)
                    finish(receipt,[destination/'comparison.mp4'])
                comparisons.append(receipt)
            monitor.update(phase='CPU_statistics_and_video',completed=n+1,total=len(v['cases']))
        f=pd.DataFrame(rows);summary=[]
        clean_count=int(f[(f.condition=='clean')].success.sum()) if len(f) else 0
        for name in expected:
            values=f[(f.condition==name)&(f.clean_success==1)] if len(f) else f
            pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':float(r.induced_failure)} for r in values.itertuples()]
            summary.append({'condition':name,'clean_successes':clean_count,'paired_completed':len(values),
                            'induced_failures':int(values.induced_failure.sum()) if len(values) else 0,
                            'CASR':float(values.induced_failure.mean()) if len(values) else None,
                            'task_start_bootstrap_95ci':paired_interval(pairs,2000,42)})
        paired=[]
        contrasts=[('radial_reverse_s42','uada500'),('radial_reverse_s43','uada500')]
        if config()['scope']=='control':contrasts=[(f'matched_relation_s{s}',f'shuffled_relation_s{s}') for s in (42,43)]
        for a,b in contrasts:
            if not len(f):continue
            left=f[(f.condition==a)&(f.clean_success==1)];right=f[(f.condition==b)&(f.clean_success==1)]
            merged=left.merge(right,on=['start','suite','task_id','state_index'],suffixes=('_a','_b'))
            pairs=[{'suite':r.suite,'task_id':r.task_id,'difference':float(r.induced_failure_a-r.induced_failure_b)} for r in merged.itertuples()]
            paired.append({'a':a,'b':b,'paired_completed':len(merged),'CASR_difference':float(np.mean([p['difference'] for p in pairs])) if pairs else None,
                           'task_start_bootstrap_95ci':paired_interval(pairs,2000,42)})
        f.to_csv(ROOT/'results/EPISODES.csv',index=False);pd.DataFrame(summary).to_csv(ROOT/'results/CASR.csv',index=False)
        pd.DataFrame(paired).to_csv(ROOT/'results/PAIRED_CONTRASTS.csv',index=False)
        status={'status':'COMPLETE' if all_complete else 'PARTIAL','scope':config()['scope'],
                'clean_starts_complete':len(f[f.condition=='clean']) if len(f) else 0,'expected_clean_starts':40,
                'clean_successes':clean_count,'native_episodes_in_table':len(f),'comparisons':len(comparisons),'summary':summary,'contrasts':paired,
                'failed_attempts':[str(p.relative_to(ROOT)) for p in (ROOT/'episodes').glob('*/*/attempt_*/FAILURE.json')]}
        atomic(status,ROOT/'results/STATUS.json')
        text=['# 原生PNG空间关系后续验证',f"状态：{status['status']}；40个起点中clean已完成{status['clean_starts_complete']}，成功{clean_count}。",
              '所有有效配对均为固定Goal checkpoint、Goal十任务，初态2–5；训练池保持四套件混合。未完成不计成功。',
              '|条件|完整配对|诱发失败|CASR|','|---|---:|---:|---:|']
        for r in summary:text.append(f"|{r['condition']}|{r['paired_completed']}|{r['induced_failures']}|{r['CASR'] if r['CASR'] is not None else '缺失'}|")
        text+=['','结果是开发背景下的新起点复验，不宣称历史从未使用。CASR按clean成功起点计算，另保留全部40个clean结果。',
               '关系对应对照使用相同数据、初始化、4,000更新及逐槽扰动长度；共同缩放保持原生动作边界。方向在同套件跨任务、跨轨迹置乱，不能当成所有非关系攻击的代表。',
               '方向置乱可能仍保留数据集共有的接近方向；无差异不直接证明策略没有空间信息。有差异也不能单独证明DINO专属机制。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        gallery=PROJECT/'research_videos'/ROOT.name
        (ROOT/'results/INDEX.html').write_text('<!doctype html><meta charset="utf-8"><h1>原生PNG后续评测</h1><p>'+status['status']+'</p><a href="RESULTS.md">结果</a> | <a href="CASR.csv">CASR</a><p>集中视频入口：'+str(gallery/'INDEX.html')+'</p>')
        if all_complete:
            for p in (ROOT/'episodes').glob('*/*/COMPLETE.json'):
                if complete(p):receipts.append(p)
            files=[ROOT/'results'/n for n in ('STATUS.json','CASR.csv','EPISODES.csv','PAIRED_CONTRASTS.csv','RESULTS.md','INDEX.html')]
            finish(ROOT/'results/COMPLETE.json',[*files,*receipts,*comparisons])
            # Portable independent copies; no change to the signed scientific run.
            import sys
            sys.path.insert(0,str(PROJECT/'research_tools'))
            from export_video_gallery import export
            export([(config()['video_label'],ROOT)],gallery,compilations=bool(comparisons))
