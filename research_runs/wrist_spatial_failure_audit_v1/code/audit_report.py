"""Single-writer CPU summaries and paired videos; no historical report lock."""
from pathlib import Path
import html
import numpy as np
import pandas as pd
from audit_common import ROOT, SOURCE, config, read, require, complete, finish, atomic, lock
from audit_math import paired_interval

def attempt(entry):
    receipt=ROOT/'replays'/entry['key']/'COMPLETE.json'
    if not complete(receipt): return None
    r=read(receipt); paths=[ROOT/k for k in r['artifacts'] if k.endswith('/REPLAY.json')]
    require(len(paths)==1,'Ambiguous replay result'); return paths[0].parent

def summary(manifest):
    rows=manifest['historical_rows']; clean={(r['suite'],r['task_id'],r['state_index']):r for r in rows if r['condition']=='clean'}
    eligible={k for k,v in clean.items() if v['success']}
    lookup={(r['suite'],r['task_id'],r['state_index'],r['condition']):r for r in rows}
    conditions=sorted({r['condition'] for r in rows if r['condition']!='clean'}); summaries=[]; contrasts=[]
    for suite in ['all']+source_suites():
        starts=sorted(k for k in eligible if suite=='all' or k[0]==suite)
        for condition in conditions:
            paired=[lookup[(*k,condition)] for k in starts if (*k,condition) in lookup]
            summaries.append({'suite':suite,'condition':condition,'clean_successes':len(starts),'paired_completed':len(paired),
                'induced_failures':sum(r['failure'] for r in paired),'casr':float(np.mean([r['failure'] for r in paired])) if paired else None})
    for seed in (42,43):
        primary=f'radial_reverse_s{seed}'
        for other in (f'initial_s{seed}','uada500',f'translation_far_boundary_s{seed}'):
            pairs=[{'suite':k[0],'task_id':k[1],'difference':lookup[(*k,primary)]['failure']-lookup[(*k,other)]['failure']} for k in sorted(eligible)]
            contrasts.append({'treatment':primary,'comparator':other,'pairs':len(pairs),
                'failure_rate_difference':float(np.mean([p['difference'] for p in pairs])),
                'task_start_bootstrap_95ci':paired_interval(pairs,config()['bootstrap_draws'],config()['bootstrap_seed'])})
    coverage=[{'suite':s,'clean_completed':sum(k[0]==s for k in clean),'clean_successes':sum(k[0]==s for k in eligible)} for s in source_suites()]
    pd.DataFrame(summaries).to_csv(ROOT/'results/CASR.csv',index=False)
    atomic({'historical_episodes_complete':len(rows),'clean_starts':len(clean),'eligible_starts':len(eligible),
            'coverage':coverage,'contrasts':contrasts,'source_summary_lock_untouched':True,
            'failure_rates_are_start_weighted':True,'intervals_exploratory_task_then_start_bootstrap':True},ROOT/'results/SUMMARY.json')
    return summaries,contrasts,coverage

def source_suites(): return read(SOURCE/'config.json')['suites']

def raw_diagnostics(manifest):
    """Use saved physical movements, paired at the clean pre-contact horizon."""
    rows=[]; grouped={}
    for e in manifest['entries']: grouped.setdefault(e['start'],{})[e['episode']['condition']]=e
    for start,entries in grouped.items():
        clean=pd.read_parquet(entries['clean']['steps']).sort_values('step')
        contact=[int(r.step) for r in clean.itertuples() if r.geometry_after['contact']]
        horizon=max(1,min(64,(contact[0]-1) if contact else 64))
        clean_progress=float(clean[clean.step<=horizon].radial_eef_displacement_m.sum())
        for condition,e in entries.items():
            steps=pd.read_parquet(e['steps']).sort_values('step'); meta=e['episode']
            h=min(horizon,len(steps)); cprogress=float(clean[clean.step<=h].radial_eef_displacement_m.sum())
            progress=float(steps[steps.step<=h].radial_eef_displacement_m.sum())
            contacts=[int(r.step) for r in steps.itertuples() if r.geometry_after['contact']]
            grasps=[int(r.step) for r in steps.itertuples() if r.geometry_after['grasp']]
            action=np.stack([np.asarray(x) for x in steps.deployed_action]); baseline=np.stack([np.asarray(x) for x in steps.same_observation_clean_chunk_action])
            delta=action-baseline; early=steps.step.to_numpy()<=128
            rows.append({'start':start,'condition':condition,'success':meta['success'],'executed_steps':len(steps),
                'paired_pre_contact_horizon':h,'clean_projected_movement_m':cprogress,'patch_projected_movement_m':progress,
                'progress_deficit_m':cprogress-progress,'first_contact_step':contacts[0] if contacts else None,
                'first_grasp_step':grasps[0] if grasps else None,
                'failure_stage':'success' if meta['success'] else ('no_target_contact' if not contacts else 'after_target_contact'),
                'rotation_command_delta_median_128':float(np.median(np.linalg.norm(delta[early,3:6],axis=1))),
                'gripper_disagreement_128':float(np.mean((action[early,6]>.5)!=(baseline[early,6]>.5)))})
    frame=pd.DataFrame(rows); frame.to_csv(ROOT/'results/RECORDED_DIAGNOSTICS.csv',index=False); return frame

def compose(start,entries,paths,folder):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    conditions=config()['video_conditions']; readers=[]; last=[]; frames=[]
    fpss=[read(paths[c]/'REPLAY.json')['fps'] for c in conditions]
    require(all(abs(x-fpss[0])<1e-9 for x in fpss),'Paired control rates differ')
    fps=fpss[0]; lengths=[entries[c]['episode']['executed_steps']+1 for c in conditions]
    try:
        for c in conditions:
            reader=imageio.get_reader(paths[c]/'video.mp4',ffmpeg_params=['-threads','1']); readers.append(reader); last.append(None)
        with imageio.get_writer(folder/'comparison.mp4',fps=fps,codec='libx264',quality=7,macro_block_size=16,
                                ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as writer:
            for index in range(max(lengths)):
                canvas=Image.new('RGB',(1344,576),(24,24,24))
                for n,c in enumerate(conditions):
                    if index<lengths[n]:
                        try: last[n]=readers[n].get_next_data()
                        except (StopIteration,IndexError): raise RuntimeError('Truncated decoded video: '+c)
                    require(last[n] is not None,'Empty source video')
                    panel=Image.fromarray(last[n]).copy()
                    if index>=lengths[n]:
                        d=ImageDraw.Draw(panel); d.rectangle((0,238,448,288),fill='black')
                        d.text((6,242),f'ENDED at {(lengths[n]-1)/fps:.2f}s; last frame held',fill='yellow')
                        d.text((6,260),f'Historical success={entries[c]["episode"]["success"]}',fill='yellow')
                    canvas.paste(panel,((n%3)*448,(n//3)*288))
                draw=ImageDraw.Draw(canvas); x,y=896,288
                for i,text in enumerate((start,f'Common clock: {index/fps:.2f}s',
                    'Saved-action reconstruction only','Goal checkpoint, seed42 preset',
                    'Left: front; right: wrist + PNG','FOV visibility: NN mask estimate',
                    'Completed clips hold labeled last frame','Ordering alone is not causality')):
                    draw.text((x+8,y+12+i*22),text,fill='white')
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers: reader.close()

def plot(start,paths,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for condition,p in paths.items():
        f=pd.read_parquet(p/'EVENTS.parquet'); axes[0].plot(f.time_s,100*f.distance_m,label=condition)
        valid=f[f.visibility_verified]; axes[1].plot(valid.time_s,valid.policy_uncovered_pixels,label=condition)
    axes[0].set_ylabel('Fixed entity center - EEF (cm)'); axes[1].set_ylabel('Uncovered target pixels (estimate)')
    axes[1].set_xlabel('Native control time (s)'); axes[0].legend(fontsize=7); axes[0].set_title(start+' | paired saved-action replay')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'curves.png',dpi=150); plt.close(fig)

def run(manifest,monitor,with_videos=True):
    with lock('report'):
        (ROOT/'results').mkdir(parents=True,exist_ok=True)
        summaries,contrasts,coverage=summary(manifest); raw=raw_diagnostics(manifest); grouped={}; cards=[]; completed=[]
        for e in manifest['entries']: grouped.setdefault(e['start'],{})[e['episode']['condition']]=e
        event_rows=[]
        for start,entries in sorted(grouped.items()):
            paths={c:attempt(e) for c,e in entries.items()}; available={c:p for c,p in paths.items() if p is not None}
            for c,p in available.items():
                v=read(p/'REPLAY.json'); event_rows.append({'start':start,'condition':c,**v['events']})
            folder=ROOT/'comparisons'/start; paired=all(paths.values())
            if paired and with_videos:
                folder.mkdir(parents=True,exist_ok=True)
                if not complete(folder/'COMPLETE.json'):
                    compose(start,entries,paths,folder); plot(start,paths,folder)
                    finish(folder/'COMPLETE.json',[folder/'comparison.mp4',folder/'curves.png',*[ROOT/'replays'/e['key']/'COMPLETE.json' for e in entries.values()]])
                completed.append(folder/'COMPLETE.json')
                cards.append(f'<h2>{html.escape(start)}</h2><video controls preload="none" width="100%" src="../comparisons/{start}/comparison.mp4"></video><img width="100%" src="../comparisons/{start}/curves.png">')
            else: cards.append(f'<h2>{html.escape(start)}</h2><p>Verified replay conditions: {len(available)}/5; paired video pending.</p>')
            monitor.update(phase='CPU_report_and_composites',starts_complete=len(completed),starts_total=len(grouped))
        event_frame=pd.DataFrame(event_rows); event_frame.to_csv(ROOT/'results/EVENT_TIMELINES.csv',index=False)
        full=with_videos and len(completed)==18
        failures=[read(p) for p in sorted((ROOT/'reports/COSTS').glob('*.json')) if read(p).get('status')=='FAILED']
        atomic({'status':'COMPLETE' if full else 'PARTIAL','historical_evaluation_complete':True,
            'historical_episodes':206,'eligible_starts':18,'replays_complete':len(event_rows),'replays_total':90,
            'paired_videos_complete':len(completed),'new_policy_episodes':0,'new_training_updates':0,
            'engineering_failures':failures,
            'next_step':'Review event ordering and video, then design one relation-specific rescue; not queued automatically'},ROOT/'results/STATUS.json')
        text=['# 空间接近实验：失败过程审计',
            '原训练和206回合已完成；此目录独立汇总，未修改历史结果或争用历史report.lock。',
            '固定Goal模型在80个起点中仅18个clean成功，全部属于Goal；本结果不证明四套件通用。',
            '视频回放保存的原生动作，不加载VLA；仅通过起点、逐步物理/接触、终止及历史关键帧核验的视频可进入配对展示。回放不是新增独立实验。',
            f'本轮视频范围：预先指定seed42，18起点×5条件。已核验{len(event_rows)}/90段，完成{len(completed)}/18个比较视频。',
            'seed43保留全部历史统计，不以seed42视频替代重复性结果。',
            '|条件|新增失败/18|CASR|','|---|---:|---:|']
        for r in summaries:
            if r['suite']=='all': text.append(f"|{r['condition']}|{r['induced_failures']}/{r['paired_completed']}|{r['casr']:.2%}|")
        text+=['','## 配对差异与不确定性（按任务→起点重采样）']
        for v in contrasts:
            ci=v['task_start_bootstrap_95ci']; text.append(f"- {v['treatment']} − {v['comparator']}：{v['failure_rate_difference']*100:+.2f} pp，探索性95%区间[{ci[0]*100:.2f}, {ci[1]*100:.2f}] pp。")
        text+=['','## 已保存的实际运动诊断']
        for condition,g in raw.groupby('condition',sort=False):
            text.append(f"- {condition}：同一clean接触前、至多64步窗口，实际径向前进减少的中位数{g.progress_deficit_m.median()*1000:.2f} mm；失败阶段{g.failure_stage.value_counts().to_dict()}。")
        text+=['','## 解释边界与下一步',
            '径向运动是实际末端位移沿当时目标方向投影之和，不是直线净位移，也不是动作命令对应的保证移动距离。',
            'FOV/补丁遮挡使用原生腕部采样时刻的实例分割，类别掩码最近邻映射到224并按官方0.9面积中心裁剪采样。它是可见性估计，不是精确重建JPEG边界；初始帧未强制刷新传感器，故可见性缺失。',
            'EVENT_TIMELINES.csv列出首次接触/抓持、首次在策略视野内不可见、首次叠加后完全不可见、仅因补丁覆盖而完全不可见及完整覆盖率。掩码为零可由遮挡或离开视野造成，不能直接断言相机转离目标；相机姿态另存于EVENTS.parquet供核对。不删除这些回合，不依据时间先后宣称因果。',
            'RECORDED_DIAGNOSTICS.csv和视频同时检查平移、姿态、夹爪；固定实体中心不是精确抓取点，多阶段任务仅解释第一交互对象。',
            '当前对照强度未匹配：远边界使用各平移维最远有效端点，关系法要求径向退离；失败率差异不能单独归因于是否使用空间信息。',
            '先人工核对全部视频/事件，选择证据支持的距离、侧向对齐或姿态关系，再单独设计恢复该关系的救援验证。本入口不自动追加训练、救援或新checkpoint评测。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        page='<!doctype html><meta charset="utf-8"><title>Spatial failure audit</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto;padding:24px;background:#eee}video,img{background:#222}h2{margin-top:42px}</style><h1>空间关系补丁：18个配对起点</h1><p>保存动作回放，不是新增策略评测。Goal模型；seed42预先指定。全部起点按任务排序，未按效果挑选。每格左前视、右腕部；结束画面冻结并明确标记。</p><p><a href="RESULTS.md">统计和限制</a> | <a href="EVENT_TIMELINES.csv">可见性/接触事件</a> | <a href="RECORDED_DIAGNOSTICS.csv">实际运动诊断</a></p>'+''.join(cards)
        (ROOT/'results/INDEX.html').write_text(page)
        if full:
            finish(ROOT/'results/COMPLETE.json',[*completed,ROOT/'results/STATUS.json',ROOT/'results/RESULTS.md',ROOT/'results/INDEX.html',
                ROOT/'results/CASR.csv',ROOT/'results/SUMMARY.json',ROOT/'results/EVENT_TIMELINES.csv',ROOT/'results/RECORDED_DIAGNOSTICS.csv'])
