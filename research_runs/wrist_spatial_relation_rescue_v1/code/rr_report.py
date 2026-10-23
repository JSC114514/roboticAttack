"""Paired task/start statistics; privileged rescue is never labeled attack CASR."""
import html
from pathlib import Path
import numpy as np
import pandas as pd
from rr_common import ROOT,config,read,require,atomic,complete,finish,artifact_folder,lock
from audit_math import paired_interval

def result_folder(case,method):
    receipt=ROOT/'episodes'/case['start']/method/'COMPLETE.json'
    return artifact_folder(receipt,'/EPISODE.json') if complete(receipt) else None

def compose(case,paths,folder):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    labels=['clean','reference_replay',*config()['methods']];clean=Path(case['baseline_videos']['clean'])
    videos=[clean]+[paths[c]/'video.mp4' for c in labels[1:]]
    results=[case['baseline_episodes']['clean']]+[read(paths[c]/'EPISODE.json') for c in labels[1:]]
    fps=results[1]['fps'];require(all(abs(r['fps']-fps)<1e-9 for r in results[1:]),'Control rates differ')
    lengths=[r['executed_steps']+1 for r in results];readers=[];last=[None]*5
    try:
        for p in videos:readers.append(imageio.get_reader(p,ffmpeg_params=['-threads','1']))
        with imageio.get_writer(folder/'comparison.mp4',fps=fps,codec='libx264',quality=7,macro_block_size=16,
                ffmpeg_params=['-threads','2']) as writer:
            for frame_id in range(max(lengths)):
                image=Image.new('RGB',(1344,576),(24,24,24))
                for i,label in enumerate(labels):
                    if frame_id<lengths[i]:
                        try:last[i]=readers[i].get_next_data()
                        except (StopIteration,IndexError):raise RuntimeError('Truncated source video: '+str(videos[i]))
                    require(last[i] is not None,'Empty video');panel=Image.fromarray(last[i]).copy()
                    if frame_id>=lengths[i]:
                        d=ImageDraw.Draw(panel);d.rectangle((0,238,448,288),fill='black')
                        d.text((6,242),f'ENDED at {(lengths[i]-1)/fps:.2f}s; last frame held',fill='yellow')
                        d.text((6,260),f'Success={results[i]["success"]}',fill='yellow')
                    image.paste(panel,((i%3)*448,(i//3)*288))
                d=ImageDraw.Draw(image)
                for i,text in enumerate((case['start'],f'Common control clock {frame_id/fps:.2f}s',
                    'Clean / PNG reference / 3 restorations','RESTORATIONS ARE PRIVILEGED DIAGNOSIS',
                    'They are not new deployed PNG attacks','Same observation: clean action reference',
                    'Front / wrist; held frames labeled')):
                    d.text((904,302+i*23),text,fill='white')
                writer.append_data(np.asarray(image))
    finally:
        for reader in readers:reader.close()

def curves(case,paths,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,11),sharex=True)
    clean_events=Path(case['baseline_videos']['clean']).parent/'EVENTS.parquet'
    clean=pd.read_parquet(clean_events);axes[0].plot(clean.time_s,clean.distance_m*100,label='clean')
    for name,p in paths.items():
        f=pd.read_parquet(p/'EVENTS.parquet');axes[0].plot(f.time_s,f.distance_m*100,label=name)
        axes[1].plot(f.time_s,f.tracking_error_peak_m*100,label=name)
        axes[2].plot(f.time_s,f.peak_robot_contact_force_n,label=name)
        valid=f[f.visibility_verified];axes[3].plot(valid.time_s,valid.policy_uncovered_pixels,label=name)
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);axes[0].set_title(case['start']+' | privileged command-restoration diagnosis')
    for ax,label in zip(axes,('Entity center - EEF (cm)','OSC tracking peak (cm)','Robot contact force peak (N)','Uncovered target pixels')):ax.set_ylabel(label)
    axes[-1].set_xlabel('Native control time (s)');fig.tight_layout();fig.savefig(folder/'curves.png',dpi=150);plt.close(fig)

def diagnostics(case,method,folder,result):
    f=pd.read_parquet(folder/'EVENTS.parquet');s=pd.read_parquet(folder/'STEPS.parquet')
    delta=np.stack([np.array(x) for x in s.actual_eef_displacement_m]);dist=np.linalg.norm(delta,axis=1)
    rows=f.iloc[1:];tail=min(64,len(s));contact_pairs={}
    for item in s.controller:
        for pair in item['contacts']:
            if pair['involves_robot']:
                key=pair['geom1']+' | '+pair['geom2'];contact_pairs[key]=max(contact_pairs.get(key,0),pair['peak_force_norm_n'])
    top=sorted(contact_pairs.items(),key=lambda x:(-x[1],x[0]))[:8]
    return {'start':case['start'],'task_id':case['episode']['task_id'],'condition':method,'success':result['success'],
        'steps':result['executed_steps'],'first_contact_step':result['events']['first_contact_step'],
        'first_grasp_step':result['events']['first_grasp_step'],'last64_median_eef_displacement_mm':float(np.median(dist[-tail:])*1000),
        'tracking_peak_m':float(rows.tracking_error_peak_m.max()),'last64_tracking_median_m':float(rows.tracking_error_final_m.tail(tail).median()),
        'robot_contact_force_peak_n':float(rows.peak_robot_contact_force_n.max()),
        'torque_clip_peak_nm':float(rows.torque_clip_peak_nm.max()),'actuator_bound_samples':int(rows.actuator_bound_samples.sum()),
        'top_robot_contact_pairs':repr(top),'uncovered_visibility_fraction':result['events']['uncovered_visibility_fraction']}

def restoration_summary(cases,lookup):
    summary=[]
    for method in config()['methods']:
        pairs=[];rescued=regressed=original_failures=successes=0
        for case in cases:
            if (case['start'],method) not in lookup:continue
            result=lookup[(case['start'],method)];old=case['episode']['success']
            pairs.append({'suite':'libero_goal','task_id':case['episode']['task_id'],'difference':result['success']-old})
            rescued+=int(old==0 and result['success']==1);regressed+=int(old==1 and result['success']==0)
            original_failures+=int(old==0);successes+=result['success']
        summary.append({'condition':method,'paired_completed':len(pairs),'expected':len(cases),'successes':successes,
            'original_failures_in_pairs':original_failures,'rescued_original_failures':rescued,'regressed_original_successes':regressed,
            'success_rate_gain':float(np.mean([p['difference'] for p in pairs])) if pairs else None,
            'task_start_bootstrap_95ci':paired_interval(pairs,config()['bootstrap_draws'],42)})
    return summary

def run(manifest,monitor):
    with lock('report_writer'):
        (ROOT/'results').mkdir(parents=True,exist_ok=True)
        lookup={};paths_by_start={};rows=[];cards=[];completed_videos=[];episode_receipts=[]
        labels=['reference_replay',*config()['methods']]
        for case in manifest['cases']:
            paths={m:result_folder(case,m) for m in labels};paths_by_start[case['start']]=paths
            for method,p in paths.items():
                if p is None:continue
                result=read(p/'EPISODE.json');lookup[(case['start'],method)]=result
                rows.append(diagnostics(case,method,p,result));episode_receipts.append(ROOT/'episodes'/case['start']/method/'COMPLETE.json')
        summary=restoration_summary(manifest['cases'],lookup)
        for case in manifest['cases']:
            paths=paths_by_start[case['start']];available=sum(v is not None for v in paths.values())
            if available==4:
                out=ROOT/'comparisons'/case['start'];out.mkdir(parents=True,exist_ok=True)
                if not complete(out/'COMPLETE.json'):
                    compose(case,paths,out);curves(case,paths,out)
                    finish(out/'COMPLETE.json',[out/'comparison.mp4',out/'curves.png',*[ROOT/'episodes'/case['start']/m/'COMPLETE.json' for m in labels]])
                completed_videos.append(out/'COMPLETE.json')
                cards.append(f'<h2>{html.escape(case["start"])}</h2><video controls preload="none" width="100%" src="../comparisons/{case["start"]}/comparison.mp4"></video><img width="100%" src="../comparisons/{case["start"]}/curves.png">')
            else:cards.append(f'<h2>{html.escape(case["start"])}</h2><p>Completed {available}/4 diagnostic conditions; missing outcomes excluded.</p>')
            monitor.update(phase='CPU_report_and_comparisons',paired_videos_complete=len(completed_videos),paired_videos_total=18)
        failures=[read(p) for p in sorted((ROOT/'reports/COSTS').glob('*.json')) if read(p).get('status')=='FAILED']
        full=len(lookup)==72 and len(completed_videos)==18
        pd.DataFrame(summary).to_csv(ROOT/'results/RESTORATION.csv',index=False);pd.DataFrame(rows).to_csv(ROOT/'results/CONTROL_DIAGNOSTICS.csv',index=False)
        atomic({'status':'COMPLETE' if full else 'PARTIAL','diagnostic_episodes_complete':len(lookup),'expected':72,
            'new_policy_episodes_complete':sum(m!='reference_replay' for _,m in lookup),'reference_replays_complete':sum(m=='reference_replay' for _,m in lookup),
            'paired_videos_complete':len(completed_videos),'restoration_is_attack_score':False,'summary':summary,
            'engineering_failures':failures,'expansion_status':'PREPARED_NOT_SUBMITTED'},ROOT/'results/STATUS.json')
        text=['# 空间关系补丁：动作恢复诊断',
            f'已完成{len(lookup)}/72个诊断回合，其中原动作回放{sum(m=="reference_replay" for _,m in lookup)}，新恢复闭环{sum(m!="reference_replay" for _,m in lookup)}，完整比较视频{len(completed_videos)}/18。',
            '原18个起点的PNG部署成功5/18、失败13/18；clean成功18/18。原clean/随机初始/UADA/同池远边界只读复用；原训练池仍为四套件混合，本轮模型和环境均为Goal。',
            '恢复命令使用同一观测下的clean输出与仿真目标方向，是有特权的机制诊断。以下是任务成功的恢复量，不是新PNG攻击效果。',
            '|恢复条件|完成配对|成功|挽救原失败|损害原成功|成功率变化|','|---|---:|---:|---:|---:|---:|']
        for r in summary:
            gain='—' if r['success_rate_gain'] is None else f"{r['success_rate_gain']*100:+.2f} pp"
            text.append(f"|{r['condition']}|{r['paired_completed']}/18|{r['successes']}|{r['rescued_original_failures']}|{r['regressed_original_successes']}|{gain}|")
        text+=['','## 配对区间','按任务→起点重采样；多个槽位和视频帧不当独立重复。缺失回合不计成功。']
        for r in summary:text.append(f"- {r['condition']}: {r['task_start_bootstrap_95ci']}")
        text+=['','## 解释及下一步',
            'A有效支持径向命令参与失败；只有B有效提示不能只考虑径向分量；C有效提示姿态/夹爪通道参与。不同恢复改变后续状态，三者不构成互斥归因，恢复无效也不能直接否定空间机制。',
            'A恢复的是原生有效XYZ命令的径向分量，受边界约束时其他分量可能一起变化，TRACE保留正交变化；它不是恢复了物体位置或真实接触关系。',
            'CONTROL_DIAGNOSTICS.csv及曲线提供末端运动、跟踪误差、接触力和限幅。CONTACTS仅为物理子步聚合峰值，不能按峰值相加解释为持续作用力；命令非零而不动需要结合这些记录核对，不自动叫作策略停止。',
            '距离是固定实体中心到EEF，不是表面间距或精确抓取误差。可见性沿用已核验的原生时刻分割和类别掩码估计；不可见可能来自遮挡或离开视野。',
            'data/EXPANSION_PREPARED.json固定准备Goal任务0–9、初态2–5、8个原生PNG条件，最多320回合。本入口不会提交扩大任务；先讨论完整配对结果，不设统一高分门槛。',
            '扩大验证用于检查既有PNG是否重复有效，不把恢复成功当作补丁变强；新增初态仅相对当前方法协议，不宣称全部历史未见。']
        (ROOT/'results/RESULTS.md').write_text('\n\n'.join(text)+'\n')
        (ROOT/'results/INDEX.html').write_text('<!doctype html><meta charset="utf-8"><title>Spatial relation restoration</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto;padding:24px;background:#eee}h2{margin-top:40px}</style><h1>空间关系补丁：动作恢复诊断</h1><p>右侧三种恢复是特权诊断，不是新补丁部署结果。18个起点全部保留；结束画面冻结并标注，使用共同控制时间轴。</p><a href="RESULTS.md">结果说明</a> | <a href="RESTORATION.csv">配对恢复统计</a> | <a href="CONTROL_DIAGNOSTICS.csv">控制诊断</a>'+''.join(cards))
        if full:finish(ROOT/'results/COMPLETE.json',[*completed_videos,*episode_receipts,*[ROOT/'results'/n for n in ('RESTORATION.csv','CONTROL_DIAGNOSTICS.csv','STATUS.json','RESULTS.md','INDEX.html')]])
