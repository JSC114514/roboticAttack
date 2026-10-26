"""Physical distance evidence, paired horizons and explicit missing coverage."""
import time
import numpy as np
import pandas as pd
from sd_common import ROOT,config,read_json,atomic_json,atomic_parquet,complete,finish,scan_receipt,scan_path,case_root,worker_complete
from sd_math import hierarchical_interval

def run(monitor):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=read_json(ROOT/'data/MANIFEST.json');cases=[];missing=[];rolls=[];offline=[];pairs=[];all_steps=[];artifacts=[]
    for e in manifest['episodes']:
        receipt=scan_receipt(e,e['state_index'])
        if not complete(receipt):
            missing.append({'episode_id':e['episode_id'],'reason':'capture_not_complete'});continue
        selection=torch.load(scan_path(e,e['state_index']),map_location='cpu',weights_only=False)
        cases.extend(selection['cases'])
        for b,reason in selection['missing_bins'].items():missing.append({'episode_id':e['episode_id'],'bin':b,'reason':reason,'cohort':e['cohort']})
    complete_cases=0
    for case in cases:
        root=case_root(case);off=root/'offline/COMPLETE.json'
        if complete(off):
            value=read_json(root/'offline/OFFLINE.json')
            offline.extend({k:case[k] for k in ('case_id','episode_id','suite','task_id','cohort','bin')}|r for r in value['rows'])
        curves={};meta={}
        for condition in config()['conditions']:
            receipt=root/condition/'COMPLETE.json'
            if not complete(receipt):continue
            attempt=ROOT/read_json(receipt)['attempt'];r=read_json(attempt/'EPISODE.json')
            rolls.append(r);meta[condition]=r;curves[condition]=pd.read_parquet(attempt/'STEPS.parquet')
            st=curves[condition].copy()
            for name in ('case_id','episode_id','suite','task_id','cohort','bin'):st[name]=case[name]
            st['condition']=condition;all_steps.append(st)
        if len(curves)==3:
            complete_cases+=1;clean=curves['clean']
            for condition in ('dino_near','dino_far'):
                altered=curves[condition];n=min(len(clean),len(altered))-1
                pc=clean.iloc[:n+1].distance_m.to_numpy();pa=altered.iloc[:n+1].distance_m.to_numpy()
                pairs.append({k:case[k] for k in ('case_id','episode_id','suite','task_id','cohort','bin')}|
                    {'condition':condition,'common_executed_steps':n,
                    'paired_end_error_advantage_m':float(abs(pc[-1]-.05)-abs(pa[-1]-.05)),
                    'paired_mean_error_advantage_m':float(np.abs(pc-.05).mean()-np.abs(pa-.05).mean()),
                    'paired_end_distance_change_m':float(pa[-1]-pc[-1]),
                    'intervention_step_fraction':meta[condition]['intervention_step_fraction']})
        if curves:
            fig,ax=plt.subplots(figsize=(7,3))
            for condition,df in curves.items():
                ax.plot(df.step,df.distance_m*100,label=condition)
                ax.plot(df.step.iloc[-1],df.distance_m.iloc[-1]*100,'o',markersize=3)
            ax.axhspan(4,6,alpha=.12,color='green');ax.axhline(5,ls='--',color='black',lw=.6)
            ax.set(xlabel='Executed control steps',ylabel='EEF to fixed anchor (cm)',title=case['case_id'])
            ax.legend(fontsize=7);fig.tight_layout();p=ROOT/f'results/curves/{case["case_id"]}.png'
            p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=120);plt.close(fig);artifacts.append(p)
    paths={name:ROOT/f'results/{name}.parquet' for name in ('ROLLOUTS','OFFLINE','PAIRED','STEPS')}
    frames={'ROLLOUTS':pd.DataFrame(rolls),'OFFLINE':pd.DataFrame(offline),'PAIRED':pd.DataFrame(pairs),
        'STEPS':pd.concat(all_steps,ignore_index=True) if all_steps else pd.DataFrame()}
    for name,df in frames.items():atomic_parquet(df,paths[name]);artifacts.append(paths[name])
    summary=[]
    for name,metrics in (('ROLLOUTS',['end_distance_m','end_error_m','error_reduction_m','net_radial_change_m',
        'longest_band_dwell_seconds','band_dwell_path_m','intervention_step_fraction']),
        ('PAIRED',['paired_end_error_advantage_m','paired_mean_error_advantage_m','paired_end_distance_change_m'])):
        df=frames[name]
        if df.empty:continue
        for group,rows in df.groupby(['cohort','bin','condition']):
            for metric in metrics:
                summary.append(dict(zip(('cohort','bin','condition'),group))|{'source':name,'metric':metric,
                    **hierarchical_interval(rows,metric,config()['bootstrap_draws'])})
    p=ROOT/'results/SUMMARY.csv';pd.DataFrame(summary).to_csv(p,index=False);artifacts.append(p)
    missing_path=ROOT/'results/COVERAGE.json';atomic_json({'selected_states':len(cases),'max_states':28,
        'missing':missing,'full_pair_cases':complete_cases,'completed_short_rollouts':len(rolls)},missing_path);artifacts.append(missing_path)
    stages_complete=all(worker_complete('capture',i) and worker_complete('mechanism',i) for i in (0,1))
    status={'status':'COMPLETE' if stages_complete else 'PARTIAL','signature':manifest['signature'],
        'selected_states':len(cases),'full_pair_cases':complete_cases,'completed_short_rollouts':len(rolls),
        'expected_short_rollouts':len(cases)*3,'missing_distance_bins':len(missing),'no_PNG_trained':True,
        'scope':'internal_DINO_mechanism_not_CASR','updated_unix':time.time()}
    atomic_json(status,ROOT/'results/STATUS.json');artifacts.append(ROOT/'results/STATUS.json')
    lines=['# 通用5厘米间距：DINO机制验证','',
        f'状态：{status["status"]}；选中{len(cases)}/最多28个状态；完整三条件配对{complete_cases}个；短回合{len(rolls)}/{len(cases)*3}。','',
        '本轮没有训练或部署PNG。128步截断及首次接触停止，不是完整任务CASR；不能用任务未完成率代替攻击成功率。',
        '所有任务固定Spatial checkpoint，其他套件的行为不能解释为其原生策略表现。',
        '5厘米是探索参数；参考点是碰撞几何中心，不等于精确抓取点。书是本轮留出对象，但有历史开发背景。','',
        '|任务/起点|距离组|条件|起始→最终距离(cm)|4—6cm最长停留(s)|期间路径(cm)|停止原因|实际干预覆盖|',
        '|---|---|---|---:|---:|---:|---|---:|']
    for r in rolls:
        lines.append(f'|{r["episode_id"]}|{r["bin"]}|{r["condition"]}|{r["start_distance_m"]*100:.2f}→{r["end_distance_m"]*100:.2f}|{r["longest_band_dwell_seconds"]:.2f}|{r["band_dwell_path_m"]*100:.2f}|{r["stop_reason"]}|{r["intervention_step_fraction"]:.0%}|')
    lines += ['', '## 当前建议','']
    val=frames['ROLLOUTS'];val=val[(val.cohort!='discovery')&(val.condition=='dino_near')] if not val.empty else val
    if not stages_complete:lines.append('先查看已完成配对及工程失败记录；缺失回合不计为成功。本轮不自动追加任务。')
    if val.empty:lines.append('尚无可解释的Validation距离轨迹，不能决定进入PNG训练。')
    else:
        far=val[val.bin=='far'];near=val[val.bin=='near'];supported=[]
        for label,g in [('far',far),('near',near)]:
            if g.empty:lines.append(f'{label}距离组缺失：不能声称双侧收敛。');continue
            groups=g.groupby(['suite','task_id']).agg(error_reduction_m=('error_reduction_m','mean'),
                radial=('net_radial_change_m','mean'),coverage=('intervention_step_fraction','mean'))
            direction=(groups.radial<0) if label=='far' else (groups.radial>0)
            n=int((direction & (groups.error_reduction_m>0) & (groups.coverage>0)).sum())
            lines.append(f'{label}组：{n}/{len(groups)}个任务的均值同时朝5厘米误差减小并沿预期方向移动；这是描述性结果，不是显著性判定。')
            supported.append(n>0)
        if len(supported)==2 and all(supported):
            lines.append('可以继续审查跨对象距离曲线和停留期间运动量，再设计共享PNG小试；不单凭终点或经过4—6厘米认定稳定控制。')
        elif len(supported)==2 and supported[0]:
            lines.append('目前最多支持“影响接近过程”；不能声称近处返回或固定间距控制。先收窄目标，不追加长训练。')
        else:lines.append('当前尚不支持用这一固定距离变换训练共享PNG；优先解释有效干预覆盖和动作—实际运动的差异，不增加训练轮数。')
    lines += ['', '## 覆盖、工程与解释限制','',
        'COVERAGE.json逐条保留缺少近/远状态的轨迹。OFFLINE.parquet保存全部8×7动作与随机对照。',
        'STEPS.parquet逐控制步保存真实距离、接触、抓持、可见性及实际执行槽位；可见性注明原生相机采样时间。',
        'PAIRED.parquet只比较两条件共同执行长度，避免接触提前停止造成终点时间混淆；截断和停止原因同时报告。',
        '虚拟穿插、移出视野时回退clean表示；覆盖低时不能把无响应解释为DINO机制不存在。',
        '移动可视几何也改变局部像素、遮挡等；不能单凭本轮响应断言找到了独立的距离编码或距离神经元。',
        '原生零位移RGB容差沿用NRMSE≤1e-4，物理/物体/相机数组及采样指纹严格配对；全部差异见每次PAIRING.json。',
        'SUMMARY.csv按任务→轨迹重采样，近/远状态先在轨迹内平均；样本少，区间不代表广泛环境泛化。',
        '代表性GIF、final.png和START.npz见evaluation下各条件的完成回执；曲线见results/curves。',
        '成本及失败记录：reports/COSTS；RSS与CUDA分列；Slurm MaxRSS由--status --slurm另报。']
    report=ROOT/'results/RESULTS.md';report.write_text('\n'.join(lines)+'\n');artifacts.append(report)
    if stages_complete:finish(ROOT/'results/COMPLETE.json',artifacts)
    monitor.update(phase='report_written',completed=1,total=1,**status)
