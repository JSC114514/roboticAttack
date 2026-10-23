"""Feature fidelity, actual PNG failures and baseline overlap, without ranking by feature MSE."""
import pandas as pd
from rgb_common import (ROOT,MATCH,config,module,require,read_json,atomic_json,atomic_parquet,
                        complete,finish,episode_path,selections)

def run(monitor):
    expected=[ROOT/'evaluation/COMPLETE.json']+[ROOT/f'training/{k}/COMPLETE.json' for k in config()['candidates']]
    missing=[]
    for path in expected:
        try:require(complete(path),'Stage incomplete')
        except Exception as error:missing.append({'path':str(path),'error':repr(error)})
    folder=ROOT/'results';folder.mkdir(parents=True,exist_ok=True)
    if missing:
        atomic_json({'status':'FAILED_OR_INCOMPLETE','missing':missing},folder/'STATUS.json')
        (folder/'RESULTS.md').write_text('本轮未完整完成，不能给出完整RGB补丁结论。详见STATUS.json及失败成本。\n')
        raise RuntimeError('RGB pilot incomplete')
    rolls=pd.DataFrame([read_json(episode_path(t,s,k).parent/'EPISODE.json') for t in config()['eval_tasks'] for s in config()['eval_states'] for k in config()['conditions']])
    paired=rolls.pivot(index=['task_id','state_index'],columns='condition',values='success')
    require(len(paired)==10 and not paired.isna().any().any(),'Missing paired conditions')
    eligible=paired[paired.clean==1];scores=[]
    reference=module('rgb_readonly_summary',MATCH/'code/report.py')
    for kind in ('initial','uada','projection','full'):
        failures=1-eligible[kind];random_failures=1-eligible.initial
        data=failures.reset_index();groups=[g[kind].to_numpy() for _,g in data.groupby('task_id')]
        scores.append({'condition':kind,'failures':int(failures.sum()),'clean_successes':len(eligible),
            'CASR_pooled':float(failures.mean()) if len(eligible) else None,
            'failures_beyond_random':int(((failures==1)&(random_failures==0)).sum()),
            'failures_overlap_random':int(((failures==1)&(random_failures==1)).sum()),
            'random_only_failures':int(((failures==0)&(random_failures==1)).sum()),
            **reference.interval(groups,draws=config()['bootstrap_draws'])})
    pd.DataFrame(scores).to_csv(folder/'CASR.csv',index=False);paired.to_csv(folder/'PAIRS.csv');atomic_parquet(rolls,folder/'ROLLOUTS.parquet')
    parts=[]
    ids=read_json(ROOT/'data/MANIFEST.json')['individual_ids']
    for cohort in ('fit','verify'):
        for row in selections()[cohort]:
            if cohort=='fit' and row['state_id'] not in ids:continue
            receipt=ROOT/f'evaluation/offline/{row["state_id"]}.json'
            require(complete(receipt),'Offline record missing')
            parts.append(pd.read_parquet(receipt.with_suffix('.parquet')))
    offline=pd.concat(parts,ignore_index=True);atomic_parquet(offline,folder/'OFFLINE.parquet')
    summary=reference.summarize(offline,['cohort','condition'],['q','q_delta','orthogonal','target_distance',
        'first_translation_delta_l2','direction_positive','direction_cos','restore_clean_dino_delta_l2','restore_clean_siglip_delta_l2'])
    summary.to_csv(folder/'OFFLINE_SUMMARY.csv',index=False)
    lines=['# 2倍几何目标的共享RGB可实现性小试','',
        '拟合240观测/48轨迹，验证240观测/48轨迹；均覆盖原24任务及5个固定进度点。验证是同任务跨轨迹，不是新任务。',
        '同一Spatial模型、seed42、共享50×50腕部PNG、固定位置；没有EOT或动作损失。正方向是探索性开发选择，不声称胜过反方向。',
        '每候选4800次候选更新，每图访问20次；另12张单图补丁各100次更新，只报告拟合诊断，预算不同，不能单凭差距证明共享冲突。',
        '新目标来自视觉位移2倍后重新渲染及编码，不是旧DINO特征差乘2。拟合与验证状态40/41、42/43；闭环44/45为已查看开发集。','',
        '|条件|失败切换/clean成功|CASR|随机成功而该条件失败|','|---|---:|---:|---:|']
    for r in scores:
        score='NA' if r['CASR_pooled'] is None else f'{r["CASR_pooled"]:.1%}'
        lines.append(f'|{r["condition"]}|{r["failures"]}/{r["clean_successes"]}|{score}|{r["failures_beyond_random"]}|')
    lines += ['', 'clean、初始随机与UADA共30回合按哈希、种子及reset数组/指纹核验后复用；只新执行两张PNG的20回合。所有条件完整保留。',
        '原生PNG前向同时影响DINO和SigLIP；只有训练梯度来自DINO目标。恢复干净分支的离线诊断不混入实际闭环回合。',
        '表示匹配、动作方向与任务失败分别报告；不能用离线特征距离替代CASR，也不能把内部2倍干预的40%写成PNG效果。',
        '损失日志、实际优化/无目标次数和PNG在training；RSS/CUDA峰值、失败成本及30秒心跳见reports。',
        '这是可实现性小试，不是新的方法命名或独立确认；不自动长训、扩任务、增加方向或调大补丁。']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json({'status':'COMPLETE','new_closedloop_episodes':20,'reused_baselines':30,'closedloop_is_native_RGB':True,'automatic_long_training':False},folder/'STATUS.json')
    finish(folder/'COMPLETE.json',[folder/p for p in ('RESULTS.md','STATUS.json','CASR.csv','PAIRS.csv','ROLLOUTS.parquet','OFFLINE.parquet','OFFLINE_SUMMARY.csv')])
    monitor.update(phase='report_written',completed=1,total=1)
