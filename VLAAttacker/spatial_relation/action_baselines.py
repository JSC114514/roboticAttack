"""Clean-reference continuous-head UADA adaptation; no expert-label claim."""
import torch

# 发布说明：输入严格为[1,8,7]的归一化原生动作。前六维范围[-1,1]、夹爪[0,1]；参考clean被detach。此实现恰在中点时取下界。
def far_targets(clean):
    if clean.shape!=(1,8,7) or not torch.isfinite(clean).all():raise ValueError('Expected finite 1x8x7 clean head actions')
    lower=clean.new_tensor([-1.]*6+[0.]);upper=clean.new_tensor([1.]*7)
    # Choose endpoints in the known native intervals. No gradient through the
    # reference; predictions in the loss remain unclipped.
    middle=(lower+upper)*.5
    return torch.where(clean.detach()>=middle,lower,upper)

# 发布说明：XYZ对照只优化前三维，全动作对照优化七维；目标基于clean预测，不是专家动作标签。
def action_loss(action,clean,method):
    if action.shape!=(1,8,7) or not torch.isfinite(action).all():raise ValueError('Invalid predicted head actions')
    if method not in ('uada_clean_xyz','uada_clean_7dof'):raise ValueError('Unknown UADA adaptation')
    target=far_targets(clean.to(action));dims=3 if method=='uada_clean_xyz' else 7
    return (action[...,:dims]-target[...,:dims]).square().mean()

# 发布说明：按套件轮换、任务均匀抽取有效观测，步数与seed固定。与关系方法配对时必须核对同一有效池及顺序。
def sample_schedule(rows,seed,steps=4000,suites=None):
    import random
    suites=suites or ['libero_spatial','libero_object','libero_goal','libero_10']
    grouped={}
    for r in rows:
        if not r['teacher']['no_target']:grouped.setdefault(r['suite'],{}).setdefault(r['task_id'],[]).append(r)
    result=[]
    for step in range(steps):
        rng=random.Random(seed*1000000+step);group=grouped[suites[step%len(suites)]]
        task=rng.choice(sorted(group));result.append(rng.choice(group[task])['state_id'])
    return result
