"""Clean-reference continuous-head UADA adaptation; no expert-label claim."""
import torch

# 发布说明：归一化clean参考动作[1,8,7]逐维选择较远端点，此实现中点取下界；detach防止目标随补丁共同移动。
def far_targets(clean):
    if clean.shape!=(1,8,7) or not torch.isfinite(clean).all():raise ValueError('Expected finite 1x8x7 clean head actions')
    lower=clean.new_tensor([-1.]*6+[0.]);upper=clean.new_tensor([1.]*7)
    # Choose endpoints in the known native intervals. No gradient through the
    # reference; predictions in the loss remain unclipped.
    middle=(lower+upper)*.5
    return torch.where(clean.detach()>=middle,lower,upper)

# 发布说明：uada_clean_xyz优化XYZ；uada_clean_7dof优化全部7维。预测不裁剪，使用实际动作头前向。
def action_loss(action,clean,method):
    if action.shape!=(1,8,7) or not torch.isfinite(action).all():raise ValueError('Invalid predicted head actions')
    if method not in ('uada_clean_xyz','uada_clean_7dof'):raise ValueError('Unknown UADA adaptation')
    target=far_targets(clean.to(action));dims=3 if method=='uada_clean_xyz' else 7
    return (action[...,:dims]-target[...,:dims]).square().mean()

# 发布说明：排除无有效teacher的记录；每步seed固定，按套件轮换并抽取任务，保证方法间样本顺序可配对。
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
