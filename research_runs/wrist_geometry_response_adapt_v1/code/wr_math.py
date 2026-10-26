# 发布注释：视角适配的B/C/R目标：动作张量为原生归一化8×7，SPAN处理各维有效跨度。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Objectives operate on normalized native 8x7 continuous actions."""
import math
import torch

SPAN=(2.,2.,2.,2.,2.,2.,1.)
# 平移/旋转跨度为2，夹爪跨度为1；先按跨度缩放，再对观测、槽位、维度平均。
def mse(a,b):
    span=torch.as_tensor(SPAN,device=a.device,dtype=a.dtype)
    return ((a-b)/span).square().mean()

# C比较同状态不同视角动作；B监督专家动作；R比较两时刻动作差。R是时序响应误差，不是移动物体的因果实验。
def objectives(camera_native,camera_changed,expert_first,expert_second,y_first,y_second):
    return dict(C=mse(camera_changed,camera_native),
                B=(mse(expert_first,y_first)+mse(expert_second,y_second))*.5,
                R=mse(expert_second-expert_first,y_second-y_first))

# BC/增强只用B；一致性用B+weight*C；约束法使用B、R相对基线的残差及非负乘子。baseline由校准集冻结。
def objective(values,baseline,spec,multipliers):
    eps=1e-6
    b=values['B']/(baseline['B']+eps)
    if spec['method'] in ('bc','augment'):return b
    c=values['C']/(baseline['C']+eps)
    if spec['method']=='consistency':return b+spec['weight']*c
    r=values['R']/(baseline['R']+eps)
    # Normalized constraints: denominators use the same epsilon as B/R.
    gb=(values['B']-(1+spec['tau'])*baseline['B'])/(baseline['B']+eps)
    gr=(values['R']-(1+spec['tau'])*baseline['R'])/(baseline['R']+eps)
    return c+multipliers['B']*gb+multipliers['R']*gr

# 按指数滑动平均更新非负乘子并限幅；这是训练约束的实现，不代表约束已在闭环中带来改善。
def update_dual(multipliers,ema,values,baseline,tau,cfg):
    for key in ('B','R'):
        residual=(float(values[key])-(1+tau)*baseline[key])/(baseline[key]+cfg['epsilon'])
        ema[key]=cfg['ema']*ema[key]+(1-cfg['ema'])*residual
        multipliers[key]=min(cfg['maximum'],max(0.,multipliers[key]+cfg['lr']*ema[key]))

def learning_rate(step,total,cfg):
    if step<cfg['warmup']:return cfg['lr']*(step+1)/cfg['warmup']
    phase=(step-cfg['warmup'])/max(1,total-1-cfg['warmup'])
    return cfg['min_lr']+(cfg['lr']-cfg['min_lr'])*.5*(1+math.cos(math.pi*phase))
