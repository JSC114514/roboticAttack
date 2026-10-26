# 发布注释：关系动作目标：区分控制命令（动作归一化后的目标增量）与真实末端运动。
# 运行来源与发布哈希分别记录在SOURCE_MANIFEST.json；历史运行目录不改动。
"""Geometric command targets; commands are never called realized motion."""
import math
import numpy as np

# clean最后一维为XYZ；unit是末端指向目标的单位向量。将命令投影到动作盒与u·v≤−margin的交集，用最小改动阻断接近；不是把物体移出视野。
def retreat_target(clean, unit, lower, upper, margin):
    """Euclidean projection onto box intersected with u.dot(v)<=-margin."""
    v=np.asarray(clean,dtype=np.float64);u=np.asarray(unit,dtype=np.float64)
    lo=np.asarray(lower,dtype=np.float64);hi=np.asarray(upper,dtype=np.float64)
    if v.shape[-1]!=3 or u.shape!=(3,) or not np.isfinite(v).all():raise ValueError('Invalid command/direction')
    if not np.isclose(np.linalg.norm(u),1,atol=1e-7) or not (np.all(lo<hi) and margin>0):raise ValueError('Invalid bounds')
    if np.sum(np.where(u>=0,lo,hi)*u)>-margin:raise ValueError('Infeasible retreat')
    def project(lam):return np.clip(v-lam[...,None]*u,lo,hi)
    a=np.zeros(v.shape[:-1]);b=np.ones_like(a)
    while np.any(np.sum(project(b)*u,axis=-1)>-margin):b*=2
    for _ in range(64):
        mid=(a+b)*.5;bad=np.sum(project(mid)*u,axis=-1)>-margin
        a=np.where(bad,mid,a);b=np.where(bad,b,mid)
    original=np.clip(v,lo,hi)
    return np.where((np.sum(original*u,axis=-1)<=-margin)[...,None],original,project(b)).astype(np.float32)

# 逐维选距clean命令更远的有效端点，等距取上界。此函数不读取物体身份或空间方向，是普通动作攻击对照。
def far_target(clean,lower,upper):
    v=np.asarray(clean);lo=np.asarray(lower);hi=np.asarray(upper)
    return np.where(abs(v-lo)>abs(hi-v),lo,hi).astype(np.float32)

# 按数据集统计把归一化动作映射为原生动作；mask为假的维度保持输入值。
def affine_actions(action,low,high,mask):
    import torch
    lo=torch.as_tensor(low,device=action.device,dtype=action.dtype)
    hi=torch.as_tensor(high,device=action.device,dtype=action.dtype)
    mask=torch.as_tensor(mask,device=action.device,dtype=torch.bool)
    return torch.where(mask,.5*(action+1)*(hi-lo+1e-8)+lo,action)

# 以teacher中的目标命令计算尺度归一化MSE；尺度来自缓存，不能将其误作实际执行位移。
def objective(command,teacher,method):
    target=teacher['targets'][method].to(command)
    scale=teacher['command_scale'].to(command)
    return ((command-target)/scale).square().mean()

def learning_rate(step,total,warmup,peak):
    if step<=warmup:return peak*step/warmup
    return peak*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))
