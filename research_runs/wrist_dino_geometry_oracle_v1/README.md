# 当前几何目标：理想DINO闭环验证

检验“完全实现现有几何目标，是否足以破坏任务”。这是内部机制验证，不是补丁部署，不训练任何新模型/补丁，不自动扩展方案。旧实验文件只读，任务4不补齐。

固定Spatial checkpoint，任务0/1/2/6/7、初态44/45，clean与dino_plus/dino_minus共30回合。沿用此前10组已完整配对起点及重置种子，各条件重新执行。条件运行顺序固定种子打乱。单4090串行完成，4CPU、12G；上轮同类进程RSS峰值约5.65GiB、CUDA allocated约14.96GiB。预计10–20分钟不含排队，新流程未实测；申请45分钟上限。CPU汇总2CPU、4G，不占GPU。

## 干预与对齐

复用relation_bias原几何代码、目标实体选择及位移规则：依任务谓词顺序选择首个腕部可见、可渲染实体；在水平世界X轴正负平移，幅度等于原30°方位弦长并截断到30mm，保留高度。没有固定改成30mm，不新增强度或动作目标。仅省略不使用的bearing图像渲染。不可见、水平距离退化等记录保留，回退原生动作并报告覆盖率。

每次决策对应反事实必须在原生相机_sensor实际采样回调中生成。复用原渲染场景geom临时位移及finally恢复，不修改物理状态，不推进仿真，不手动刷新相机。零位移重现原RGB的NRMSE上限仍为1e-4，渲染前后物理/物体/相机/传感器时钟指纹必须完全一致。回调画面必须逐像素等于env.step返回画面。

原生硬重置会重建sensor绑定，因此在reset和set_init_state之后安装钩子；仍执行原10步等待，只在最后一步采样第一决策的反事实。随后每8动作块仅最后执行槽位开启采样，与已核验的旧缓存实现一致。异常时恢复sensor和场景钩子。

提取反事实腕部DINO原生预处理后的block22空间输出，完整替换当前观测腕部DINO分支输出。前视、腕部SigLIP、语言、本体状态不动。每决策核验实际消费DINO等于目标且SigLIP等于clean；同时保存同当前观测的原生clean动作参考，不执行参考动作。无目标时不替换。不是在动作输出后添加偏差。

GPU首批检查：身份替换重现原生动作；单独提取CF DINO等于CF原生前向消费表示；CF双分支替换等于原生CF RGB动作；DINO-only保留clean SigLIP；模型冻结。工程检查不设置攻击效果阈值。

配对在动作执行前比较reset后完整sim_state、双相机画面、本体状态。保持逐项严格一致；START.npz保存原数组，失败时PAIRING日志分别给出形状、最大差值和不同元素数量。不得将工程不一致记成攻击成功。

## 手动提交与监控

```bash
bash research_runs/wrist_dino_geometry_oracle_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_oracle_v1/run.sh --dry-run
bash research_runs/wrist_dino_geometry_oracle_v1/run.sh --submit
bash research_runs/wrist_dino_geometry_oracle_v1/run.sh --status --slurm
bash research_runs/wrist_dino_geometry_oracle_v1/run.sh --watch --slurm --interval 30
```

--submit仅手动运行时提交1个GPU评测任务及afterany依赖CPU汇总。查询不到队列时停止提交；本实验已有活动任务则去重；不自动取消、重试不确定提交或开启训练。相同命令按已完成回合续跑，失败回合保留独立attempt目录；报告未完成时明确列出缺失，不静默删任务。

每30秒心跳记录当前任务/初态/方向、回合与步骤数、预计剩余时间及渲染成本。reports/COSTS记录全部成功/失败尝试的RSS HWM、CUDA allocated/reserved峰值、设备和耗时；Slurm sampled MaxRSS单独查看。

## 结果解释

results/RESULTS.md给出本轮clean成功、两个方向的CASR、目标可用决策数及实际执行槽位干预覆盖率。CASR.csv同时报告计数比例、任务等权均值及任务→初态重采样区间。5任务/10起点是已查看开发集，不宣称独立泛化，不按显著性门槛筛结果。

TRACE保存完整8×7动作、实际执行槽位、同观测clean参考、目标实体/位移/可见性、相机采样时钟、末端与夹爪关节。关键帧第0/4/12块是front/clean腕部/CF腕部拼图（CF图仅是DINO目标来源，SigLIP仍看clean）；最终帧为原画面。每回合最多4张图，不把动作块当独立重复。

若内部目标能造成失败，支持进一步研究共享RGB如何实现；若未失败，仅否定本数据、当前幅度与覆盖下的目标，不否定全部DINO空间信息。即使替换精确，也不是任意DINO干预的上界。不会根据本轮结果自动训练、选种子、扩任务或扩大位移。
