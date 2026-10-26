# DINO相对位置读出目标：上午小试

只检验物体—末端XYZ相对位置。Spatial冻结，block22空间表示，固定50×50腕部PNG、xy=[26,137]、无EOT。历史实验只读。新目录不继承旧实验的科学结论。

## 固定协议

缓存来自`wrist_dino_geometry_rgb2_v1_fix1`的Spatial子集：任务0/1/2/4/6/7，fit初态40/41、verify42/43，各60观测/12轨迹/五进度点。几何位移逐条读取`geometry.deltas.plus`；不再乘2、不重新渲染。无目标记录保留，候选次数与实际更新次数分报。

冻结读出复用`wrist_dino_geometry_pilot_v1/readouts/b22.npz`的前三列XYZ；原拟合仅使用atlas Discovery的600观测/120轨迹。新verify与其episode ID、任务/重放种子和实际初态数组均检查不重叠；同任务邻近初态并非独立场景。旧4×4池化和float16缓存舍入在可微实现中复现，梯度不中断。方向检查也记录本缓存上的clean读出误差和渲染位移读出误差，不把域外读出偏差当作真实物理变化。

方向检查固定取fit每轨迹frame_rank=1，共12条，不按响应选样。将物理单位读出写成q(h)=hW+b，在原始池化特征的欧氏范数下构造`delta_h=delta_r @ pinv(W)`。4×4变化均匀展开到16×16空间token，这是给定池化变化的最小范数展开；CLS/register及其他分支保持原值。正负关系干预、4组seed42随机零空间方向均执行完整原生动作头。随机方向投影到W零空间，并在展开特征范数上配对。数学约束先以float64检查；模型实际dtype量化后的读出泄漏、范数与目标误差另报，不冒称量化后精确为零。

每观测以原渲染DINO正负反事实的首槽平移动作差为参考方向。关系分数是正负关系干预的首槽动作差在该单位方向上的投影；随机分数是4组对应投影的平均。只在有效观测上`median(relation)>0`且`median(relation-random)>0`时训练；这只是预定探索性预算门槛，无统计显著性或因果唯一性声明。渲染无目标、位移为零、参考动作差为零均保留并标注不适用；没有适用观测则不训练。

通过后训练唯一seed42补丁2000次候选更新，Adam lr=.002，warmup20后cosine，沿用MATCH的配对初始PNG、每轮seed42+轮次的固定shuffle。损失是`sum((q(Fpng)-q(Fclean)-actual_delta_xyz)^2)/.03^2`；无动作/整块特征/EOT附加损失。只经腕部DINO回传。每100步保存补丁、Adam、Python/NumPy/Torch/CUDA随机状态及计数，按候选步数续跑；只用最终2000步PNG。

verify完整原生双分支PNG前向，比较clean、initial、UADA、relation，并对relation恢复干净DINO。报告厘米误差、首槽及8槽动作、实际DINO/SigLIP不变检查和恢复诊断。闭环任务0/1/2/6/7、初态44/45共10起点；220步上限、8步执行。clean/initial/UADA仅在checkpoint代码、原初态文件、种子、PNG和四个START数组证据匹配时复用；不匹配则该起点全部四条件重新配对，不放宽重置检查。

## 入口和资源

```bash
bash research_runs/wrist_dino_relation_readout_pilot_v1/run.sh --preflight
bash research_runs/wrist_dino_relation_readout_pilot_v1/run.sh --dry-run
bash research_runs/wrist_dino_relation_readout_pilot_v1/run.sh --submit
bash research_runs/wrist_dino_relation_readout_pilot_v1/run.sh --status --slurm
bash research_runs/wrist_dino_relation_readout_pilot_v1/run.sh --watch --slurm
```

只有用户手动调用`--submit`会写Slurm，重复调用按已封存阶段续跑并拒绝重复活动链；不会取消任务。gate12G→train16G→evaluate数组2×12G→CPU report(2CPU/4G)。每GPU worker 1×4090、4CPU，最大并发2张。依赖等待不占GPU；科学门槛未通过时后继worker在导入CUDA/加载模型前记录SKIPPED退出，只有短暂调度启动成本。report使用afterany，即使前级工程失败也生成部分报告。

每30秒心跳、RSS HWM、CUDA allocated/reserved峰值、设备、吞吐、ETA及失败成本；预估未实测，不保证11点完成。10:45 CPU中期汇总独立运行，不持有GPU或等待评测锁；最终汇总覆盖更新。CPU检查通过不替代Slurm内GPU接口检查。只改读出数值不称空间控制成功，已查看起点不称独立确认；不会自动增加方向、预算或训练规模。
