# 最后一波：DINO几何动作教师

固定Spatial checkpoint、50×50腕部RGB PNG、位置[26,137]、无EOT。复用RGB2 fix1已核验的240 fit/240 verify缓存，24任务、每轨迹5进度点。fit初态40/41，verify42/43；不按动作响应选图。不扩建反事实目标缓存。

教师是干净观测仅替换腕部DINO为真实重新渲染的2倍正方向几何表示后产生的归一化动作，其他输入保持干净。优化全部8槽平移MSE；不放大动作、不匹配完整特征、不加几何辅助损失。实际PNG前向同时通过DINO和SigLIP，教师法仅从腕部DINO反传；基线允许两分支梯度。部署没有内部替换。

四组：geometry_teacher/clean_far_boundary × seed42/43；主组固定geometry_teacher_s42。同种子初始PNG及顺序完全相同。clean_far_boundary用clean预测而非专家动作构造最远有效端点：前6维[-1,1]，夹爪[0,1]，等距选上界；不是原专家参考UADA，旧Spatial UADA500另作真实PNG参照。该方案是目标构造/梯度路径实验，不预宣称独立方法创新或CASR提升。

每组24,000候选更新，batch1；Adam lr0.002，warmup20次候选更新，随后cosine。无有效几何目标记录四组同规则跳过，保留计数。无效定义为渲染no_target或原投影实现的特征方向总能量≤1e-12；预审为fit8/240、verify9/240，前者包含2条渲染标志正常但特征方向为零的记录。每1000次保存补丁/Adam/RNG/调度时钟，启动时验证完整来源及接口。实际动作头的非线性前向及分段VJP；语言层/视觉层以non-reentrant checkpoint包装，仍为eval。前视不接收补丁梯度；被测模型权重冻结。STE只用于PNG量化/预处理的反向，前向与部署像素相同。

北京时间2026-10-20 05:30训练停止，最后完整更新另保存。每个种子选两组共同完成的最大1000步边界，不按验证/闭环选；一方工程失败仍保留成功方末次checkpoint，但不宣称配齐对照。首个启动gate失败立即停该worker，没有自动重试/换卡/放宽阈值。未完成预算明确标注。

闭环固定Spatial模型评四suite全部40任务初态0/1，优先Spatial，8槽实际执行，各环境上限Spatial220/Object280/Goal300/Long520。四训练PNG+两初始PNG+clean+原Spatial UADA500，最多640回合；重新评测基线，逐起点验证相同seed、物理状态、本体、相机和observable时钟，RGB差异完整记录。未证实相同条件的历史结果不复用。相同状态的轻微渲染差异沿用已审计的完整状态指纹证明，不提高旧RGB阈值。失败/缺失不计攻击成功。离线240验证观测完整原生前向及恢复干净分支诊断。旧场景有历史开发接触，不称完全独立确认或任意环境通用。

每30秒心跳、RSS HWM、CUDA allocated/reserved、设备名、失败成本；每个闭环起点完成更新阶段报告，07:50独立CPU作业再汇总，所有评测结束自动最终汇总。按task→start/trajectory重采样，不以动作槽位为独立样本。CASR和超出各自随机初始补丁的失败分别报告。预估尚无新反传4090吞吐实测，不能承诺08:00全部完成；未完报告仅用完整配对起点。

资源分波次：cache≤2×(4090/4CPU/12G) → train≤4×(4090/4CPU/16G) → CPU选预算 → eval≤4×(4090/4CPU/12G) → CPU汇总(2CPU/4G)。全流程GPU≤4；等待依赖不占GPU。准备只读分片，不加载2.82GiB原大bank。--submit/--resume在锁和Slurm去重后仅提交未完成阶段，不取消任何任务；不自动提交，用户需手动运行。SELECT封存后续跑仅补完该次固定检查点的评测，不再改变训练或按已有评测结果重选检查点。

```bash
bash research_runs/wrist_dino_geometry_action_teacher_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_action_teacher_v1/run.sh --dry-run
bash research_runs/wrist_dino_geometry_action_teacher_v1/run.sh --submit
bash research_runs/wrist_dino_geometry_action_teacher_v1/run.sh --watch --slurm --interval 30
bash research_runs/wrist_dino_geometry_action_teacher_v1/run.sh --resume
```

输出：results/RESULTS.md、CASR.csv、OFFLINE_SUMMARY.csv、STATUS.json；training按候选分目录，reports保存启动检查、来源、Slurm提交回执、心跳与成本。CPU测试不能证明GPU接口通过或补丁有效；GPU接口检查随正式长任务启动运行。
