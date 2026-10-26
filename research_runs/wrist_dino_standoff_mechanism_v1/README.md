# 通用5厘米间距：冻结Spatial模型的DINO机制验证

本轮只验证动态内部干预，不训练PNG，不自动安排后续训练，不报告CASR。手动提交后先按历史动作重放14条轨迹选样，再运行原生动作对照及最多84个128步短回合，最后CPU汇总。最多同时2张4090。

## 固定目标和数据

- 末端至交互位置的距离目标5cm；靠近干预沿当前真实三维目标—末端连线移动目标可视几何5cm，反向对照远离5cm。没有世界X轴平移、角度目标或读出器拟合。
- Spatial任务0黑碗、Object任务0汤罐、Goal任务0中层抽屉把手：40/41为Discovery，42/43为Validation。Long任务5书只用42/43，为本轮对象留出；不是全新历史独立测试。
- 每轨迹最多选较远和较近两个原生决策时刻：6–12cm中最接近8cm、1–4cm中最接近3cm，并列取早。只在首次目标接触/抓持之前且腕部中心裁剪视野可见时选；无有效状态保留缺口、不换轨迹、不按动作挑选。
- 自由物体用碰撞几何在物体坐标系中的AABB中心；把手用已核对木柜资产的三块把手碰撞box联合中心，不使用柜体/抽屉区域中心。标注来自资产几何，不等于精确抓取点。每个状态保存双视角叠加图。

## 因果检查与执行口径

只修改原生策略消费的腕部DINO block22空间输出。前视、SigLIP、CLS/register及原腕部block22前向结果均逐数组核对不变。对照包括clean、恒等替换、零位移原生RGB、靠近/远离的DINO替换、相应原生RGB，以及4组同实际BF16变化范数随机方向（seed42）。完整8×7输出存档；动作命令不能解释为实际米数。

每个状态分别执行clean/dino_near/dino_far，条件顺序固定种子打乱；每次重新reset并按原保存动作重放到选定时刻，严格比较物理、物体、相机、本体数组及相机采样指纹。RGB容差沿用已核验NRMSE≤1e-4并完整记录；没有放宽物理状态比较，也不通过set_state拼造相机样本。

渲染只改owned scene中的geom，异常后恢复。虚拟穿插检测在独立MjData中用有符号几何距离检查新出现/加深的穿插，绝不修改live sim/model；新增/加深阈值10微米为工程浮点容差。不可见、出裁剪视野、穿插或方向退化时回退clean表示，记录覆盖率，不切换实体。反事实画面并不保证物理可实现，不能把它当真实轨迹。

真实距离小于5cm时，严格按向量公式移动5cm会越过末端并改变目标方位，不等于把非负距离简单减去5cm；不偷偷截断位移。靠近末端时也可能与夹爪发生虚拟穿插，因此尤其报告近距离有效覆盖；覆盖消失不等于机制被否定。

短回合最多128控制步，8步动作块；逐控制步记录距离、末端、接触/抓持、相机可见性采样时刻及实际执行槽位。首次目标接触、抓持或任务结束立即停止。每次决策动态重算几何干预；它需要仿真真值，不能称静态共享PNG。GIF为每两动作块及停止时的原生观测抽样，不冒充逐帧录像。

报告较远接近、较近返回、4–6cm停留及同期移动量，保留停止/截断原因。主要配对比较使用共同执行长度，避免clean接触提前结束带来的时长混淆。区间任务→轨迹重采样，近/远起点在轨迹内聚合；不把动作槽位、重复帧或随机方向当独立轨迹。不按Validation调强度或选检查点。机制结果与PNG可实现性分开。

## 资源与监控

capture/mechanism每worker 1×4090、4CPU、12G，最多2并发；两个阶段串接，依赖等待不占GPU。capture只用GPU渲染，不加载VLA。CPU准备/汇总2CPU、4G。GPU部分预算20–40分钟尚未实测，不含实现及排队，单阶段Slurm上限45/60分钟是超时上限。每30秒心跳，完成首批状态后更新实际吞吐/ETA；记录进程RSS HWM、CUDA allocated/reserved、失败耗时。Slurm采样MaxRSS另报。

登录CPU直接导入已安装MuJoCo出现非法指令，故CPU preflight不导入MuJoCo或加载模型；几何纯函数、伪环境及推理钩子在CPU测试，真正渲染、原生双分支一致性和冻结权重检查在GPU任务启动时运行。失败即退出，不反复重试、换卡或放宽阈值。

## 手动入口

```bash
bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --preflight
bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --dry-run
bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --submit
bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --status --slurm
bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --watch --slurm --interval 30
```

同一提交命令（或`--resume`）按轨迹、状态、条件完成回执续跑；队列中已有本实验任务则不重复提交。仅显式`--submit/--resume`调用sbatch；worker不提交、不取消任务。提交响应不确定时停止，不自动重试。某capture分片失败，其后继机制worker在加载VLA前退出；report仍输出已完成配对，缺失不计为成功。

无需GPU可手动生成当时的部分报告：`bash research_runs/wrist_dino_standoff_mechanism_v1/run.sh --report`。

主报告`results/RESULTS.md`，距离曲线`results/curves/`，覆盖缺口`results/COVERAGE.json`，完整动作`OFFLINE.parquet`，逐步物理证据`STEPS.parquet`，等长配对`PAIRED.parquet`，区间`SUMMARY.csv`；每条短回合的GIF、初态数组及配对证据按完成回执关联。
