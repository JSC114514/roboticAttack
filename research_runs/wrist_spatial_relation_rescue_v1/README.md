# 空间关系补丁：原生动作恢复诊断

独立目录。历史`wrist_spatial_approach_pilot_v1`和`wrist_spatial_failure_audit_v1`全部只读，不重新训练、不改变四套件混合训练池，也不把干预结果写成PNG攻击效果。采用冻结Goal checkpoint、原主结果`radial_reverse_s42/checkpoint_004000.png`，50×50腕部位置[26,137]，不加EOT。

## 固定回合与三种操作

原18个clean成功起点全部保留，原始种子/初态/等待步数和300步上限保持一致。先重放18个已有补丁回合、补录控制器与全部接触记录；只执行保存动作，不调用策略。全部回放通过后，加载一次冻结模型，进行18×3=54个新恢复闭环。每起点三种条件按seed42预先随机排列，续跑沿用原顺序。

每个决策时刻，分别对干净图像和实际PNG图像做原生前向，得到各8×7原生反归一化动作。原生夹爪阈值/反转流程原样执行。XYZ按原生控制器输入限幅[-1,1]及0.05米比例处理，不混用归一化动作头或把命令写成实际位移。

|条件|处理|
|---|---|
|restore_radial|该决策起点的实体中心→EEF关系取朝目标方向u，8槽共用。每槽在有效XYZ盒内求离补丁命令最近、且u投影等于clean命令的点；80次二分求解。记录径向误差及限幅引入的正交变化。方向距离≤1e-8米时保持补丁动作并记录。|
|restore_xyz|XYZ换为有效clean命令，姿态/夹爪保留补丁命令。|
|restore_pose_grip|XYZ保留补丁命令，姿态/夹爪换为clean命令。|

以上是有干净图像、仿真信息的特权诊断；没有声称恢复真实空间状态。对照为当前状态下的clean动作，不是另一条clean轨迹的动作。干预导致后续状态不同，三者不是互斥归因；恢复失败不能直接证明空间机制不存在。

## 日志、视频和工程检查

- 复用原生相机回调内的分割/零渲染核验；异常恢复相机钩子。首次动作前可见性标缺失，不强制刷新原生传感器。
- 包装原有`sim.step`，只在实际物理子步完成后读取：OSC目标位置/姿态、实际跟踪误差、原始/限幅力矩、执行器触边、关节剩余范围，以及MuJoCo接触力。每个控制步按接触对聚合峰值/时间/次数，不保存巨大逐子步日志，不额外step或forward；异常恢复方法。
- 接触包含全部实体对，并标注是否涉及机器人。全场接触力与机器人接触力分别记录，不能把桌面物体的静态接触误算为机械臂碰撞。
- 初始物理数组、本体状态严格一致，物理/相机/时钟指纹与旧配对证据相同；回放逐步坐标误差≤1e-9米、接触/抓持/结果及历史关键帧RGB核验保持原规则(NRMSE≤1e-4)。原生首次动作块与历史clean/PNG各自动作最大绝对误差≤1e-5。A求解的径向误差≤1e-9米；仍经过官方float32处理，并逐槽核验实际命令误差不超过预定IEEE舍入界`0.05*eps(float32)*sum(abs(u))+1e-9`米，误差与界全部保留。模型始终冻结/eval。失败立即释放GPU，不自动重试、放宽阈值、换卡或补样本。
- 视频全部双视角，实际PNG保留，原生共同控制时钟，提前结束的画面明确标记并冻结。配对页面包含clean、原补丁回放、三种恢复及距离/跟踪误差/机器人接触力/可见性曲线。
- 每30秒记录心跳、阶段、回合/步数、吞吐、剩余时间、RSS和CUDA allocated/reserved峰值。EGL全卡显存另外采样，不能当作单进程显存。接触峰值不相加解释为持续力，实体中心距离不是抓取/表面误差。

## 资源与提交

最多1张4090，GPU worker 4CPU/12G；对照旧原生评测实际RSS约5.6GiB、CUDA allocated约14.7GiB。此流程新增子步日志及视频，暂估GPU30–60分钟，CPU报告2CPU/4G再用5–15分钟，不含实现及排队。首2个回放、首3个恢复闭环输出实测吞吐和更新估时；GPU申请2小时上限，按实际完成释放，不因依赖等待占GPU。CPU报告使用独立非阻塞锁，不重复旧汇总锁等待问题。

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_spatial_relation_rescue_v1/run.sh --preflight
bash research_runs/wrist_spatial_relation_rescue_v1/run.sh --submit
bash research_runs/wrist_spatial_relation_rescue_v1/run.sh --status --slurm
bash research_runs/wrist_spatial_relation_rescue_v1/run.sh --watch --slurm
# 中断后只补缺失回合和合成：
bash research_runs/wrist_spatial_relation_rescue_v1/run.sh --resume
```

`--dry-run`只准备来源和打印命令，不提交。提交先查队列，无法检查去重则拒绝提交；不取消任何任务。代码与输入哈希不符则拒绝续跑。回合全部完成但末尾封存中断时，CPU补封存，不为此重申请GPU。GPU后自动依赖CPU报告，GPU失败时仍输出明确标记缺失的部分报告，不把缺失计为成功。

## 结果口径与扩大准备

`results/RESULTS.md`、`RESTORATION.csv`报告每组完成数、成功数、挽救原13个失败数、损害原5个成功数，以及按任务→起点重采样的配对成功率差异区间。`CONTROL_DIAGNOSTICS.csv`和`INDEX.html`提供全部逐任务诊断、视频和曲线。重复动作槽位和视频帧不当独立样本；不挑选原本失败、始终可见或表现最好的子集。原18个起点属于开发诊断，不包装为独立确认。

`data/EXPANSION_PREPARED.json`仅准备Goal任务0–9、初态2–5的40个起点，clean、两个关系补丁、两个初始补丁、两个同池远边界、UADA共8条件，最多320个原生PNG回合；核验初态与PNG来源。它们未用于当前方法协议的拟合/验证/评测，不宣称历史从未见。无扩大提交入口或预排任务，先讨论本轮完整配对证据；不设统一高分门槛，不把恢复成功当作新补丁增强，不自动扩展到其他checkpoint或真实机械臂。
