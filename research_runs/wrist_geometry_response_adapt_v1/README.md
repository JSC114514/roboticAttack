# 腕部视觉鲁棒适配

目标：相机位置变化时维持正确动作，同时保留机器人面对真实任务变化时的控制能力。冻结联合 OpenVLA-OFT，只训练腕部 DINO blocks 18–22 的低秩参数；前视、SigLIP、语言模型和原生动作头保持冻结。

## 训练使用什么数据

两类数据各有用途，不把它们冒充逐帧对齐的数据：

- **专家演示**：本地四套件 RLDS 的真实演示动作，保护正确操作。每个观测都有未来 8 条动作；两段观测相隔 8 条保留记录，用于约束动作响应变化。
- **同状态相机对**：从已有保存状态、种子和动作重放，只移动腕部相机，保持物理状态、前视、本体和语言一致。训练移动范围是相机局部 x/z 方向的 ±5、±10 毫米。

正式训练任务采用原有每套件 6 个任务，共 24 个任务；每套件另外 2 个任务调参、2 个任务不参加本次适配。联合基础模型原本已见过四套件，后两类任务表示“本次适配未使用”，不是基础模型从未见过。

| 阶段 | 专家训练 | 相机训练状态 | 闭环评测 |
|---|---:|---:|---:|
| 小规模 pilot | 8 个任务 ×8 轨迹 ×8 对 =512 对 | 80 个状态 | 8 个调参任务 ×4 初态 ×5 条件 =160 回合/模型 |
| 正式 main | 24 个任务 ×20 轨迹 ×8 对 =3,840 对 | 720 个状态 | 40 个任务 ×3 初态 ×5 条件 =600 回合/模型 |

具体可用数由准备报告核实；缺少轨迹或有效未来动作会明确报告，不能复制样本补齐。专家验证轨迹与训练分开；相机验证初态为 42/43。新实验不修改历史结果或基础模型。

## 比较哪些方法

小规模固定 seed42，每配置 1,000 次更新：

1. 仅专家行为克隆。
2. 专家行为克隆加常规 RGB 增强。
3. 专家行为克隆加同状态动作一致性，权重 0.1、1、10。
4. 约束适配：最小化同状态换相机引起的动作变化，同时限制专家动作误差和真实动作变化误差；允许误差增长比例 0、10%、25%。

共 8 个训练配置，加原模型基线。所有配置使用相同腕部适配容量、初始化、专家顺序及更新次数。主方法的约束权重根据训练中实际违反程度更新；相机一致性、专家误差和响应误差都通过完整原生动作头计算。

正式训练采用小规模为每类方法选出的参数，各运行 seeds 42/43/44、5,000 次更新。选择分数为“原生成功率与四种相机扰动平均成功率的调和平均”；相同分数优先原生成功率，再比较实际 GPU 时间。最终检查点固定，不以测试集选 checkpoint。

## 手动入口

在项目根目录运行。以下入口不会隐式提交另一个阶段：

```bash
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --check
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --dry-run --stage pilot
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --submit --stage pilot
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --status --stage pilot --slurm
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --watch --stage pilot --slurm
```

中断后仅补齐未完成工作：

```bash
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --resume --stage pilot
```

2026-10-24 修复：相机准备曾因共享渲染器默认要求 3090，而实际分配 4090 退出。
本轮 `run.sh` 现明确设置 `OFT_STAGE_GPU=4090`，并按 Slurm 分配的单卡编号设置 EGL。
这项启动修复不改变模型、训练数据和科学配置，已完成的专家数据通过原收据核验后复用。
失败日志保留；使用上述 `--resume` 重新提交未完成阶段。

同日续跑在原生图像预处理导入时遇到 Diffusers/PEFT 版本检查错误。
`run.sh` 已沿用成功评测入口的 `_CHECK_PEFT=0` 设置：本实验使用 OFT L1 动作头和自有低秩模块，不调用 Diffusers 的 PEFT 适配功能。保留现有依赖版本，无需升级环境。

续跑去重已修复：本阶段提交记录中的旧 CPU 汇总任务单独排队时，不阻止补齐失败阶段；GPU 任务及正在运行的汇总仍阻止重复提交。旧任务不自动取消，下一份提交记录保留其编号。

相机配对使用本次原生重放画面及其同状态相机变换。旧 RGB（部分由 A800 生成）仅作为审计参照，记录逐视角 NRMSE、平均/最大灰度差和变化像素数，不要求跨显卡逐字节相等。重放物理/相机数组、本体、传感器时钟和采样时间仍严格匹配；零相机位移的渲染误差仍须不超过 `1e-4`。炉灶等动态 site 颜色在原生采样回调中一并保存，随后只恢复到渲染副本。

小规模结果位于 `results/pilot/RESULTS.md` 和 `SELECTION.json`。结果讨论后，正式阶段由用户另行手动提交：

```bash
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --submit --stage main
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --status --stage main --slurm
```

`--prepare` 是显式的本地 CPU 专家数据准备，会读取 RLDS；通常直接提交 pilot，让 CPU 任务在 Slurm 中完成准备即可。`--report --stage pilot` 可随时汇总已有完整回合，不需要 GPU。`--check` 运行静态、CPU 契约检查，不加载大模型、不运行仿真。

## 调度、资源与耗时

调度顺序为：CPU 专家准备 → GPU 相机配对准备 → 原模型闭环 → 共享基准误差校准 → 适配训练 → 适配模型闭环 → CPU 汇总。训练和评测不会同时占用两套 GPU 配额。依赖等待不分配 GPU；pilot 最大 2 张 4090，main 最大 4 张。

先用原模型回答实验动机：在起点目标可见且相机变化后仍可见的条件中，是否至少出现一次“原生成功、改变相机后失败”。没有这项现象时，入口停在原模型报告并释放资源，不继续训练；依据与起点清单写入 `results/pilot/MOTIVATION.json`。全部回合仍在结果表中。

| 工作 | 每 worker |
|---|---|
| 专家准备、汇总 | 2 CPU、4G，无 GPU |
| 相机准备、误差校准、评测 | 1 张 4090、4 CPU、12G |
| 训练 | 1 张 4090、4 CPU、16G |

pilot 暂计训练 8–20 GPU 小时、评测 6–18 GPU 小时；两卡串接约 7–19 小时，加准备和排队。新流程首批 100 次更新和完整回合后，按实测吞吐更新估时。正式实验连同消融暂留 80–180 GPU 小时，讨论 pilot 后再决定投入。

每 30 秒更新 `reports/PROGRESS/`，保留 RSS、CUDA allocated/reserved 峰值、耗时、失败成本、梯度及约束状态。训练保留优化器、调度、随机状态，每 250 次更新保存检查点。提交前查询本实验队列，已有活动任务则不重复提交。源代码或配置改变时，旧收据不能继续当作当前版本已完成。

## 结果和视频

- `results/<stage>/RESULTS.md`：成功率、完成数量、缺失和参数选择。
- `SUMMARY.json`：逐配置和配对区间；main 另含跨种子汇总。
- `EPISODES.csv`、`TASK.csv`、`SUITE.csv`：逐回合、任务和套件。
- 每套件等权，重采样以任务和初态为单位；同一起点的相机条件始终配对。缺失回合不计成功或失败。
- 视频集中在项目 `research_videos/wrist_geometry_response_adapt_v1_singleview/`，按方法、相机视角分类。每回合前视、腕部分开，保留 MP4 与 IDE 用 WebM，不拼接。

首轮回答的是正常视觉几何变化下的闭环鲁棒性。已有强补丁和重新优化补丁的压力测试将在主效果成立后单独配置，不混入本轮调参分数。

## 2026-10-25 监控退出修复与评测续跑

八组 pilot 均已完成 1,000 次更新并保存最终适配器。训练指标中包含 RSS/CUDA 字段，旧监控在合并同名资源字段时抛出 TypeError，导致 Slurm 将训练标为失败，依赖评测未启动。现已修复为字典覆盖合并，增加正常退出及异常保留测试；不改变训练目标、参数和模型。

完成资产按监控修复前后的源码核验兼容迁移：原始收据及适配器元数据保存在 `reports/repairs/20261025_monitor_exit/original/`，参数张量与优化器状态不变，207 份完成收据通过核验。续跑计划仅包含 evaluate 和 report，复用专家数据、相机缓存、原模型 160 回合与八组最终适配器。

```bash
bash research_runs/wrist_geometry_response_adapt_v1/run.sh --resume --stage pilot --workers 2
```

最多两张 4090，每 worker 4 CPU、12G；本次不再训练。按已完成 base 评测吞吐，剩余 1,280 个闭环回合及离线指标预留约 4–6 小时，不含排队和 CPU 视频导出。
