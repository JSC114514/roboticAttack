# Wrist-DINO 修复对抗训练与高维 RGB 去噪（2026-10-02）

状态：用户已授权更新方案、实现及 CPU/mock 预检，完成至手动提交命令。正式 GPU 作业只由用户提交，不自动 sbatch，不使用重置卡。此目录是本轮唯一入口，旧实验保持原位。

## 1. 已完成证据与本轮目标

上一轮 `wrist_dino_transfer_v1` 五组、1000/2000 检查点共 4000 攻击回合完整。
最佳开发点估计为 diverse_1000：Object/Spatial/Goal/Long CASR=100/23.556/58.889/81.000%；同轮原始 UADA=39.667/16.444/32.111/81.250%。diverse_2000=100/22.667/50.667/78.000%。单独动态评分没有稳定优于均匀采样。

当前 TV 为 `relu(TV(P)/0.4-1)^2`，TV<=0.4 时梯度为零；diverse 最终 TV=0.406122，噪点未完全解决。历史空间对应错位、Object-Decoy、局部 Jacobian、ILA/ILPD 与归档 CID-CF 均已有记录，不包装成新方法。

本轮主要目标：改善 Spatial 迁移，同时保留 Goal 效果并减少细碎噪点。Spatial 35–40%、Goal >=50% 是推进目标，不是预测或模型选择后补写的成功阈值。Spatial/Goal 40–49 初态是已使用的开发评估；不称完全盲的确认集。Long 完整报告、次要解释。

## 2. 五条件与训练来源

| 条件 | 方法名 | 内容 |
|---|---|---|
| A | diverse_reference_1000 | 已完成的 diverse_1000，冻结复用 |
| B | clear | 动态 DINO 采样+现有增强+持续去噪 |
| C | uniform_clear | 均匀采样+同样增强+持续去噪 |
| D | dino_causal_clear | B 的采样/增强/去噪，训练前向仅保留腕部 DINO 的受扰效应，无修复内循环 |
| E | dino_repair_clear | D+小幅 RGB 修复内循环，主候选 |

另复用同场 original_uada_1000 作外部攻击基线。主要配对比较：B-A（去噪代价）、B-C（采样作用）、D-B（DINO 因果训练前向）、E-D（修复内循环）、E-B（完整新方法）。同时报告全部新方法对原始 UADA 与 A 的差值，不把组合收益单独归因给某个组件。

不重跑已完成 A 或原始 UADA。B–E 的起点复用上一轮已经完成的 **250 轮随机初始化前缀**及其 Object 回放库；前缀的 12500 次更新计入总 1000 轮/50000 次补丁更新。恢复完整 patch、Adam、scheduler、RNG，不重置优化器。既不读取 1000/2000 的学习补丁作为初始化，也不是 1000+1000 续训。共同前缀和固定库是配对控制的一部分，不声称是新的独立训练重复。

前缀源配置：`configs/exploring_oft_dino/object_dino_transfer2000_v1.json`；前缀状态 `research_runs/wrist_dino_transfer_v1/training/warmup/resume.pt`；库 `research_runs/wrist_dino_transfer_v1/data/WINDOW_BANK.pt`。在新配置/manifest 中冻结这些文件与旧评估的 hash，禁止使用其他 warm start。

同一 seed42、Object 300 源记录、同一任务/阶段轮转、三决策片段、50 inner/outer、dim0 八步反向 q01/q99 边界 MSE。总训练到 1000；cosine **总周期仍为2000**，保持与 A 的1000检查点日程对齐，不在1000提前衰减至零。保存0/250/500/750/1000。动态条件在500后用各自当前RGB更新评分；1000已结束，不再无效刷新。C不刷新，均匀抽样。所有分数仍只来自Object。

## 3. 增强与持续去噪

保留 Exploring 派生现有 RGB 贴图与仿射，避免把渲染器替换混进本轮效应。前25 inner固定部署位置(26,137)，后25 inner随机合法腕部位置、原旋转/剪切、亮度/对比度0.8–1.2。Front干净，50x50 RGB、7500可训练参数，不使用低维网格或生成器。

新外观项是持续的 edge-preserving TV 与轻度高频残差：

`Lapp = lambda_tv * TV_charbonnier(P) + lambda_hf * mean((P-G_sigma(P))^2)`。

`TV_charbonnier`使用相邻差异的`sqrt(diff^2+1e-6)-1e-3`均值；Gaussian sigma=0.8、5x5、reflect padding。对参数P直接约束，部署保存的仍是P；不是训练后再模糊。

为避免原始损失量纲导致去噪压过动作梯度，B–E 在固定共同前缀、固定10条Object记录（每task取第一条）上统一校准两个lambda：分别使TV与HF梯度范数中位数为原正常UADA动作梯度中位数的0.15；之后冻结，所有新组相同。校准由第一个用户提交的4090准备作业执行，报告输入hash、全部10条记录、梯度范数及lambda；不能在Spatial/Goal结果后调权重。要求全部梯度有限且三个中位数非零；个别动作MSE梯度为零可以是已达边界的合法饱和，记录零梯度数量，不筛掉记录，也不以单条饱和中断训练。动作梯度中位数为零则不能完成校准，不能仅靠外观梯度假称可训练。

记录TV、HF能量、RGB标准差、饱和比例、主损失/外观梯度、实际更新量和PNG变化。外观由用户看保存图片判断；低HF或低TV不自动记为外观PASS，单色退化也不记为清晰成功。训练结束自动测试，不额外等待人工外观门。

## 4. DINO修复内循环与因果动作训练

方法依据：WHEN/UPA-RFAS第3.4节的两阶段鲁棒增强，https://arxiv.org/html/2511.21192v3#S3.SS4 。本轮是DINO修复目标与因果动作目标的适配，不是完整WHEN复现，也未实现其PAD/PSM/InfoNCE。

每个inner先且只先采样一次EOT，得到受扰腕部RGB及真实支持mask。D/E使用同一状态的干净腕部SigLIP特征，Front的两分支保持正常干净特征；受扰腕部DINO经过正常projector/LLM/action_head。仍最小化同一个UADA动作边界MSE，不新增外层DINO范数损失。

E额外先固定P，优化样本特有eta：初值0、2步sign-PGD、每步2/255、Linf<=4/255，只允许改动当前贴图支持之外的腕部RGB。目标最小化受扰DINO dense输出与干净DINO的逐token归一化平方差，排除覆盖token，避免将局部遮挡本身当全局表征破坏；覆盖规则保守排除任何覆盖token。不用物体mask/关系probe/目标suite信息。

eta优化调用OFT实际腕部DINO，模型权重与P均不更新。选择2步路径中（含eta=0）修复目标最小的eta，保证不会把一次失败更新冒充修复。然后detach eta，用同一EOT重新构造有P梯度的RGB，执行DINO因果动作前向更新P。RGB需clamp[0,1]，eta可见区域与幅度必须逐步检查。计算干净DINO参考时不施加patch颜色变化。

记录修复前后DINO误差、eta范数、实际有效修复比例、额外DINO调用/PGD次数与耗时。内外目标不同，是借鉴WHEN的两阶段方法，不声称严格同一目标的零和min-max。D作为必要对照区分修复内循环与训练前向本身。

测试没有eta、特征替换或其他训练hook；只使用正常OFT和最终量化RGB补丁。不能把训练时干净SigLIP替换记为攻击者部署权限。

## 5. 代码与正确性要求

新建Exploring派生训练文件与wrapper，原UADA.py保持只读与冻结SHA；旧配置、训练代码和完成产物不回写。新run/config/source签名与完成状态保护，按完整outer断点恢复，半轮失败回到上一完整outer；恢复需包含动态分数、Adam/调度器/RNG、累计补丁更新及修复成本。

GPU启动嵌入真实接口检查：直接DINO输入与模型消费的腕部输入对齐；干净SigLIP替换闭合；eta=0与D无修复前向/梯度相容；sigma不进入Front/patch支持；正常动作梯度有限且非零。使用有界数值容差，避免重复历史CUDA逐元素梯度断言问题。CPU只做mock和纯张量测试，不加载真实VLA。

## 6. 自动测试、统计与资源

流程：CPU输入与来源预检→用户提交的4090共同去噪校准→4090四新组训练（最多4卡，<=用户5卡上限）→CPU产物核验→3090四套件自动测试（最多8卡）→CPU统计。

新测试4方法x4suitex10任务x10初态=1600回合；复用A与original_uada_1000各400旧回合，合计2400。冻结clean有效分母Object99/Spatial90/Goal98/Long95，每条件382有效，总2292。每任务array顺序处理四suite，最多10对测试日志，支持已完成suite-task跳过。复用前核对patch SHA、runner/runtime源码、模型路径、clean资格、任务/初态/seed、贴图位置与量化。派生合并保留旧experiment_id/condition/source provenance，不改旧数据。

旧数据是历史同网格对照而非本轮并发对照。历史manifest可核对109份相关依赖，但有5个共用评估runtime文件没有当时的独立hash记录；本轮冻结其当前hash，明确保留历史指纹缺口，不声称能追溯验证遗漏文件的旧版本。旧raw/traces、检查点、配置和初态/seed逐条核对均可执行。

任务等权CASR；配对task→episode bootstrap10000次、95%区间。报告Spatial/Goal等权平均与worst-suite、正/负任务数、全部成本。只有一个训练seed；checkpoint/决策块不当独立训练重复。自动核验只代表工程完成，不代表机制/外观成功。

预计复用输入准备CPU数分钟、GPU校准5–15分钟、四组并行训练约3–5小时（E增加DINO内循环，实际启动后更新）、3090八卡测试约2–4小时、CPU统计数分钟；总约5–9小时另加排队。源自上一轮4090耗时，修复内循环尚无真实计时，不能保证完成时刻。

## 7. 结果导航与提交入口

新结果统一`research_runs/wrist_dino_repair_v1/`：`training/<method>/outer_1000.png/.pt`、`training/<method>/training.parquet`、`logs/`、`reports/`、`evaluation/four_suite/`。

新增文件：`VLAAttacker/white_patch/UADA_OFT_DINO_REPAIR.py`（Exploring派生训练副本）、`oft_dino_repair_objectives.py`（持续去噪/局部RGB修复）、独立wrapper、`object_dino_repair1000_v1.json`、`scripts/dino_repair_common.py`、独立正常OFT测试与统计脚本、102–106号Slurm脚本和两份CPU/mock测试。原始UADA.py及既有训练器未修改。

手动入口（先给用户预计耗时再给命令）：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_repair1000_v1.sh
```

仅预检、不查询队列也不提交：同一目录执行`bash submit_exploring_oft_dino_repair1000_v1.sh --preflight`。正式用户提交会串联校准→四组训练→自动核验→四套件测试→统计；逐次记录作业ID，前置提交失败时不继续制造无效依赖。详细实施预检见`reports/PREFLIGHT.md`。

当前无新GPU结果，不自动提交。提交前给用户时间预估和完整`cd ...; bash ...`命令。
