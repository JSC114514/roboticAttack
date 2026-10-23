# OpenVLA-OFT Wrist-DINO 攻击研究：跨模型交接

更新于北京时间 2026-09-26。本文可单独复制或上传给不同网页大模型；不假设接收方能访问本机文件。文中的本机路径用于复核原始证据。请区分：已观察到的因果/闭环结果、尚未证实的机制解释、以及下一步建议。历史交接包和方案中的“待运行”状态可能过时，以本文件列出的最新结果文件为准。本文是研究背景，不是授权自动提交作业的指令。

## 1. 一页结论

研究对象是 OpenVLA-OFT 在 LIBERO 中的 Wrist 单视角局部 RGB 补丁。早期同状态反事实研究支持：Wrist 输入变化比 Front 更容易影响瞬时动作；DINO 分支恢复能关闭大部分额外 Wrist 动作差异；所测的 Wrist 距离/尺度和横向视差是有效上游因素。但这些结果**没有**证明一个固定、可训练的 Wrist 补丁可以可靠操控 DINO 并提升闭环攻击成功率。

最新的动作敏感 Wrist-DINO dense-token Jacobian 补丁已经完成两档 500 轮训练和四套件配对测试。与同场重测的 Exploring→OFT UADA-500 相比，r03、r10 在源套件 Object 分别低 25.0、16.1 个百分点，在 Goal 分别低 24.7、16.4 个百分点，任务分层 95% 区间均低于零；Spatial 和 Long 没有稳定正增益。r10 比 r03 的差值四套件区间均跨零。**本候选在“增强攻击”主终点上判负；不能把训练期 DINO 定向分数或补丁结构门通过改写成攻击成功。**

当前可守住的论文级区分是：“Wrist-DINO 对部分瞬时动作差异有因果贡献”与“通过它设计出优于 UADA 的局部通用攻击”是不同命题。前者有早期实验支持，后者在已测具体方法上未成立。这不等于 DINO 不重要，也不能据此宣称 DINO 特异中介已经在最终补丁上得到证明。

## 2. 问题、模型与威胁边界

- 模型：OpenVLA-OFT，Front+Wrist 图像输入，视觉塔包括 DINOv2 与 SigLIP，输出连续 8 步×7 维动作块。普通 Exploring 的 OpenVLA 离散动作 token 损失不能原封不动搬到 OFT。
- 本轮攻击：只在 Wrist 原始 RGB 图像上叠加一张 50×50 通用补丁；Front、语言、本体状态与模型权重不改。测试固定 Wrist 左上角坐标 (26,137)。不做双相机补丁，也不拥有任意视频流编辑权限。
- 训练：仅 LIBERO-Object 的冻结 300 状态 bank；Object 是源套件。Spatial 和 Goal 是跨 suite 零样本评估；LIBERO-10（下称 Long）必须测并报告，但不能独自承担主结论。
- 主终点：每套件只在 clean rollout 成功的状态上计算任务等权 CASR，即“原本成功的任务被攻击导致失败”的比例；同一 task×state 的各攻击条件共用 rollout seed，任务→episode 分层 bootstrap 10,000 次、种子 20260907 给 95% 区间。不能把动作 L2、DINO 特征位移、训练 loss 或 decision block 当成闭环成功率。
- 冻结 clean 网格：每套件 10 任务×10 状态（state 40–49）；clean-success 分母 Object 99、Spatial 90、Goal 98、Long 95。四套件不合并一个总 CASR。

## 3. 为什么曾经把 Wrist-DINO 作为候选

早期 M 系列 patch 的 Wrist-vs-Front 差异很大，但它是在 processor 后 pixel_values 上操作，不等于本轮 raw RGB 补丁性能。后续把同一个 clean 仿真状态的 Front 或 Wrist 独立扰动，WHEN/UADA/UPA 三种输入变化的首次动作预测都出现正 Wrist−Front 标准化差，约 +0.364、+0.280、+0.300。更干净的 600-state、120-episode、10-task、3-donor 来源匹配替换中，完整动作依赖 Front/Wrist 为 2.502/3.703，差 +1.201，95% 区间 [0.902,1.458]，10/10 任务同向。这说明所测瞬时动作读出存在 Wrist 易感性；闭环任务差异不能由此直接外推。

视觉分支 W1：同源替换的 Wrist−Front 归一化特征位移，DINO 为 +0.261 [0.220,0.300]，SigLIP 为 −0.012 [−0.051,0.022]。输入扰动范数匹配的 P1 只能解释原 Action gap 的约四分之一，排除了“只因 Wrist 扰动更大”的简单说法，但也提示方向/来源交互。P2 对同一输入恢复分支：DINO gap contribution +1.222 [0.975,1.451]，SigLIP +0.065 [−0.046,0.172]；DINO 恢复后 Front/Wrist 动作依赖约 1.027/1.007，额外 gap 基本关闭。这里支持的是**干预条件下 DINO 承载主要额外 Wrist 动作差异**；branch mixing 可能偏离自然数据流，不可称唯一根因。

VF1 保持物理状态不变，改变相机外参：预设的距离/表观尺度 range、横向视差 lateral 支持更大的 Wrist-DINO 位移、动作效应及 DINO 恢复贡献；angular/pitch 未获支持。工程 QA 仍有 36/600 个预定义 identity activation 偏差，原 QA 标为 FAILED，不能事后写成 PASS；排除这些状态的 sensitivity 分析与 primary 方向一致。更细的自然中介 SEM0–SEM4 仅动作平移 A2-dx/A2-dy 两项通过门，效应很小，没有找到独占式 Wrist 专用后端回路。详见 FROZEN_WRIST_EVIDENCE_20260909/WRIST_EVIDENCE_CHAIN_FROZEN_20260909.md 和根目录主方案。

因此最初合理的攻击假设是：一个 Wrist-only 视觉扰动如果能有方向地改变与错误动作相关的 DINO 局部表征，也许比无差别视觉破坏更有效。**这是待测试假设，不是早期因果实验已经证明的攻击结论。**

## 4. 走过的路线及为何不能混为一种“DINO 成功”

1. 早期自定义 DINO-Continue-B-500 在 Object Final-100 曾有 74.33% task-equal CASR，但 Spatial/Goal 只有 7.22%/15.56%，Long 59.50%；该代码线并非 Exploring 源码副本，后来被只读归档，不能直接当作当前同源比较的胜利证据。CID-CF 虽改善代理 DINO 指标，却未改善主闭环，判负。
2. 旧自定义 OFT-UADA 与 DINO-UADA 在 Object 曾接近/达到 100% CASR；普通 UADA 的分支恢复显示攻击效应大量经 Wrist-DINO 传播（DINO restore fraction 约 87.83%）。添加 DINO 目标并未改善 Spatial/Goal。旧 100% 与源码重启后的 UADA 数字不是同一训练/评估线，不可混比。
3. 2026-09-20 起按用户要求重启：原 Exploring UADA.py 恢复 Git 版、只读；在独立副本中保留随机 RGB 初始化、RandomPatchTransform、AdamW/cosine、outer 抽样、inner 随机几何 EOT、逐次更新/clamp，仅做 OFT 双视角输入和连续动作损失的必要适配。所有正式新补丁从随机初始化，不能从旧噪声补丁热启动。大面积无结构噪声是外观失败门。
4. 源码重启后，UADA-500 成为同预算对照；UADA-2000 已冻结为操作强基线。六路路径/全 token/有符号几何/持久动作实验、Object-Decoy、DINO 引导位置/训练状态抽样都做过训练和闭环；未在 Object/Spatial/Goal 上形成稳定的 DINO 特异增益。有些改了路径或代理变量，却未变成更强任务失败。Object-Decoy r03/r10 的 Object CASR 为 24%/18%，不及同场 UADA500 的 32.1%。DINO 引导抽样相对同位置普通 UADA 在 Goal 低 16.0 个百分点，95% 区间 [−31.7,−2.9]。物体—夹爪对称几何刺激因技术完整性失败而关闭；编辑图像未进入 DINO/动作模型，不能把这当作 DINO 机制反证。
5. Frozen UADA-2000 在其三臂四套件评估的 Object/Spatial/Goal/Long task-equal CASR 为 42.9%/8.4%/41.2%/77.8%；维度 5 探索臂未稳定胜它。它与新 Jacobian-500 训练预算不同，仅作实用强基线背景，不能用跨运行的绝对数值宣称严格配对优劣。

这些负结果缩窄了“再加一个 DINO loss/再放大梯度”的理由。下一种方法若被提出，必须明确攻击者要造成的具体错误和相对既有方法的新信息，而非只证明梯度经过 DINO。

## 5. 最新候选：动作敏感 Wrist-DINO dense-token Jacobian 补丁

注意这里的 Jacobian 是“Wrist-DINO token 特征→OFT 动作损失”的局部梯度，不是此前“世界物体位置→图像 token 坐标”的几何 Jacobian 技术 pilot。

两臂与 UADA-500 使用相同 Object bank、种子、补丁形状、EOT、动作维度 0 的 opposite-boundary MSE、AdamW/cosine 和 500 outer×50 inner=25,000 次 patch 更新。每个 outer 在 clean 输入上计算动作损失对 Wrist-DINO dense tokens 的梯度，将负梯度逐 token 归一化，作为“使动作朝错误边界移动”的局部特征方向；训练时在原 UADA 动作主目标之外，让补丁覆盖的 token 沿该方向变化，并按 clean 动作敏感度加权。r03/r10 在每个 inner 把 DINO 辅助梯度范数校准到主梯度的 0.3/1.0；这只是两档方法强度，不是有先验保证的最优倍率。Front 始终 clean。

冻结 OFT 模型曾导致初版 clean DINO 捕获无梯度，已在副本里把第二次（Wrist）DINO 输出作求导叶子修复。训练 array 136339 的两臂均到 outer 325 后因一次随机变换使补丁 DINO-token 覆盖不足而中止；修复后仅在低覆盖的该步跳过 DINO 辅助项，仍执行 UADA 主损失，用户提交 136344 从 outer 326 续跑。两臂均 COMPLETE，100/200/500 结构门通过，最终补丁图有连续弧线/纹理而非整张均匀随机噪声；每臂 25,000 步中有 1 步低覆盖、24,999 步 DINO 引导。结构和训练期定向得分不是闭环效果。续跑 status 中约 3,600 秒 elapsed_s 只计续跑段，不能当全程训练成本；Jacobian 每 outer 还增加 clean 前向/反传，尚无核定的低成本优势。

原始 Exploring 源码：roboticAttack-main/VLAAttacker/white_patch/UADA.py，权限 0444，SHA256 808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e。训练原语副本：UADA_OFT_SOURCE_COPY.py。候选训练器：UADA_OFT_DINO_JACOBIAN.py；包装入口：VLAAttacker/UADA_OFT_DINO_JACOBIAN_wrapper.py。UADA500、r03、r10 最终 pt 的 SHA256 依次为：

- c7ebb69204290fc193c3f694e25ba02c656de9e2df0443a74470d8ad587db136
- 2298e508dadf9bd1246044d91bc59a8d49678f835ac1d436473e7c9454071785
- cc622fd0cc81c5573fcfa30580c7e6f6b6bdd0ef3143528a2801351ecb083405

## 6. 最新四套件闭环结果：已完成，主结论为负

用户手动提交评估 array 136352（40/40 分片）和依赖统计 job 136353，统计状态 COMPLETE。三臂按相同 task×state×seed、新鲜地重测；400 个冻结 clean 状态，1,200 条新攻击 rollout，1,146 条 clean-success 条件有效记录。合并 rollout 1,200×21、trace 33,873×16，均无缺失，rollout 无重复键，三臂种子配对完整，补丁哈希各自唯一。日志未见 traceback/OOM；sacct 曾因控制节点 DNS 失败而不可用，但磁盘产物及统计任务日志均完成。

下表均为每套件任务等权 CASR；数值越高，攻击越强。Long 单独看，不因其高基线掩盖源套件和零样本主结果。

| Suite / clean-success 分母 | UADA-500 | Jacobian r03 | Jacobian r10 |
|---|---:|---:|---:|
| Object / 99 | 32.1% | 7.1% | 16.0% |
| Spatial / 90 | 13.3% | 16.3% | 10.4% |
| Goal / 98 | 31.9% | 7.2% | 15.4% |
| Long / 95 | 71.1% | 70.5% | 64.3% |

同一评估内，Jacobian−UADA500 的配对任务等权差值，单位为百分点；方括号是任务分层 bootstrap 95% 区间：

| Suite | r03−UADA500 | r10−UADA500 |
|---|---:|---:|
| Object | −25.0 [−44.0,−7.0] | −16.1 [−30.0,−2.2] |
| Spatial | +3.0 [−10.0,+20.0] | −2.9 [−19.8,+9.2] |
| Goal | −24.7 [−45.4,−7.0] | −16.4 [−31.1,−4.0] |
| Long | −0.6 [−21.0,+22.4] | −6.9 [−18.2,+5.0] |

r10−r03 在 Object +8.9 [−3.0,+22.0]、Goal +8.2 [−1.9,+20.0]，Spatial/Long 为负，四套件区间均跨零。不能据 r10 比 r03 的 Object 点估计较高就宣称“加大 DINO 权重有效”。Object 和 Goal 的两路相对 UADA 都更弱；Spatial/Long 的区间跨零表示没有确定的改进证据，**不是等效性证明**。

这些区间按冻结的任务分层方法计算，未对四套件和多组差值另作多重比较校正；报告的重点是主套件未见增强且源套件/Goal 方向明显不利，不把某个单独区间包装成普遍定律。

额外可重复性提醒：相同 UADA-500 patch、同一 400 状态与 rollout seed 在本次重测和 9 月 21 日旧 Scale-500 评估之间有 10/400 个成功标签不一致（Long 9、Spatial 1；Object/Goal 0），patch hash 与 seed 均匹配。尚未定位原因，可能涉及运行时非确定性，但不能把“可能”写成已证实原因。正因如此，本候选的判断以这次新鲜三臂配对结果为准，不拼接旧运行里的 UADA CASR。

结论边界：本结果反驳的是**这一种局部线性 Jacobian 教师 + 覆盖 token 定向辅助损失**能增强 UADA-500 的主张，不反驳早期 Wrist-DINO 因果敏感性；没有最终补丁的同输入 DINO/SigLIP restore 机制读出，不能声称其失败由 Front 补偿、DINO 教师方向不稳定或梯度冲突导致。也不能宣称已经证明真实物理场景灾难性后果。

## 7. 证据文件、代码与当前操作状态

工作根目录：/home/caijiasheng25/robopatch_defense；代码仓库：其下 roboticAttack-main。以下相对路径均从代码仓库起算，除非另有说明。

| 内容 | 路径 |
|---|---|
| 最新评估配置 | configs/exploring_oft_dino/object_dino_jacobian500_cross_suite.json |
| 最新评估完整状态 | outputs/exploring_oft_dino/object_dino_jacobian500_cross_suite/STATUS.json |
| 最新分 suite CASR | outputs/exploring_oft_dino/object_dino_jacobian500_cross_suite/DINO_JACOBIAN500_XSUITE_PER_SUITE.csv |
| 最新配对差值/区间 | outputs/exploring_oft_dino/object_dino_jacobian500_cross_suite/DINO_JACOBIAN500_XSUITE_CONTRASTS.csv |
| 最新原始 rollout/有效行 | 同目录 DINO_JACOBIAN500_XSUITE_ROLLOUTS.parquet / DINO_JACOBIAN500_XSUITE_ELIGIBLE.parquet |
| 两路训练结果 | outputs/exploring_oft_dino/object_dino_jacobian500_v1/各 method 目录的 status.json、training.parquet、outer_0500.png、outer_0500.pt |
| 评估/统计脚本 | scripts/run_exploring_oft_dino_jacobian500_cross_suite.py；scripts/analyze_exploring_oft_scale500_cross_suite.py |
| Slurm 日志 | logs/slurm/expl_oft_jac500_xsuite_136352_*.out/.err；expl_oft_jac500_stats_136353.out/.err |
| 冻结 UADA2000 | configs/exploring_oft_dino/FROZEN_UADA2000_BASELINE_20260923.json；outputs/exploring_oft_dino/three_arm_object2000_cross_suite/ |
| 早期因果证据 | 根目录 FROZEN_WRIST_EVIDENCE_20260909/WRIST_EVIDENCE_CHAIN_FROZEN_20260909.md |
| 长期方案/历史 | 根目录 OFT_DINO_MECHANISM_CLOSED_LOOP_MASTER_PLAN.md（目前到 31.26，最新 Jacobian 结果不在其中） |

历史交接 DEEPSEEK_HANDOFF_OFT_DINO_20260925.md 写于训练前，末节“无补丁/无 CASR”已过时。NEW_SESSION_HANDOFF_CID_CF_20260918/ 也是更早快照。不要据旧文件重新提交、重复训练或以其中“待运行”覆盖本次 COMPLETE 状态。

当前没有正在等待助手自动提交的任务。用户规则：正式 Slurm 作业只由用户手动提交；任何新实验需先讨论并取得用户方向。最多并发 5 GPU；不改只读原 UADA.py；不清理整个脏工作树；不要为了漂亮结果重选已测位置/任务/权重；Long 照看但不是唯一成功证据。应用场景目前仅作解释备忘，不改变攻击主线。

## 8. 给多个网页模型的讨论题

请先独立审核第 6 节统计和第 3 节因果证据的逻辑边界，不要把“DINO 参与普通攻击传播”误说成“显式 DINO 目标必然增强”。尤其回答：

1. 这个具体 Jacobian-500 方法的负结果最合理地排除了什么、仍不能排除什么？哪几种失败解释目前只是猜测？
2. 当前资料能否支撑“有新攻击贡献”或“更低成本”？若不能，哪些论断仍有研究价值，如何避免把早期因果发现与近期攻击负结果混淆？
3. 如果还建议一种新方法，请给出**与已失败的路径约束、全 token 特征破坏、Object-Decoy、几何对齐、DINO 抽样和本次局部 Jacobian 辅助项实质不同**的具体攻击目标、攻击者权限、最小必要对照和一次性闭环判据。不要只说“提高 DINO 权重”“继续查梯度”或事后从同一结果中挑最佳 suite。

请先讨论，不要假设已获授权继续训练、提交任务或改动主方案。
