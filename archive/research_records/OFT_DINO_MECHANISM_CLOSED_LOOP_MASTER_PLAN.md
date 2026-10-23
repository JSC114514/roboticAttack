# OpenVLA-OFT Wrist-DINO 机制增强闭环研究主方案

> **2026-10-13 当前状态**：第53/54/55节两批250轮闭环均完成。最新注意力/多层/关系目标Object、Spatial、Goal、Long CASR为0/10/5/40、0/15/40/80、5/35/35/80；同轮UADA为5/25/5/85，Key为5/20/40/85，生成器cos为5/10/45/90。关系Spatial局部提升、Object仍未改善。第56节最后补充Value与生成器跨图像InfoNCE（含同B8余弦对照）三张250轮筛选已实现，14项新CPU测试/25项回归和资产预检通过。独立`roboticAttack-main/research_runs/wrist_dino_value_contrast_screen250_v3/`，手动入口`bash submit_exploring_oft_dino_value_contrast_screen250_v3.sh`，未自动sbatch，无新GPU效果。

> **2026-09-20 重启声明（当前最高优先级）**：代码溯源确认 2026-09-15 至 2026-09-20 的自定义补丁训练路线并非在 Exploring UADA/UPA 源码上适配，且后续补丁共同继承噪声型 DINO-Norm-1500。第 1–29 节保留为历史记录，不再作为新主线的方法证据或执行授权。旧代码和结果只读归档于 `ARCHIVE_CUSTOM_PATCH_LINEAGE_20260915_20/`。新实验必须从 `roboticAttack-main/VLAAttacker/white_patch/UADA.py` 的训练结构出发，从相同随机初始化重新训练，禁止使用任何旧学习补丁热启动。

更新日期：2026-10-13
文档性质：持续更新的唯一主方案；后续版本在本文“修订记录”中追加，不另建相互冲突的方案。  
研究边界：仅用于 OpenVLA-OFT 鲁棒性、因果机制与闭环失效研究，不用于真实系统部署。

## 0. 修订记录

| 版本 | 日期 | 内容 | 执行状态 |
|---|---|---|---|
| v1.0 | 2026-09-17 | 冻结 CL-Dev 连续暴露筛选；预注册后续两条机制增强分支 | 已完成实现与校验，待用户手动提交；后续分支待结果 |
| v1.1 | 2026-09-17 | 记录 CL-Dev 250 episodes 结果并按预注册规则选择路线 | CL-Dev 完成；建议 Branch B，尚未授权执行 |
| v1.2 | 2026-09-17 | 补充阶段条件 DINO 方向定义及与 Exploring/WHEN 的可比性边界 | 结论冻结，Branch B 尚未授权执行 |
| v1.3 | 2026-09-17 | 冻结 Branch B 的方向定义、等预算对照、development流程和资源流程 | Branch B development 获准并实现；Final-100未授权 |
| v1.4 | 2026-09-17 | 记录 Branch B 训练、机制读出与250-episode闭环结果 | Development完成；定向项未优于等预算continuation；Final-100未授权 |
| v1.5 | 2026-09-17 | 冻结独立Final-100确认实验、同口径官方基线与主要比较 | Final-100获用户授权；待手动提交 |
| v1.6 | 2026-09-17 | 记录800-episode Final-100结果与冻结解释 | Final-100完成；主方法三个主要比较均为正；停止自动扩展 |
| v1.7 | 2026-09-18 | 冻结Object训练补丁的三suite零样本迁移和四suite汇总 | 获用户授权；待手动提交 |
| v1.8 | 2026-09-18 | 记录2400-episode跨suite零样本结果及异质性结论 | 已完成；Object补丁未形成稳定跨suite迁移；停止自动训练 |
| v1.9 | 2026-09-18 | 冻结单源DINO反事实因果攻击路线及跨suite失效诊断 | 获用户授权；先执行诊断，训练目标不按目标suite结果修改 |
| v1.10 | 2026-09-18 | 记录360单元DINO反事实诊断并定位跨suite失效层级 | 已完成；DINO通路仍主导，但像素→DINO扰动强度显著衰减 |
| v1.11 | 2026-09-18 | 设计单源Context-Invariant DINO-CF正式方法与分阶段评估 | 仅完成设计；未实现、未提交训练 |
| v1.12 | 2026-09-18 | 落地CID-CF四条件训练与Stage A真实峰值测试链 | 实现及CPU契约测试完成；Stage A待用户手动提交 |
| v1.13 | 2026-09-18 | 记录CID-CF Stage A结果并冻结Stage B实测资源 | Stage A通过；Stage B提交脚本已就绪、待用户手动提交 |
| v1.14 | 2026-09-19 | 记录Stage B四路训练并冻结Stage C机制读出协议 | Stage B完成；Stage C已实现并通过预检、待用户手动提交 |
| v1.15 | 2026-09-19 | 记录Stage C结论并冻结三suite Stage D闭环；Long降为非门控可选探针 | Stage D完成；CID无独立优势且源域效力显著下降 |
| v1.16 | 2026-09-19 | CID-CF假设判负；冻结Exploring-UADA/UPA与Wrist-DINO结合的H1/H4二因子实验 | 获用户授权；先实现Object-only训练与Final-100 |
| v1.17 | 2026-09-20 | 记录OFT-UADA Object-100%结果；冻结UADA机制读出及Spatial/Goal零样本评估 | 获用户授权；按机制读出→跨suite两阶段执行 |
| v1.18 | 2026-09-20 | 记录UADA机制与迁移结果；冻结跨suite DINO层级失效诊断 | 获用户授权；只做诊断，不自动训练新方法 |
| v1.19 | 2026-09-20 | 根据层级诊断冻结Aggregate-Mid-DINO与Base-Anchor实验；恢复Long次要外部测试 | 获用户授权；先执行Stage A importance/probe，不提前训练 |
| v1.20 | 2026-09-20 | 记录AG/Base Stage A资源与梯度门结果；授权Object-only Stage B训练 | Stage A通过；Stage B实现并预检后由用户手动提交 |
| v1.21 | 2026-09-20 | 记录AG/Base Stage B训练；冻结复用OFT-UADA旧行的Stage C机制读出 | Stage B完成；Stage C实现并预检后由用户手动提交 |
| v1.22 | 2026-09-20 | 记录Stage C阴性机制证据；冻结Stage D主闭环与Stage E强制Long闭环 | Stage C完成；D→统计→Long→统计完整闭环待用户手动提交 |
| v1.23 | 2026-09-20 | 回填AG/Base Stage D与Long最终结果；冻结应用场景、攻击权限和风险口径；提出下一阶段定向感知通路攻击的待确认设计 | AG/Base分支已关闭；新设计仅供讨论，未实现、未提交 |
| v1.24 | 2026-09-20 | 将应用场景降为可修改解释备忘；重写下一阶段为Wrist-DINO定向可控性与像素可实现性实验 | 仅完成讨论设计；未实现、未提交 |
| v1.25 | 2026-09-20 | 实现定向可控攻击Stage A0接口/梯度审计，冻结Object-only单记录GPU探针 | CPU预检与94项全测试通过；待用户手动提交A0，A1未实现 |
| v1.26 | 2026-09-20 | 记录A0首次端点门失败并修正clean基线：bank仅冻结观测，因果比较统一使用同一causal runtime | 首次作业135380无科学结论；修正版预检后待手动重提，A1未实现 |
| v1.27 | 2026-09-20 | 记录修正版A0通过并冻结/实现Stage A1多状态潜空间可控性实验 | A0接口门通过；A1实现、99项全测试及check-only通过，待用户手动提交 |
| Long-v2 | 2026-10-08 | 审核Long-v1采集工程失败；原生RGB前向/STE反向、等量区域子集配对、原始观测恢复与重提锚点 | 用户授权修复；助手只本地检查，正式GPU由用户提交 |
| Long-v2-evalfix1 | 2026-10-09 | 三组500轮已完成；修复正式测试EGL上下文绑定/生命周期并增加真实单卡检查 | 仅评测修复；原训练和配置冻结，GPU由用户手动提交 |
| Long-v2-evalfix2-3090 | 2026-10-09 | 第一版真实启动检查仍失败；恢复旧评测初始化顺序、恢复线程OpenGL API并记录GL/EGL细节；仅测试改3090≤5 | 新独立副本CPU回归/预检完成，实际GPU与闭环结果待用户提交 |
| Long-final8-v1 | 2026-10-09 | 冻结8方法：Long UADA、DINO采样、频率步预条件、XYZ-TMA及两组真实RGB教师蒸馏；0→2000保存500/1000/2000；4090训练≤8、3090自动测试≤8 | 独立副本已实现；26项CPU/mock、18文件AST、shell与真实资产预检通过；GPU未提交，用户手动入口已就绪 |
| Feature-screen250-v1 | 2026-10-13 | 停止新增DINO选样；Key、去中心化dense、公开cos分支生成器+同预算UADA；250×50与自动400回合 | 27项CPU/toy、21份AST、7份shell及本地资产预检通过；训练4卡后测试4卡，未提交 |
| Feature-screen250-cudnnfix-v2 | 2026-10-13 | Key/dense完成、UADA运行；生成器首次前向cuDNN错误，提前禁cuDNN且只补生成器/修复依赖 | 独立启动副本，38项CPU/toy及资产预检通过；复用三组和缓存，未自动提交 |
| Feature-screen250-cudnnfix-v2 | 2026-10-13 | 修复G首次前向在禁cuDNN之前及失败整个array依赖；复用三组/Long缓存，只补G，等待UDA子任务后测试 | 新11项/总38项CPU通过；独立手动入口，未提交修复 |

## 1. 当前证据

1. Wrist 输入相对 Front 具有更强的动作传播增益，主要经 DINO 视觉编码路径传播。
2. DINO-Norm-1500 在 LIBERO-Object Wrist-only fresh test 上达到 54.22% task-equal CASR。
3. RAMP-T 相对 DINO-Continue 在 development open-loop 上将轨迹累计漂移提高
   `+1.407 [0.822, 2.002]`；RAMP-TG 提高 `+1.139 [0.511, 1.750]`。
4. 一次带补丁动作块后移除补丁的 recovery probe 未证明 RAMP 更难恢复。
5. recovery probe 不等价于持续攻击：正式数字补丁在每个闭环决策步持续出现。因此在重新训练前，必须先测量现有机制增强是否提高持续闭环失败率。

## 2. 总体科学问题

v1.0–v1.23首先研究了如何提高Wrist-DINO扰动向持续闭环任务失败的转化效率；该路线已经完成并证明：
OFT-UADA会自然经由DINO通路产生强源域失败，但继续叠加DINO辅助目标没有建立独立收益。v1.24起的当前
科学问题更新为：

> 已验证的Wrist-DINO→Action因果通路能否被攻击者主动指定方向并实现双向可控，而不只是被动传递一种
> 无方向、事后才被解释为DINO介导的攻击？

当前核心因果链为：

```text
attacker-specified action direction (+d / -d)
→ Wrist-only bounded intervention optimized through DINO-only action path
→ DINO-specific directional displacement
→ signed full-model action change and bidirectional flip
→ direction-consistent closed-loop state change
```

攻击载体暂不固定为patch，应用场景和危险后果也暂不进入主假设。任何只能提高动作L2或普通CASR、却不能
随`+d/-d`翻转，或不能被DINO restore特异关闭的方法，只能作为一般攻击对照，不能支持主动利用机制的结论。

## 3. Stage CL-Dev：现有补丁持续闭环筛选（已完成）

### 3.1 目的

回答 RAMP 的 open-loop 累计漂移增益，在补丁持续存在时是否转化为更高闭环失败率。

### 3.2 数据与配对

- Suite：LIBERO-Object，10 tasks。
- 每任务使用 R2 在 `state_index 30–49` 中按升序固定选出的前5条 clean-success trajectory。
- 共50个配对初始状态；这些是 development，不产生最终确认性结论。
- 每个初始状态 Clean 固定先运行；4个攻击条件按冻结 seed 在状态内随机排序。

### 3.3 条件

```text
Clean
DINO-Norm-1500
DINO-Continue-100
RAMP-T-100
RAMP-TG-100
```

所有攻击在每个 policy decision block 持续修改 Wrist；Front、语言、proprioception 和模型权重不变。

规模：`10 tasks × 5 states × 5 conditions = 250 episodes`。

### 3.4 指标

Primary：task-equal CASR，以本次配对 Clean 成功 episode 为分母。  
Secondary：attacked success rate、raw failure、episode steps、per-task CASR。  
Mechanism-linked：将 episode failure 与相同 state/condition 的 R2 DINO displacement、DINO restore reduction、trajectory cumulative drift 配对，不把6个phase state当独立样本。

主要差值：

```text
RAMP-T   - DINO-Continue
RAMP-TG  - DINO-Continue
RAMP-TG  - RAMP-T
DINO-Continue - DINO-Norm-1500
```

统计：task→paired episode hierarchical bootstrap，10,000 draws，seed 20260907。

### 3.5 计算资源与执行约束

- 模型推理与 LIBERO 仿真：RTX 3090，每个 task 1 GPU、4 CPU、8 GiB host memory。
- 数组 `0–9%5`：最多并发5张 GPU；每张卡只运行一个 task shard。
- 复用相同模型链路已完成的峰值报告：RTX 3090 reserved GPU memory 约20.49 GiB，
  host MaxRSS 约6.20 GiB；8 GiB host memory 包含约25%余量。
- CPU 仅运行提交前契约测试与最终 bootstrap 统计，不在 CPU 上运行模型或仿真。
- 每个 state 完成后原子写入 partial parquet，重提时只续跑完整 state 之后的内容。

### 3.6 结果后的预注册分支

- 若 RAMP-T 相对 DINO-Continue 的 CASR 与任务方向显示有实际增益：进入 Branch A，将 RAMP-T 机制训练扩展到500 trajectory updates，DINO-Continue 同预算对照。
- 若 RAMP-T 无闭环增益：进入 Branch B，不再增加无目标漂移；改用阶段条件 DINO 因果方向。
- CL-Dev 只负责路线选择；不得挑选 development 上最优 checkpoint 作为最终结论。

## 4. Branch B 预定义：阶段条件 DINO 因果方向

对训练状态定义 task-critical action score，并计算：

\[
u_{k,t}^{critical}=\operatorname{normalize}
\left(\frac{\partial S_{critical}}{\partial z_{DINO}}\right)
\]

主方法优化：

\[
J=D_{state}+\lambda_D\|\Delta z_{DINO}\|_2
+\lambda_C\langle\Delta z_{DINO},u_{k,t}^{critical}\rangle
+\lambda_TD_{trajectory}
\]

对照为 DINO-Continue、Action-Critical-only、DINO-Critical-T。三者从同一
DINO-Norm-1500 开始，Primary 为500 trajectory updates；checkpoint 100/250/500仅诊断。

该分支须同时验证 DINO alignment、critical action deviation、DINO restore closure 与闭环失败。
本节是 CL-Dev 前冻结的路线预定义；CL-Dev 已完成，Branch B development 已获授权，
当前冻结执行协议见第10节。

## 5. 训练时间解释与后续预算

当前 RAMP continuation 只有100 trajectory optimizer steps、600 state backward exposures，且从 DINO-Norm-1500 开始、固定位置、无EOT。Exploring/WHEN 本地配置通常为
`2000 outer × 50 inner ≈ 100,000 patch updates`，并包含随机几何、较大batch、周期验证；WHEN还可包含8步RAUP。因此当前快速完成不代表已达到同等优化预算。

后续机制训练若启动，使用500 trajectory updates，即每方法3000 state backward exposures；先做完整峰值测试，首选RTX 3090，最多并发5张GPU。

## 6. 隐蔽性与实际场景（暂缓）

待机制增强在闭环成立后，再固定真实载体并加入正常纹理距离、平滑度、可打印色域和EOT。隐蔽化后仍必须保留 DINO displacement、DINO restore closure 与闭环效应，不能只报告外观或成功率。

## 7. 当前执行边界

Stage CL-Dev 已完成。当前获准执行第10节 Branch B development：方向构建、三路等预算训练、
development机制读出和持续闭环。完成后停止，由用户决定是否进入 Final-100。
不得自动访问冻结 final states、跨suite、物理或隐蔽性实验。

## 8. Stage CL-Dev 结果（2026-09-17）

### 8.1 完整性

- 250/250 episodes 完成；10/10 task shards 完成。
- 50/50 Clean rerun 成功，因此 CASR 分母没有因 clean failure 缩减。
- 共6956条 decision-block traces；GPU renderer 检查通过；统计任务完成。

### 8.2 闭环结果

| 条件 | CASR | task→episode 95% CI | 失败数/50 |
|---|---:|---:|---:|
| DINO-Norm-1500 | 58% | [36%, 80%] | 29 |
| DINO-Continue-100 | 68% | [50%, 84%] | 34 |
| RAMP-T-100 | 72% | [52%, 90%] | 36 |
| RAMP-TG-100 | 68% | [46%, 88%] | 34 |

预注册主要对比：

- RAMP-T − DINO-Continue：`+4 pp [−10, +16]`；4个任务为正、1个为负、5个持平。
- RAMP-TG − DINO-Continue：`0 pp [−16, +14]`；3正、2负、5平。
- RAMP-TG − RAMP-T：`−4 pp [−16, +6]`。
- DINO-Continue − DINO-Norm：`+10 pp [−4, +26]`。

配对 episode 中，RAMP-T 相对 DINO-Continue 新增4次失败，但同时使2次原有失败恢复成功，
净增仅2/50；RAMP-TG 双向各改变4次，净增为0。

### 8.3 机制解释与路线判定

持续施加 Wrist-DINO patch 本身已能造成较高闭环失败率，证明 open-loop 动作偏差可以转化为
闭环任务失败。但 RAMP-T 的额外累计漂移只带来小幅且不确定的 `+4 pp`，RAMP-TG 没有净增益；
而且按闭环成败分组后，R2 的 DINO displacement、动作偏差和累计漂移未呈现稳定的单调分离。
因此当前瓶颈不是“扰动幅度仍不够大”，而是扰动是否沿当前任务阶段真正关键的动作方向传播。

按3.6节预注册规则，CL-Dev 不足以支持直接把 RAMP-T 延长到500 updates，建议进入
**Branch B：阶段条件 DINO 因果方向**。该判断是路线筛选，不是否定 Wrist-DINO 机制；相反，
下一步仍以 DINO 为必需机制，只把无方向的幅度目标改为 task/phase-critical alignment。
该路线判定随后在 v1.3 获得用户授权；具体执行协议见第10节。

### 8.4 与基线工作的比较边界

同一 MGWP-v2 fresh-test runner、同一批约100个 Object Wrist-only episodes 的直接对照为：

| 方法 | task-equal CASR |
|---|---:|
| Random patch | 11.00% |
| Exploring E-UADA-1to3 official patch | 9.00% |
| WHEN official patch | 7.11% |
| Persistent-action baseline | 38.00% |
| DINO-Norm-1500 | **54.22%** |

因此 DINO-Norm 在这组同口径 fresh test 上分别高于 Exploring
`45.22 pp [27.22, 61.33]`、WHEN `47.11 pp [29.11, 64.33]`；两者为事后补充的
task→episode paired bootstrap 对比。相对 Persistent baseline 为
`+16.22 pp [−0.78, 32.00]`，尚不能确认稳定超越。另一组官方补丁跨-suite评估在 LIBERO-Object 上得到
Exploring UADA-1to3 Wrist-only 11%、WHEN Wrist-only 16%，也明显低于 DINO-Norm 的54.22%；
但其 state split/placement 与 MGWP-v2 不完全相同，只作外部参照。

CL-Dev 的 DINO-Norm 58%、DINO-Continue 68%、RAMP-T 72%、RAMP-TG 68%使用50个
development states。它证明持续机制攻击很强，但 Exploring/WHEN 没有在同一 CL-Dev grid 重跑，
所以不能用72%直接宣称新的公平 head-to-head 超越，也不能宣称跨模型/跨suite SOTA。

## 9. Branch B 的“任务阶段条件 DINO 方向”定义

原方法主要最大化 DINO 表征位移、全动作偏差或跨状态累计漂移；它只关心偏差大小，不区分
该偏差是否破坏当前阶段决定任务成败的动作自由度。新的方向定义为：对 task `k`、阶段 `φ`
和状态 `t`，先定义阶段关键的任务进展分数 `S_{k,φ,t}`，再在 clean DINO 表征处计算使该分数
下降最快的方向：

\[
u^{fail}_{k,\phi,t}=-\operatorname{normalize}
\left(\frac{\partial S_{k,\phi,t}}{\partial z_{DINO}}\right).
\]

训练仍使用同一个固定 Wrist patch，但增加表征投影项，使补丁引起的
`Δz_DINO` 与 `u_fail` 对齐。任务条件决定正在操作什么目标；阶段条件区分 approach、pre-contact、
contact、post-contact 等阶段。例如 approach 更关注朝目标移动，contact 更关注夹爪闭合，
post-contact 更关注稳定抬升/运输。最终部署不要求在线更换补丁或读取阶段，task/phase 只在训练时
用于构造方向监督。

与原方法的核心差别是：

```text
原方法：让 DINO / action / trajectory 偏差尽量大
Branch B：让 DINO 偏差优先投向“此任务、此阶段最可能导致失败”的子空间
```

Branch B 仍必须通过 DINO restore closure 证明作用路径经过 DINO；它是机制的定向增强，
不是抛弃机制改做纯闭环黑盒优化。

## 10. Branch B Development 冻结执行协议（v1.3）

### 10.1 科学问题与实验单位

问题：在相同初始化、数据和500 trajectory updates 下，把 Wrist-DINO 偏差对齐到
task/phase-specific failure direction，能否比无方向 DINO-Continue 更有效地转化为闭环失败？

训练单位是一条六状态trajectory；闭环独立重复单位是task内的initial state。六个phase readout
是重复测量，统计前必须合并到state/trajectory，禁止当作独立样本。

### 10.2 方向的可计算定义

对每个训练状态，令 clean action chunk 经动作尺度标准化后为 `a_clean`，局部失败动作方向为：

\[
b^{fail}_{k,\phi,t}=-\frac{a^{clean}_{k,\phi,t}}
{\|a^{clean}_{k,\phi,t}\|_2}.
\]

在已验证的 RANGE/LATERAL 自然 Wrist 几何 DINO 二维子空间内，用对称有限差分得到局部
DINO→action响应矩阵 `R∈R^(56×2)`，并取：

\[
c^{fail}=\operatorname{normalize}(R^Tb^{fail}).
\]

再由 `c_fail` 组合两个自然 DINO basis，得到 `u_DINO_fail`。这里 task 条件来自语言和任务状态，
phase 条件来自该阶段 clean policy 实际要执行的action chunk；不根据闭环攻击结果反向挑方向。
RANGE/LATERAL 的自然尺度复用 MGWP-v2 artifact；Branch B 的局部对称差分半径独立冻结为
自然尺度的0.25倍。源 artifact 的历史0.5倍差分半径仅作为 provenance 记录，不约束本分支。

### 10.3 等预算条件

三个条件都从同一 DINO-Norm-1500 patch 开始，使用相同50条训练trajectory、相同顺序、
学习率和500 trajectory updates（每方法3000 state backward exposures）：

1. `DINO-Continue-B`：继续优化动作偏差和DINO norm，无定向辅助项。
2. `Action-Critical-B`：在相同base objective上加入 failure-action alignment。
3. `DINO-Critical-B`：加入等总梯度预算的 failure-action + failure-DINO alignment。

checkpoint 100/250/500只用于训练诊断；主比较固定使用update 500，不按结果挑checkpoint。

### 10.4 Development评估

- 使用 Stage CL-Dev 已冻结的每任务5个clean-success states，共50 states；均与历史
  `state 40–49` final split不重叠。
- Open-loop：4 conditions × 50 states × 6 phases = 1200 rows。条件为DINO-Norm和三个B方法。
- Continuous closed-loop：Clean + 4 attack conditions，共250 episodes；Clean固定先运行，
  四个攻击在state内按冻结seed随机排序。
- Primary：task-equal paired CASR。
- Mechanism：critical action alignment、critical DINO alignment、DINO displacement、
  DINO restore reduction、trajectory cumulative drift。
- 主要对比：DINO-Critical−DINO-Continue、DINO-Critical−Action-Critical、
  Action-Critical−DINO-Continue、DINO-Critical−DINO-Norm。
- 统计：task→paired episode hierarchical bootstrap，10,000 draws，seed 20260907。

### 10.5 资源与停止规则

先在RTX 3090运行完整六状态DINO-Critical反向传播峰值任务，再以实测host MaxRSS×1.25
申请正式内存。训练最多3张3090并行，评估最多5张；CPU仅做契约测试和bootstrap统计。

Branch B development统计完成后写入 `FINAL_100_NOT_AUTHORIZED` 并停止。冻结的historical
Final-100为每任务state 40–49；只有用户再次授权后，才在同一runner中比较Clean、Exploring官方、
WHEN官方、DINO-Norm、DINO-Continue-B和DINO-Critical-B。

## 11. Branch B Development 结果（2026-09-17）

### 11.1 完整性

- 三种方法均完成500 trajectory updates，即每方法3000次state backward exposures。
- Open-loop完成1200/1200 rows；持续闭环完成250/250 episodes和7221条decision traces。
- 50/50配对Clean episodes均成功，四种攻击各有50个有效闭环分母。
- 统计使用冻结的task→paired episode hierarchical bootstrap；完成后按协议停止在
  `FINAL_100_NOT_AUTHORIZED`。

### 11.2 持续闭环结果

| 条件 | CASR | task→episode 95% CI | 失败数/50 |
|---|---:|---:|---:|
| DINO-Norm-1500 | 58% | [34%, 80%] | 29 |
| DINO-Continue-B-500 | **84%** | [68%, 96%] | 42 |
| Action-Critical-B-500 | 82% | [64%, 98%] | 41 |
| DINO-Critical-B-500 | 82% | [64%, 96%] | 41 |

预注册主要闭环对比：

- DINO-Critical − DINO-Continue：`−2 pp [−16, +10]`；2个任务为正、2个为负、6个持平。
- DINO-Critical − Action-Critical：`0 pp [−8, +8]`；1正、1负、8平。
- Action-Critical − DINO-Continue：`−2 pp [−16, +10]`。
- DINO-Critical − DINO-Norm：`+24 pp [+8, +42]`；6正、0负、4平。

### 11.3 机制读出

DINO-Critical相对同预算DINO-Continue：

- critical action alignment提高 `+0.275 [0.233, 0.322]`，10/10任务同向；
- critical DINO alignment仅 `+0.000047 [−0.000772, +0.000892]`，没有可辨别增益；
- action deviation为 `−0.002 [−0.123, +0.121]`，基本不变；
- DINO displacement反而降低 `−111.90 [−123.27, −101.00]`；
- DINO restore reduction差值为 `−0.0017 [−0.0047, +0.0015]`，两者绝对值仍约85%；
- trajectory cumulative drift为 `−0.371 [−1.553, +0.802]`，没有稳定变化。

### 11.4 Development判定

500-step continuation将同一development grid上的描述性CASR从DINO-Norm的58%提高到84%，说明
延长机制训练本身仍能显著增强持续闭环失效。可是当前failure-direction辅助项虽然稳定提高了
动作在预定义critical direction上的投影，却没有控制DINO表征沿目标方向移动，也没有在闭环上
超过等预算DINO-Continue。因此本版Branch B不能支持“阶段条件DINO定向优于普通DINO继续训练”
这一主张；它支持的更窄结论是：Wrist-DINO持续攻击可达到很高development失败率，但当前瓶颈仍是
如何把可控DINO方向转化为额外闭环失败，而不是继续增加一个action-space alignment项。

本结果不自动启动Final-100，也不据此挑选checkpoint。用户随后明确授权第12节冻结的
Final-100确认实验。

## 12. Final-100 独立确认实验（v1.5）

### 12.1 冻结目的和主方法

Final-100只回答：在完全未参与训练、方法选择和超参数选择的初始状态上，预先选定的
`DINO-Continue-B-500`能否保持高闭环失败率，并在同一runner中公平超过官方Exploring、WHEN
和原始DINO-Norm。主方法固定为update 500，不依据Final结果改选checkpoint或Branch B变体。

### 12.2 样本、条件和运行顺序

- Suite：LIBERO-Object，10 tasks。
- Final states：每任务固定使用`state_index 40–49`，共100个独立初始状态；它们未参与训练、
  development筛选或方法选择。
- 条件：Clean、Random、Exploring E-UADA-1to3官方补丁、WHEN官方补丁、DINO-Norm-1500、
  DINO-Continue-100、DINO-Continue-B-500、DINO-Critical-B-500。
- 共`10 tasks × 10 states × 8 conditions = 800 episodes`。
- 所有攻击使用同一50×50 RGB支持、同一Wrist-only位置`(x,y)=(26,137)`，Front、语言、
  proprioception、模型权重和仿真初态不变。
- 每个state固定先运行Clean；随后7个攻击用seed 20260916在state内随机排序。每个条件前均从
  相同initial state和rollout seed精确reset，因此condition order不改变初态。

### 12.3 指标和冻结比较

Primary：task-equal paired CASR，仅以本次Clean rerun成功的state为分母。

Primary contrasts：

1. DINO-Continue-B-500 − Exploring official；
2. DINO-Continue-B-500 − WHEN official；
3. DINO-Continue-B-500 − DINO-Norm-1500。

Secondary contrasts：DINO-Continue-B-500分别减Random、DINO-Continue-100和
DINO-Critical-B-500。另报告attacked success、raw failure、每任务CASR、episode steps及
正/负/持平任务数。统计使用task→paired episode hierarchical bootstrap，10,000 draws，
seed 20260907；六个phase或decision traces不作为独立样本。

### 12.4 解释边界

- Final-100若保持高CASR并在主要配对差值上为正，才支持“在该冻结LIBERO-Object Wrist-only
  threat model下稳定优于官方基线”。
- 该实验不支持跨suite、跨模型、物理世界或隐蔽性SOTA主张。
- Branch B定向方法已在development落后于普通continuation，因此仅作为机制消融，不能在
  Final结果出来后替换主方法。

### 12.5 资源和停止规则

复用已完成的RTX 3090完整训练峰值报告以及同runner 250-episode闭环实测：推理显存低于已测
17.60 GiB训练峰值，host MaxRSS约6.20 GiB，因此每个GPU shard申请1张RTX 3090、4 CPU和8 GiB
host memory。数组`0–9%5`，最多并发5张GPU；CPU仅进行提交前测试和最终bootstrap。

Final统计完成后停止，不自动启动跨suite、物理、EOT或隐蔽性实验。

## 13. Final-100 结果（2026-09-17）

### 13.1 完整性

- 800/800 episodes、20,033条decision traces和10/10 task shards完成，无重复key或运行错误。
- 100次Clean rerun中99次成功；唯一Clean failure为task 8、state 43，按预注册规则不进入CASR分母。
- 每个攻击条件均有99个paired clean-success episodes；统计单位为task→paired state，不把trace当重复。
- 主方法、主要比较和checkpoint均在结果产生前冻结；结果状态为
  `FINAL_100_COMPLETE_NO_AUTOMATIC_EXTENSION`。

### 13.2 Final闭环结果

| 条件 | task-equal CASR | 95% CI | 失败事件/99 |
|---|---:|---:|---:|
| DINO-Continue-B-500（冻结主方法） | **74.33%** | [57.33%, 90.00%] | 74 |
| DINO-Critical-B-500（预注册消融） | 79.33% | [64.11%, 92.00%] | 79 |
| DINO-Continue-100 | 70.56% | [53.11%, 86.00%] | 70 |
| DINO-Norm-1500 | 54.22% | [36.44%, 72.00%] | 54 |
| Random | 10.00% | [1.00%, 24.00%] | 10 |
| Exploring E-UADA-1to3 official | 9.00% | [1.00%, 20.00%] | 9 |
| WHEN official | 7.11% | [1.00%, 14.89%] | 7 |

冻结的三个Primary contrasts全部为正且95% CI排除0：

- Continue-500 − Exploring：`+65.33 pp [+47.00, +81.67]`，10正、0负、0平；
- Continue-500 − WHEN：`+67.22 pp [+47.55, +85.00]`，10正、0负、0平；
- Continue-500 − DINO-Norm：`+20.11 pp [+8.00, +33.11]`，8正、0负、2平。

Secondary contrasts：

- Continue-500 − Random：`+64.33 pp [+43.11, +84.56]`；
- Continue-500 − Continue-100：`+3.78 pp [−8.78, +17.00]`；
- Continue-500 − DINO-Critical-500：`−5.00 pp [−16.00, +4.00]`。

### 13.3 冻结解释

Final-100确认：在冻结LIBERO-Object、Wrist-only、50×50、固定位置和同一OFT runner的威胁模型下，
基于Wrist-DINO机制训练的主方法具有很高且可泛化的闭环破坏率，并稳定高于直接迁移的Exploring、
WHEN官方补丁和原始DINO-Norm。相对DINO-Norm的`+20.11 pp`说明机制继续训练带来额外闭环收益。

但是，Final数据不能确认500-step一定优于100-step，因为差值区间跨0。DINO-Critical在Final上的
描述性CASR为79.33%，高于冻结主方法5 pp，但该secondary差值同样跨0，且development阶段未显示
优势；因此不得事后把它替换为主方法或宣称定向方法更优。

对Exploring和WHEN的结论限定为“官方released patch在本研究统一Wrist-only runner与共享位置上的
直接迁移基线”，不等价于否定其原论文威胁模型、重新训练版本或其他放置方式。当前不自动启动
新的调参、跨suite、物理、EOT或隐蔽性实验。

## 14. Cross-Suite Zero-Shot Transfer（v1.7）

### 14.1 目的与冻结边界

不修改Object训练得到的任何补丁像素，直接在LIBERO-Spatial、Goal、Long（`libero_10`）各自的
OFT checkpoint上测试，回答Wrist-DINO机制攻击能否跨任务分布和suite-specific模型零样本迁移。
本阶段不训练multi-suite补丁、不调参、不做隐蔽性或EOT。

### 14.2 设计

- 新评估suite：Spatial、Goal、Long；Object复用已冻结Final-100，不重复消耗GPU。
- 每suite 10 tasks，每task固定state 40–49，共300个新初始状态。
- 条件与Final-100完全一致：Clean、Random、Exploring、WHEN、DINO-Norm、Continue-100、
  Continue-B-500、DINO-Critical-B-500。
- 新运行量：`3 suites × 10 tasks × 10 states × 8 conditions = 2400 episodes`；合并Object后为
  3200 episodes。
- 所有攻击统一使用50×50 Wrist-only像素位置`(26,137)`，以隔离suite/checkpoint变化；官方补丁
  suite-native placement的既有结果只作外部参照，不与共享位置主要比较混合。
- 每state先运行Clean，7个攻击按seed 20260916在state内随机排序；每个条件精确reset到同一初态。

### 14.3 指标与比较

Primary：每suite task-equal paired CASR，以及四suite中suite→task→paired-state等权汇总。
冻结主方法仍为DINO-Continue-B-500。主要比较仍为其相对Exploring、WHEN、DINO-Norm；另报告
相对Random、Continue-100、DINO-Critical的secondary contrasts。零样本结论主要依据三个新suite，
Object只作为in-domain reference。

统计在suite内使用task→state bootstrap；跨suite汇总使用suite→task→state hierarchical bootstrap，
10,000 draws，seed 20260907。Clean失败state不进入CASR分母，decision traces不作为独立样本。

### 14.4 资源与停止

复用同runner已完成的RTX 3090峰值和Final-100实测，每个task shard申请1张3090、4 CPU、8 GiB
host memory；数组30 shards、最多并发5张GPU。统计完成后停止，不自动训练multi-suite补丁，也不
启动隐蔽性、EOT或物理实验。

## 15. Cross-Suite Zero-Shot 结果（2026-09-18）

### 15.1 完整性

- 30/30 GPU shards完成，共新增2400/2400 episodes和68,328条decision traces；合并Object
  Final-100后共3200 episodes。
- Spatial、Goal、Long的Clean成功数分别为90/100、98/100、95/100；仅Clean成功state进入
  paired CASR分母。Object沿用Final-100的99/100。
- 条件、补丁像素、Wrist位置、state范围和主要比较均按第14节冻结方案执行；未发现traceback、
  OOM、重复episode key或缺失shard。
- 结果状态为`XSUITE_ZERO_SHOT_COMPLETE_NO_AUTOMATIC_TRAINING`。

### 15.2 各suite闭环结果

下表为task-equal paired CASR；Object是训练suite内的冻结参考，另外三列才是零样本迁移结果。

| 条件 | Spatial | Object（in-domain） | Goal | Long |
|---|---:|---:|---:|---:|
| DINO-Continue-B-500（冻结主方法） | 7.22% | **74.33%** | 15.56% | 59.50% |
| DINO-Critical-B-500 | 14.22% | 79.33% | 19.78% | 59.50% |
| DINO-Continue-100 | 7.22% | 70.56% | 21.56% | 58.75% |
| DINO-Norm-1500 | 12.22% | 54.22% | 20.89% | 57.64% |
| Exploring official | 15.56% | 9.00% | 38.33% | 83.25% |
| WHEN official | 13.56% | 7.11% | 31.22% | 95.00% |
| Random | 10.22% | 10.00% | 4.22% | 37.31% |

### 15.3 三个零样本suite的冻结比较

三个新suite等权汇总时，冻结主方法CASR为`27.43% [5.93%, 56.42%]`。其他条件为：
Exploring 45.71%、WHEN 46.59%、DINO-Norm 30.25%、DINO-Critical 31.17%、
Continue-100 29.18%、Random 17.25%。

冻结的三个Primary contrasts均未支持主方法的跨suite优势：

- Continue-500 − Exploring：`−18.29 pp [−31.22, −5.04]`，2正、19负、9平；
- Continue-500 − WHEN：`−19.17 pp [−37.00, −3.37]`，3正、20负、7平；
- Continue-500 − DINO-Norm：`−2.82 pp [−10.00, +3.19]`，5正、8负、17平。

Secondary contrasts为：相对Random `+10.18 pp [−2.52, +23.70]`，相对Continue-100
`−1.75 pp [−7.57, +2.58]`，相对DINO-Critical `−3.74 pp [−10.94, +1.83]`。

若将Object也纳入四suite等权汇总，Continue-500为39.15%，Exploring为36.54%，WHEN为
36.72%，DINO-Norm为36.24%；但三个主要差值的95% CI均跨0。该汇总受Object上的巨大
in-domain优势驱动，不能替代三个新suite的零样本结论。

### 15.4 冻结解释

这是一个清晰的异质性/负迁移结果。Object训练的Wrist-DINO补丁在Object上保持74.33% CASR，
并在Long上迁移到59.50%，但在Spatial和Goal上分别只有7.22%和15.56%。因此，现有证据支持
“Wrist-DINO是Object攻击能够被增强的重要机制”，却不支持“同一个Object像素补丁是稳定的
跨suite通用攻击”。机制本身可能跨suite存在，但当前优化得到的具体表征方向和像素模式明显带有
suite/checkpoint依赖。

不得只挑选Long宣称普适迁移，也不得事后将DINO-Critical替换为主方法。v1.8结果产生时曾将
suite-balanced multi-suite训练列为候选下一步；用户指出其不能证明面对新增suite的泛化后，
v1.9正式以第16节的单源训练路线取代该建议。当前仍不启动隐蔽性实验。

## 16. 单源 DINO 反事实因果攻击路线（v1.9）

### 16.1 对v1.8结论的修正

Object、Spatial和Goal中分别存在Wrist pixels→DINO representation→Action的因果通路，并不推出
同一个像素方向跨suite共享。v1.8只能说明当前Object补丁没有形成稳定迁移，不能仅凭闭环CASR区分：

1. 补丁在目标场景中未产生足够DINO位移；
2. DINO位移存在，但落入目标checkpoint的低action-gain方向；
3. DINO-only动作效应存在，但没有命中会在闭环累积成失败的阶段或方向。

因此，当前补丁可称为“在Object上利用DINO高增益通路”，尚不能称为“跨suite共享DINO机制补丁”。

### 16.2 威胁模型与泛化边界

- 正式方法只允许使用一个源suite：LIBERO-Object及其OFT checkpoint、Wrist图像和动作输出。
- Spatial、Goal、Long checkpoint、图像、梯度和统计均不得进入补丁更新、checkpoint选择或超参数选择。
- 不进行四suite联合训练。四suite联合训练最多作为oracle upper bound，不能成为主方法。
- Spatial/Goal的既有结果已用于发现迁移问题，因此后续应表述为预先固定目标域上的zero-shot transfer，
  而不是从未观察过的benchmark。Long不参与本节机制诊断，保留为相对独立的最终迁移确认。
- 新方法与训练变体在本次诊断结果产生前冻结，诊断只解释失效层级，不触发目标域调参。

### 16.3 先行机制诊断

使用冻结主补丁`DINO-Continue-B-500`，在Object、Spatial、Goal上运行同一套读出：每suite 10 tasks，
每task从既有Final state 40–49中按state index选择前2个Clean成功state，重放clean轨迹并取6个
阶段代理，共`3×10×2×6=360`个读出单元。补丁固定为50×50、Wrist-only、位置`(26,137)`。

每个单元记录：

- DINO与SigLIP feature RMS displacement；
- 完整攻击Action L2；
- `DINO-only`反事实Action L2：patched DINO + clean SigLIP；
- `SigLIP-only`反事实Action L2：clean DINO + patched SigLIP；
- DINO restore与SigLIP restore reduction；
- restore-both Action closure，作为hook工程闭合检查。

phase是同一trajectory内重复测量，不作为独立样本。先对6 phases取state均值，再对state取task均值；
报告task-equal point estimate、task bootstrap 95% CI和目标/Object比值，不设置事后PASS阈值。

### 16.4 冻结的新训练目标

新方法暂称`DINO-CF-CVaR`。对Object源样本，令：

\[
a_{D\text{-only}}(P)=F(z_D(x\oplus P),z_S(x)),
\]

即补丁只通过DINO分支进入反事实前向，SigLIP保持clean。冻结目标为：

\[
L=L_{\text{CF-mean}}+\lambda_TL_{\text{CF-tail}}+\lambda_FL_{\text{full-action}},
\]

其中`CF-mean`最大化平均DINO-only标准化Action L2，`CF-tail`提高源域低效状态的下尾表现，
`full-action`约束反事实效应在真实双编码器前向中仍存在。训练使用Object内10 tasks与阶段均衡采样、
Wrist位置/尺度/旋转EOT；不把此前未显示优势的critical action direction写入主损失。

### 16.5 冻结对照与后续顺序

同一Object数据、初始化、更新数和EOT预算下比较：

1. 现有DINO-Norm/Continue；
2. Action-only；
3. DINO-CF；
4. DINO-CF-CVaR（完整方法）。

checkpoint及超参数只按Object development选择。先完成第16.3节诊断并检查实现闭合；随后实现上述
四条件单源训练，最终在Object held-out states、Spatial/Goal零样本和Long保留确认集上评估。
Primary同时报告目标suite Macro-CASR与Worst-Suite CASR，不能用Object高值掩盖目标域失败；机制主张
还要求目标suite中DINO-only effect及DINO restore closure与闭环提升方向一致。

## 17. DINO反事实跨suite诊断结果（2026-09-18）

### 17.1 完整性

- Job `134908`的10/10 task shards完成；每个task顺序处理Object、Spatial、Goal，未产生额外小日志。
- 共360/360行：3 suites × 10 tasks × 2 clean-success states × 6 phases；无重复键、NaN、Inf、
  traceback或OOM。
- restore-both Action closure为360/360精确0；GPU renderer、hook替换和输出闭合通过。
- Job `134909`完成task-level bootstrap统计；phase先在state内聚合，未作为独立样本。

### 17.2 主要结果

| 指标（task-equal mean） | Object | Spatial | Spatial/Object | Goal | Goal/Object |
|---|---:|---:|---:|---:|---:|
| DINO RMS displacement | 8.416 | 2.309 | 27.4% | 3.185 | 37.8% |
| SigLIP RMS displacement | 0.834 | 0.734 | 88.0% | 0.901 | 108.0% |
| Full Action L2 | 5.659 | 2.064 | 36.5% | 2.898 | 51.2% |
| DINO-only Action L2 | 5.370 | 1.538 | 28.6% | 2.537 | 47.3% |
| SigLIP-only Action L2 | 0.813 | 0.455 | 55.9% | 0.558 | 68.7% |
| DINO-only / Full Action | 99.6% | 79.6% | — | 87.3% | — |
| DINO restore reduction | 4.846 | 1.609 | 33.2% | 2.340 | 48.3% |

相对Object的task-bootstrap差值均显示DINO链路明显衰减：

- Spatial DINO displacement：`−6.107 [−6.373, −5.839]`；DINO-only Action：
  `−3.833 [−4.511, −3.197]`；
- Goal DINO displacement：`−5.231 [−5.501, −4.963]`；DINO-only Action：
  `−2.833 [−3.585, −2.080]`。

与此同时，SigLIP displacement并未出现同等幅度下降：Spatial为Object的88.0%，Goal为108.0%。
因此不能用整体图像扰动变弱解释结果，衰减具有明确的DINO分支特异性。

### 17.3 冻结解释

诊断否定“当前补丁在目标suite完全没有利用DINO机制”这一强说法。Spatial和Goal中，
DINO-only反事实效应仍分别占完整Action效应的79.6%和87.3%，恢复DINO也仍能去除主要动作偏差；
所以DINO因果通路在目标suite中仍然主导。

真正的首要瓶颈出现在更上游：同一Object补丁在Spatial/Goal图像中只能产生Object的27.4%/37.8%
DINO位移，随后DINO-only Action效应同步下降到28.6%/47.3%。这说明当前像素图案学到的是
Object-context-specific的DINO扰动，而不是跨场景稳定的DINO触发模式。动作头/checkpoint方向差异
可能仍有次要贡献，但现有结果不再支持把它作为第一解释。

这也暴露了第16.4节原始`DINO-CF-CVaR`的不足：若只最大化Object中的DINO-only Action效应，仍可能
继续强化源域特定像素方向。后续正式方法应同时包含source-only的跨状态/EOT DINO扰动下尾约束和
跨context方向一致性，使补丁在不同Object背景、阶段和变换下都产生稳定DINO位移；DINO-CF负责保证
该位移能通过因果通路改变动作。由于该修正来自Spatial/Goal诊断，二者后续只能作为development
target；最终可迁移性主张必须由未参与该修正的Long或新增模型/任务确认。当前不自动启动训练。

## 18. Context-Invariant DINO-CF 单源正式方法设计（v1.11）

### 18.1 科学问题与威胁模型

科学问题：只访问一个源suite时，能否把已验证的Wrist-DINO因果脆弱性转化为跨场景、跨OFT
checkpoint仍有效的通用补丁，而不是记忆源suite的视觉背景？

- Source knowledge：LIBERO-Object OFT checkpoint、Object Wrist图像、语言和动作输出；允许白盒梯度。
- Target knowledge：训练、checkpoint选择和超参数选择期间不访问Spatial、Goal、Long权重、图像或梯度。
- Intervention：一张共享50×50 RGB补丁，仅作用于Wrist输入；Front、语言、proprioception和模型权重不变。
- Transfer setting：同一补丁直接迁移到suite-specific OFT checkpoints，不做target adaptation。
- 方法定位为source-white-box / target-zero-shot cross-checkpoint transfer，不宣称完全model-agnostic。

Spatial/Goal已参与失效诊断，后续只作为development targets；Long不进入方法实现或选择，作为保留
确认目标。若需要最强的完全未见泛化主张，后续必须增加新的checkpoint、模型或benchmark。

### 18.2 数据划分与独立单位

- Source train：复用冻结`GEOMETRY_TRAIN_BANK.pt`，Object 10 tasks × 5 trajectories × 6 phases；
  其state主要位于10–17。
- Source validation：另取Object state 20–29中的clean-success轨迹，只做训练稳定性和源域读出，
  不用Spatial/Goal选择checkpoint。
- Source final：Object state 40–49，沿用Final-100口径。
- Development transfer：Spatial/Goal state 40–49，用于解释改进来自哪一层，不作为完全未见证据。
- Confirmation transfer：Long state 40–49；补丁、训练步数和报告口径在查看新方法Long结果前冻结。

phase、EOT draw和action chunk均为trajectory/state内重复测量；统计先phase→state，再state→task，
不能把它们当独立样本扩大样本量。

### 18.3 训练批次与source-only context构造

每个optimizer update按固定seed抽取两个不同task的源trajectory，每条包含6个阶段，共12个source
contexts。每个context独立采样一次Wrist patch EOT：

- translation：横纵各`[-8,+8]` pixels；
- scale：`[0.85,1.15]`；
- rotation：`[-15°, +15°]`；
- brightness/contrast：各`[0.85,1.15]`；
- 变换后仍限制在Wrist图像内，不改变Front和proprioception。

该EOT用于在单一Object suite内部制造位置、成像和局部背景变化，不使用任何目标suite图像。

### 18.4 四个可解释目标

对第i个source context，记clean与patched DINO/SigLIP表征为
`z_D^0,z_D^P,z_S^0,z_S^P`，动作尺度标准化后定义：

1. **DINO反事实动作效应**

\[
J_{CF}=\mathbb E_i\,D\big(F(z_D^P,z_S^0),a_i^0\big).
\]

只允许patched DINO进入动作前向，直接优化已验证的因果路径。

2. **DINO下尾位移**

\[
m_i=\operatorname{RMS}(z_{D,i}^P-z_{D,i}^0),\qquad
J_{tail}=\operatorname{mean}\big(\text{lowest-25\%}\{m_i\}\big).
\]

优化一个batch中最弱的3/12个context，避免均值被少数Object高增益状态主导。

3. **跨context方向一致性**

\[
u_i=\frac{z_{D,i}^P-z_{D,i}^0}{\|z_{D,i}^P-z_{D,i}^0\|_2},\quad
p=\operatorname{stopgrad}\left(\frac{\sum_i u_i}{\|\sum_i u_i\|_2}\right),\quad
J_{cons}=\mathbb E_i\cos(u_i,p).
\]

它要求同一补丁在不同任务、阶段和EOT下产生相近的DINO扰动方向，直接针对v1.10发现的
Object-context-specific问题。

4. **真实双分支动作效应**

\[
J_{full}=\mathbb E_i\,D\big(F(z_D^P,z_S^P),a_i^0\big),
\]

防止反事实效应无法在正常双编码器推理中实现。

完整方法最大化：

\[
J=J_{CF}+\lambda_TJ_{tail}+\lambda_CJ_{cons}+\lambda_FJ_{full}.
\]

实现时分别求四项对补丁的梯度，以`J_CF`梯度范数为参考进行固定比例缩放；记录梯度余弦、有效
比例和裁剪前后范数，但不根据Spatial/Goal结果调权重。暂定辅助梯度比例为tail 0.5、consistency
0.25、full-action 0.5；若源域训练数值不稳定，只能依据Object validation修改并完整记录。

### 18.5 训练条件与公平对照

所有新训练均从同一个`DINO-Norm-1500`初始化，使用同一Object bank、task-pair schedule、EOT draw、
AdamW、学习率和500 optimizer updates；主结果固定使用update 500，100/250仅作收敛曲线，不按
目标suite结果挑checkpoint。

| 条件 | CF | DINO tail | consistency | full action | 回答的问题 |
|---|---:|---:|---:|---:|---|
| Action-EOT | 0 | 0 | 0 | 1 | 普通动作攻击在同预算下能否迁移 |
| DINO-CF | 1 | 0 | 0 | 0 | 只利用因果路径是否足够 |
| DINO-CF-Tail | 1 | 1 | 0 | 1 | 防止DINO幅度塌缩是否有效 |
| CID-CF（完整方法） | 1 | 1 | 1 | 1 | context一致性是否带来额外迁移 |

冻结的DINO-Continue-B-500、DINO-Norm、Exploring和WHEN不重新训练，作为既有参照。Exploring/
WHEN只能标注为released-patch direct-transfer baseline；同预算因果比较主要依靠上述四个新条件。

### 18.6 分阶段评估

**Stage A：GPU峰值与数值可训练性**

对完整12-context update执行一次真实forward/backward/optimizer step，记录GPU peak、host MaxRSS、
各损失、四项梯度范数与余弦。该步骤只判断实现和数值是否有效，不根据攻击结果筛方法。

**Stage B：四条件Object单源训练**

四个条件各500 updates，可分别占用1张RTX 3090并行；训练过程不访问目标suite。保存100/250/500，
但primary固定500。

**Stage C：机制读出**

先在Object validation确认训练没有数值退化；随后在Spatial/Goal development targets上测DINO RMS、
DINO-only Action、restore reduction和context consistency。重点观察当前27.4%/37.8%的DINO传播比例
是否提高，以及提高是否转化为DINO-only Action，而不是仅增加SigLIP或非机制效应。

**Stage D：闭环评估**

只评估四个新条件的update 500，并复用既有Clean和冻结基线轨迹：Object、Spatial、Goal各100 states；
方法及分析冻结后再运行Long 100 states。每state内随机化四个新条件的顺序并精确reset。

### 18.7 指标与可支持的结论

Primary：

- Spatial/Goal development Macro-CASR与Worst-Suite CASR；
- Long confirmation CASR；
- CID-CF相对Action-EOT、DINO-CF及冻结DINO-Continue的task-paired差值。

Mechanism-linked：DINO RMS target/source ratio、DINO-only Action target/source ratio、DINO restore
reduction、restore-both closure，以及这些指标与闭环failure在task层级的方向一致性。

若CID-CF只提高Object而不提高Spatial/Goal，只能判定source mechanism enhancement；若提高
Spatial/Goal但Long不复现，只能判定benchmark-informed development improvement；只有Long或新增未见
目标也提高，才能支持single-source transferable mechanism attack。无论结果如何，不用四suite联合训练
补救，也不在本阶段加入扩散模型、纹理伪装或物理载体约束。

## 19. CID-CF 实现状态（v1.12）

第18节已在`oft_ramp_recovery_aware`中落地，但尚未运行GPU Stage A、尚未训练，也未提交任何作业。
实现包括：

- 四个冻结条件`Action-EOT`、`DINO-CF`、`DINO-CF-Tail`和`CID-CF`；
- 每次update从两个不同Object task各取一条六阶段trajectory，共12个source contexts；
- 每个context独立的平移、缩放、旋转、亮度和对比度Wrist EOT；
- 两遍式低显存更新：第一遍冻结最低3/12 contexts和stop-gradient一致性原型，第二遍逐context累加
  `J_CF`、`J_tail`、`J_cons`和`J_full`对patch的独立梯度；
- 以`J_CF`梯度范数为参照的固定辅助比例、全部梯度余弦/范数、裁剪前后范数和EOT范围日志；
- update 100/250/500原子checkpoint、每10 updates恢复点和单文件Parquet训练日志；
- Stage A真实12-context forward/backward/optimizer-step峰值与数值报告。

CPU静态编译、输入检查和全部38项契约测试已经通过。训练配置只包含Object checkpoint和source bank；
Spatial/Goal没有训练路径，Long仅以锁定名称存在且没有checkpoint或数据路径。当前唯一允许的下一执行是由
用户手动运行`bash submit_cid_cf_stage_a.sh`。该作业完成后必须先检查GPU峰值、Slurm MaxRSS、四项loss、
梯度范数/余弦和optimizer update是否有效，再依据实测资源生成Stage B四路训练提交脚本；不得提前提交
Stage B或访问Long。

## 20. CID-CF Stage A结果与Stage B资源冻结（v1.13）

Stage A job `134951`已完成。两个不同Object task的12个contexts均进入双遍前向、四项独立梯度、
梯度组合和一次真实optimizer step；未访问Spatial、Goal或Long。四项梯度和全部记录数值均有限，
patch update L2为0.169，说明optimizer step有效。`J_CF`、`J_tail`、`J_cons`和`J_full`分别为
9.236、1.891、0.488和12.566；总梯度L2为1.586，未触发10.0裁剪。辅助梯度缩放后严格满足相对
`J_CF`的0.5、0.25和0.5冻结比例。

RTX 3090峰值allocated/reserved显存分别为20.46/20.55 GiB，实测process MaxRSS为6.20 GiB。
将clean-SigLIP cache从Stage A的12项外推至完整300项后，host memory约6.35 GiB；乘1.25余量为
7.94 GiB。因此Stage B每路资源冻结为1张RTX 3090、4 CPU、8 GiB host memory、4小时；四个方法以
array `0-3%4`并行，未超过5卡上限。按Stage A单update 18.08秒估算，每路500 updates约2.5小时，
正式预期2.5–3.5小时，不含排队。

Stage B提交链只运行四路Object-only训练，不自动进入Stage C/D，也不访问Spatial、Goal或Long。
v1.13时的状态为提交脚本已通过契约测试、等待用户手动运行`bash submit_cid_cf_stage_b.sh`；
该训练随后已完成，现行状态见第21节。

## 21. CID-CF Stage B结果与Stage C冻结协议（v1.14）

### 21.1 Stage B训练完整性

Array job `134952`的四路方法均完成500/500 Object-only updates；每路500行训练记录、6000次context
backward exposures，共2000行和24000次exposures。四路均保存update 0/100/250/500 checkpoint，最终
patch hash与完成报告一致；无重复update、NaN、Inf、traceback、OOM或残留resume文件。训练数据、
task-pair schedule和EOT draws在方法间完全一致。GPU allocated峰值为20.48 GiB，每路训练约
1.74–1.76小时；Slurm控制端DNS故障导致正式MaxRSS暂不可查询。

训练contexts上，最后100 updates相对Action-EOT的描述性均值显示：DINO-CF-Tail将DINO下尾提高
0.147，CID-CF提高0.150；CID-CF将方向一致性提高0.0218。CID-CF相对DINO-CF-Tail的一致性增量为
0.0106，而CF、下尾和full-action基本保持。该结果只证明优化目标按预期工作，不作为独立迁移证据，
也不据此选择checkpoint；四路primary仍固定为update 500。

### 21.2 Stage C机制读出冻结协议

Stage C在查看新读出结果前冻结如下：

- 方法：四个新方法update 500，并加入冻结`DINO-Continue-B-500`作为同状态机制基线；不重新训练基线。
- Object validation：每任务在state 20–29中按state index顺序选择前两个可复现clean-success轨迹；
  选择只依赖clean结果，不依赖任何patch读出。
- Spatial/Goal development：复用DINO反事实诊断已冻结的每任务两个clean-success states 40–49；
  不根据新方法结果更换状态。
- 每个state取六个phase proxy；固定50×50 Wrist-only位置`(26,137)`，不在读出时使用EOT，保证与
  v1.10固定位置诊断可比。
- 规模：`3 suites × 10 tasks × 2 states × 6 phases × 5 methods = 1800 rows`。
- 指标：DINO/SigLIP RMS、full/DINO-only/SigLIP-only Action L2、DINO/SigLIP restore reduction、
  restore-both closure、target/source ratio及DINO扰动方向一致性。
- 方向一致性同时报告task内12 contexts的局部值和全部120 contexts的suite级全局值；方向向量只在
  task shard artifact中保存，统计时合并。
- 统计先phase→state，再state→task；方法差值使用task-paired bootstrap，target/source ratio对两个
  suite的task分别重采样；10,000 draws、seed 20260907。

Stage C使用10个task shards、数组`0-9%5`，每个shard顺序处理Object、Spatial和Goal并按suite恢复。
完成后只运行CPU统计并停止；不得自动进入Stage D闭环或访问Long。实现与全部46项仓库契约测试已经
通过，当前等待用户手动运行`bash submit_cid_cf_stage_c.sh`。该句记录的是v1.14时点状态；Stage C现已完成。

## 22. Stage C结论与Stage D冻结边界（v1.15）

Stage C is complete (1,800 rows; restore-both closure exactly zero). The causal
readout supports a narrower conclusion than the original universal-transfer
hypothesis: source-only EOT/mechanism objectives reduce the Object-to-Spatial
collapse and improve Spatial action effects, but Goal does not show a stable
absolute improvement, and CID-CF has not separated reliably from Action-EOT at
the action level.

The user therefore changed Long from a mandatory confirmation gate to an
optional external probe. This is scientifically consistent with the already
weak Long causal contrast: a weak or null Long gain is interpretable and must
not veto the Object-to-Spatial/Goal result. The corresponding claim boundary is
now **benchmark-informed Object→Spatial/Goal development transfer**, not
universal unseen-suite transfer.

The next authorized experiment is Stage D closed-loop evaluation:

- freeze the four update-500 patches (Action-EOT, DINO-CF, DINO-CF-Tail,
  CID-CF); no optimization or retraining;
- evaluate Object, Spatial, and Goal only, ten tasks × states 40–49;
- reuse the existing Clean and frozen baseline rollout grid and its exact reset
  seeds; add only 1,200 new episodes;
- report suite-specific task→state bootstrap CASR, Spatial/Goal Macro-CASR,
  Worst-Suite CASR, CID-CF paired contrasts against DINO-Continue-B and all
  three new ablations, plus Object source retention;
- use at most five RTX 3090 GPUs and stop after CPU statistics;
- do not access Long automatically. Any later Long run requires a separate
  user decision and is non-gating.

The manual submission entry point is
`oft_ramp_recovery_aware/submit_cid_cf_stage_d.sh`.

## 23. CID-CF判负与H1/H4二因子主线（v1.16）

### 23.1 已有假设的正式判断

Stage D已完成1200个新增闭环episodes。CID-CF在Spatial/Goal上的Macro-CASR为17.56%，低于
Action-EOT的22.61%、DINO-CF的23.00%和DINO-CF-Tail的20.00%；相对三者的配对差值置信区间均跨
零。同时四个新方法的Object CASR仅12–20%，显著低于冻结DINO-Continue-B-500的74.33%。因此：

- “CID方向一致性可带来独立闭环增益”的核心假设不成立；CID-CF不再作为后续主方法；
- EOT/CF带来的Spatial改善是旁支发现，不能用来替代CID-CF的原假设；
- 不继续围绕CID权重、tail比例或跨suite均衡做局部修补；
- 后续重新以闭环攻击成功率为首要目标，借用Exploring已验证的UADA/UPA攻击逻辑，并加入已验证的
  Wrist-DINO因果通路。

WHEN/PAD注意力假设、双视角语义—控制分工假设均因缺少必要中间证据而暂停。关于背景、光照、几何
导致跨suite衰减的具体归因也未被验证，因此不启动最坏背景搜索或RAUP实验。

### 23.2 两个独立、可证伪的假设

**H1：DINO-routed UADA。** Exploring式动作远边界攻击若同时保留完整动作主目标，并用DINO-only
反事实分支引导其走Wrist-DINO因果通路，将比不带DINO路由的OFT-UADA产生更高Object闭环CASR。
DINO项只能作为辅助，不能替代完整动作目标；若CASR不高于OFT-UADA，H1判负。即使CASR提高，若
DINO restore不能消除新增动作偏差，也只能判方法有效而不能支持H1机制。

**H4：phase-conditioned UPA。** 对末端`xyz`轨迹方向和距离进行阶段条件化攻击，将比逐动作维度远
边界产生更高Object闭环CASR。现有phase读出没有显示统一的contact优势，因此六个phase等量采样，
不得预设contact/post-contact权重更高；pre阶段攻击`xyz`轨迹几何，contact/post阶段在`xyz`之外加入
gripper反边界。若CASR不高于OFT-UADA，H4判负。

### 23.3 冻结的2×2条件

| 条件 | 输出攻击逻辑 | DINO路由辅助 |
|---|---|---|
| `oft_uada` | 完整OFT action chunk逐维反向数据边界 | 无 |
| `dino_uada` | 与`oft_uada`相同 | 有 |
| `phase_upa` | phase-conditioned xyz方向/距离与gripper转换 | 无 |
| `dino_phase_upa` | 与`phase_upa`相同 | 有 |

所有条件从同一DINO-Norm-1500 patch开始，只访问Object训练bank，使用相同的两个不同task/12-context
update schedule、AdamW、学习率和500 updates；保存100/250/500但primary固定为500。第一轮固定
Wrist位置`(26,137)`且不开EOT，以免把迁移鲁棒性混入H1/H4。四个条件都执行full与DINO-only两条
前向以匹配计算预算；DINO条件将route梯度投影去除与full主梯度冲突的分量，再按full梯度范数的0.5
加入。UADA/UPA在OFT action-head归一化输出空间计算；每个DoF的反向边界和尺度冻结为300个训练
contexts共2400个clean action rows的q01/q99，不混用反归一化物理动作统计。训练不访问Spatial、Goal
或Long。

### 23.4 主要比较、终点与停止规则

第一终点仅为Object Final-100 clean-success条件下的task-equal闭环CASR：

1. H1 primary：`dino_uada - oft_uada`；
2. H4 primary：`phase_upa - oft_uada`；
3. interaction secondary：`dino_phase_upa - dino_uada`及`dino_phase_upa - phase_upa`；
4. practical benchmark：四个新条件分别与冻结`DINO-Continue-B-500`比较。

复用已有Final-100的Clean、DINO-Norm、DINO-Continue-B、Exploring和WHEN结果及其精确rollout seed，
只新增四个方法共400 episodes。state内随机化四个新条件顺序；统计保持task→state层级bootstrap、
10,000 draws、seed 20260907。若没有新条件超过DINO-Continue-B-500，则不扩展到Spatial/Goal/Long，
也不做额外机制读出。只有CASR成功的候选才进入DINO restore或phase几何机制验证；机制指标不能替代
CASR Gate。正式Slurm仍由用户手动提交，最多并发4张训练GPU或5张评估GPU。

## 24. OFT-UADA结果与机制—迁移验证（v1.17）

### 24.1 H1/H4结果和结论边界

四条件均完成500 Object-only updates及Final-100。`oft_uada`和`dino_uada`在99个clean-success states
上均达到100% task-equal CASR；`phase_upa`为99%，`dino_phase_upa`为98%。相对冻结
`DINO-Continue-B-500`的差值分别为`+25.67 pp [10.44,42.56]`、
`+25.67 pp [10.67,43.22]`、`+24.67 pp [9.00,42.44]`和
`+23.67 pp [9.00,41.11]`。

预注册H1与H4均不成立：`dino_uada - oft_uada = 0`，显式DINO辅助未提高Object终点；
`phase_upa - oft_uada = -1 pp [-4,0]`，阶段轨迹目标未优于逐DoF反边界。组合也无额外收益。因此
后续冻结的实用主方法为更简单的`oft_uada_u0500`，`dino_uada_u0500`仅作为机制/迁移对照，不得把
Object天花板解释为H1成立。

这里验证的是将Exploring/UADA的反向动作边界原理重新实现于OpenVLA-OFT连续动作头并在Object OFT
模型上原生训练；不等于将论文released patch直接迁移到OFT，也不等于已证明所有suite都达到高CASR。

### 24.2 Stage M：Object因果机制读出

目的：判断普通OFT-UADA的100%攻击是否自然利用已发现的Wrist-DINO通路，而非继续选择或修改补丁。

- 方法：冻结`oft_uada_u0500`、`dino_uada_u0500`和`dino_continue_b_u0500`；不训练。
- 数据：严格复用CID-CF Stage C已选中的Object validation states及rollout seeds；10 tasks × 2 states，
  每state六个phase，共`10×2×6×3=360` rows。
- 读出：DINO/SigLIP RMS、full action、DINO-only action、SigLIP-only action、DINO/SigLIP restore
  reduction、restore-both closure，以及DINO restore fraction。
- 统计：先phase→state，再state→task；主要描述`oft_uada`自身的DINO restore fraction，并成对比较
  `dino_uada - oft_uada`与`oft_uada - dino_continue_b`；10,000 task bootstrap、seed 20260907。
- 解释：若OFT-UADA的DINO restore effect很大，只能说明UADA自然使用DINO通路；显式DINO辅助仍因
  Object CASR无增益而不能宣称有效。若DINO restore effect小，则不得把UADA强度归因于DINO。

Stage M是解释性读出，不能替代闭环CASR，也不依据读出结果修改现有补丁。

### 24.3 Stage X：Spatial/Goal零样本闭环

目的：检验Object上100%的OFT-UADA能否迁移，以及显式DINO路由是否只在迁移端产生价值。

- 方法仅为`oft_uada_u0500`和`dino_uada_u0500`，补丁及checkpoint均冻结。
- 训练suite固定为Object；评估suite固定为Spatial和Goal；Long仍为非门控可选探针且本阶段不访问。
- states固定40–49；精确复用既有cross-suite Clean的reset seed与clean-success eligibility。
- 只新增`2 suites × 10 tasks × 10 states × 2 methods = 400 episodes`；Clean、DINO-Continue、
  Exploring、WHEN等旧基线不重跑。
- 第一终点：各suite task-equal CASR和Spatial/Goal suite-equal Macro-CASR。
- 主要比较：`dino_uada - oft_uada`的各suite及Macro差值；实用比较为两者各自相对
  `dino_continue_b_u0500`。Worst-Suite作为辅助终点。
- 若OFT-UADA本身跨suite高且DINO-UADA无增益，保留简单OFT-UADA；若只有DINO-UADA迁移更强，
  才支持“DINO路由的价值位于迁移端”这一新结论；若两者均弱，则停止并判定Object强度不迁移。

Stage M与Stage X分别提交、分别停止。Stage X不得因Stage M结果修改方法、checkpoint、state或主要比较；
两阶段均不自动启动Long、EOT、物理实验或新一轮训练。

## 25. UADA跨suite层级失效诊断（v1.18）

### 25.1 新证据与尚未验证的归因

Stage M完成360/360行且restore-both精确闭合。`oft_uada_u0500`在Object上的DINO restore fraction为
87.83% `[84.97%,90.37%]`，DINO-only/full action比为98.21% `[92.77%,103.55%]`，说明未显式加入
DINO辅助的UADA仍自然利用Wrist-DINO因果通路。`dino_uada - oft_uada`的restore fraction为
`+0.60 pp [-0.79,2.05]`，完整Action L2差值也跨零；显式DINO辅助没有建立独立优势。

Stage X完成400/400个新增闭环episodes。OFT-UADA在Spatial、Goal的task-equal CASR分别为12.22%和
22.67%，DINO-UADA为13.22%和19.78%；两者Macro差值为
`dino_uada - oft_uada = -0.94 pp [-5.50,3.28]`。因此Object上的100%强度没有稳定迁移，显式DINO
路由也没有挽救迁移。Long未访问且不是本结论的门槛。

但现有数据只在Object读取了UADA的DINO因果效应。此前Spatial/Goal的“像素→DINO位移衰减”来自
`DINO-Continue-B-500`，不能未经验证直接归因给OFT-UADA。下一步必须区分：

1. 浅/中层DINO响应已经跨场景塌缩；
2. 中层仍稳定、只在DINO后层表征塌缩；
3. 最终DINO位移保留，但DINO-only Action映射跨checkpoint衰减；
4. DINO-only Action仍保留，但没有转化为持续闭环失败。

本地checkpoint核查显示Object、Spatial、Goal均使用`dinosiglip-vit-so-224px`，但抽查的多个DINO与
fused视觉Transformer中间attention权重并不相同，最大绝对差约0.034–0.038；部分position/register
参数相同。因此诊断同时面对输入分布变化、视觉权重变化和动作映射变化，不能把问题简化为背景变化。

### 25.2 冻结诊断设计

本阶段仅做解释性层级读出，不训练、不更新、不选择新补丁：

- 冻结方法：`oft_uada_u0500`与`dino_uada_u0500`；二者同状态配对。
- suites：Object、Spatial、Goal；Long不进入配置、模型加载或结果文件。
- 独立单位：沿用CID-CF Stage C冻结的每suite每task两个clean-success states及精确rollout seeds；
  每state的六个phase只是重复测量，先聚合为state，不能作为独立样本。
- 规模：`3 suites × 10 tasks × 2 states × 6 phases × 2 methods = 720 rows`。
- Object也重新读取中层激活，因为Stage M只保存最终DINO/SigLIP，不能为中层target/source ratio提供分母。
- 最终因果指标：DINO/SigLIP RMS、full/DINO-only/SigLIP-only Action L2、restore reductions、
  DINO restore fraction和restore-both closure。
- 层级指标：DINO blocks 5、11、17、23的block-output RMS，以及相同block attention-qkv线性输出按
  最后一维三等分得到的q/k/v RMS。四层在查看任何新结果前按24-block架构的近似四分位固定，不能事后
  挑选最好看的层。
- 统计：phase→state后进行task→state bootstrap；每suite报告两方法均值及
  `dino_uada - oft_uada`配对差值。Spatial/Object、Goal/Object ratio在task层级分别重采样，
  10,000 draws、seed 20260907。

### 25.3 解释分支与停止规则

- 若早期block及q/k/v ratio已经与最终DINO一样低，支持“像素模式跨场景激活失败”，后续才考虑
  global/local self-universality；
- 若早期ratio较高而随深度下降，支持“后层源checkpoint过拟合”，后续才考虑base-anchored、
  aggregate-gradient中层DINO目标；
- 若最终DINO ratio较高但DINO-only Action ratio低，瓶颈在DINO→Action映射，不应继续增强DINO位移；
- 若DINO-only Action ratio仍高而闭环CASR低，瓶颈在动作效应→持续任务失败，才考虑连续轨迹目标。

这些分支是机制解释，不是预设数值PASS阈值。Spatial/Goal已经参与路线发现，只能称development
diagnostic；任何后续方法不得把它们重新称为完全未见确认。完成统计后必须停止，不自动实现
base-anchored/FIA/self-universality方法，不访问Long，也不追加闭环实验。

## 26. Aggregate-Mid-DINO / Base-Anchor UADA（v1.19）

### 26.1 层级诊断结论

720/720行层级诊断完成且restore-both精确闭合。OFT-UADA在Spatial中从block 5的71.4% Object保留率
逐步下降至block 23的48.6%、final DINO的46.3%、DINO-only Action的25.7%；Goal对应为81.2%、
64.2%、62.2%和42.1%。q/k/v三类facet均随深度衰减，v仅略高，不支持事后选择单一facet。

因此浅层像素激活并未完全失效，主要瓶颈包含两个连续环节：DINO后层的suite/checkpoint特异衰减，
以及final DINO→Action映射的进一步衰减。SigLIP RMS在Spatial/Goal仍为Object的88.0%/109.2%，说明
衰减具有DINO分支特异性。下一方法必须同时保留UADA动作主目标并攻击跨变换、跨base/source视觉塔
稳定的动作重要中层特征；不得退回DINO norm、固定q/k/v facet或final-layer方向一致性。

### 26.2 两个可证伪假设与三个冻结条件

**H-AG：Aggregate-gradient mid-DINO。** 在Object源数据上，对DINO blocks 5/11/17的UADA动作目标
关于中层输出的梯度，在四个随机mask视图上先分别单位化再聚合。该aggregate importance应抑制
context-specific梯度噪声。若`AG-Mid-UADA`不提高Spatial/Goal Macro-CASR，则H-AG判负。

**H-BA：Base anchor的增量价值。** 在相同aggregate importance下，让补丁同时通过Object-OFT DINO
和原始OpenVLA base DINO中层产生同向投影，若`BA-AG-Mid-UADA`不高于`AG-Mid-UADA`，则H-BA判负。
base只提供冻结视觉塔，不使用base动作头、target checkpoint、target图像或target统计。

条件固定为：

1. `oft_uada_u0500`：已有冻结对照，不重训；
2. `ag_mid_uada`：UADA主目标 + 两路独立source-mask中层辅助；
3. `ba_ag_mid_uada`：UADA主目标 + 一路source-mask和一路base-DINO中层辅助。

AG与BA-AG都执行两个视觉塔辅助前向；AG第二路是独立source-mask sham，BA-AG将该路替换为base DINO，
避免把额外计算本身混入base-anchor比较。两路辅助先平均，再经冲突投影并统一缩放到UADA主梯度范数
的0.5；BA-AG不得获得更大辅助梯度预算。两种新方法从同一DINO-Norm-1500初始化，使用相同Object
training bank、task-pair/phase schedule、AdamW、学习率和500 updates；primary固定update 500。

### 26.3 Aggregate importance定义

对每个冻结Object训练context i、层l∈{5,11,17}和四个由seed固定的mask视图r：

\[
g_{i,l,r}=\nabla_{z_{i,l,r}}J_{UADA},\qquad
w_{i,l}=\operatorname{normalize}\left(\frac{1}{4}\sum_r
\operatorname{normalize}(g_{i,l,r})\right).
\]

mask保持token空间对应，只对Wrist clean图像中非补丁区域做10% Bernoulli pixel drop；不改变Front、语言、
proprioception或clean action target。importance在训练前一次性冻结为float16 artifact，mask重复不是独立
统计单位。训练中对patched中层输出使用`<w,z^P>`的有符号投影；clean项对patch梯度为常数，故无需
重复缓存clean中层输出。blocks 5/11/17作为预先固定的多层集合共同使用，不按Spatial/Goal结果挑层。

### 26.4 分阶段执行与资源门

1. **Stage A1：importance bank。** 10个task shards，每task 30 contexts、每context四个mask、三层；
   输出300个context directions及完整hash/finite/shape检查。
2. **Stage A2：真实单update probe。** 对AG与BA-AG分别运行同一12-context真实forward/backward/
   optimizer step，验证base/source形状对应、两路梯度非零、冲突投影、patch确实更新，并记录RTX 3090
   peak allocated/reserved、process MaxRSS和耗时。Stage A失败时停止，不生成正式训练提交。
3. **Stage B：Object-only训练。** Stage A通过后才生成/提交两路500-update训练；不访问任何target。
4. **Stage C：机制读出。** 冻结update 500，在Object/Spatial/Goal读出中层ratio、final DINO、
   DINO-only/full Action和restore closure。
5. **Stage D：主要闭环。** Object/Spatial/Goal；复用既有OFT-UADA/Clean/基线，只新增两个方法共
   `3 suites × 10 tasks × 10 states × 2 = 600 episodes`。Primary为Spatial/Goal suite-equal
   Macro-CASR；H-AG比较AG−OFT-UADA，H-BA比较BA-AG−AG；Worst-Suite辅助，Object单列源域保持。
6. **Stage E：Long次要外部测试，必须执行。** 在方法、update、统计和结论口径冻结后，对
   OFT-UADA、AG、BA-AG运行`10 tasks × 10 states × 3 = 300 episodes`。Long单列并报告相同配对差值，
   但不得用于选择方法/checkpoint，不能替代Spatial/Goal主结论，也不能因阴性结果否决已成立的主比较。

Stage D与Stage E可在补丁完全冻结后顺序执行；不得因为Stage D好坏取消Long。Spatial/Goal和Long均已在
历史工作中被观察，Long在本版本的科学角色是“必报的次要外部异质性测试”，不是完全未见确认集。
所有闭环统计以task→clean-success state为独立层级；phase、mask、action chunk和重复前向不得作为
独立样本。当前只授权实现并手动提交Stage A，Stage B–E不得由Stage A依赖链自动启动。

### 26.5 Stage A结果与Stage B资源冻结（v1.20）

Stage A1已生成10/10 task shards、300/300 Object context importance directions并通过finite、shape、单位范数
和hash检查。Stage A2在RTX 3090上完成AG与BA-AG各一次真实12-context update：AG/BA-AG峰值reserved
显存分别为17.62/18.19 GiB，process MaxRSS均为6.65 GiB，单update模型加载后耗时分别为6.68/5.71秒；
两路主梯度、辅助梯度和patch update均非零有限。冲突投影后两条件的辅助/主梯度有效范数比均精确为0.5。

base与Object-OFT的blocks 5/11/17 attention-qkv最大绝对参数差分别为0.03015、0.02832、0.02588，
所以base anchor是实质不同的冻结视觉塔。上述结果只建立实现和资源可行性，不支持H-AG或H-BA，也不得
比较不同视觉塔的原始辅助objective数值来判断方法优劣。

Stage B据此固定为两个独立GPU作业，每方法500 Object-only updates、12 contexts/update，申请1张RTX 3090、
12 GiB host memory和3小时上限；保存updates 100/250/500及断点。训练完成后CPU audit只核验500行、数值、
辅助预算、phase平衡和最终patch hash，然后停止。Stage B依赖链不得加载Spatial、Goal或Long，也不得自动
启动Stage C–E。

### 26.6 Stage B结果与Stage C复用设计（v1.21）

AG与BA-AG均完成500/500 Object-only updates和6000次context backward exposures，最终patch hash分别为
`e0bbd8725867c5633e806b641b07384b2680c6fa1f553e44fd92422efc210081`与
`f53d8b7a678264fca6614b909976d0a0ac9cbc12ef687da7556518a5b74dfa4b`。两路全程finite、无梯度裁剪，
冲突率为53.4%/53.2%，最终饱和比例为11.05%/11.00%。这些是训练健康性证据，不是H-AG/H-BA结果。

Stage C固定复用既有UADA layer diagnostic中OFT-UADA的360行，不重跑reference；只为AG、BA-AG新增
`3 suites × 10 tasks × 2 states × 6 phases × 2 methods = 720`行。三方法合并1080行后先phase→state、
再state→task统计，报告blocks 5/11/17、final DINO、DINO-only/full Action、restore reduction/fraction及
restore-both closure。机制对比固定为AG−OFT-UADA与BA-AG−AG；Stage C不得选择或修改patch，也不得因
机制指标跳过Stage D或Long。其依赖链只到CPU统计，不访问Long、不跑闭环。

### 26.7 Stage C结论与闭环收束（v1.22）

Stage C完成720行新读出并复用360行OFT-UADA，共1080行，restore-both Action L2最大值为0。AG相对
OFT-UADA在Spatial的final DINO、DINO-only Action和full Action绝对效应分别降低0.0107、0.0553和
0.0555；Goal三项差值均跨零。部分target/source ratio的小幅提高主要伴随Object分母下降，不能解释为
target绝对增强。BA-AG相对AG进一步降低Spatial full Action 0.0535，并降低Goal final DINO、DINO-only
Action和full Action 0.0563、0.0587和0.0707。机制证据不支持H-AG，并对H-BA呈负面趋势；但两假设的
冻结判据仍是闭环CASR，所以不得以Stage C替代Stage D。

最终闭环固定为单一手动依赖链。Stage D只新增AG/BA-AG在Object、Spatial、Goal的600 episodes，复用
已有300 Clean、300 OFT-UADA以及DINO-Continue-B；primary为Spatial/Goal suite-equal Macro-CASR，
H-AG比较AG−OFT-UADA，H-BA比较BA-AG−AG，Object单列。Stage D统计成功后，不依据结果做任何选择，
直接执行Stage E：Long上新增OFT-UADA、AG、BA-AG共300 episodes并复用既有Clean。Long单独报告相同
配对差值，禁止选方法、替代或否决Spatial/Goal主结论。Stage E统计完成即关闭本实验分支，不自动设计或
训练下一攻击族。

## 27. AG/Base最终闭环与分支结论（v1.23）

### 27.1 Stage D主终点

Stage D完成600个新增闭环episodes。各方法在Object、Spatial、Goal上的task-equal CASR如下：

| 方法 | Object | Spatial | Goal | Spatial/Goal Macro | Worst-Suite |
|---|---:|---:|---:|---:|---:|
| DINO-Continue-B | 74.33% | 7.22% | 15.56% | 11.39% | 7.22% |
| OFT-UADA | 100.00% | 12.22% | 22.67% | 17.44% | 12.22% |
| AG-Mid-UADA | 98.89% | 4.22% | 20.56% | 12.39% | 4.22% |
| BA-AG-Mid-UADA | 96.89% | 14.22% | 19.67% | 16.94% | 14.22% |

冻结主比较为`AG-Mid-UADA - OFT-UADA = -5.06 pp [-17.11,3.89]`，不支持H-AG；
`BA-AG-Mid-UADA - AG-Mid-UADA = +4.56 pp [-3.94,16.22]`，不足以建立H-BA。结合Stage C中AG未提高
target绝对机制效应、BA-AG还进一步降低若干target效应，不能把比例分母变化表述为跨suite机制增强。

### 27.2 Long强制次要测试

Long在95个clean-success episodes上的task-equal CASR分别为：DINO-Continue-B 59.50%、OFT-UADA
57.14%、AG-Mid-UADA 59.50%、BA-AG-Mid-UADA 56.25%。`AG - OFT = +2.36 pp
[-3.00,8.67]`，`BA-AG - AG = -3.25 pp [-10.75,2.50]`，均未建立清晰差异。Long按预注册角色只说明
外部异质性：它必须报告，但不选择方法、不替代Spatial/Goal主终点，也不否决其他suite上的已成立结果。

### 27.3 正式收束

AG/Base机制增强分支判负并关闭。现有最简实用方法仍为OFT-UADA；其Object攻击强度和Wrist-DINO因果
中介已经建立，但跨suite攻击强度低且异质。后续不得继续以“给UADA再加一个DINO辅助损失”为默认路线，
也不得把OFT-UADA自然覆盖DINO机制包装成新方法贡献。新的科学问题应从“提高无方向失败率”转为：攻击者
能否利用已验证的Wrist-DINO→Action因果入口，对动作和闭环行为施加攻击者指定、可翻转的定向控制。

## 28. 应用场景解释备忘（v1.24，非实验约束）

本节只保存未来与老师讨论时可能使用的解释框架，不是当前主假设、威胁模型或方法选择的冻结条件。
其中的行业、攻击入口、危险事件和物理载体均可随机制实验结果及后续讨论修改；不得因为本节的例子，
提前把实验限定为某个应用、某种攻击载体或某类危险结果。

### 28.1 暂定解释示例

可暂以智能制造或仓储中的视觉语言机械臂作为解释示例：腕部相机提供操作对象、接触关系和末端附近局部几何，
图像经相机固件、ROS图像topic或边缘视觉节点送入OpenVLA-OFT，再由模型输出连续动作块。研究对象不是
“一张贴纸能否让机器人失败”，而是：攻击者不能直接控制机器人动作时，若获得最关键局部感知入口的有限
控制权，是否仍能借助Wrist-DINO因果通路影响机器人行为。该叙述目前只帮助说明研究意义，不决定实验终点。

### 28.2 候选部署入口

未来可讨论的部署入口包括相机固件、ROS图像topic、边缘预处理/压缩节点及远程运维视觉接口。它们只能
用于解释“为什么Wrist视觉通路可能形成攻击面”，当前机制实验仍可采用白盒反事实干预或数字像素扰动，
不因这些部署入口而提前冻结现实攻击者权限。

屏幕、投影、结构光、恶意标签、反射材料或数字传感流都只是候选载体。当前不要求使用补丁，也不要求现在
选定物理实现；以后若选择具体载体，必须另行冻结权限、成像条件和结论边界。

### 28.3 可保留的解释原则

完全遮挡、黑屏或断流不能证明攻击者利用了Wrist-DINO通路。无论以后选择何种应用，机制结论都必须来自
DINO特异的干预与恢复证据；隐蔽性、物理性和危险性是后续独立维度，不能由低像素范数或普通CASR代替。

### 28.4 候选应用后果（暂不进入当前实验）

以后若需要把机制结果映射到应用，可考虑下列安全相关事件：

1. 抓取成立后定向松爪并造成掉落；
2. 物体进入预先冻结的错误区域或污染区域；
3. 末端执行器进入预先冻结的禁入区；
4. 在关键阶段连续若干控制步向危险方向运动。

这些例子目前不作为训练目标、主要终点或Stage Gate。只有在用户后续选定应用解释后，才另行定义自动状态
判据、触发窗口和指标；不得看完rollout后反向挑选最像“危险”的案例。

### 28.5 环境结果的暂时解释

已有Object 100%、Spatial 12.22%、Goal 22.67%、Long 57.14%的OFT-UADA结果只说明固定通用像素模式
具有明显suite异质性，不能被表述为稳定跨环境攻击。它可以作为未来讨论适应性或部署风险的背景，但不自动
把“环境适应”升级为下一实验的主要任务。Long仍应在正式跨suite测试中报告，但不承担主方法选择职责。

### 28.6 当前边界

当前只能围绕Wrist-DINO因果机制提出和检验实验结论。没有完成具体应用实验前，不宣称任意现实攻击者、
物理落地或灾难性后果。应用解释以后单独整理，不反向改变已经冻结的机制证据。

## 29. 下一阶段待确认设计：Wrist-DINO定向可控攻击（v1.24）

本节只设计如何利用既有因果机制，不预设应用场景、物理载体或“灾难”定义，不构成实现或提交授权。

### 29.1 核心问题

已有证据说明无方向的OFT-UADA会自然经过Wrist-DINO通路，但这不足以形成新贡献。下一问题是：

> Wrist-DINO→Action是否只是被动传递任意扰动，还是一个可以被攻击者主动选择方向、产生相反输出并在
> 完整模型与闭环中保持的可控攻击通道？

“攻击者指定”用成对相反的标准动作方向定义，不绑定具体应用。第一轮固定为归一化动作空间中的
`±x、±y、±z`以及gripper open/close方向；rotation方向作为后续扩展，不在第一轮看结果后挑选。
对同一clean观测分别指定`+d`与`-d`，若输出只能变大但不能随指令翻转，就不能称为定向可控。

### 29.2 四个可区分、可证伪的假设

**H-C（DINO潜空间可控性）。** 在分支内相同相对RMS预算下，Wrist-DINO定向干预相对DINO随机/打乱
方向干预，能在完整动作输出中产生更高的目标方向投影，并能对同一state实现`+d/-d`符号翻转。SigLIP
定向干预作为分支特异性参照单独报告，不把两种不同维度表示的原始范数直接当作公平尺度。该假设只检验
因果通路的控制上界，不代表像素攻击已经实现。

**H-R（Wrist输入可实现性）。** 在固定Wrist像素预算下，通过DINO-only动作路径优化得到的输入扰动，能
把完整模型输出推向指定`d`，且优于等预算随机扰动和方向打乱sham。若潜空间成立而像素端失败，只能说明
通路可控但当前输入载体不可达。

**H-M（DINO机制特异性）。** 对已成功定向的同一攻击样本，恢复clean Wrist-DINO应显著关闭目标投影，
恢复clean Wrist-SigLIP不应产生同等关闭，restore-both应闭合剩余动作差异。若像素攻击成功但该中介关系
不成立，只能称Wrist定向攻击，不能称利用了DINO机制。

**H-CL（闭环转化）。** 连续施加同一攻击者指定方向时，单步DINO定向控制能够转化为相同符号的累计
末端位移或gripper状态改变，而不只是一次动作向量扰动。普通任务CASR是辅助结果，不替代方向一致的闭环
行为指标。应用意义在H-CL之后另行解释，不预先写入本假设。

### 29.3 攻击形式与目标函数

第一轮不使用固定patch，也不训练生成器；采用每个观测独立优化的Wrist-only有界数字扰动，作为检验像素
可实现性的最直接载体。它不要求跨观测通用，因此不能与通用补丁混称同一威胁模型。若机制闭环成立，后续
再讨论是否蒸馏为生成器、稀疏区域、补丁或物理载体。

令clean完整动作为`a_0`，攻击者指定的单位方向为`d`。`a`在冻结的OFT归一化动作空间中表示完整
action chunk，`d`把同一轴和符号重复到chunk各时间位后再单位化；gripper正负号在A0由controller定义核验。
主要目标不是扩大总动作L2，而是提高
`T=(a_adv-a_0)^T d`，同时约束正交泄漏`||P_perp(d)(a_adv-a_0)||`。DINO-path条件在优化时固定clean
SigLIP表示，只允许梯度经Wrist-DINO→Action传播；最终一律在未做任何restore的完整模型上评估。这样
“通过DINO设计攻击”与“事后发现DINO参与”在实验上可被区分。

### 29.4 分阶段实验

1. **Stage A0：接口与尺度审计。** 只确认归一化动作坐标、DINO/SigLIP注入位置、`+d/-d`定义、特征与
   像素范数尺度及restore闭合；不选择最好看的动作轴。
2. **Stage A1：潜空间可控性上界。** 在同一批冻结clean states上执行DINO-target、DINO-shuffled、
   SigLIP-target和等范数随机特征干预。每个state配对测试全部预注册方向。主要终点为目标投影、符号命中率、
   正交泄漏和双向翻转率。H-C失败则停止，不进入像素攻击。
3. **Stage B：小规模像素可实现性。** 条件固定为clean、随机像素、direction-shuffled DINO-path、
   full-path targeted、DINO-path targeted；OFT-UADA只作为无方向攻击参考，不参与定向假设的公平主比较。
   两个targeted条件匹配`L_inf`、允许像素范围、初始化、优化步数和步长。先在Object development states
   调试唯一预算，再冻结算法。
4. **Stage C：机制中介确认。** 在与Stage B分离的states上，对成功与失败样本都做DINO restore、SigLIP
   restore和restore-both，避免只挑成功案例。报告目标投影的restore fraction以及DINO−SigLIP关闭差值。
5. **Stage D：短闭环转化。** 对`±x、±y、±z`记录累计末端位移投影，对open/close记录gripper状态变化；
   对同一initial state配对运行相反方向。只有方向符号、累计效应和DINO中介同时成立，才支持H-CL。
6. **Stage E：跨suite报告。** 冻结方法后在Object、Spatial、Goal分别报告H-R/H-M/H-CL指标；Long同样
   测试但作为必须报告的次要模块，不用于挑选方向、预算或方法，也不要求它一定增强。

### 29.5 指标与关键对照

- **Target Projection（主连续指标）**：`(a_adv-a_0)^T d`；越大表示越沿指定方向。
- **Directional Hit Rate**：目标投影超过预注册最小效应且符号正确的state比例。
- **Orthogonal Leakage Ratio**：非目标方向变化相对目标投影的大小，防止把无方向大扰动误称为定向控制。
- **Bidirectional Flip Rate**：同一state的`+d/-d`条件是否产生相反符号；这是区别于UADA的关键指标。
- **DINO Mediation Gap**：DINO restore关闭量减去SigLIP restore关闭量。
- **Closed-loop Signed Displacement**：短闭环状态变化在指定方向上的累计投影；gripper单独按open/close定义。

关键因果比较分别是：DINO-target对DINO-shuffled检验“指定方向是否有效”；DINO-target对SigLIP-target检验
“DINO通路是否更可控”；DINO-path pixel对full-path pixel检验“显式沿DINO设计是否损失或增强完整模型的
定向控制”；DINO restore对SigLIP restore检验实际中介。OFT-UADA只回答无方向失败强度，不能作为H-C、
H-R或H-M的阳性对照。

### 29.6 独立单位、公平性与停止规则

独立单位为task内的clean-success initial state；动作chunk维度、优化迭代、正负方向、控制步和restore前向
均为state内重复测量。方法和`+d/-d`在同一state上配对，task作为上层block，运行顺序用固定seed在state内
随机化。Stage B development states与Stage C/D确认states必须分离；所有阈值、预算和主要方向集合在进入
确认states前冻结。统计采用task→state层级bootstrap，按suite分别报告，并报告方向间异质性，不能把八个
方向当成八倍独立样本。

顺序停止规则为：H-C失败则否定当前定向可控假设；H-C成立但H-R失败则停止像素闭环；H-R成立但H-M失败
则保留一般Wrist攻击结论、否定DINO机制攻击；H-M成立但H-CL失败则结论止于单步动作控制。只有四层证据
依次成立，才进入具体应用、危险后果、载体与隐蔽性的下一轮讨论。

### 29.7 Stage A0实现状态（v1.25）

Stage A0已实现为单个Object冻结记录上的真实GPU接口探针，不重跑旧实验，也不检验H-C。输入严格复用
既有300-context训练bank的record 0；只加载Object checkpoint，不访问Spatial、Goal或Long。探针固定检查：

1. OFT动作块形状为`8×7`，坐标顺序为`x/y/z/roll/pitch/yaw/gripper`；第一轮方向集合固定为
   `±x/±y/±z/open/close`，同轴正负方向必须构成单位反向向量；
2. Wrist-DINO与Wrist-SigLIP真实Hook均触发并记录shape/dtype；用clean两分支替换时官方动作必须数值闭合；
3. 八个方向关于Wrist-DINO替换值的梯度均有限且非零，同轴正负目标的梯度余弦必须不高于`-0.999`；
4. 以clean DINO RMS的1%做一次直接特征步，仅用于确认bfloat16替换和官方动作端点可见，不是科学预算、
   不得用于选择方向或声称H-C成立。

新增配置、runner、合约测试及Slurm文件分别为`configs/DINO_CONTROL_A0.json`、
`scripts/run_dino_control_a0.py`、`tests/test_dino_control_a0_contracts.py`和
`slurm/46_dino_control_a0.sbatch`。5项A0合约测试及仓库全套94项测试均通过；check-only预检确认冻结bank
包含300条记录、probe action为`1×8×7`、clean DINO为`1×256×1024`，bank SHA256为
`7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8`。

正式A0仍须由用户手动运行`bash submit_dino_control_a0.sh`。该脚本只提交一个RTX 3090、1小时上限的GPU
作业，生成`reports/DINO_CONTROL_A0.json`后停止；报告明确写入`scientific_hypothesis_tested=false`和
`next_stage_authorized=false`。在检查A0的shape、闭合、梯度和显存前，不实现或提交Stage A1。

### 29.8 A0首次失败与同运行时基线修正（v1.26）

首次A0作业`135380`正常加载模型并完成clean双分支restore、八个方向的有限非零DINO梯度及直接特征步，
但在写报告前被`frozen bank clean action`与当前causal runtime动作头的逐值一致性门终止；最大差为
`0.0078125`。因此该作业没有正式报告，既不支持也不否定H-C。

核查确认bank中的clean action由历史differentiable wrapper预处理路径生成，而A0干预使用当前causal
runtime。冻结bank仍可合法固定观测、task/state和phase身份，但跨预处理路径的缓存动作不应作为当前干预
的clean反事实基线。修正版据此冻结以下规则：

1. bank只提供冻结观测及身份；bank action差异以
   `clean_action_head_max_abs_vs_frozen_bank_audit_only`记录，不参与PASS/FAIL；
2. clean action、定向DINO干预和所有restore必须在同一个causal runtime、同一个prepared输入上计算；
3. 同一runtime两次clean动作的最大绝对差必须不高于`1e-7`，clean双分支restore也继续严格闭合；
4. 不放宽梯度有限性、非零性、正负方向反平行或Hook触发门，不改变1% RMS仅作数值可见性检查的角色。

这不是对阴性科学结果调参，而是修复不应存在的跨实现基线混用。修正版仍运行同一个Object记录，不访问
新数据、不重跑旧实验，也不授权A1。

### 29.9 A0通过与Stage A1冻结实现（v1.27）

修正版A0作业`135384`通过：同runtime clean replay与restore-both最大动作误差均为0；四对正负目标的DINO
梯度余弦为`-0.999897`至`-0.999921`；1%相对DINO RMS的数值可见性步在八个方向均产生正确符号的官方
动作变化。峰值allocated/reserved显存约17.05/17.09 GiB。该结果只通过接口门，不把单状态直接干预解释为
H-C成立。

Stage A1据此冻结为Object-only的配对随机区组设计：10 tasks各选择3个不同clean-success initial states，
固定使用trajectory ranks 1/2/3并分别取pre/contact/post的第一个phase记录；A0使用的rank 0完全排除。
真正独立单位为task内state，八个方向和四个条件均为state内重复测量。方向仍为
`±x/±y/±z/open/close`；条件固定为`dino_target`、等RMS的`dino_shuffled`、等RMS的`dino_random`和
相对自身branch clean RMS匹配的`siglip_target`。正负方向共用同一基础干预并只翻转符号，避免sham本身
破坏双向配对。相对特征RMS固定为1%，沿用A0数值可见性预算，不查看A1结果选预算。

总规模为`10×3×8×4=960`行。主要连续终点为标准化动作空间的Target Projection；同时报告Orthogonal
Leakage Ratio、标准化Action L2、各方向效应及Bidirectional Flip。H-C冻结通过门为：DINO-target相对
DINO-shuffled和DINO-random在Target Projection及Bidirectional Flip两个成对差值上的task→state层级
bootstrap 95%区间下界均高于0。DINO-target对SigLIP-target只作branch-specific secondary，不作为H-C门。

新增`configs/DINO_CONTROL_A1.json`、`scripts/run_dino_control_a1.py`、
`scripts/analyze_dino_control_a1.py`、两份Slurm文件及`submit_dino_control_a1.sh`。A1以10-task GPU array
执行、最多并发5张RTX 3090，成功后才运行CPU统计；任一shard失败则统计不启动。A0/A1专项10项、仓库全套99项
测试及真实artifact check-only均通过。该链不实现像素攻击、不跑闭环、不访问Spatial、Goal或Long；统计完成
后必须停止并检查H-C，不能自动进入Stage B。实现与本冻结一致：30 states、960行，并额外保存标准化
Action L2；H-C程序门同时要求DINO-target相对shuffled/random的Target Projection与Bidirectional Flip
四个95%区间下界均高于0。

## 30. Exploring 源码基础上的 OFT-DINO 重启实验（v2.0）

### 30.1 科学问题与对照

在尽量保留 Exploring UADA 训练结构的前提下，只做 OFT 连续动作头和双视角输入所必需的适配，并检验加入已发现的 Wrist-DINO 机制是否改变攻击及其中介特征。

- `exploring_oft_uada`：Exploring UADA 对 OFT 的最小适配；
- `exploring_oft_dino_cf`：完全相同的完整模型 UADA 主目标，再加入 DINO-only 反事实 UADA 辅助项；辅助前向把 Wrist-SigLIP 恢复为 clean，因此辅助动作效应只能经受攻击的 Wrist-DINO 传播；
- 两路共享随机补丁、训练状态顺序、每次 EOT 随机种子、优化器和学习率；方法是唯一处理差异；
- 两路均从随机 `3x50x50` RGB 补丁开始，禁止加载 DINO-Norm 或任何旧学习补丁；
- 保留原复现 `maskidx=0`，只攻击平移动作 x 维；OFT 的 8 步 action chunk 对该维共同应用相反边界目标；
- DINO-only 反事实辅助梯度在第一次有效更新按主梯度范数的 0.3 倍冻结尺度，不看结果调参。

### 30.2 保留项与必要适配

保留 Exploring 的随机 RGB 补丁、AdamW、cosine warmup、外循环取新训练样本、inner-loop 内重复随机位置/仿射 EOT、每次 inner-loop 更新并 clamp、同种子可复现检查点等结构。

必要适配仅包括：OpenVLA 离散动作 token 换成 OFT 连续 action chunk；单视角输入换成 Front+Wrist 且只在 Wrist 贴补丁；离散 token opposite-boundary MSE 换成连续动作边界 MSE；加入 DINO hook 供机制分支使用。时间受限筛查用 batch=1，不把它包装成原论文完整规模复现。

训练状态文件只提供已冻结的干净 Object 观测与任务文本；缓存的旧 `clean_action`、`clean_dino`、方向和补丁均禁止进入新损失。每个 outer iteration 必须用当前 OFT checkpoint 现场重新计算 clean action 与 clean Wrist-DINO 锚点。

### 30.3 截止时间与规模

2026-09-20 22:37 至 2026-09-21 10:00 只允许完成最小可信闭环：两路各 200 个 outer iterations、每个 outer 50 个 inner updates，共 10,000 次 patch optimizer updates；两路并行各占一张 RTX 3090。旧完整复现为 2,000 outer、50 inner、batch 8，实际单卡运行 122,425 秒（约 34.0 小时），本轮不得声称达到该完整规模。

### 30.4 噪声失败门槛

检查点固定为 outer 0/25/50/100/200。结构门槛以旧 Exploring 轨迹为参考：outer 100 时初始相关系数必须低于 0.95 且 TV 比不得高于 0.97；outer 200 时相关系数必须低于 0.85 且 TV 比不得高于 0.90。任何一路不满足即标记 `STRUCTURE_GATE_FAIL`，停止把该补丁送入攻击效果评估。自动指标通过后仍必须人工查看补丁；若仍以大面积无结构高频噪声为主，同样判失败，不以 CASR 挽救。

### 30.5 截止前允许的结论

本轮只形成代码溯源、训练可运行性、结构化收敛和小规模同状态配对诊断。若时间允许，再做 Object 小规模闭环筛查。不得在明早前宣称 Final-100、跨 suite 泛化或最终机制贡献；这些必须在服务器恢复后按正式规模重跑。

## 31. 源码重启训练结果、Object筛查与Scale-500扩展（v2.1）

### 31.1 Outer-200训练结果与人工门

两路均完成200 outer×50 inner=10,000次patch更新。`exploring_oft_uada`训练耗时3989.11秒，
`exploring_oft_dino_cf`为7869.07秒；状态均为COMPLETE。outer 200时前者的初始相关系数/TV ratio为
0.59160/0.73122，后者为0.58603/0.72827，均通过预注册0.85/0.90结构门。两路补丁从随机彩色噪声
逐渐形成稳定弧形和带状结构。用户于2026-09-21完成人工查看并判定“起码不是全是噪声”，因此授权将
这两张outer-200补丁送入攻击效果筛查。结构通过不能替代CASR，当前仍不得声称攻击有效或DINO辅助更优。

### 31.2 机制利用的精确定义

普通路在patched Front+Wrist完整模型上计算DoF 0跨8步action chunk的opposite-boundary MSE。
DINO-CF路保留完全相同的完整模型主损失，并用相同EOT随机种子做第二条前向；第二条前向把attacked
Wrist-SigLIP替换为同一观测的clean Wrist-SigLIP，只保留attacked Wrist-DINO，因此其相同动作边界损失
对patch的梯度只经Wrist-DINO→Action传播。第一次有效更新把该梯度范数冻结为主梯度的0.3倍，之后不看
结果调权。该设计合理之处是机制项服务于真实动作攻击且具有反事实分支隔离；局限是它可能与普通UADA
梯度冗余，必须以配对CASR和后续DINO restore验证，不能用训练损失或补丁外观证明独立机制增益。

### 31.3 Outer-200 Object Final-100配对筛查

筛查固定复用既有Object Final-100的10 tasks×states 40–49共100个clean episodes及其reset seeds，不重跑
clean；只新增两张outer-200补丁的200个episodes。固定Wrist位置(26,137)，Front/语言/proprioception不变，
每个state内用seed 20260921随机化两种攻击顺序。CASR只在99个既有clean-success states上计算，统计为
task→state层级bootstrap 10,000 draws、seed 20260907。第一比较为
`exploring_oft_dino_cf_o0200 - exploring_oft_uada_o0200`；同时分别报告两路CASR和task异质性。本实验只回答
Object outer-200攻击效果，不支持跨suite或完整Exploring复现主张。

使用10-task array、最多并发5张RTX3090。根据既有同runner的40 episodes/task约9分钟实测，当前每task仅
20 episodes，预计每个5-task wave约5–7分钟，两波加模型加载和CPU统计约15–25分钟，不含排队。

### 31.4 Scale-500扩大训练

扩大实验从同一seed 42随机初始化重新训练，不从outer-200续跑。原因是outer-200的cosine scheduler已经在
200处退火到零，修改总步数后继续会改变预定义学习率轨迹并造成不可解释的断点。两路保持唯一处理差异，
规模扩大到500 outer×50 inner=25,000 updates，batch仍为1；保存0/25/50/100/200/500。该实验是2.5倍
时间扩展，仍不是旧完整复现的2000 outer、batch 8。

按outer-200实测线性估计，普通路约2.8小时，DINO-CF约5.5小时，两路并行由后者决定；Slurm上限7小时。
训练后自动生成结构报告，但Scale-500补丁仍需人工查看才可进入CASR，不把依赖链自动接到闭环评估。

### 31.5 代码继承边界与后续四suite测试

代码继承的严格表述为：`ExploringOFTOpenVLAAttacker`直接继承原
`white_patch/UADA.py::OpenVLAAttacker`并复用其`RandomPatchTransform`；OFT端口逐段保留原函数的随机
patch、AdamW/cosine、outer取新样本、inner随机几何变换、逐inner backward/step/clamp及outer scheduler
顺序。由于原函数的离散action-token labels、logits和ActionTokenizer不能用于OFT L1连续action chunk，
端口新增`patchattack_unconstrained_oft`并只替换模型输入、动作损失和checkpoint/恢复接口。不得表述成原
函数未经修改直接运行，也不得再实现独立于该继承链的新训练器。Scale-500继续使用完全相同端口文件，
只改变预注册outer总数和新增不影响优化的梯度诊断列。

Outer-200 Object筛查后，若这两张补丁具备非平凡CASR，其后续闭环一次性覆盖Object、Spatial、Goal和Long。冻结同一两张
补丁、固定Wrist位置、各suite各10 tasks×10 states，复用既有400个clean rollouts及reset seeds，只新增
`4 suites×10 tasks×10 states×2 methods=800`个攻击episodes。四suite可由40个suite-task shards以
array `0-39%5`执行；预计约60–100分钟，不含排队。统计按suite分别做task→clean-success-state层级
bootstrap；Object是source，Spatial/Goal是主要跨suite诊断，Long为必须报告的次要异质性模块。不得把
四suite简单池化成一个主平均数，也不得因Long较弱而删除或把它升级为否决其他suite结论的门槛。

### 31.6 Scale-500五路机制筛查（v2.2）

为避免普通UADA与单一DINO-CF结果相似后再事后设计方法，Scale-500在查看任何新CASR前扩展并冻结为
五路。五路全部使用同一`UADA_OFT_DINO.py`源码端口、同一完整UADA主目标、随机初始化、训练记录顺序、
EOT draws、AdamW/cosine和25,000 updates；唯一差异是第二条反事实梯度：

1. `exploring_oft_uada`：无第二机制梯度，原始动作攻击对照；
2. `exploring_oft_dino_cf`：attacked DINO+clean SigLIP对相反动作边界的平均损失，检验DINO路径充分性；
3. `exploring_oft_dino_cf_worst_chunk`：同一DINO-only路径，但最小化8步chunk中当前最大边界误差，检验
   是否能避免平均损失掩盖某些时间位、形成更持续动作块；
4. `exploring_oft_dino_restore`：完整攻击逼近相反边界，同时clean-DINO+attacked-SigLIP反事实应恢复到
   clean action，检验DINO作为攻击必要中介而非仅可用分支；
5. `exploring_oft_siglip_cf`：clean DINO+attacked SigLIP仍逼近相反边界，作为同额外前向和同0.3梯度
   预算的分支对照。若它不弱于DINO路由，不支持DINO特异优势。

四个机制/对照路都在第一次有效更新按主梯度范数0.3冻结辅助尺度，同时记录主/辅助梯度余弦和实际缩放
范数比。预注册闭环比较为DINO-CF−UADA、WorstChunk−DINO-CF、DINO-Restore−UADA及
DINO-CF−SigLIP-CF。CASR仍是第一终点；梯度、损失和restore只解释机制，不能替代攻击效果。

五路以array `0-4%5`并行各占一张RTX3090，预计墙钟仍由四个双前向条件决定，约5.5–6小时。Scale-500
人工结构门通过后，对五路一次性进行Object/Spatial/Goal/Long闭环：预计新增
`4 suites×10 tasks×10 states×5 methods=2000`个episodes，最多5卡，约1.5–2.5小时。四suite分别报告，
不得用池化总均值挑选方法；Long继续是必须报告的次要模块。

### 31.7 同尺度机制检测与身份校验（2026-09-21补充）

不能把训练中的辅助损失直接当作UADA与DINO-CF之间可比较的机制读数：UADA没有该辅助目标，五路的辅助
目标也不相同。因此Scale-500在outer-200和outer-500的冻结补丁上，对五路统一运行同一训练bank的
每任务首条记录、同一patch变换随机种子，分别计算完整路径、attacked-DINO+clean-SigLIP、
clean-DINO+attacked-SigLIP的动作边界误差与相对clean动作偏移。输出每任务原始值、各方法均值、
相对UADA的task-paired差值；这些是训练bank描述性探针，不是留出集CASR，更不能独自证明因果必要性。
Object outer-200短测试则严格校验补丁文件内`experiment_id`与`method`对应condition，并在相同
task/state/reset上报告DINO-CF−UADA的配对CASR。若CASR相近、机制探针也相近，应如实承认目前没有
证据表明DINO机制优化优于普通UADA，不能因为方法名称而宣称机制优势。

### 31.8 源码对齐与Scale-500提交前预检（2026-09-21）

Object outer-200实测中，UADA为9/99、DINO-CF为5/99；配对差值-4个百分点，95%区间[-13,+3]个百分点。
该差异不能证明任一方法优越，也不能仅据训练辅助损失解释原因。此前所谓“原版代码修改”应精确表述为
OFT模型/损失适配端口：它继承`UADA.py`类且复用随机贴片变换，但原版token训练函数不能直接用于OFT。
本次把原版UADA和OFT端口共同调用的随机补丁初始化、AdamW+cosine调度及optimizer/clip/zero-grad单步
更新抽到`UADA.py`；原版离散token前向与损失不变，OFT专属双视角前向及连续动作/反事实损失仍在适配器。
这是共享优化核心，不声称两者前向或损失代码完全相同。

实际checkpoint的`timm_model_ids`依次为DINOv2、SigLIP，模型构造分别映射到`featurizer`和
`fused_featurizer`。部分OFT通用源码注释写成反序，不能据注释判断；Scale-500 dry-run现读取checkpoint
配置断言此顺序，防止后续更换checkpoint时误把两支交换。此次不事后修改五路DINO损失或0.3辅助比例；
正式训练后以统一机制探针的主/辅梯度关系及闭环CASR决定是否增加新方法。

### 31.9 Scale-500五路完成与四套件统一闭环测试（2026-09-21）

五路Scale-500训练及outer-500结构门全部完成，统一机制探针齐全，状态为
`READY_FOR_MANUAL_VISUAL_REVIEW`。用户确认五张最终补丁的人工视觉结构门通过。训练bank每任务首条记录的
outer-500探针显示：普通UADA与DINO-CF的完整路径相反边界MSE分别为1.0177与1.1069，DINO-CF的
DINO-only clean-action shift MSE分别为0.02563与0.03054。这仅说明该探针上更大的DINO路径动作偏移
没有转化为更接近相反动作边界；它不是闭环CASR，不据此筛除任何方法。

统一闭环测试固定五张outer-500补丁一次性覆盖Object、Spatial、Goal、Long，不根据Object结果事后
挑方法。四suite各10 tasks×10 states 40–49，每个suite/task/state用相同冻结clean rollout seed测试五
条件，条件顺序按固定seed在state内打乱；位置保持Wrist (26,137)，Front、语言和proprioception不变。
复用既有400个clean episodes，不重跑clean；新增2000个攻击episodes，数组`0-39%5`，最多五张RTX3090。
四suite clean-success分母按既有数据分别为Object 99、Spatial 90、Goal 98、Long 95，即共有1910行
eligible attack rows。完整50行/suite-task、配对seed、补丁身份与hash、NVIDIA renderer都必须校验。

CASR为首要终点，每suite分别进行task→state分层bootstrap并报告五路CASR及四项预先冻结的task-paired
对比：DINO-CF−UADA、WorstChunk−DINO-CF、DINO-Restore−UADA、DINO-CF−SigLIP-CF。Object为
训练源suite，Spatial/Goal为跨suite零样本诊断，Long必须测试并单独报告但不作为主结论；不做四suite
池化主均值。统一runner、统计、Slurm脚本和`CHECK_ONLY`预检位于`roboticAttack-main`，提交由用户手动
执行；该测试作业已完成，其结果见31.10。此前“尚未提交”仅是最初冻结方案时的状态，不再适用。

### 31.10 四套件结果后的单一缺失对照：Full-WorstChunk（2026-09-21）

五路outer-500闭环已经完成，以下为各suite在冻结clean-success状态上的CASR（%），不是训练指标：

| 方法 | Object | Spatial | Goal | Long |
| --- | ---: | ---: | ---: | ---: |
| 普通Exploring-OFT UADA | 32.1 | 14.3 | 31.9 | 74.2 |
| DINO-CF mean | 12.1 | 11.3 | 11.4 | 65.8 |
| DINO-CF WorstChunk | 33.2 | 12.3 | 31.9 | 75.0 |
| DINO restore | 9.2 | 13.4 | 11.3 | 55.0 |
| SigLIP-CF | 19.1 | 7.3 | 7.3 | 53.9 |

这些结果支持的是：DINO-CF mean明显弱于普通UADA，而DINO-CF WorstChunk在Object/Goal/Long
恢复到普通UADA附近；不能据此断言恢复由DINO专属机制带来，因为“时间聚合改变”与“DINO分支隔离”
仍纠缠。Long保留为单独测试套件，不以它作为机制主结论。已有训练bank探针还显示DINO-CF相对UADA
的完整路径相反动作目标投影平均为负（约-0.0289，10个任务中7个为负）；这是解释性线索，
并非留出集因果证明。

因此只补一张**Full-WorstChunk**对照补丁：保留Exploring原UADA共享初始化、AdamW/调度、EOT放置、
数据、seed、25,000次更新和完整模型均值主目标；第二次同EOT前向仍走完整DINO+SigLIP路径，
与DINO-WorstChunk使用完全相同的每时间块误差最大值辅助目标和首次梯度范数0.3比率。两者唯一
关键差异是第二次前向是否把Wrist-SigLIP恢复为clean。新训练使用独立输出目录，不修改或重跑
已完成五路。`CHECK_ONLY=1`预检已通过17项测试、OFT依赖导入、共享UADA更新核心和配置干运行；
作业由用户手动提交，本文记录时尚未确认作业ID或训练结果。

判读在看结果前固定：首要比较是Full-WorstChunk与DINO-WorstChunk在Object/Spatial/Goal上的
同state CASR；Long照常测试但单独报告。如果两者相近，应承认目前的WorstChunk收益主要可由
通用时间聚合解释，不能声称DINO专属攻击优势。如果DINO-WorstChunk在多个主suite稳定领先，
再结合同输入机制探针，才可进一步研究DINO路径是否提供增益；单次点估计领先不构成充分证据。
训练后先过结构门与人工补丁外观审阅，再做闭环评估；不以外观或训练损失代替CASR。
两臂四suite闭环脚本已准备：新Full-WorstChunk与旧DINO-WorstChunk在同一冻结clean成功state、同一
rollout seed下各跑400次攻击，共800次新攻击episodes，最多五张3090卡；保留Object、Spatial、Goal
和Long逐suite统计，不重跑clean，也不重训旧补丁。评估配置当前显式标为
`PASS_USER_FULL_WORST500`（用户于2026-09-22确认补丁外观通过），提交前预检已通过10项测试、补丁身份及四套件冻结clean数据检查。统计预注册为每suite两臂CASR
及DINO-Worst−Full-Worst配对task→state bootstrap对比，不做跨suite池化主结论；评估提交仍由用户执行。

### 31.11 仅补 Exploring-OFT UADA 基线的2000轮训练与测试（2026-09-22）

用户明确要求先把普通 UADA→OFT 基线从500 outer补到原 Exploring 常用的2000 outer，**不**延长其余五路机制补丁。新基线从seed=42随机初始化，使用原 UADA共享的随机补丁、RandomPatchTransform、AdamW/cosine、clamp与每inner一次更新；OFT连续动作头适配及第0动作维度相反边界目标均与现有500轮基线相同。固定2000 outer×50 inner=100,000次更新、同一Object训练bank300条、相同data_seed/EOT/学习率/补丁位置、独立输出目录；不加载500轮补丁，也不沿用500轮余弦学习率状态。第2000轮为唯一闭环主补丁，500/1000/1500轮只存结构与机制诊断，不按闭环结果选择。

本实验只匹配原 Exploring 的**outer/inner轮次**，不称完整原论文复现：原OpenVLA配置batch=8，本OFT适配batch=1；300条bank在2000 outer约重复6.7次。2026-09-22用户要求切换4090：已取消先前3090训练job 135799及其依赖结构job 135800，两者均从队列消失；该次仅约4分49秒运行的`.out/.err`日志已删除，三个初始/续训文件移到`roboticAttack-main/outputs/exploring_oft_dino/cancelled_job_135799_135800_20260922/`可恢复归档，活动输出目录腾空。替代训练job 135802已在`rtx4090`节点`aias-compute-4`运行，依赖结构job 135803待其结束；1张4090、16小时上限。3090的500轮实测2.8小时仅作粗略参考，**4090实际速度必须从新任务日志测量，不能由规格推断固定倍速**。训练后自动结构报告，仍须用户人工确认补丁不是大面积无结构噪声，才可运行CASR。

评估已准备为同批同state的`UADA-500`与`UADA-2000`两臂对照：Object、Spatial、Goal、Long各10任务×10个冻结state40–49，复用400个clean episodes，新增800个攻击episodes；同一state两臂使用相同rollout seed，顺序固定seed随机化。只在clean-success状态计算任务等权CASR和2000−500配对差值，task→state分层区间；Long报告但不作为机制主结论。测试配置当前为`PENDING_USER_VISUAL_REVIEW`，不得因训练或结构门通过就自动提交评估。此实验只回答训练预算是否改变普通UADA攻击强度，不回答DINO专属机制增益。

### 31.12 根据2026-09-22汇报PPT讨论的下一种攻击方向（候选，尚未授权实现）

过去六路主要改变反事实前向经过DINO还是SigLIP，以及8步损失的时间聚合；这是**路径约束**，不是攻击者要制造的具体视觉/动作效应。PPT第15、17、18页提示回到已观察到的自然Wrist RANGE/LATERAL几何变化。候选具体攻击是：仅控制Wrist输入，使场景在模型表征中持续呈现错误的远近或横向位移线索，同时保留任务失败为终点。载体可以是可训练局部图像补丁，或另行定义的Wrist相机流几何欺骗；两者的威胁模型不同，不能混为一个方法。

必须先做只读可行性诊断：在训练/开发状态上用已验证的真实Wrist相机几何±RANGE/±LATERAL渲染，固定符号后检查DINO有符号方向、完整动作投影、DINO恢复效应及跨任务/状态的一致性，并审计现有UADA补丁是否已诱导同类效应。不得在留出闭环结果上逐状态选最有利符号。若方向不稳定，一张通用、固定纹理补丁没有明确可学习目标，应停止该假设而不是强加对齐损失。

若可行性通过，才讨论在Exploring原UADA训练循环中给完整动作主目标加入**有符号几何方向**辅助项，并配一个等额、打乱方向的sham；三臂均从随机初始补丁、相同2000×50更新及EOT训练，保留UADA为基线。机制读出必须测实际DINO方向、动作方向及restore，闭环主终点仍是分套件CASR。历史MGWP-v2已用RANGE/LATERAL DINO对齐但Object CASR约40%，低于其DINO-Norm对照约54%；历史DINO-Critical也提升动作方向投影而未提高闭环CASR。因此不能把PPT的方向项包装成未尝试的新机制，也不能仅凭对齐提高宣布成功；新方案与旧失败方案的实质区别、可行性门和公平对照须先经用户审定。当前只实现31.11基线，不自动启动31.12候选攻击。

### 31.13 2026-09-22 扩展判据：DINO也可降低达到相同攻击效果的成本（讨论稿，未实施）

不再把“2000轮时CASR严格超过UADA”当作唯一成功条件。成本需分开记账：离线准备/训练GPU小时、OFT完整前反传次数、在线每帧运算、闭环测试episode数；不能只报优化轮次，更不能把换4090带来的加速记为DINO贡献。主指标仍是分suite、clean-success条件下任务等权CASR。可以预先固定两种不同结论：同等训练与测试预算下更高CASR是**攻击增强**；达到预设相同CASR水平所需GPU小时更少、同时机制证据成立是**成本优势**。若两者都未证实，不宣称DINO方法更好。

优先候选A是**DINO引导的低维Wrist相机流几何欺骗**，不是六路路径约束补丁：攻击者控制数字Wrist图像流，对每帧施加同一个小幅尺度变化和横向平移（RANGE/LATERAL，边界和插值规则预先固定），Front/文本/proprioception不动；无需学习RGB纹理。用训练状态自然几何反事实得到带符号的DINO方向及DINO→action局部响应，按训练状态上“朝不利动作方向的DINO投影”从固定小网格中选一个**全局**变换，不看闭环测试结果选符号或幅度。先在留出离线状态检验该固定变换能否稳定复现DINO方向和动作变化、且DINO restore能消除相应动作偏差；若否即停止，不进入闭环。若是，与同一变换网格上的随机/打乱DINO方向选择及完整模型动作分数选择对照，记录全部搜索计算量。之后同一冻结state上测Object/Spatial/Goal/Long配对CASR；UADA-2000只作外部攻击参照，因纹理补丁与相机流控制权不同，**不能把两者的CASR差解释为同一威胁模型下的算法优越性**。这种攻击若成立，具体优势是无需100,000步补丁训练，在线仅有图像变换；不是预先保证更强。

备选候选B是**DINO几何敏感状态驱动的UADA样本选择**，才保持与Exploring补丁相同威胁模型：仍用原UADA随机补丁、动作主损失、EOT、AdamW/cosine和单步更新，只将训练bank的状态采样概率按预先冻结的DINO几何—动作敏感度排序/分层；匹配随机采样、按完整模型动作敏感度采样两种对照，三臂同初始补丁、同EOT种子、同卡型。用固定早期checkpoint和训练GPU小时画CASR—成本曲线，主终点按预先定义的预算点比较，不在测试集挑最佳checkpoint。离线DINO打分成本必须并入总成本。该方法的假设是把计算集中到机制敏感状态可更快学到有效补丁，**不是**已有事实；若敏感度在留出状态不稳定或旧UADA本已等价聚焦，则不继续。两候选彼此是不同攻击入口，不能混称一种DINO方法；A优先做离线可行性，B须先审计与旧MGWP/六路工作是否重复，再决定实施。

### 31.14 2026-09-22 用户要求收窄为补丁并并行训练：DINO选攻击目标的探索性实验

用户认为控制数字视频流权限过强，改用与基线相同的Wrist-only 50×50补丁威胁模型；允许DINO带来更高CASR或更低达到相近CASR的成本，要求基线训练时并行获得候选。上一节相机流欺骗不再作为本轮主候选。先审计旧MGWP-v2：它**已经**把有符号RANGE/LATERAL方向当DINO表征辅助损失，并用动作漂移、梯度投影和动态权重训练，不能原样再做并称新方法。其冻结训练状态方向artifact（SHA256 `58ddb6df5c279c89213d40b11c7b5e5f88b27d54abc01b73ceacad2d788faf31`）的8步动作响应绝对值均值按零基维度0–5约为`[0.0064,0.0219,0.0059,0.1046,0.1957,0.2107]`；现行UADA主损失只攻击维度0，方向响应最强为维度5。

因此本轮**探索性DINO-target5补丁**只做一项冻结改动：在完全相同的Exploring-OFT UADA随机补丁、Object 300条bank、样本顺序、EOT种子、AdamW/cosine、2000 outer×50 inner、50×50 Wrist补丁与单卡4090训练下，把完整模型动作主目标的`maskidx`由`[0]`改为`[5]`；仍每inner一次UADA动作损失反传，**不加入DINO方向损失、不锁DINO通路、不加载旧补丁**。这种做法利用DINO几何响应**选择攻击目标**，并不保证补丁本身模拟相机远近/横移，也不保证作用专经DINO。训练job 135805已与UADA-2000 job 135802并行运行，结构job 135806待其结束，二者都在`rtx4090`。新输出与旧结果隔离；自动结构检查后仍须人工看补丁再做闭环，不自动提交评估。

预定读出：两张第2000轮补丁在Object/Spatial/Goal/Long相同冻结task×state×seed上配对测clean-success任务等权CASR；Long报告但不作机制主结论。若target5胜出，先只能说“由DINO几何响应选择的动作目标优于原维度0目标”，**不能**仅据此说DINO因果机制带来增益，因为动作维度固有难度与DINO响应混杂。必须再做DINO-restore前后动作/成功相关性及同类非DINO目标维度对照，才有更强机制归因。若target5不胜出，可快速否定“直接选最大DINO响应维度即可增强攻击”的简单假设。低成本结论也需在同一4090记录训练GPU小时，并计入旧方向artifact的372秒准备成本；只用同一预定checkpoint的成功率—成本曲线，不用测试集挑最佳轮次。

### 31.15 DINO 局部空间关系误判补丁：阶段 0a 因来源错配撤回（尚不训练）

用户认可“固定 Wrist 补丁使 DINO 错误表示物体—夹爪相对位置”的假设，并要求从前提验证、短训练到完整规模。完整方案见 [`OFT_DINO_SPATIAL_RELATION_PATCH_EXPERIMENT_PLAN_20260922.md`](OFT_DINO_SPATIAL_RELATION_PATCH_EXPERIMENT_PLAN_20260922.md)。**2026-09-22 来源审计撤回 job `135810` 的全部机制结论**：真值 mask/RGB 来自 LIBERO-10、`openvla-7b-oft-finetuned-libero-10`，脚本却加载 LIBERO-Object checkpoint；同编号任务语言不同。原 STATUS 的 `pass=true` 无效，原始输出保留供审计，脚本已新增 checkpoint/unnorm_key 一致性门并拒绝原配置。错误细节见 [`OFT_DINO_SPATIAL_RELATION_STAGE0A_RESULT_20260922.md`](OFT_DINO_SPATIAL_RELATION_STAGE0A_RESULT_20260922.md)。当前 Object 主线仍无有效局部空间关系前提验证，不进入补丁训练；必须先取得与 Object checkpoint、任务和当前 bank 一致的标注，再检验动作及 DINO 恢复。该设计与 MGWP-v2 的全局 DINO 向量对齐、target5 的动作维度选择不同。所有后续 Slurm 作业仍须用户亲自提交。

单轨迹 Object bank 精确重放 3090 作业 `135836` 已通过：task 0、trajectory rank 0、初始状态 10 的六个冻结决策点，任务语言、Front/Wrist 像素、proprio、完整动作均一致，clean rollout 成功，三个 Wrist instance mask 均非空；固定补丁及相邻 token 排除后目标/夹爪仍有非零 token 质量。凭据为 `roboticAttack-main/outputs/exploring_oft_dino/spatial_relation_object_bank_mask_pilot/task_0/trajectory_rank_0/RECEIPT.json`。此结果只确认来源对齐，**尚非机制验证**。已准备 3090 array `roboticAttack-main/slurm/exploring_oft_dino/18_object_bank_mask_all10_3090.sbatch`，每任务重放 5 条轨迹×6 状态，共 300 状态；按整轨迹冻结每任务一条留出，task 0 不把已看过的 pilot rank 0 当盲留出。CPU 预检和测试通过，**array 尚未提交，仍须用户亲自提交**。收集后先审计掩码可见性，再进入局部 DINO 读数、几何扰动及动作/DINO 恢复门槛；门槛前不训练新补丁。

2026-09-22 后续：用户提交的 array `135837` 完成十任务精确重放，300/300 状态和掩码可用。CPU 留出读数目标/夹爪整体中位误差 0.708/0.766 token，满足原绝对定位门槛；但静态坐标先验在目标 6/10、夹爪 10/10 任务更准，且留出轨迹内相对位移的 DINO 误差仅 4/10 任务优于零位移先验。这是对“DINO 动态追踪物体—夹爪相对位置”强解释的关键反证，不能因整体定位误差低而跳过。结果见 `roboticAttack-main/outputs/exploring_oft_dino/spatial_relation_object_bank_cpu_readout/STATUS.json` 和 `.../spatial_relation_object_bank_cpu_controls/STATUS.json`。当前暂停 RANGE/LATERAL、动作恢复和新补丁训练；后续若继续此假设，需另冻独立、大位移轨迹及动态优于静态的门槛，不能用当前留出反复调参。没有自动提交任何新作业。

### 31.16 三臂 2000 轮闭环结果与物体位移校准边界（2026-09-23）

用户人工确认 UADA-2000 与探索性 target5-2000 补丁外观通过后，提交四套件三臂配对评估 `135870`，依赖统计 `135871` 成功完成。数据审计：1200/1200 攻击 episode 完整，1146 条 clean-success 条件记录；每套件×条件 100 条，无重复、无缺失，同一 `task×state` 三臂 rollout seed 一致。任务等权 CASR（UADA-500 / UADA-2000 / target5-2000）分别为：Object `0.321/0.429/0.300`，Spatial `0.133/0.084/0.153`，Goal `0.319/0.412/0.323`，Long `0.711/0.778/0.813`。2000−500 仅 Goal 的任务分层 95% 区间排除 0（`+0.093 [0.010,0.191]`）；Object `+0.108 [-0.084,0.329]`、Spatial `−0.049 [-0.200,0.053]`、Long `+0.066 [-0.028,0.161]` 均不确定。target5−UADA2000 在 Object/Spatial/Goal/Long 为 `−0.129/+0.069/−0.089/+0.035`，所有区间跨 0。故完整训练预算只在 Goal 得到明确增强，不能称全面增强；target5 未稳定优于普通 UADA-2000，也不支持 DINO 专属机制。Long 必测且绝对 CASR 很高，但仍是次要解释。凭据在 `roboticAttack-main/outputs/exploring_oft_dino/three_arm_object2000_cross_suite/`。

物体位移 oracle-only 校准 `135867` 的 12 条条件完整，但等世界距离的 ±X/±Y 在 Wrist token 空间高度不对称：正向可过 ≥2 token，负向在 0.06 m 内均不过，因此预设的同轴对称对为空，不能挑正方向进入模型。下一步不再扩同一世界轴幅度或降低门槛，而改为**相机 token 坐标目标的局部 Jacobian 校准**：在每个状态用不进入模型分析的小探针估计世界 XY→目标 token 质心的 2×2 局部映射，再在预设世界位移上限内求得 ±水平/±垂直、等 token 幅度的四个反事实；选择仅依赖 oracle mask、可见性、碰撞/支撑和非目标画面完整性，不依赖 DINO/动作。先在建模轨迹做技术 pilot；通过后，冻结目标 token 幅度、世界位移上限及求解规则，再在按整轨迹留出的状态上检验 DINO signed response 和干净 DINO 恢复的动作分量。若局部映射病态、任一正负方向不可达或完整性失败，则该状态无效，不以单方向替代。所有新 Slurm 作业仍由用户提交。

### 31.17 冻结 UADA-2000 基线与局部 Jacobian 技术 pilot（2026-09-23）

按用户决定，`exploring_oft_uada_o2000` 从本节起冻结为后续补丁攻击的统一操作基线。唯一补丁、SHA256、训练预算、四套件任务等权 CASR、结果表哈希及使用边界写入 `roboticAttack-main/configs/exploring_oft_dino/FROZEN_UADA2000_BASELINE_20260923.json`。这里“冻结为基线”是实验治理决定，不改写31.16的统计结论：2000−500只有 Goal 的任务分层95%区间排除0，不能宣称四套件全面增强。以后不得根据新测试集结果重训或替换这张基线；新方法应复用同一clean-success状态、rollout seed、位置和逐suite任务等权CASR，Long仍单独完整报告。

Jacobian pilot 已实现为 `scripts/pilot_object_pose_jacobian_oracle.py`，只在 Object task 0 的 discovery trajectory rank 1、decision 5 做技术可行性检查。固定规则为：±0.01 m中央差分估计世界XY到Wrist token XY的2×2 Jacobian；条件数上限30；四个目标为相机token坐标的±水平/±垂直2.5 token；允许误差0.5 token、方向余弦至少0.95、世界XY位移范数最多0.08 m、最多两次固定Jacobian修正。四个方向必须全部通过，不降低原来的2-token有效位移边界，也不以单侧结果代替。 altered renders 不进入 DINO 或动作模型，因此该作业只能验证反事实刺激能否对称构造，不能提供机制或攻击证据。3090脚本为 `slurm/exploring_oft_dino/23_object_pose_jacobian_oracle_3090.sbatch`，由用户手动提交。

### 31.18 物体—夹爪关系与 DINO 的相关工作边界（2026-09-23检索）

相邻方向已有充分先例，但本轮聚焦检索未发现与本项目完全等价的工作。关系建模方面，Dense Object Nets（CoRL 2018）用自监督密集描述子建立物体点对应并服务抓取；Transporter Networks（CoRL 2020）直接把操控写成物体、部件或末端执行器之间的空间位移；基于鲁棒关键点的视觉RL（arXiv:2202.06027）明确用关键点编码 tool/target 的几何与空间关系。DINO方面，原始DINO与DINOv2证明了无监督ViT的像素/patch级语义和对应能力；DINOBot（arXiv:2402.13181）用DINO图像级检索和像素级对齐，使末端执行器对齐新物体；D3Fields（arXiv:2309.16118）把视觉基础模型特征融合为动态3D描述场用于零样本重排；DINO-WM（arXiv:2411.04983）直接预测DINOv2空间patch特征并在该空间规划。它们共同支持“DINO类特征可以承载与操控相关的局部空间结构”，但不自动证明当前OFT Wrist-DINO分支动态编码了物体相对夹爪位移——本项目静态先验对照已经说明必须单独验证。

攻击方面，Exploring VLA vulnerabilities（arXiv:2411.13587）已有动作/位置目标的彩色补丁，EDPA（arXiv:2510.13237）已有通用视觉嵌入破坏补丁，另有工作分析DINO ViT本身的对抗脆弱性（arXiv:2206.06761）。这些工作分别覆盖VLA攻击、嵌入破坏和DINO鲁棒性；本轮检索未见将三者闭成“固定Wrist补丁 → 有符号地误表征物体相对夹爪几何 → 通过DINO路径改变连续动作 → 闭环失败”的因果链。因此潜在贡献不应表述为首次发现物体—夹爪关系或DINO密集对应，而应是对这条**关系特异、路径可干预、闭环可验证的攻击机制**进行严格验证；若Jacobian刺激或独立留出机制门失败，就不能靠相邻文献替代本项目证据。

### 31.19 单状态 Jacobian 结果与多任务 discovery 可达性审计（2026-09-23）

用户提交的单状态技术 pilot `135924` 完成，约耗时1分20秒。局部Jacobian条件数1.95，说明该点附近世界XY到Wrist token XY映射数值可辨；但四方向门失败：图像X+经一次修正达到2.185 token且方向余弦0.996，X−固定Jacobian修正发生严重非线性过冲并伴随目标token质量降至0.265，图像Y±的线性解各需约0.105 m、超过0.08 m上限。因此这不是DINO假设检验——altered renders没有进入模型——而是当前单状态、世界XY-only对称刺激构造失败。不得挑X+继续，也不得降低2-token门槛或放宽世界位移上限。

为区分“单状态几何/求解器失败”与“该构造普遍不可行”，冻结一次多任务discovery审计：10个Object任务各保留既有整轨迹holdout不看，只用其余4条独立初始状态轨迹；每轨迹按bank标签确定性选择最后一个pre-contact状态，共40个轨迹级实验单位。每状态仍用±0.01 m中央差分、2.5±0.5 token目标、方向余弦≥0.95、世界XY范数≤0.08 m及条件数≤30。根据单状态暴露的数值过冲，求解器在查看多任务结果前改为有界固定Jacobian线搜索：初始反解沿同一射线检查预设比例1、3/4、1/2、1/4，后续更新投影到0.08 m球内并只测试预设比例1、1/2、1/4、1/8，最多两轮，均按oracle目标误差选择；另要求目标token质量至少为clean的25%且至少1 token，防止物体近乎出画仍被接受。选择不使用DINO或动作。

停止门在结果前冻结：40条discovery轨迹中四方向全通过比例至少50%，且至少8/10任务各有一条全通过，才允许把完全相同规则一次性用于10条未触碰holdout并开始DINO/action机制门；否则关闭世界XY-only构造，不在holdout调阈值。GPU数组脚本为`24_object_pose_jacobian_discovery_3090.sbatch`，CPU汇总为`25_object_pose_jacobian_discovery_summary.sbatch`；均由用户手动提交。

### 31.20 关闭世界XY-only构造，转入诊断性图像空间关系干预（2026-09-23）

多任务discovery作业`135926`及汇总`135927`完成，40/40轨迹完整、10条holdout未触碰。只有2/40轨迹四方向全通过，仅2/10任务有任何全通过，远低于预注册的20/40与8/10门槛。局部Jacobian并不病态（条件数中位2.08、最大3.37）；方向可达性高度不对称：图像X+为40/40、Y−为39/40，而X−仅8/40、Y+仅5/40。失败评估中365次触发目标可见质量不足，另106次目标或夹爪在固定排除后不可见。因此按冻结规则关闭世界XY-only刺激构造，不运行holdout、不调阈值。该结果仍不是DINO机制反证，因为altered renders从未进入模型。

下一步严格区分诊断干预与攻击权限：先用模拟器目标mask和“目标物体移出场景后的背景”在Wrist图像中合成精确对称的物体位移，只用于验证物体相对夹爪关系是否沿DINO影响动作，不能把这种图像编辑称为攻击。技术pilot冻结为task 1、discovery rank 3、decision 3；该状态在上一轮仅凭oracle四方向可构造性入选，先前未查看任何altered-image DINO或动作。将目标及2像素支持环在原始256图上分别平移±38像素，得到约2.5±0.3 token的±水平/±垂直位移；Front、proprio、语言及支持区域外像素保持clean，零位移重建必须逐像素等于clean，移动后目标不得与夹爪重叠。

单状态pilot在运行前冻结五道门：四刺激oracle方向余弦≥0.995；四方向DINO关系读出相对clean的方向余弦≥0.5且有符号投影≥0.5 token；±水平和±垂直动作块L2间距各≥0.02；四条件动作相对clean变化均≥0.01；以clean DINO替换后动作变化的中位恢复比例≥50%。只有全部满足才设计多状态discovery/holdout机制检验，仍不立即训练补丁；任一门失败即停止物体—夹爪关系机制分支。脚本为`pilot_object_gripper_image_cf.py`和`26_object_gripper_image_cf_pilot_3090.sbatch`，提交仍由用户执行。

### 31.21 2026-09-23 提交前设计复核与解释边界修正

对31.15–31.20的审计认为，原来的来源一致性、整轨迹切分、对称oracle刺激和世界XY路线的预注册停止是合理的；世界XY路线已按2/40而关闭。原31.20末段把单个合成图像状态的任一数值门失败写成“停止整条物体—夹爪机制分支”，这一推断过强，现撤回。先前静态先验已经优于部分DINO动态读数；单状态位置读出、动作变化和clean-DINO替换都可能由物体消失、图像剪贴边缘或一般Wrist视觉变化造成，不能独自归因于相对几何。此前的五道数值门保留为探索性诊断列，不能作为假设成立或否定的判决器，尤其动作块L2大不等于朝任务失败方向改变。

提交前的`pilot_object_gripper_image_cf.py`因此修订为：任务1的pilot整条轨迹从原型拟合中移除；四个平移方向各增加同源擦除、同目标支持区域的“只贴背景、不贴物体”匹配sham。DINO位置读出和动作效应首先报告物体存在图像减去相应sham图像，clean-DINO恢复也在这对匹配输入间比较；保留原始相对clean变化供审计。零位移重建逐像素一致，检查移走物体后夹爪像素稳定、位移目标不覆盖机器人instance像素，严格记录改动区域及重渲染干净帧相对冻结bank的像素差距。单状态仅判定合成刺激是否技术有效，所有关系和动作门均标为探索性；无论数值方向如何都需要按轨迹重复和独立holdout，且不能据单状态训练补丁。若背景渲染、可见性或合成完整性失败，则先修技术问题并记录，不把程序异常解释为机制失败。攻击闭环主线继续以冻结UADA-2000作对照。

### 31.22 图像空间pilot第一次提交的技术失败与v2修正

用户提交的job `135946` 在task 1/rank 3/decision 3重放及模型加载后，检查`image_x_minus`时发现平移后的目标覆盖机器人instance像素，因完整性门中止；没有`STATUS.json`，也没有可解释的四方向DINO/动作结果。这暴露了之前只检查夹爪局部mask、未在提交前覆盖整个机器人掩码的预检缺口，不是关系机制反证。v2改用上一轮oracle-only审计中另一条四方向可构造的discovery状态task 9/rank 0/decision 3，固定±38原始像素与所有阈值，不借用旧作业未完整保存的模型读数选状态。v2先完成**全部四方向**的目标—机器人遮挡、边界、相对位移与匹配sham检查，然后才对任何altered图像运行DINO/动作；几何失败写明`IMAGE_CF_TECHNICAL_STIMULUS_INVALID`并保存原因，不作机制结论。独立输出目录和新3090脚本为`object_gripper_image_cf_pilot_v2`、`27_object_gripper_image_cf_pilot_v2_3090.sbatch`，仍由用户手动提交。

### 31.23 v2结果与冻结的批量几何筛查

用户提交的v2作业`135948`在task 9/rank 0/decision 3的`image_x_minus`发现目标覆盖3个机器人instance像素，写出`IMAGE_CF_TECHNICAL_STIMULUS_INVALID`，没有对任何编辑图像做DINO/动作推理。两条按既有oracle可达性选出的状态均因同一方向的机器人遮挡失败；这不是DINO假设检验，也不据3像素事后放宽零遮挡门槛。下一步停止逐状态手选，按同一冻结规则批量审计Object十任务的40条discovery轨迹，每条确定性取最后一个pre-contact状态；10条整轨迹holdout保留。各任务只重放干净OFT以定位状态，依次检查重渲染、目标移除背景、±38原始像素四方向边界、目标—机器人完整instance零重叠、约2.5±0.3 token有符号位移和匹配sham。编辑图像不进入DINO或动作模型。

批量几何可行门在运行前固定为至少20/40轨迹且至少8/10任务有一条四方向有效轨迹；通过才为后续多轨迹DINO机制实验提供广覆盖开发样本，否则不再以少数手选状态宣传关系机制，转向其他DINO候选。两个数字是本轮技术覆盖要求，不是统计显著性或对DINO假设的检验。审计脚本`audit_object_gripper_image_geometry.py`、汇总脚本`summarize_object_gripper_image_geometry.py`，3090数组脚本`28_object_gripper_image_geometry_discovery_3090.sbatch`及依赖汇总脚本`29_object_gripper_image_geometry_summary.sbatch`；所有作业仍由用户提交。

### 31.24 关闭物体平移刺激路线，转向直接DINO特征攻击（2026-09-23）

用户提交的几何审计array `135949` 已完成10任务、40条discovery轨迹；0/40四方向全有效。35条在移走目标物体生成背景时夹爪像素变化（最大像素差28–68），其余5条至少一个方向与机器人重叠。编辑图像没有进入DINO/动作模型，所以这**不能否定DINO机制**，却足以按预设20/40和8/10技术门关闭该四方向刺激路线。不再逐状态换位置、放宽零遮挡、修改阈值或使用holdout继续修补。先前300/300精确重放有用；静态先验优于部分DINO动态读数是关系假设的重要反证；Jacobian和合成图像的后续多轮主要提供技术失败边界，没有提供攻击增益证据。

新的、一次性有截止门的攻击候选采用Exploring源代码及现有OFT适配器：仍用随机初始化的Wrist-only RGB 50×50补丁、同一Object 300状态bank、样本顺序/transform seed、每outer取一个样本、每inner做一次EOT与一次AdamW更新、cosine调度和clamp；训练预算500 outer×50 inner=25,000更新。完整OFT模型的UADA维度0相反边界MSE为**不变主目标**。在同一次attacked前向捕获Wrist视觉塔特征，辅助目标最小化clean与attacked的逐token余弦相似度，即最大化特征差异；辅助梯度首次更新的范数固定为主梯度的0.3，后续沿用同一权重。两条等预算新臂只改变目标分支：`exploring_oft_dino_feature`（Wrist-DINO）和`exploring_oft_siglip_feature`（Wrist-SigLIP）。不恢复/替换分支、不改变动作损失、不加载任何历史补丁。全局逐token损失可能被补丁覆盖区域主导，故特征差异增大本身**不等于**场景几何被误读，也不等于闭环失败。

500轮历史普通Exploring-OFT UADA补丁是固定比较臂，SHA256 `c7ebb69204290fc193c3f694e25ba02c656de9e2df0443a74470d8ad587db136`；训练配置审计锁定与其相同的bank、checkpoint、seed、EOT、优化器、预算及维度0。新训练只创建`UADA_OFT_FEATURE.py`、`UADA_OFT_FEATURE_wrapper.py`、`object_feature500.json`、`30_train_feature500_3090.sbatch`及测试文件，不修改原`UADA.py`或既有`UADA_OFT_DINO.py`。用户**禁止自动提交**；当前仅完成代码和预检，补丁尚未训练。由用户手动运行`sbatch slurm/exploring_oft_dino/30_train_feature500_3090.sbatch`后，先检查结构门、训练日志和两张最终补丁外观；任何一路出现明显大量噪声/结构门失败即不拿它宣传成功。只有用户确认外观后，才准备同task×state×seed四套件闭环评估；Object/Spatial/Goal主比较，Long必须测试但单独解释。首要比较是DINO特征臂−普通UADA500的CASR，DINO−SigLIP用于分支特异性；训练GPU小时和额外前反传一并记账。若DINO臂无闭环增益也无可验证成本优势，则不扩到2000轮、不再围绕同一全局特征差异目标诊断。若有增益，再用同输入DINO/SigLIP恢复评估分支因果中介；若SigLIP同样强，则不能称DINO专属。

方法依据与新意边界：Exploring（arXiv:2411.13587）提供补丁与UADA动作目标；EDPA（arXiv:2510.13237）已经提出VLA视觉嵌入差异攻击，其论文使用patch级对比损失与视觉—文本对齐损失；本轮仅采用更简单的逐token余弦差异辅助项，**不是EDPA复现**。DINO自监督ViT对抗研究（arXiv:2206.06761）仅说明DINO特征可受对抗扰动，不能直接外推OFT任务失败。因此本轮不是宣称首次做嵌入攻击，而是在相同Exploring-OFT威胁模型和源代码训练循环中检验“DINO分支特征扰动是否比通用动作攻击及SigLIP对照更有效/省成本”。后续文献探索只提出可被同一闭环终点反驳的具体攻击目标，不再用无终点的代理几何审查替代攻击实验。

### 31.25 取消两臂任务、恢复只读源代码并冻结六臂联合比较（2026-09-23）

用户在两条Feature-500子任务约outer 130时手动取消job `135967/135968`，要求清除断点并将相关候选一次性公平训练。取消产物、日志以及更早取消的`135799/135800`断点已删除；来源错配的Stage0A原始输出和两次无模型读出的技术无效image-CF v2产物也已清理，结论边界仍保留在本文。形成有效负结论的Jacobian discovery、冻结基线、正式补丁及跨suite结果没有删除。

审计发现此前为复用初始化/优化器/更新步骤曾直接重构`VLAAttacker/white_patch/UADA.py`，虽然未改`weighted_loss`且默认语义等价，仍不符合“源文件不改”的治理要求。现已恢复Git版本，SHA256为`808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e`，权限设为`0444`；复制出的训练原语位于`UADA_OFT_SOURCE_COPY.py`，所有OFT派生代码只调用副本。wrapper在每次训练前同时检查源文件哈希及owner不可写位。

新的联合实验固定500 outer×50 inner、随机初始化、Object 300状态bank、样本/EOT seed、AdamW/cosine、50×50 Wrist-only补丁与3090卡型，冻结UADA-500补丁仅作外部同预算比较，不热启动。六个新臂为：

1. `dino_feature_uniform`：维度0 UADA主损失 + 全部Wrist-DINO token余弦差异；
2. `siglip_feature_uniform`：匹配的Wrist-SigLIP分支对照；
3. `dino_signed_geometry`：维度0 UADA主损失 + 旧MGWP-v2冻结的动作敏感RANGE/LATERAL有符号DINO方向；
4. `dino_signed_geometry_sham`：逐状态保持方向数值与范数、但以冻结seed打乱空间token对应的方向对照；
5. `full_persistent6d`：修改主损失，在前6个连续动作维度按各自q01–q99范围归一化，以clean chunk均值确定每维单一反向边界，同时优化8步逐步误差和chunk均值持续误差；
6. `dino_persistent6d`：与第5臂完全相同，再加入attacked-DINO+clean-SigLIP路径上的相同Persistent-6D目标。

有辅助项的臂均把首次辅助梯度范数冻结为主梯度的0.3；以后不动态放大。第3臂是对旧定向思想在Exploring训练底座上的重做，不得包装成从未尝试的新机制；其相对第4臂的差值才回答真实方向是否优于等预算sham。第5与第6臂的差值隔离DINO路径增量，不能只用第6臂相对旧维度0 UADA的差异声称DINO贡献。两条全特征臂回答广泛视觉表征破坏；四类问题不得混成单一“DINO方法”。

实现为`UADA_OFT_JOINT500.py`、`UADA_OFT_JOINT500_wrapper.py`、`object_joint500_v1.json`和`31_train_joint500_v1_3090.sbatch`。六臂是一个手动提交的array，最多六张3090并发；每臂结构门仍在outer 100/200/500检查，不通过即停止该臂。全部最终补丁须先人工检查外观，再统一构造四suite测试；Object/Spatial/Goal为主，Long完整报告但不承担方法选择。用户禁止自动提交，当前仅完成实现和CPU预检。

### 31.26 六臂训练完成与七臂配对闭环测试（2026-09-24）

用户手动提交的训练array `136009` 已完成六个子任务：每臂500 outer×50 inner=25,000次优化，全部`status.json=COMPLETE`，outer 100/200/500结构门全部通过。用户已查看六张最终补丁的预览并答复“ok”，授权准备闭环测试。训练完成只证明优化程序和外观门通过，不能用训练损失大小排序CASR，也不能把定向DINO得分当作因果路径证据。

本次采用旧`run_continuous_cldev.run_episode`和冻结状态40–49；Object、Spatial、Goal、Long各10任务×10状态。六张新补丁与冻结UADA-500共七臂，状态内共享clean rollout seed、随机化攻击顺序；四套件各700条新攻击episode，合计2800条。既有400条clean rollout用于确定配对的clean-success分母，分别为Object 99、Spatial 90、Goal 98、Long 95；预期有效攻击记录为2674条。每个task×state为配对单位，task分层自助法给逐suite置信区间；8步决策块不能当独立episode。固定Wrist位置`[26,137]`、50×50补丁、同套件checkpoint/归一化及原最大步数。

预定五组差值：DINO全token−UADA-500、DINO全token−SigLIP全token、真实有符号几何−打乱方向sham、完整模型Persistent-6D−UADA-500、DINO路径Persistent-6D−完整模型Persistent-6D。五组均逐suite报告，探索性多比较不事后挑最有利结果作为唯一主结论。前两组检验广泛特征攻击和分支特异性；第三组检验已尝试过的旧几何方向在Exploring训练底座上的增量；第四组检验修改动作主损失；第五组隔离DINO路径对该动作目标的增量。Long必须完整测试、单独解释。闭环CASR是攻击效果终点；若路径定向臂胜出，再进行同输入DINO/SigLIP恢复才可进一步讨论中介。

独立评估入口`roboticAttack-main/scripts/run_exploring_oft_joint500_cross_suite.py`、统计入口`analyze_exploring_oft_joint500_cross_suite.py`，配置`joint500_v1_cross_suite.json`逐补丁固定SHA256与用户外观确认。3090测试array为`32_joint500_v1_cross_suite_3090.sbatch`（40 shard，最多并发5，单shard上限8小时）；依赖统计为`33_joint500_v1_cross_suite_stats.sbatch`。CPU预检已通过，返回`JOINT500_XSUITE_PREFLIGHT_PASS`、2800计划episode、2674有效记录。所有Slurm作业仍须用户亲自提交。

## 32. 2026-09-26 审计后：空间对应错位与高维补丁外观对照

31.26之后的Decoy、Sampling、Jacobian结果见`WEB_MODEL_HANDOFF_OFT_DINO_20260926.md`；解释须同时阅读`audit_dino_history_20260926/AUDIT_REPORT.md`。最新Jacobian500 Object/Goal显著弱于同场UADA500，不扩2000。保留UADA2000及所有实际CASR；P1是Projector范数匹配，不是像素范数匹配。旧几何教师受单位错误影响，本轮不复用，也不整体重跑历史实验。

用户已批准先更新文档，再实现至手动提交命令。新方案唯一实施附件为`OFT_DINO_CORRESPONDENCE500_PLAN_20260926.md`：A复用UADA500，B=UADA+阈值TV，C=UADA+DINO空间对应右移，D=C+阈值TV，E=同区域普通DINO特征破坏+阈值TV。全部直接优化50×50×3高维RGB；不降维、不热启动、不修改只读原UADA。先seed42、500×50更新筛选四个新臂；外观和CASR分别验收，四套件完整测试，Long次要。仅允许CPU预检；正式GPU作业用户手动提交，禁止自动提交及自动使用重置卡。

### 32.1 四臂完成、外观通过及五臂四套件闭环（2026-09-27）

用户提交136423已完成B–E，每臂500×50更新、约2小时49–53分钟。完整性及配对检查通过，用户已确认外观。空间诱导仍弱：C/D最后100轮匹配右移0.0366/0.0310 token，预设目标约0.9713；动作主损失未优于B，不预判闭环结果，也不扩2000。具体日志统计与哈希见第32节实施附件第7节。

准备五臂（同场重测A及B–E）四suite测试，固定40–49状态、位置(26,137)，40个task分片共2000攻击rollout，clean分母99/90/98/95，预计1910有效记录。六组逐suite配对差值C−A、D−B、D−E、D−A、B−A、D−C全部报告，Long次要。评估51号array最多5张3090，52号依赖CPU统计；提交入口`roboticAttack-main/submit_exploring_oft_correspondence500_cross_suite.sh`，由用户执行。

### 32.2 五臂四套件闭环完成，关闭空间对应候选（2026-09-27）

136436/136437完成2000攻击episode、1910有效记录。A/B/C/D/E的Object CASR为32.1/26.0/12.0/15.0/25.1%，Spatial为14.3/9.3/14.4/10.6/11.4%，Goal为31.9/43.4/15.6/28.9/41.3%，Long为74.3/79.0/77.1/79.9/85.4%。Object C−A与D−A的配对95%区间低于零；Goal D−B也低于零。TV-only Goal收益为+11.6pp[+2.1,+23.4]，不归因于DINO。空间错位实际只约0.03token，不能称充分实现错位后机制失败。具体统计见第32节附件第8节；不扩2000、不删除负结果。

## 33. 2026-09-27 UADA500起点的DINO参考方向强化

用户批准更新文档、设计并实现至手动提交命令。唯一附件为`OFT_DINO_ILA_REFINEMENT_PLAN_20260927.md`，参考ILA（ICCV2019）将已有对抗特征变化作为方向。三臂都从同一冻结UADA500的全RGB补丁开始，各追加200×50更新：原动作目标+TV、有方向DINO投影+TV、同区域无方向DINO破坏+TV。共同起点及追加优化器重启/EOT配对，使C−B回答增强/成本、C−D回答参考方向价值。明确这是500+200的强化，不混称随机从头500轮。

使用OFT实际DINO dense读出，clean/基线teacher detach，基线与当前共享EOT，排除贴图覆盖tokens，逐token归一化避免范数作弊。特征臂每inner只调用腕部DINO，参考计算成本也计入；初始接口对齐检查及模型加载单独记录。Front始终干净、高维7500 RGB参数和外观确认不变。原UADA只读，新增副本及严格恢复签名。只做CPU实现预检，GPU任务仍须用户手动提交，禁止自动使用重置卡。

### 33.1 独立实现与CPU预检完成，待用户提交（2026-09-27）

新训练器`UADA_OFT_ILA.py`、目标模块`oft_ila_objectives.py`、wrapper、冻结配置及53号3090三臂array已完成。10项CPU测试通过，包括三臂断点恢复逐元素一致、同EOT像素/梯度一致、投影方向及覆盖排除、腕部通道映射和累计成本恢复；干跑通过，300训练状态、三臂各10000追加更新/35000总更新、原文件只读和基线哈希均核对。真实模型接口校验嵌入正式GPU任务，未在CPU运行真实VLA，未提交GPU任务。手动入口`bash roboticAttack-main/submit_exploring_oft_dino_ila_refine200.sh`（从项目父目录执行时）；完整命令、代码差异与结果路径见第33节附件第6节。训练后先检查外观，再准备四组四套件评估。

### 33.2 V1启动校验失败，修正独立DINO输入路径（2026-09-27）

用户提交136494三臂均在clean对齐校验失败，追加outer/updates均为0，没有新补丁或方法效果结果。原错误无差值记录；代码存在新clean CPU ToTensor与旧GPU浮点转换的实现偏差及独立切片布局差异，但现有日志不足以确认唯一根因。V2统一clean转换和完整12通道cast/slice布局，用实际DINO输入pre-hook先验证逐像素一致，再保留原特征容差；失败也记录详细差异，未放宽阈值或移除校验。

新54号3090脚本及`object_dino_ila_refine200_v2.json`保持原三臂预算/主目标/TV不变，使用独立V2结果目录。V1源码快照、日志和零步断点保留；不跨签名强制恢复，重用冻结UADA500而不是重训基线。通用手动提交入口已指向V2；GPU实际对齐仍待用户提交后检查。细节及预检见第33节附件第7节。

### 33.3 V2训练完成、外观确认与四组闭环测试准备（2026-09-27）

136499三臂均完成追加200×50，总预算500+200轮/35000更新，真实clean/reference-EOT像素及特征对齐误差均0；起点与冻结UADA500一致、训练序列配对、最终张量有效。用户确认三张最终PNG。追加训练B/C/D耗时65.65/6.10/6.49分钟，含已记录基线和本次加载约233.79/174.23/174.65分钟（缺历史基线加载）。C投影改善仅为代理指标，尚无闭环攻击增强或效率结论。

冻结A原UADA500+B/C/D四臂四套件评估：1600新攻击episodes、复用400 clean、1528有效行，位置/初态/seed/官方动作处理保持；Long完整但次要。C−B/C−D及B−A/C−A/D−A五类配对对比，任务等权CASR和训练成本一起报告；A不是700轮同预算臂。新增独立配置、loader、统计、CPU/mock测试、55号3090 array与56号afterok统计脚本，预检后仅提供用户提交命令。附件第8–9节记录训练结果与测试设计；冻结UADA2000基线不变。

实施预检已完成：9项CPU/mock测试全通过，四patch哈希/共同热启动/实际GPU接口记录及99/90/98/95 clean分母核对；合成1600回合验证16条摘要/20条对比和统计符号，不是正式攻击结果。原UADA.py保持0444与冻结SHA256。手动入口`bash submit_exploring_oft_dino_ila_refine200_cross_suite.sh`（在roboticAttack-main目录），一次提交40分片3090评估及afterok统计，最多5卡并行；仅执行CHECK_ONLY，未自动提交。完整命令和输出位置见附件9.1节。

### 33.4 ILA闭环完成（136504/136505）

1600回合、1528有效行及所有身份/数据核对通过。A/B/C/D的Object为32.1/35.2/36.0/10.0%，Spatial为13.3/6.4/13.4/17.4%，Goal为31.9/31.9/28.9/23.0%，Long为74.3/76.4/86.5/77.4%。C−B主套件区间跨零；Object C−D为+26pp[+8,+46]；Long C−B为+10.1pp[+1.4,+19.6]，次要开发结果且未多重校正。保留低成本候选的意义，但不宣称主套件明确增强或非劣，不直接扩2000。

## 34. UADA与Wrist-DINO交替、滚动局部参考强化（2026-09-27）

用户批准先更新方案、再实现预检直至手动提交命令。唯一附件`OFT_DINO_ALTERNATING_REFINEMENT_PLAN_20260927.md`。新增固定参考交替及滚动参考交替两臂，从相同UADA500开始各追加200×50；每outer先25次原动作目标、再25次DINO投影。固定臂沿原UADA500−clean方向；滚动臂沿本outer动作更新后−前的归一化DINO变化，以动作更新前为局部起点。每inner参考三输入共享EOT，snapshot detach，排除前/后/当前支持并集；沿用真实V2接口、7500 RGB、Wrist-only和TV。

先做两臂训练，不重训已完成纯UADA/固定ILA；外观通过后新增两臂共800回合四suite闭环，条件一致时复用冻结配对对照。Long完整次要，CASR及全部成本一起报告。新独立文件与目录，原UADA只读；按完整outer断点恢复，失败半轮重做，源码签名与累计成本保留。仅CPU预检；57号3090双臂脚本由用户提交，禁止自动提交和自动使用重置卡。

### 34.1 实现及CPU预检完成，待用户手动提交

新增独立训练器`UADA_OFT_ALTERNATING.py`、三支持投影目标、wrapper、JSON、测试、57号Slurm脚本及helper。11项CPU/mock测试通过：同EOT/参考detach/三支持排除、两臂25+25阶段预算、动作或特征阶段中断恢复的最终patch及训练记录逐元素一致、配置与源码漂移拒绝、原V2像素接口继承及手动提交门。dry-run确认300 Object记录、7500 RGB、每臂10000追加更新（5000动作+5000特征）/35000总更新、269源码文件，两臂NOT_STARTED；无真实CPU VLA前向，无GPU提交。

命令（在roboticAttack-main）：`bash submit_exploring_oft_dino_alternating_refine200.sh`，最多2张3090，只训练两臂；PNG见新目录`object_dino_alternating_refine200_v1/<method>/outer_0200.png`。原UADA保持0444和冻结SHA，原ILA文件及已完成补丁保留。完整设计、代码差异、验证和路径见附件第6节。

### 34.2 交替训练完成；用户授权带噪声问题先测试

136551两臂完成追加200×50，各37.12分钟、5000动作+5000DINO更新；实际GPU像素/特征对齐误差0。固定组保留起点结构；滚动组用户指出噪声较多，尚不满足外观要求。用户明确“先测试吧”，仅授权效果诊断，不记为外观PASS，不修改冻结训练补丁或重新训练。固定/滚动末20outer平均投影1.065/0.523，滚动99.44%特征步骤有梯度；这些均非CASR结论。

两新臂四suite800新回合，复用ILA V2的A原UADA500/B纯UADA续训/C固定ILA共1200旧回合；五条件共2000、预计1910有效行。复用前冻结/核对旧raw、manifest、共享runner/runtime、模型路径、同初态/seed/clean资格，保留来源身份。历史对照及单seed开发状态局限保留，Long完整次要。测试独立保存外观反馈和测试授权；即使滚动CASR提高也不能称满足非噪声约束。按附件第7节实施至手动提交命令，禁止自动提交与重置卡。

独立测试实现及10项CPU/data/mock测试、CHECK_ONLY预检均通过；无真实CPU VLA前向、未自动提交。手动入口（在roboticAttack-main）：`bash submit_exploring_oft_dino_alternating_refine200_cross_suite.sh`，58号40分片array最多5张3090，59号afterok CPU统计。只跑800新攻击，不重跑旧对照/clean/训练；合并保留旧experiment_id及来源，20条suite摘要/28条配对差值，滚动外观未接受标记随结果保存。完整命令及路径见附件7.1节。

### 34.3 交替闭环完成，停止滚动、固定不扩轮次

136560/136561完成800新+1200复用=2000攻击回合、1910有效行、20条摘要/28条对比，原始结果与身份核验通过。固定交替Object/Spatial/Goal/Long CASR=37.111/9.333/29.889/88.750%；滚动=7.000/6.333/2.000/44.167%。固定相对纯UADA追加200的主套件差值+1.889/+2.889/−2.000pp，区间均跨零；Long+12.361pp[+3.750,+22.222]为次要开发结果。滚动明显退化且外观未获接受，结束候选。固定不直接扩2000。原DINO因果脆弱性与方法利用有效性分开，不据此推翻原因果结论。历史配对对照、单seed、未校正多对比局限保留。

## 35. Wrist-DINO ILPD训练时扰动衰减，正常RGB闭环测试

用户批准先更新方案、再实现预检至手动提交命令，并要求整理结果路径。唯一新附件：`roboticAttack-main/research_runs/wrist_dino_ilpd_v1/PLAN.md`；该根目录集中training/logs/reports/evaluation，旧outputs保留原位并补导航索引，不移动既有证据。

在Exploring派生独立副本上，四组同UADA500起点各追加200×50，7500RGB、Wrist-only、Front干净、同状态/EOT/优化器。动作主损失不变；原前向UADA、DINO仅反向×0.5、DINO前向Z0+0.5(Zp−Z0)、SigLIP前向同样衰减四组，后一组只作通用机制对照。借鉴ILPD NeurIPS2023，不新增特征辅助损失，不将梯度缩放重命名为新方法。全部同TV和相对起点±16/255像素盒约束，仍高维全RGB，不保证外观通过。测试必须正常模型、无特征hook、只贴RGB。

CPU预检及GPU作业内一次启动接口检查、完整outer断点/签名/成本/COMPLETE保护；不自动提交、不使用重置卡、不串联测试。训练完成先人工看图，再四suite配对闭环；主套件Object/Spatial/Goal、Long完整次要。只有代理或Long收益则结束，不再堆诊断；主套件和外观满足后再讨论2000轮。

### 35.1 独立实现与预检完成，手动提交入口

新增UADA_OFT_ILPD.py、oft_ilpd_features.py、独立wrapper/JSON/tests、60号3090四臂array及submit_exploring_oft_dino_ilpd_refine200.sh；未修改既有训练器或原UADA/EOT。临时hook对实际模型消费的dense输出进行变换，测试不得加载hook。每步仍为原连续动作MSE，四组同±16/255像素盒与TV。

新14项CPU/mock测试全通过，含四组启动校验、完整训练循环、同状态/EOT和bitwise恢复；连同既有ILA/交替回归共36项通过。CHECK_ONLY确认300唯一Object记录、共同起点/包与273源码文件、7500参数、10000追加动作/0独立特征更新/35000总更新、四组NOT_STARTED；model_loaded=false、sbatch_submitted=false。原UADA.py仍0444与冻结SHA。真实GPU接口及梯度检查在正式作业启动执行，并非已通过GPU测试。

手动命令（roboticAttack-main目录）：`bash submit_exploring_oft_dino_ilpd_refine200.sh`；最多4张3090，每臂4CPU/10G、4小时调度上限。结果统一`research_runs/wrist_dino_ilpd_v1/training/<method>/outer_0200.png/.pt`，日志统一同根logs/。先审四图，再准备四suite闭环；不自动提交测试。旧outputs未移动或删除，关键历史路径见outputs/README_RESEARCH_INDEX.md；详细改动与预检见当前根reports/PREFLIGHT.md。

### 35.2 V1零步启动失败；独立V2工程修复完成

用户提交136602四组均FAILED，追加0轮/0更新。clean/reference-EOT像素和DINO特征差异0，α=1特征和动作前向一致；严格逐元素RGB梯度比较失败，普通UADA重复自身也失败。日志未记录梯度差值幅度，CUDA反向非确定性为疑似因素而非已证实唯一原因。不能解释成ILPD攻击无效。四resume均等于原UADA500且optimizer state为空，基线未改。

2026-09-27用户授权此类同范围工程错误直接修复、预检、给命令，今后不再重复请求修复许可；仍不自动提交GPU、不使用重置卡、不借修复改科学设计。协作规范已更新。

新增独立UADA_OFT_ILPD_V2.py/wrapper/config/tests/61号3090脚本，通用helper指向V2。仅启动校验覆盖；原train/_modified_forward函数原样继承。普通前向三参考梯度校准数值波动，α=1输出同Tensor对象、严格特征/动作一致、α=0.5同图局部导数与有限非零RGB梯度保留；身份梯度使用测得普通波动范围并设严重差异上限，完整数值先记后判，不自动放宽。正常启动只比旧预期多2次完整前向，无新增独立诊断作业。工程容差明列附件第4节，非攻击超参数。

V2与V1科学设置全部相同，配置仅变experiment_id/output_root。新输出统一原实验根training_v2/，新日志ilpd_v2_train_<ID>_<arm>；旧training/及136602八日志不覆盖，36证据哈希冻结于reports/FAILED_V1_136602.json。不跨源码签名恢复V1零步断点，从同冻结UADA500重启。

29项V1/V2 CPU/mock测试通过，连同ILA/交替联合回归51项通过；覆盖微小数值波动兼容、严重漂移/identity计算图/局部导数/动作前向错误拒绝并保留诊断、四组RGB/优化器/调度/RNG bitwise恢复。CHECK_ONLY核对300状态/278源码/7500参数/10000追加与35000总更新，四组V2 NOT_STARTED、model_loaded=false、sbatch_submitted=false；原UADA仍0444与冻结SHA，旧证据及V1/EOT代码核对未变。真实GPU启动通过仍待用户提交确认，不宣称已实际修复GPU行为。

手动入口保持：roboticAttack-main内`bash submit_exploring_oft_dino_ilpd_refine200.sh`。只提交V2四组训练，不串联测试。详细预检见原实验根reports/PREFLIGHT_V2.md；训练完成先看training_v2/<method>/outer_0200.png，再四suite闭环。

### 35.3 V2训练完成，四套件闭环测试待用户提交

用户提交的136609四组均完成追加200轮/10000 Adam更新，累计UADA500+200；真实GPU启动对齐均PASS，最终PT/PNG、状态、训练表与日志齐全。四张图与共同起点看起来接近，但8-bit PNG平均绝对变化约6.8–8.0/255，训练不是零更新。±16/255限制使外观变化较小；不能由图像变化、训练损失或hook检查预言闭环攻击成功率。

用户2026-09-28要求先测试。已准备四补丁、四套件、同初态同seed配对正常OFT闭环：Object/Spatial/Goal为主，Long次要；每套件10任务×10初态×4攻击=1600新攻击回合，复用400冻结clean回合。只在腕部贴最终RGB补丁，Front与状态不变，推理不带训练hook。主对比DINO-ILPD−同约束UADA，另比较DINO反向缩放与SigLIP-ILPD对照；CASR在clean成功状态计算，训练成本另列。新测试脚本、配置、62/63号3090作业及`submit_exploring_oft_dino_ilpd_refine200_cross_suite.sh`集中在项目内，输出归`research_runs/wrist_dino_ilpd_v1/evaluation/v2_four_suite/`。CPU CHECK_ONLY含6项测试与补丁/训练证据/冻结clean预检均PASS；尚无CASR结果、未提交任何新GPU测试。所有提交仍由用户手动执行。

## 37. 2026-09-30 当前实施：从头2000轮动态DINO回放

最新附件：`roboticAttack-main/research_runs/wrist_dino_transfer_v1/PLAN.md`。五组原Exploring→OFT UADA/均匀回放/静态DINO/动态DINO/动态打乱，同2000轮日程，自动测试1000与2000。单腕高维RGB、Object源域训练、四suite含Long；A800训练、4090采集及测试，阶段依赖保证本流程总并发≤5 GPU。不自动提交。WHEN参考与未实现部分、同预算边界、更新监测、时间预算见附件；跨模型与实机延后。

## 36. 受扰轨迹回放与DINO恢复引导的Wrist通用补丁

ILPD评估136613/136614已完成，1600攻击回合/1528有效行及身份、初态、统计核验通过。同约束UADA对DINO-ILPD任务等权CASR在Object为32.333对27.111%、Spatial 14.333对16.556%、Goal 34.222对29.889%、Long 79.750对73.500%；四个配对差值区间均含零，ILPD增强候选结束，不扩轮次。最新唯一实施附件为`roboticAttack-main/research_runs/wrist_dino_replay_v1/PLAN.md`。

用户要求明日组会前准备新攻击。新方案以冻结Exploring→OFT UADA2000为共同起点，在源Object初态10–17采集现有补丁实际闭环访问而未攻破的状态。DINO恢复同状态动作给每个replay状态评分，只将评分用于训练抽样。四组均沿用Exploring派生副本的动作主损失、RGB/EOT和200×50追加更新：普通clean续训、均匀replay、DINO评分replay、任务/阶段内打乱评分replay。硬±16/255盒移除、TV与[0,1]保留并在四组保持一致，补丁外观仍由用户审核。比较B−A、C−B、C−D，四套件含Long、同场UADA2000，所有GPU作业由用户亲自提交。训练使用4090、测试最多8张3090；执行门、成本与停止规则见附件。

### 36.1 独立实现与CPU预检完成，待用户手动提交

新方案实现于同名实验根的`PLAN.md`附件、独立采集/合并/训练/评估/统计脚本与64–68号Slurm脚本。6项CPU/mock回归通过；采集预检10任务×5源初态PASS，训练正确等待replay bank，评估正确等待完整训练/人工看图，冻结clean分母99/90/98/95。原UADA.py仍0444与冻结SHA，源码哈希清单可建立；无GPU作业、无真实采集/训练/测试结果。训练手动入口为项目目录内`bash submit_exploring_oft_dino_replay_train.sh`，内部用户执行后才排采集→CPU bank→4090四臂训练；不会自动提交3090评估。四张最终PNG由用户先审，详细许可及8张3090四套件测试命令见附件最后一节。A/B/C/D各2000+200轮同预算，冻结2000轮UADA只作异预算操作参照，不可混称同预算增强。

### 36.2 真实4090采集启动失败与3090收集替代

用户首次提交4090采集array136660，GPU模型加载成功，但至少七分片在`renderer_metadata()`因空OpenGL renderer报错，无task bank产物；下游CPU bank报`DependencyNeverSatisfied`，训练未开始。64号缺失EGL变量是初查问题，补上后用户又以136672单分片重试，仍同位置失败；因此不能宣称已修好4090 EGL。为不再消耗4090仿真队列，新建69号3090采集脚本（旧闭环3090/EGL设置），训练66号仍4090，评估67号仍最多8张3090；训练helper改为69→65→66。用户提交的69号分片136677已完成task0：5条仍成功轨迹、30条评分记录、21条正分，finite/哈希检查PASS，用时1.17分钟；其余9任务尚未跑。按用户请求删除22份失败作业采集日志，仅保留本段故障摘要、方案以及成功的136677日志与数据；助手不自动提交/取消。详细命令和边界见附件开头与实施清单。

### 36.3 受扰回放四套件完成及下轮连续片段方案

136678采集、136679合并、136680四组4090训练、136701四套件3090评估与136702 CPU统计均已完成。50源轨迹、234回放记录、29仍成功轨迹；四组各追加200 outer×50内层更新，用户审核四张PNG通过。四套件2000新攻击回合，1910条clean-success有效攻击。任务等权CASR（Object/Spatial/Goal/Long）：UADA2000冻结42.9/8.4/40.2/74.6%，clean续训52.7/8.3/62.9/86.8%，均匀回放68.4/11.3/71.9/74.3%，DINO回放73.0/17.3/60.7/80.0%，打乱DINO72.8/11.3/68.0/79.6%。受扰回放和追加训练有效；DINO组在Object几乎与打乱相同，Goal低于均匀回放11.2百分点，不能宣称DINO评分带来稳定独占增强。Spatial相对均匀回放的6百分点中5百分点来自一个只有2条clean-success初态的任务。详细原始结果、配对差与状态见`roboticAttack-main/research_runs/wrist_dino_replay_v1/evaluation/four_suite/`。

用户要求基于这轮结果设计并准备下轮实验，只由用户提交。新实验附件为`roboticAttack-main/research_runs/wrist_dino_window_v1/PLAN.md`：以本轮均匀回放2200轮补丁为共同起点，用其重新采集Object源初态的受扰轨迹，把各中心前后连续3个决策块构成短片段。DINO恢复分数仍仅用于抽样；四组普通续训/均匀片段/DINO片段/任务阶段内打乱评分片段，两个固定训练种子、每组再加200 outer，Exploring派生RGB动作主损失与单腕贴图不变。评估仍含四套件与Long，视觉审查之后才允许用户手动提交测试；现有40–49测试状态已影响选法，属于开发集复验。独立目录、脚本、70–74号Slurm入口与预检状态见附件，不回写旧产物或只读源。

### 36.4 连续片段完成及从头500/750/1000学习曲线

136819/136820完成3600回合、3438有效行。两seed平均Object/Spatial/Goal/Long CASR：普通续训70.9/13.3/68.4/87.6%，均匀片段67.3/10.9/62.9/81.1%，DINO片段74.3/13.3/65.4/78.6%，打乱70.4/10.3/62.8/84.1%。DINO相对均匀在两seed前三套件均正，Object分别+6.89/+7.11pp；配对区间多数含零，不能称全面最强或已证实DINO独占增益。

用户希望次日中午组会前完成从头曲线。新唯一附件`roboticAttack-main/research_runs/wrist_dino_scratch_v1/PLAN.md`：seed42随机RGB、总1000轮统一日程，先共享250轮UADA，再仅用本次250轮补丁采集新片段，三臂普通UADA/均匀/DINO从完整250轮优化状态续至1000，在500/750/1000保存共9张补丁。共享前缀计入总预算，不读取历史2000/2200 RGB或其回放bank。新同设置UADA对照消除旧位置/TV/日程差异；旧基线750不存在，不虚构。训练4090、采集3090，后续四套件含Long；预估训练+准备4–5h、测试2–3h另加排队及看图。GPU仍只由用户提交。

## 38. 2026-10-02 Spatial迁移与高维RGB去噪（已完成，结论见38.2）

2000轮迁移实验已完成4000回合，数据/检查点/clean资格核对通过。最佳开发点为diverse_1000：Object/Spatial/Goal/Long=100/23.556/58.889/81.000%；2000点为100/22.667/50.667/78.000%。动态DINO采样单独未稳定超过均匀采样，不能把增强组合收益归因于DINO评分；补丁仍有细碎噪点。

用户批准先更新文档、再实现/预检至手动提交命令。最新唯一附件为`roboticAttack-main/research_runs/wrist_dino_repair_v1/PLAN.md`。复用已完成且计入预算的随机250轮完整前缀和Object回放库，新四组到总1000轮、保持2000周期日程；冻结A最佳1000及原UADA1000参照，不重跑旧实验。B持续TV/HF去噪、C均匀采样去噪、D腕部DINO因果动作训练、E再加入样本特有DINO修复内循环；都保持7500 RGB、Wrist-only、正常OFT闭环。新方法借鉴WHEN两阶段训练，D/E训练时干净SigLIP替换只用于因果目标，部署不携带hook或修复扰动。4090最多4训练卡、3090最多8测试卡，训练后自动测试；所有正式作业仍由用户手动提交。Spatial/Goal仍为开发套件，Long完整次要；不自动使用重置卡。

### 38.1 独立实现与CPU预检完成，待手工提交

新增Exploring派生`UADA_OFT_DINO_REPAIR.py`、RGB去噪/修复objectives、独立wrapper/config/common、正常OFT评估和配对统计、102–106号Slurm与提交helper。源UADA仍0444与冻结SHA，SOURCE_COPY及旧训练器SHA和旧manifest相同。300源记录、258回放记录、随机prefix250完整Adam12500步和2000周期日程核对通过；1000/2000学习补丁不作新初始化。

新训练、workflow与直接transfer/window回归共29项通过；8个Python AST、总`--preflight`入口与5个sbatch语法通过，可建立309份来源文件快照。校准采用固定10任务梯度中位数而非单条饱和硬失败；runtime单条已达动作边界时独立相反边界接口探针明确标注且不进入训练损失。恢复及hook、实际RGB更新保护均已测试。扩大检查另发现归档replay测试2轮mock/200轮固定结尾与float32容差两项历史问题，未修改旧代码或重跑旧实验，详见本轮`reports/PREFLIGHT.md`，不称全仓库回归全绿。

测试自动串联1600新攻击+800历史对照，共2400回合/2292有效行，四suite含Long。109份历史来源可核对、5个旧runtime缺历史hash的限制保留。当前`WAITING_FOR_TRAINING`是正常预期；未加载真实VLA、未查询队列或提交GPU。预计校准5–15min、4090四组并行3–5h、3090八卡测试2–4h，合计5–9h另加排队；手动命令为项目目录内`bash submit_exploring_oft_dino_repair1000_v1.sh`。结果统一`research_runs/wrist_dino_repair_v1/`，真实GPU接口、外观与攻击提升需运行后判断。

### 38.2 完成与负结果闭环

137321–137325整链完成；40个suite-task分片、1600新攻击、2292合并有效行完整且无重复，统计复算一致。B/C/D/E均有实际RGB更新；连续去噪明显有效，但攻击性退化。任务等权CASR（Object/Spatial/Goal/Long）：原UADA1000=39.667/16.444/32.111/81.250%，旧A=100/23.556/58.889/81.000%，B=28/9.667/12.444/49.889%，C=13/8.556/8.556/55.028%，D=10/11.111/7.222/45.417%，E=12/5.111/4.222/41.167%。E内层DINO误差下降11.1%，不代表动作反制有效或CASR收益；clean-SigLIP替换与连续去噪未与repair解耦。停止这四组，不扩2000；不能据此否定早期DINO因果敏感性。

## 39. DINO指导的动作反制500轮六组（已完成，2026-10-02）

用户批准先更新文档、实现/预检到提交命令。最新唯一附件为`roboticAttack-main/research_runs/wrist_dino_action_counter_v1/PLAN.md`，取代38节旧执行入口，不删除历史证据。一个新方法与无η/匹配随机η两个对照，交叉原TV预算0.4与温和预算0.3，共六组；每组同随机RGB/空Adam从0到500×50更新、统一2000日程，不复用250轮优化前缀。共用已有Object源bank和旧A冻结源采样表，绝不加载旧训练P作初始化。

内层固定P/EOT，仅在补丁外腕部RGB用Wrist-DINO偏导增大原错误边界动作MSE；每η重算真实SigLIP、只截内层梯度，候选含0。外层完整正常OFT最小化同一动作目标；不替换clean特征、不叠DINO损失，不称严格全梯度min-max或保证clean恢复。删除持续TV/HF，有限hinge固定尺度。正常四suite闭环自动串联，原UADA500/旧A500同轮次参照，Long必测次要，Spatial/Goal仍是开发套件。

仅新增Exploring副本派生文件，原UADA只读；4090训练≤5卡、3090测试≤8卡，阶段依赖隔离。预计整轮7–9h加排队。助手不提交、不用重置卡；链停500统计，1000/2000和实机/跨模型等待另行授权。

### 39.1 独立实现及最终预检完成，等待用户手动提交

新增动作反制训练器/objective、独立wrapper/config/common、正常OFT评估/配对统计、107–111号Slurm/helper及三组测试文件。55项联合CPU/mock测试通过，9个Python AST与6个shell语法通过，真实`--preflight`核对300 clean/258 replay/500源采样行、两张既有500轮参照和冻结四suite资格。独立拦截验证无模型加载、CUDA初始化、queue或sbatch调用；未自动提交。原UADA仍0444与冻结SHA，12个关键旧文件SHA未变，旧结果不覆盖。

准备签名、来源/实际HF缓存校验、非连续随机η、双表中断恢复问题已修复；DINO与随机/无反制各交叉两档TV，从零独立训练，正常模型部署测试。最多3200攻击回合、3056资格行、400冻结clean；单seed/开发套件/Long次要边界不变。真实GPU启动、外观及增强效果尚待运行。

用户在`roboticAttack-main`执行：`bash submit_exploring_oft_dino_action_counter500_v1.sh`。入口自动串联CPU准备→六组4090训练→CPU验证→3090四套件测试→CPU统计；失败依赖取消，部分提交失败不提交下游。训练约5–6.5h、测试约1.5–2h、CPU10–30min，整链7–9h另加排队。输出统一`research_runs/wrist_dino_action_counter_v1/`，完整导航/审查记录见其中README与reports/PREFLIGHT.md。

### 39.2 完整闭环；停止反制路线扩轮次

137356–137360完成3200攻击回合、3056资格行；80原始分片、40实际runtime身份和统计复算核对通过，无需重做。原UADA500/旧diverse500的Object/Spatial/Goal/Long为11/6.333/30.778/61.806和99/22.556/53.667/77%。N0/N1/R0/R1/D0/D1分别为9.111/13.556/15.778/50.167，87/9.556/71.889/80.139，51/17.444/24.667/54.5，58/18.444/26.889/63.25，62.222/12.333/56.667/69.639，39/7.333/38.111/79.639%。D0−R0的Goal+32pp，但Spatial−5.111pp；主Spatial/Goal平均区间仍跨零。N1最强主平均40.722%，反制没有全面超越。六组都有真实RGB更新；温和TV降低噪声但仍不能宣称外观已完全通过。DINO内层目标改善不等于动作反制或攻击增强。保留结果，不扩这六组。

## 40. 最新唯一实施：恢复受扰采样与高维低噪声约束

用户批准先更新文档、再实现至手动提交命令。唯一最新附件`roboticAttack-main/research_runs/wrist_dino_sampling_clear_v1/PLAN.md`。恢复已验证Exploring动作训练、Object受扰片段采样和多样EOT；不继续WHEN反制内循环，只参考跨条件覆盖。四组旧动态DINO+软TV、旧动态DINO+硬外观、双视图稳定DINO+硬外观、组内打乱+硬外观，同全RGB修正后随机起点、空Adam从0独立到2000×50，无旧250轮优化前缀。评分只用于选样，250/500/1000/1500后由各组当前P动态更新；Object数据之外不混入目标suite。

硬外观采用固定3×3 σ0.5高斯残差≤16/255、TV≤0.3，仅超限修正并核验实际量化PNG；仍7500参数，不锁住初图。具体可行性恢复及舍入余量见附件，不冒称精确欧氏投影或外观/ASR保证。正常模型四suite自动测试1000/2000，与既有同轮次原UADA/diverse比较；共4800攻击回合、4584资格行，Long完整次要。四4090训练≤5、3090测试≤8，阶段隔离。只新增副本；旧结果、只读源不改，不自动提交、不使用重置卡，跨模型/实机延后。实施/CPU预检完成后才给用户命令。

### 40.1 独立实现和预检完成，等待手动提交

新训练器/objective/wrapper/config/common、正常OFT四suite runner与统计、112–116号Slurm和helper完成。新增85项加选定旧代码62项，共147项CPU/mock检查通过；其中146项最终联合通过，独立冷启动提交入口隔离检查另行通过。9个AST、6个shell语法通过；300/258源记录、同轮次既有参照、完整4800/4584模拟网格及48条suite摘要核对。原UADA仍0444与冻结SHA，12个关键旧文件SHA未变。没有真实GPU接口通过、补丁或新CASR，不混称已跑实验。

修正共同起点和量化评分/PNG哈希的CPU规范化，固定顺序高斯避免TF32边界差异；严格PNG预算检查、完整outer原子恢复和每轮自身动态评分抽样重构均测试。入口实际CHECK_ONLY退出0，独立阻断CUDA初始化/模型加载/queue/sbatch检查通过。报告和导航见新根README与reports/PREFLIGHT.md。

预计训练7–9h、测试2–3h、CPU10–30min，整链9–12h另加排队。用户在roboticAttack-main执行`bash submit_exploring_oft_dino_sampling_clear2000_v1.sh`，自动串联准备→四4090训练→验证→最多八3090四suite测试→统计，1000/2000与同轮次旧UADA/diverse正常模型比较。助手未提交，链后不扩任务。

### 40.2 原链末尾工程中断；仅恢复剩余1–2轮

2026-10-03用户授权修复。原链137409–137413中A完成2000轮，B/D原子提交1998轮、C提交1999轮；三硬组最后学习率约1e-9时触发每轮RGB必须移动的过严保护，下游测试取消，无本轮CASR。三硬组已提交步实际RGB更新率均超过99.8%，1000到最近断点平均通道变化约10.55–11.43/255，不是训练未启动或预算修正持续抹掉更新。原日志、断点和来源证据保留。

恢复附件为`roboticAttack-main/research_runs/wrist_dino_sampling_clear_v1/RECOVERY.md`。只修末尾float32舍入健康判据及有界来源兼容；不改科学PLAN、目标、采样、数据、优化器、日程或外观预算。A不重训、不改写；B/C/D保留完整Adam/scheduler/评分/RNG，仅补1–2轮到原2000轮。实际新执行来源与原前缀来源由独立ledger分开记录，来源检查不全面绕过。新手动恢复链为CPU准备→最多3张4090尾轮恢复→完整验证→最多8张3090正常四suite测试→统计；命令和最终CPU检查在附件补充。助手不自动提交。

末尾检查修复18项及原85项CPU/mock回归通过，恢复workflow最终18项和runner6项通过，共127项分组通过；7个Python AST、6个shell语法、独立源码审查和实际新helper `--preflight`通过，待补更新共250次。用户执行`bash submit_exploring_oft_dino_sampling_clear_recovery_v1.sh`；补训/CPU验证约5–15分钟、测试约2–3小时，排队另算。旧失败证据和A全部字节保留，原UADA0444及SHA不变，配置/PLAN不改；来源迁移只许可审定trainer/common两文件SHA替换与八个新执行文件。恢复GPU和CASR尚未运行，详细记录见本轮reports/RECOVERY_PREFLIGHT.md。

### 40.3 已完成的完整结果（2026-10-03）

恢复链137425–137429完成4800回合、4584有效行，无需重做。2000轮A/B/C/D的Object/Spatial/Goal/Long分别为100/8.333/64.667/68.389，20/6.333/9/54.806，23/3.222/10.111/47.806，21/11.333/12.111/49.556%；旧diverse2000为100/21.667/51.667/76.889%。硬组84–87%更新触发外观修正，攻击性明显下降，C未胜过D。关闭硬外观路线扩轮次；外观不再作为新增主损失或硬预算。

## 41. 当前唯一新增实施：状态—动作坐标联合采样

用户批准先更新文档、再实现/预检到手动提交命令。唯一附件`roboticAttack-main/research_runs/wrist_dino_state_axis_v1/PLAN.md`先于代码冻结：三个seed42原始RGB/空Adam独立从0到2000×50，均匀固定坐标0、归一化动作剩余误差联合XYZ采样、DINO恢复贡献×剩余误差联合XYZ采样。保留Object源clean/三帧replay与训练匹配EOT；无TV/HF、旧训练前缀或新增特征loss。DINO恢复仅用于当前P评分，正常训练和部署不替换特征。

M−R1为DINO增量对照；R1−R0同时改变坐标覆盖与难例选样，不单独归因。正常四suite自动测试1000/2000并同场重测原UADA/diverse：4000回合/3820有效行，Spatial单独首要、Goal及SG平均同时报告、Long完整次要。最多3张4090训练、8张3090测试，预计9–12h加排队。助手不自动提交、旧源与结果保留，跨模型/实机延后。当前仅方案冻结，CPU/mock和实际GPU结果分开记录。

### 41.1 独立实现与CPU预检完成，待用户手动提交

新增独立配置、训练器`UADA_OFT_DINO_STATE_AXIS.py`、wrapper、common、正常OFT四suite runner、配对统计脚本、122–126号Slurm脚本和`submit_exploring_oft_dino_state_axis2000_v1.sh`。原只读Exploring `UADA.py`保持0444和冻结SHA；不读取历史训练补丁作初始化。训练三组为`uniform_axis0`、`action_remaining`、`dino_remaining`，全部从同一seed42随机RGB和空Adam开始；唯一像素约束为Adam步后的[0,1] clamp，无TV/HF/平滑/硬投影。

实现细节与41节一致：250/500/1000/1500按各组当前量化PNG刷新Object replay评分；R1按两视图归一化剩余动作误差采样，M按`max(0,e_restore-e_patch)*e_patch`采样，恢复clean Wrist-DINO只出现在no_grad评分中。训练内层仍是正常OFT动作MSE，对单一选中XYZ坐标更新50步；部署和测试均无hook/恢复。

预检已通过：`py_compile`、5个sbatch语法、wrapper dry-run和helper `--preflight`。dry-run核对300条Object clean、258条Object replay、三方法、2000 outer/100000更新、无历史patch初始化、无共享训练前缀、`model_loaded=false`、`cuda_initialized=false`、`sbatch_submitted=false`。正式GPU接口、补丁外观和CASR尚无结果。

用户在`roboticAttack-main`目录执行：

```bash
bash submit_exploring_oft_dino_state_axis2000_v1.sh
```

预计CPU准备/验证/统计10–30分钟，3张4090并行训练6–8小时，最多8张3090测试2.5–3小时，整链9–12小时另加排队。输出统一在`research_runs/wrist_dino_state_axis_v1/`；最终统计在`evaluation/four_suite/`。

### 41.2 提交前审核：修复训练日志导致的自动测试阻塞

2026-10-03用户要求复核最新代码和脚本。发现确定性工程bug：`selected_*`只存在于已评分replay行，clean/bootstrap行缺键，Parquet补NaN，完整训练后的CPU验证必失败。已统一每轮日志字段、增加评分/恢复误差适用标志，统计只汇总适用行；CPU验证重构每次选样并核对分数和标志。训练目标、候选、数据、分数公式、预算均不变。同步将报告元数据对齐为Spatial首要/M−R1主比较、SG联合指标，CPU线程与申请2核一致。

新增18项CPU行为回归全部通过，覆盖真实贴图梯度、三组RGB更新、当前量化补丁评分正负符号/归一化/相同像素与SigLIP保持、中断恢复的patch/Adam/RNG/评分一致、训练表序列化与采样重构、模拟Slurm依赖与失败回执；6个AST和6个shell语法通过。代码继承Exploring派生副本，原UADA仍0444及冻结SHA，旧实验不重跑不覆盖。真实GPU启动、外观及CASR仍待用户手动提交；命令与时间估计沿用41.1。

## 2026-10-05 当前唯一新增实施：Batch8补充与WHEN-DINO推进

用户确认四组从零2000×50训练：Exploring→OFT原始适配UADA-B8、uniform_axis0回放-B8、dino_remaining状态/XYZ采样-B8、WHEN特征核心腕部DINO-dense-B8；复用三种同方法B1的1000/2000补丁权重，只重测。最新唯一附件为 `roboticAttack-main/research_runs/wrist_dino_batch8_when_v1/PLAN.md`，取代旧方案的新增执行入口，历史结果保持。

每更新平均8条不同记录梯度，微批1×累积8后一次Adam更新；100000次更新/800000次受扰样本计算，不能称与B1等计算。DINO采样保持旧评分与XYZ选择；WHEN以非补丁dense的L1偏移+反向token InfoNCE为主目标，引用WHEN §3.3，非完整WHEN复现。Object-only、单腕部50×50全RGB、无外观正则/平滑/旧补丁热启动。

1000/2000七条件正常RGB四suite自动评测，5600回合/5348资格行；Long必测次要。4张4090训练（≤5）、8张3090测试、阶段依赖隔离，先等待用户已有GPU链；用户手动提交，助手不sbatch或使用重置卡。保守动作B8训练45–60h、测试3–4h，排队另算。四组独立副本与142–146号Slurm脚本已实现，38项CPU测试及真实冻结资产预检通过，未启动真实GPU。手动入口为 `bash submit_exploring_oft_batch8_when2000_v1.sh`，目录导航/审核记录见新增目录README和 `reports/IMPLEMENTATION_REVIEW.md`。预计无法保证10月6日中午前整批闭环，不把CPU/mock结果当成攻击效果。

## 42. 2026-10-08 Object版准备：未运行，后被43节Long版替代

用户确认把攻击目标从“泛化的动作/特征距离”收敛为一个可闭环解释的错误：当干净 OFT 的夹爪输出本应闭合或保持闭合时，单腕部 50×50 通用 RGB 补丁诱导夹爪张开，从而造成未建立抓取、接触丢失或掉落。训练沿用 Exploring TMA 的随机 RGB、AdamW、每个 outer 取新状态、每状态 50 次 EOT 内循环和逐步 `[0,1]` clamp；只在独立副本中把离散动作 token 目标适配为 OFT 连续 8×7 动作块中的 gripper 目标。实际 OFT 控制链约定为原始预测 `0=闭合、1=张开`，不能沿用源码中相互矛盾的注释。Front 视角始终不贴补丁，部署时也只有一张腕部 RGB 补丁。

本轮从相同 seed42 随机 RGB、相同 Object 成功源轨迹、相同有效 close/hold 槽位、相同状态顺序和相同 EOT 随机数，独立训练三组 500 outer×50 inner（每组25000次 Adam 更新）：A `plain_gripper_tma` 只最小化正常完整 OFT 的开夹爪目标损失；B `task_region_dino_tma` 同时要求正常完整 OFT 和只保留“物体+夹爪”区域 Wrist-DINO 扰动的区域反事实 OFT 都产生该错误；C `background_region_dino_tma` 使用与B等 token 数的非任务/背景区域作为机制对照。B/C 反事实定义为 `D_cf=D_clean+M*(D_adv-D_clean)`，Wrist-SigLIP恢复干净、未选 Wrist-DINO token 恢复干净；Front、文本和本体状态保持原输入。每一步优化两条分支中更难达到目标的一条，即最小化 `max(L_full,L_region_cf)`，而不是把 DINO 作为小权重附加损失。两条分支的特征都必须由同一张实际 RGB 补丁和同一次 EOT 变换产生，禁止独立优化特征张量。

区域掩码由成功 Object 源轨迹重放得到：物体实例掩码与机器人夹爪近端实例掩码映射到模型实际消费的 16×16 Wrist-DINO dense token 网格；C在非任务区域中按冻结顺序选取与B相同数量的 token。若只能从机器人实例与投影末端位置近似夹爪区域，报告必须标注为近似任务区域，不得写成精确指爪分割。训练池只保留干净模型在相应 action-chunk 槽位确实输出 close/hold 的槽位；正常释放槽位不设为攻击目标。三组共享同一批选中记录和槽位，不能把B/C的差异混入不同数据。

本轮不再优化 XYZ、DINO范数、register、注意力、采样分数或外观正则，也不从 UADA2000/2200 热启动。TMA 的 open-gripper 目标本身来自 Exploring，不能作为本文新颖性；新增可检验点是“任务实体 dense 区域是否足以承载具体闭环错误”，以及任务区域相对等 token 背景对照是否能增强正常 RGB 补丁。之前DINO恢复能解释总体动作变化，不等于已经证明其能控制gripper，因此本轮仍是假设检验，不预设正结果或漂亮外观。

训练结束自动进入正常 OFT、uint8 PNG、无 hook/无反事实替换的四套件测试。Object/Spatial/Goal为主，Long必须完整测试但作为次要结果。主指标仍为clean-success条件下CASR；同时记录干净要求close而攻击输出open的错配、抓取建立失败、接触丢失/疑似掉落等轨迹代理。动作错配本身不能写成真实掉落；Long的正常子目标释放不能自动判失败。测试状态与既有40–49开发网格重叠，须标注为开发集复验，不称全新held-out。

流水线按2026-10-08最新资源要求冻结为CPU准备→最多3张A800区域标注→CPU组池→最多3张A800并行三组训练→CPU训练证据核验→最多5张4090四套件测试→CPU统计；阶段不重叠，不依赖3090。助手只生成并预检入口，不自动提交、不使用重置卡。500轮是本轮停止点：若A本身不能诱导具体错误，或B未优于A与等token背景C，按负结果闭环，不自动扩到2000轮或继续堆诊断。真实GPU接口、掩码质量、补丁外观和CASR只有用户提交后才能确认。

### 42.1 批准实施范围与具体冻结附件

2026-10-08用户确认执行。当前唯一实施目录为`roboticAttack-main/research_runs/wrist_dino_gripper_tma500_v1/`，先冻结其中`PLAN.md`与`config.json`再创建代码。三组名为`plain_gripper_tma`、`task_region_dino_tma`、`background_region_dino_tma`，从零500×50、B1，不自动扩轮。历史分割输出已删除，本轮重建标签采用MuJoCo geom-ID与visual/contact geom映射，替代近端圆盘近似；三组共用有效记录和close槽位。

旧UADA500来自2000轮日程中间点，且目标/训练池与本轮不同，只作参考重测，不宣称严格预算匹配；主要对照B−A、B−C。四suite包含新clean与四张受扰patch，共2000回合，冻结资格总382。wrong-open按同一实际状态逐执行槽计算；接触/疑似掉落明确是模拟轨迹代理，正常释放不自动判错。最新资源为标签≤3张A800、训练≤3张A800、测试≤5张4090，阶段隔离，估计8–14h加排队。详细数据、EOT、区域排除、损失、停止点和论述边界见PLAN。此时仅方案/实现准备，不代表GPU或增强效果已通过。

### 42.2 独立实现与手动提交入口（2026-10-08）

已新建完整只读TMA/贴图/模型加载器快照、独立连续gripper-TMA循环、真实RGB donor区域CF、geom标签收集、共同训练池、恢复、普通RGB评测与配对统计。新代码全部在`research_runs/wrist_dino_gripper_tma500_v1/`，原始TMA/贴图源码字节未改，贴图源码补设只读。RGB无关的保守affine支撑保证三组区域/EOT重抽不随学习颜色漂移。B/C释放正常语言模型图后重算同一RGB donor，通过两阶段链式求导优化max，不直接优化latent。启动仅核对接口，不因一个状态的零局部梯度判负；真正RGB未更新会在训练核验停链。

旧UADA500真实身份由500行日志的25000次更新及RGB结构统计核实；该旧保存器的`outer_iterations=2000`是总日程字段。参考PT、manifest、history共同冻结，不重训旧UADA。自动测试四suite各100初始状态，clean及四种补丁共2000回合；同决策状态成对预测，执行到哪一槽只统计哪一槽，不按分叉轨迹时间硬配对。正常释放、缺测与掉落代理分开，开发网格与Long边界保留。

手动入口为项目根`bash submit_exploring_oft_gripper_tma500_v1.sh`；加`--preflight`只CPU检查，不查队列或提交。147–153号新脚本及阶段依赖都集中在新目录；用户执行提交入口后等待已有GPU/OFT链，最新标签3×A800→训练3×A800→测试5×4090隔离，不自动占用GPU、不使用重置卡。预计标签1.5–3h、并行训练3–5h、测试3–5h、CPU10–30min，排队另计。CPU/mock验证和资产预检不是科学结果，真实图像标签、GPU接口、外观及CASR仍待用户提交。

### 42.3 资源迁移与Long来源后续安排（2026-10-08）

用户因资源占用要求当前训练改A800≤3卡、测试改4090≤5卡。只变资源，不改Object源记录、三组目标、B1、500×50预算、学习率、初始化、EOT或四suite网格。标签重建也改A800≤3卡且先于训练；整链峰值按阶段分别受限，不依赖3090。旧准备时的4×3090标签/3×4090训练/8×3090测试安排已被覆盖。A800沿用已有可参考脚本的`partition=a800/qos=a800/gres=gpu:a800:1`，4090为rtx分区。当前没有本轮PREPARATION/提交回执/训练输出；只读查询用户队列为空，不取消任务、不自动提交。原始Exploring源码和历史结果保留。

资源复核新增9项，完整CPU/mock共43项通过；20个AST、JSON、七个sbatch/入口语法及真实资产预检通过。提交CLI同时显式锁定分区/GPU/QOS，阶段3/3/5限制与配置/脚本/运行设备守卫一致。只读确认账号默认emerg6不在A800分区允许列表，账号具有a800，因此显式`qos=a800`不可遗漏；rtx允许默认QOS。查询QOS每用户6/16GPU配额高于本轮3/5卡限制，但未预留空闲资源。详细证据见本轮`reports/IMPLEMENTATION_REVIEW.md`；科学效果尚未测试。

Long来源采样放在当前500轮夹爪TMA闭环之后，作为独立新批次，不在本批同时更换方法与数据。拟最小三组原始适配UADA/均匀回放/DINO回放，B1，每组同随机初始化、空Adam从0到2000×50，保存500/1000/2000检查点；各方法训练及测试串行衔接，不从旧Object学习补丁续训。三检查点是一条训练曲线，不是三次独立实验；所有同轮次比较保持相同2000总日程。Long状态池、对应OFT-Long模型和动作统计必须配套重建，不能将Object池改名。Long转为源域结果，Object/Spatial/Goal为来源Long的迁移结果，旧来源结果分别保留；不能预设换Long就提升外观或攻击性。本次只记录后续安排，Long代码/数据/作业尚未创建，完成当前批次后再冻结细节。

## 43. 2026-10-08 当前唯一新增实施：直接切换Long来源的区域夹爪TMA500

用户最新决定覆盖42节执行计划：Object尚未提交/启动，不再占用GPU，直接换Long（libero_10）。唯一当前附件`roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v1/PLAN.md`。三组普通gripper-TMA/任务区DINO-CF/等token背景CF的损失、随机7500 RGB、B1、500×50、LR0.002/warmup20/500日程、EOT与正常部署均保持；不是改学习率或旧补丁续训。只在新独立副本适配Long来源；原始Exploring代码只读，历史结果不覆盖，Object版保留为未运行准备记录，其根提交入口明确停用并提供新命令。

真正配套切换数据与模型：A800采集OFT-Long干净成功轨迹，使用libero_10_no_noops真实动作/本体统计。Long十任务、init0–39中每任务5成功轨迹、全程均匀6个不同决策，共300条新RGB/state/sim-state/native-action记录；eval40–49不入来源。失败/过短/尝试状态记录，固定来源网格不足则停，不用Object/Spatial/Goal补足。是Long来源成功状态训练，不是完整RLDS Long训练集或完整Exploring复现；旧Object bank/mask_runtime不再输入训练或标签。

区域适配Long多主体和Close/Turnon器件，取goal操作主体的真实visual/contact geom并集+夹爪；器件region映射父器件的整区域，不称精确活动门/旋钮分割。目的地容器可能在背景，背景不预设因果无关。共同资格池和状态/EOT/槽位配对保持；源bank新生成、SHA冻结，报告按任务/早中晚覆盖与排除，训练只优化实际腕部RGB。旧Object自然DINO因果证据不能直接当Long已证实，Long视角差异较弱的历史边界保留。

自动正常四suite测试仍2000回合；Object/Spatial/Goal为Long来源迁移主结果，Long为完整来源内次要结果，不称Long零样本。每suite仍是自己的OFT微调checkpoint，属于补丁跨suite/同架构权重迁移。旧UADA500明确为Object来源/2000日程500中间点的不同源历史参考，不能当公平Long-UADA基线；主增量B−A、B−C。原始行、摘要和差值保存训练来源/参考角色。同实际状态逐执行槽wrong-open与接触/疑似掉落代理、已查看开发网格限制继续保留。

资源：CPU准备→A800采集≤3→CPU组池→A800训练≤3→CPU验证→4090测试≤5→CPU统计；afterok隔离，无3090依赖。A800显式partition/qos/gres=a800/a800/gpu:a800:1，rtx=gpu:4090:1，每任务一卡。助手不自动提交、不用重置卡。预计Long采集2–4h、训练3–5h、测试3–5h、CPU10–30min，总9–15h加排队，新链尚无实测GPU计时。500闭环后停；Long来源UADA/均匀/DINO采样500/1000/2000另议，不随该链运行。换Long是扩大来源阶段覆盖的试验，不预设跨suite或外观必然更好。

### 43.1 独立实现与CPU预检完成，待用户手动提交

新Long目录已包含完整只读来源快照、Long源采集/逐轨迹恢复、多主体/器件geom映射、新bank/共同池、显式Long模型训练、按来源标注的正常评测及147–153号七阶段。旧Object入口已停用且原入口归档；没有删除任何训练结果。61项CPU/mock、22个AST/JSON、七个sbatch/新入口bash语法、实际十个Long任务及四套件模型/真实统计资产检查通过，保护源码SHA与只读保持。CPUmock包括新源采集和中断恢复，不当真实轨迹/外观/CASR结果。

用户在项目目录执行`bash submit_exploring_oft_gripper_tma500_long_v1.sh`；加`--preflight`只CPU检查，不查队列/模型/渲染/提交。Long库需用户首个GPU阶段新采集，不混旧Object记录。预算、时间、停止点按43节；当前没有GPU/补丁或效果结果，未自动提交。全部输出集中新目录，完整说明见README及reports/IMPLEMENTATION_REVIEW.md。

### 43.2 Long-v1实际工程失败与审核结论（2026-10-08）

用户提交链138446–138452。138446准备完成；138447源数组任务1/2/3/4/5/9完成，0/6/7/8在collect_labels.py的0.02原生/可微动作接口检查退出；138448组池、138449训练、138450核验、138451测试、138452统计因依赖失败取消，未运行。228/300条记录（38条成功轨迹）保留，其中六任务180条完成、其余48条已逐轨迹提交。采集阶段19:03:47至19:12:45约9分钟，是工程失败，不是新补丁、DINO假设或闭环效果负结果。

已确认适配缺口：engine.base覆盖了官方processor像素，torch近似漏原生中心裁剪后的uint8转换；同一真实图CPU/实际processor比较BF16前向最大差达到0.03125。失败帧/逐维差值未保存，尚不能唯一归因每个失败帧；不以放宽0.02阈值绕过。已保存记录未发现夹爪close方向误标，实际raw0关/raw1开的仿真校准通过。另一个Long适配问题是要求背景数量≥全部任务token，任务2的17条close状态因此全被筛掉；实际共同池尚未生成。61项mock测试绕过真实预处理，不能证明真实运行接口一致。原Exploring只读文件SHA不变，不据此推翻旧研究结果。

### 43.3 当前Long-v2工程修复与恢复边界

用户授权继续修复。新独立目录wrist_dino_gripper_tma500_long_v2，失败v1代码、PREPARATION、数据与日志不变。新增native前向/straight-through反向：clean直接用wrapper.prepare_hf_inputs；RGB/EOT按PNG部署同样的round/clamp先转uint8，随后真实官方crop和实际processor，torch裁剪/归一化只提供代理导数。不声称量化可微或近似前向等于真实部署；同一Front、实际DINO donor及正常评测保持一致。保留0.02动作容差，增加close方向一致性、逐维max误差及对应槽位/DoF/观测记录。

区域仍为原goal操作主体整实体/器件+夹爪，不更换新的分割假设。patch排除后n=min(可用任务token,可用背景token)，以两份状态固定排列各取n；背景不足时从任务区取等量子集，不声称覆盖全任务区域。修复等token比较与Long整区域不兼容的问题，损失/候选/来源状态网格/500×50预算不变。不再因为背景少于任务区删除记录；只排除no-close、任务完全不可见或背景为空，报告区域子集比例及任务/阶段覆盖。任务区/背景区仍同实际RGB/EOT，且与补丁颜色/损失无关。

RECOVERY_SOURCE.json冻结十个v1原始分片、旧config和PREPARATION的SHA，并检查模型/真实统计/环境/开发网格未变。只恢复RGB/proprio/sim-state/native-action/原geom掩码/尝试前缀；所有旧clean_action、DINO/SigLIP锚点和close标签重新提取。228条不是旧补丁热启动；六个已完成任务不重演成功轨迹，四个未完成任务继续原状态前缀补缺72条。恢复按完整轨迹保存，失败每task单个候选PT+JSON，禁止混用v1和v2锚点或通过修改旧准备签名绕过。

原三方法随机seed42/B1/LR0.002/warmup20/500 cosine、单腕部RGB7500参数、A正常夹爪MSE、B/C max(full,region_cf)、几何EOT、无TV/blur、25000更新/方法和正常四suite2000回合均保持。资源A800恢复/训练各≤3、4090测试≤5，afterok阶段隔离，不自动提交、不用重置卡、不取消其他任务；500闭环后停。预估恢复/补采10–30min、并行训练3–6h、测试3–5h，总6.5–12h加排队；训练/测试尚无本批实测，不保证效果/外观。仅用户运行新命令`bash submit_exploring_oft_gripper_tma500_long_v2.sh`，--preflight/--dry-run不启动GPU或查/提交Slurm；不重新执行旧v1入口。

修复本地核验完成：72项CPU/mock/真实processor回归通过，26个AST、两个静态JSON、九个shell语法及最终资产预检通过。真实HF缓存processor、真实官方crop函数、真实v1观测上，BF16原生前向差为零且RGB代理梯度有限/非零。旧raw掩码/原生gripper的无patch配对核算从81条可用增至107条，任务2从0恢复17条；这是CPU原始数据核算，不是新GPU正式池或攻击结果。源原文件仍444且SHA不变、旧228条数据及旧代码签名不变；未生成新GPU标签/训练/测试结果，未自动提交。详情见v2 reports/IMPLEMENTATION_REVIEW.md。

### 43.4 2026-10-09：Long-v2训练完成，仅修复正式测试

实际作业138492–138498：准备、全部十任务源标注/补采、组池、三组训练和自动核验完成；新来源300条，共同资格池148条。A/B/C均500 outer×50 inner=25000更新。A800训练实测约1h21m/2h27m/2h48m，三组并行最长约2h48m。训练日志/补丁确有更新，但训练错开夹爪率不是闭环CASR，外观也仍有高频纹理/噪声，不能声称已增强成功。

测试数组138497十个任务全部失败且没有rollout数据；六个在无当前GL上下文时执行`glGetString(...).decode()`触发NoneType错误，四个SIGSEGV（无原生堆栈，暂不能唯一归因）。统计138498因依赖失败取消。此为评测工程失败，不是DINO方法失败。已训练补丁、旧代码、配置、原PREPARATION/RUN_MANIFEST/核验、源数据与失败日志全部保留。

用户授权先修测试。新增嵌套独立目录`wrist_dino_gripper_tma500_long_v2/evaluation_fix_v1`，复制评测器、模型运行器和统计器后修改，不刷新原签名。所有新结果只写新目录。原Exploring源码444/SHA及Long-v2冻结训练源保持。明确绑定当前环境实际offscreen EGL上下文后查询renderer；None给出具体错误，不用另建context或CPU renderer伪造通过。hard reset前绑定旧context、reset后清理旧对象并绑定新context，close前绑定。移除创建后的额外未配对reset，但每回合原hard-reset/seed/init-state/10步等待保持。每suite在同一卡上串行独立子进程，记录阶段进度、返回码和faulthandler原生堆栈。SIGSEGV修复是否充分必须通过真实启动检查，不能由CPU/mock断言。

新链只CPU准备→1×4090四suite真实检查→最多5×4090正式数组→CPU统计；不申请A800/3090、不再采集训练。检查复用原task0/init40五条件，四suite20回合，成功配对状态写断点，完整数组复用，因此最终仍2000唯一回合。检查失败afterok阻断整批；只完整五条件配对状态可提交断点，损坏/变更来源不覆盖。清理失败后的检查重试还须重新检查环境/模型/close，不仅凭已保存回合冒充启动通过。

同时确认已安装LIBERO ControlEnv.reset的finally:continue会吞掉非RandomizationError异常、不断重试。新评测副本调用同一内部reset，保留原摆放随机化异常重试，但EGL等其他异常上报并停止，避免隐性卡死；不修改安装包或原训练调用。

实验范围/来源/权重/条件/seed/部署/指标与43.3及冻结PLAN不变：B−A、B−C主要比较，旧Object-UADA500不同来源历史参考；Object/Spatial/Goal为Long来源迁移主结果，Long必测源域次要；40–49是已查看开发网格；wrong-open只统计实际执行槽，接触/掉落为仿真代理。500闭环后停，不自动扩轮或执行跨模型/实机。

用户手动项目根`bash submit_exploring_oft_gripper_tma500_long_v2_evalfix1.sh`；`--preflight`/`--dry-run`只本地CPU，不查/提交Slurm、不加载GPU模型/渲染。预计单卡启动检查10–30min、完整测试3–5h、统计5–15min，排队另计；测试无成功实测，时间不是保证。助手不自动提交、不用重置卡、不删除历史数据。具体本地核验记录见新目录`reports/IMPLEMENTATION_REVIEW.md`，真实GPU渲染与闭环效果待用户提交。

### 43.5 2026-10-09：第一版启动失败后，仅评测切3090的独立修复

第一版用户提交138591–138594：CPU准备成功9秒，138592单4090启动检查21秒退出，阶段environment_create、0个rollout；数组/统计依赖取消。该版显式make_current没有报错，但GL返回值未通过非空bytes守卫；日志未保存原始值和类型，不能断言两者都None，更不能把“未绑定”或4090型号当唯一根因。原138497四个SIGSEGV也未获真实修复证据。

用户授权修复并允许3090评测。新独立嵌套副本evaluation_fix_v2_3090复制第一版评测/运行器/统计/流水线后修改，第一版代码、PREPARATION、输出与Long-v2训练全部保持原字节。恢复旧成功3090评测的模型加载→环境创建→reset产生真实camera帧→renderer核验顺序。预先加载与robosuite相同的GL/EGL入口；安装版context.free调用eglReleaseThread、make_current不恢复API，因此绑定本环境自己的context前显式eglBindAPI(OpenGL)，校验线程API、实际current-context指针与display。此为合理修复假设，不称驱动唯一根因已证实。保留原每回合seed/hard-reset/init-state/10步等待、无CPU替代及四suite串行独立子进程。

失败记录补齐GL renderer/vendor/version的原值与类型、current/expected context、线程API/display、平台/GL/EGL库名称、已加载驱动库路径与白名单环境变量。make_env尚未返回的故障也写入task_XX_PROGRESS.json。保留只重试RandomizationError的同内部reset，不修改安装包，不放宽模型/初始状态/资产校验。

GPU覆盖仅在新副本evaluation_resources与回执中声明：rtx/gpu:3090:1、启动1卡、正式数组0–9%5，上级冻结科学config中的4090训练历史字段不回写。CPU准备→单卡四suite20个原网格配对启动回合→最多5卡完整2000唯一回合→CPU汇总，afterok隔离；启动失败阻断整批，成功配对状态复用。科学函数frozen_clean/paste_png/run_episode、模型加载/观测/四元数转换与bootstrap统计通过AST逐函数一致核验。B−A/B−C主要比较、历史Object-UADA500角色、Long来源迁移与已查看开发网格边界不变。

新副本46项CPU/mock回归、6个Python AST、6个shell语法以及真实训练资产/原签名预检通过；第一版已有PREPARATION也重新核对源与训练未变。只运行--preflight与CPU tests，未加载GPU模型、渲染、查询/提交Slurm，不生成GPU_SMOKE_PASS。实际GPU成功与攻击效果必须等用户提交后分别确认。入口见本文首行；预估启动10–30min、正式3090测试3–6h、CPU准备/汇总5–20min，排队另计。

噪声结论按上级reports/noise_diagnosis_20261009_v1冻结记录：三张补丁确实更新、PNG与PT一致、相关字段有限；B/C真正优化的目标在88.4%/95.2%的outer记录仍≥0.95，末100轮full训练wrong-open仅9.74%/7.21%，不是CASR。初始高频79.9%降到50.0%/60.3%/64.4%，仍保留很多随机起点纹理。这支持“具体目标未稳定达成”，不支持“数值爆炸”“根本没更新”或“所有历史代码均正确”。前向适配与评测真实工程问题不能被研究假设掩盖；接口/梯度/保存检查通过不再表述为机制利用成功。训练调整留待本批闭环后另行讨论，不在修测试中更换攻击设计或预算。

## 44. 2026-10-09 最后一批：Long来源8方法，原始动作目标与可实现DINO教师

本节是当前执行合同，取代旧pending入口。研究仅限有权限的本地LIBERO仿真；不进入真实机械臂、不执行跨模型或新增方法搜索。用户自己提交，助手只更新副本、检查代码并提供命令。

### 44.1 已完成结果与新问题

Long-v2 evalfix2作业138595–138598已完成：2000唯一回合，5张3090正式测试墙钟1h42m48s。A/B/C任务等权CASR（Object/Spatial/Goal/Long）分别7.11/14.33/3.22/70.50、5/12.33/3/47.67、7/7.11/1.11/38.19%；区域方案不扩轮。旧Object-UADA500是不同来源历史参考，不能替代新Long同预算基线。噪声并不等于未训练：历史diverse补丁与初始随机纹理相关仅约0.008；弱目标、优化偏好和部署接口要分开判断。

### 44.2 八组方法与比较

|编号|目录名|方法|主要比较|
|---|---|---|---|
|1|uada_original|原Exploring-UADA原则适配OFT：无补丁预测的相反统计边界、X维MSE、干净状态|Long同预算基线|
|2|uada_dino|Long replay三决策窗口+动态DINO restore评分采样+legacy diverse EOT|2−1为训练包总收益，不是孤立DINO效果|
|3|uada_dino_freq|2+全维RGB更新频率预条件|3−2|
|4|tma_xyz|固定归一化XYZ=(0,0,0)的连续目标；与5同候选池和EOT，replay均匀取样|TMA适配基线，不称完整官方复现|
|5|tma_xyz_dino|4+相同XYZ动作目标的动态DINO restore评分采样|5−4|
|6|tma_xyz_dino_freq|5+与3相同的全维频率预条件|6−5|
|7|tma_xyz_dino_teacher|5+真实RGB样本教师的DINO dense特征蒸馏，不叠加频率预条件|7−5|
|8|uada_dino_teacher|2+独立UADA RGB教师蒸馏，不叠加频率预条件|8−2|

UADA不把动作简单乘−1，目标按clean输出相对于Long来源q01/q99中点选远端；保持既有maskidx=[0]，覆盖8个未来时间步。原论文使用专家动作与动作token软概率；本适配使用同状态无补丁连续预测与MSE，不能混称原样token复现。TMA直接把三个平移增量拉向同一个动作头目标，其他DoF未直接监督但可能间接变化；动作头归一化零对应统计区间中点，不宣称物理停止。

### 44.3 数据、训练与DINO利用

复用Long-v2已验证300条干净记录（10任务×5轨迹×6决策，source init0–39），不使用148条gripper筛选池、不引入Object数据。重新从这些记录的实际Long连续clean输出计算q01/q99。当前实验是Long来源状态bank，不是全Long RLDS训练集原样复现。

所有学生使用seed42随机3×50×50 RGB，7500维全部可学习；Front/语言/proprio/模型权重不变；B1、LR0.002、weight_decay0、warmup20 outer、2000 outer×50 inner=100000更新，scheduler每outer一步。保存0/250/500/1000/1500/2000，正式测试500/1000/2000。1先从0训250用于本批Long replay采集，随后恢复自身Adam/scheduler/RNG续至2000；其余7组仍从相同随机0开始（不加载该前缀像素）。前缀计入1总预算；不加载任何旧学习补丁。

Long replay用本批250快照、50个原source初态采集；每轨迹6窗口，每窗口3相邻决策，成功轨迹优先（任务无成功则全保留），该选择规则显式记录。2/3/8沿用legacy diverse：clean/replay交替、task/phase平衡、正DINO评分前半候选权重2其余1，500/1000/1500后用当前PNG量化补丁刷新。评分=恢复clean Wrist-DINO后目标MSE−正常补丁目标MSE，Wrist-SigLIP仍攻击、Front仍干净；不是仅按特征幅度挑样本。2/3/8保留旧soft-TV阈值0.4/权重1，不将其称为无外观项。4/5/6/7同候选池、同EOT，无任务/背景区域minmax、无额外TV。6不是重跑区域路线。

3/6是AdamW所提RGB更新的全频正增益预条件：rFFT增益(1+8f²)^−0.75，全部频率>0，无硬截断/降维/后处理模糊；首次非零更新按clamp后RGB L2步长与未预条件提议对齐，标量此后固定并保存。目标损失不改变，不保证低噪声或更强CASR。

7/8各自教师bank：对clean+replay去重的每个Long状态，用同权限同尺寸的样本RGB补丁固定100更新优化各自主动作目标；教师不能跨UADA/TMA混用。用固定source评估EOT比较初始/最终动作目标MSE，再恢复clean Wrist-DINO判定该特征改变是否帮助目标；无效教师不蒸馏，但状态仍参与主动作训练。保存实际消费者DINO特征及教师RGB；学生每次按完全匹配的EOT重新取得detach的教师dense特征，不跨状态平均方向，不复制教师像素初始化。主动作MSE+归一化DINO变化匹配；辅助系数只在Long来源首个有效样本按RGB梯度范数比0.1校准一次、固定并记录，不按测试结果调参。所有学生均可正常RGB部署，测试无教师、无特征替换。教师额外计算单独报告。

### 44.4 部署、结果与资源

沿用成功3090 evalfix2的真实EGL/OpenGL上下文恢复和原生uint8/PIL前向；训练使用明确STE反向，不把Torch近似像素当官方前向。所有正常部署都只贴量化PNG到Wrist (26,137)，8步动作块；前向接口/有限梯度检查嵌入训练，不设独立smoke门或外观人工门。

Object/Spatial/Goal是Long来源迁移主结果，Long必测源域次要；各suite10任务×init40–49，配对种子取既有网格，已有看过的development网格不称新盲测。24个攻击快照×400=9600攻击回合，加本批fresh clean400=10000回合。以本批clean-success计算task-equal CASR，保存失败率/逐任务结果与配对差值；不能用历史clean替代分母。按task/state两层配对bootstrap输出预设比较，不扩展参数/种子扫描。

阶段CPU准备→1×4090前缀→≤8×4090采集→CPU bank→≤8×4090教师分片→CPU教师合并→≤8×4090八路学生→CPU资产核验→≤8×3090测试→CPU统计。每作业一GPU，不是单补丁8卡DDP；各GPU阶段afterok串行，任一阶段失败不推进后阶段。每个方法/来源任务可断点续跑，不自动取消其他作业、不使用重置卡。默认先等待用户已有GPU作业结束，避免总占用超上限；调度不可达时拒绝提交。提交中途失败会显示已提交job ID，保留日志和数据供修复。

初步时间（实际启动后，不含排队/代码实现）：前缀约1–2h，采集与bank0.5–2h，8卡分片教师3–6h，8卡学生8–16h，8卡3090测试5–8h，CPU组织约0.5h；合计约18–35h，建议预留两天。根据首批实际进度更新，不保证4090必然快于旧A800。教师/学生均使用保守Slurm上限，不以任务时限当耗时预测。

参考：Exploring(https://arxiv.org/html/2411.13587v4#S3.SS4)、Distillation-Enhanced Physical Adversarial Attacks(https://arxiv.org/html/2501.02232v1#S3.SS4)、Feature Visualization预条件(https://distill.pub/2017/feature-visualization/#preconditioning)。后两者不是VLA成功证据。完成本批即整理OFT方法，不自动新增第9/10方法。

### 44.5 实现完成与本地核验

独立目录已包含完整原源快照、来源SHA、配置、Long replay/教师/学生训练、PNG部署、fresh-clean CASR统计、十阶段Slurm与手动根入口。26项CPU/mock通过，18文件AST与全部shell通过，300条Long来源和四suite模型资产预检通过；原源仍0444且SHA不变，TMA快照只补EOF换行。修复了检查发现的整型XYZ目标写入浮点动作张量dtype问题。成功3090 evalfix2的实际环境创建/重置/上下文逻辑逐函数AST保持一致，只泛化阶段GPU型号；不把该代码继承当4090已GPU成功。详细核验见独立目录reports/IMPLEMENTATION_REVIEW.md。

根入口`bash submit_exploring_oft_long_final8_2000_v1.sh`包含自动测试；`--dry-run`只CPU、本地只读，不查询/提交Slurm、不加载模型/渲染。助手未sbatch、未取消任何作业、未使用重置卡。当前controller查询不可达，无法保证实时8卡空闲，18–35h是实际启动后的估算，排队另计。等待用户手动提交后检查真实GPU接口、教师有效比例、更新进展和闭环结果。

### 44.6 首次执行结果与采集工程修复 v2（2026-10-09）

v1手动提交作业138663–138672；原始UADA前缀138664实际完成250 outer/12500 Adam更新，GPU计时2937.279s（48分57秒）。250补丁相对随机初始化RGB MAE=0.277837，最后一outer移动MAE=0.0091353、50/50实际RGB更新，不能称完全没训练，也不能据此声称攻击成功。采集数组138665十任务均失败于首次环境OpenGL信息读取：8个`GL.glGetString`段错误、2个空/无效vendor/renderer；未产出replay分片，八路正式训练和测试尚无科学结果，不将工程失败当假设失败。

明确适配遗漏：v1 collector先导入模型/TF辅助代码与加载OFT模型，才预载EGL；此前成功的evaluate则在模型前预载。崩溃发生在高层GL字符串查询，导入/库分派顺序是修复依据，不宣称已证明全部唯一根因。

独立修复目录`roboticAttack-main/research_runs/wrist_dino_long_final8_v2/`，新手动入口：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_long_final8_2000_repair_v2.sh
```

v2只改工程：采集在模型/辅助导入前缓存预加载同一个EGL/GL平台；使用该EGL `eglGetProcAddress`解析`glGetString`与精确C ABI，不走旧失败高层查询；仍验证LIBERO当前真实context、NVIDIA/已分配GPU与真实uint8相机帧，不禁用检查/不CPU渲染。初始化phase和Python错误证据提前落盘，C段错误留前一phase。另修此前为空的`evaluation.seed_sources`：引用已完成evalfix2网格，只使用初态/种子，本批fresh clean仍全新执行；CPU检查四suite共400唯一(task,state)，不借旧成功率。

经来源SHA、模型JSON、参数与250状态核对后，CPU续接该同批原始UADA断点：0/250 PNG像素字节不变，Adam moments/12500 step、outer cosine、全部RNG、历史、时间不变；仅桥接新版本signature和测试种子配置，记录`reports/PREFIX_ADOPTION.json`。其余七学生仍seed42随机0；不是用旧补丁热启动新方法。v1源码/补丁/失败日志与Exploring原源只读SHA未改。

调度改为CPU准备/续接250→原定task0的5条真实Long轨迹→原定task1–9的45条轨迹→原教师/学生/自动测试/统计。没有新增smoke或额外回合，不再提交250训练，避免未成功就同时启动八个采集失败任务。GPU仍阶段afterok串行，训练最多8×4090、测试最多8×3090，八方法/损失/轮次/规模不变。

35 CPU/mock、19文件AST、shell、真实本地资产与prefix桥接核验通过；学生/教师/损失/数据处理/正常测试/统计代码与v1逐字节相同。真实4090渲染需用户重新提交后的task0确认，不能将CPU检查等同GPU成功。新日志/补丁/结果均在v2目录。当前Slurm查询不可达，助手没有自动提交或取消；若旧下游仍在队列，用户可只取消旧ID138665–138672后运行新入口。剩余预计17–34h，排队另计；详细核验见v2 `reports/REPAIR_REVIEW.md`。

### 44.7 当前执行状态：八路2000已完成，仅评测改4090（2026-10-11）

v2已完成真实4090采集50条轨迹、replay/teacher bank及八路正式训练。2026-10-10下午学生训练中断，8份RGB/Adam/scheduler/RNG断点完整；独立恢复入口未改冻结科学代码，手动恢复批138818完成八路2000 outer/100000更新。2026-10-11约00:36–01:39训练陆续正常完成，各路累计记录计算约6.70–7.29h；138819于01:41完成24个500/1000/2000 PNG/PT自动核验。不是因训练失败而重做方法，目前本批闭环CASR尚未产生。

138820测试仍PENDING/ReqNodeNotAvail；唯一3090节点aias-compute-01为DOWN+NOT_RESPONDING，Slurm理由Node unexpectedly rebooted。138821等待afterok为正常依赖。重新提交相同3090请求不能修复节点。用户已授权**仅评测改最多8张4090**；不重训、不换数据/目标/种子/初态/补丁位置/动作接口，不新增smoke，不改变4suite×10task×10init×25条件=10000回合、fresh-clean CASR及配对统计。

独立评测副本位于`roboticAttack-main/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/`；根入口`bash submit_exploring_oft_long_final8_eval4090_v1.sh`。仅CPU准备→4090数组0–9%8→CPU统计。先由用户仅取消旧未启动测试138820及统计138821，助手不取消/提交。入口检测重复/旧活动作业，调度不可达拒绝提交，其它本人GPU作业先等待。

评测复制父v2已在实际4090采集中跑通的运行器，保留同EGL preload、精确glGetString C ABI/NULL检查、实际LIBERO context/OpenGL API与真实camera、hard-reset重绑定/close绑定，明确要求实际CUDA4090和NVIDIA4090渲染器。正常episode函数与task_casr/paired_bootstrap函数AST保持；新增phase/原生子进程退出码、失败清理和完整close后再写COMPLETE。四suite独立子进程，完整episode按断点恢复，原8路补丁/模型参数和父训练config/身份不动。

17项CPU/mock、5 Python AST、全部shell及真实24 PNG/PT SHA/父来源预检通过；CPU/mock不是新的4090闭环成功证据。父训练signature保持225b6161194a8eb2048ba5992273715e41d949fbefeef49bcec6a596f76e5cc2，新评测signature为f3381b995746936c8f22bc3537b2cd589b9309e97ff587b0a68d634a3eac7290，分别记账。预计启动后测试5–8h、CPU约0.5h，排队另计。新日志/结果全部集中评测子目录，最终汇总为`results/FINAL8_SUMMARY.csv`。详细核验见该目录`reports/IMPLEMENTATION_REVIEW.md`。

## 45. 2026-10-11：官方专家数据八路1000轮补齐与DINO特征推进（已完成历史批次）

用户已确认并授权更新文档、实现、CPU检查到提供手动命令；助手不自动提交、不查询/取消作业、不使用重置卡、不删历史结果。本轮仅在有权限的本地LIBERO仿真；跨模型和实机暂不执行。第44节为保留的上一批历史，不再是新任务入口。详细冻结合同见`roboticAttack-main/research_runs/wrist_dino_official_long8_1000_v1/PLAN.md`。

### 45.1 八路及本轮目的

|ID|method|方法|主要比较|
|---|---|---|---|
|A1|uada_original|官方专家标签决定相反归一化边界，X维8步MSE|新UADA-OFT基线|
|A2|uada_dino|A1＋当前补丁的Wrist-DINO恢复贡献采样|A2−A1|
|A3|uada_dino_freq|A2＋全秩频域RGB更新预条件|A3−A2/A1|
|B1|tma_xyz|固定归一化XYZ=(0,0,0)，8步MSE|新TMA-OFT基线|
|B2|tma_xyz_dino|B1＋同目标DINO贡献采样|B2−B1|
|B3|tma_xyz_dino_freq|B2＋相同频域预条件|B3−B2/B1|
|C1|tma_xyz_geometry|B1＋动作相关的DINO尺度/横移方向|C1−B1|
|C2|tma_xyz_attention|B1＋动作敏感dense query对补丁key的native Q/K引导|C2−B1|

C1/C2不叠加采样或频域优化。WHEN-DINO、教师蒸馏、旧任务/背景区域TMA、旧补丁/前缀及额外uniform replay臂均不做。普通基线仍需随机抽官方train窗口，不将这一正常数据读入称为额外uniform方法。

上一批使用clean-policy rollout/replay bank，不是本轮官方专家标签数据。旧结果保留，不用旧基线冒充新数据同预算基线。A1/B1是Exploring源码副本上的连续OFT适配：保留RGB参数、AdamW、外内循环、outer cosine及源几何变换；离散token目标改成连续动作目标，因此不称完全未经修改的官方论文复现。具体文件及差异见本轮`SOURCE_PROVENANCE.md`。

### 45.2 数据与共同预算

原数据目录`/home/caijiasheng25/Desktop/roboticAttack/dataset/libero_10_no_noops/1.0.0`：32个TFRecord，379官方train episode。建立完整有效8步窗口池，固定seed抽1000×4候选，缓存实际需要的当前双视角/语言/8维proprio和未来8步专家动作；不采仿真测试数据、不用模型预测充当GT，不改写原数据。官方LIBERO gripper转换、Long checkpoint BOUNDS_Q99/mask和原生JPEG/lanczos3/中心crop/PIL processor均保持。

八路从同一seed42随机3×50×50高维RGB开始；B1、1000outer×50inner=50000次AdamW更新，lr0.002/wd0/warmup20/cosine1000，scheduler每outer一次。所有方法共同每outer前25固定位置(26,137)、后25源随机位置/旋转/剪切和0.8–1.2亮度/对比度。共同EOT是OFT适配配置，不与旧不同EOT结果混称只改变单一因素。没有TV/HF辅助损失、低维补丁或外观否决门。保存0/250/500/750/1000 PNG/PT；每10轮保存RGB、Adam、scheduler、所有RNG和历史，服务器中断恢复完整点。

模型/Front/语言/proprio冻结，唯一优化参数是腕部RGB。新运行器在作业内存中将官方checkpoint自动同步助手替换为只读一致性检查，模型配置/代码不自动改写；未修改原wrapper/安装包/模型权重。原Exploring源文件及本轮完整源快照保持只读。

### 45.3 两条特征推进路线

C1每个源窗口现场取clean动作对Wrist-DINO的梯度；只改变腕部图像的尺度0.9/1.1和横移±4px形成四个DINO参考。在归一化XYZ目标空间，按局部动作损失下降方向选择参考，RGB补丁匹配其非补丁dense token变化。没有正方向就保留正常动作主训练并记录。它是二维几何代理，不称精确物理相机/物体距离干预；不沿用旧混合物理单位Jacobian/教师文件。

C2同一clean动作梯度范数选择dense query，读取真实DINO第8/16层native Q/K，保留q_norm/k_norm及scale，在softmax之前吸引任务动作敏感query到实际补丁keys；不以CLS/register为query，不重写推理注意力。方法依据为Attention-Fool，动作敏感query是本轮适配。C1/C2辅助RGB梯度范数每步为主梯度10%，不做强度搜索/冲突投影；保存实际比例、夹角、投影/分数及主目标损失。

### 45.4 自动闭环、输出与资源

只正式测试1000：fresh clean＋8张正常PNG，四suite×10任务×init40–49，共3600唯一回合，同状态同reset seed、正常8步部署。测试无训练hook、DINO替换、参考几何或教师。Long来源内，其余三套件为补丁跨suite迁移；40–49是已有开发网格，不宣称新的盲测初态。分别报告各suite clean/attack任务成功率、task-equal clean-success CASR、对应基线差值及配对区间，不做跨suite平均主结论。

CPU官方数据准备→最多8×4090八路训练→CPU自动核验→最多8×4090四suite测试→CPU统计，GPU阶段afterok串行，不重叠16卡。使用上一批实际4090成功运行器的相同EGL preload/真实context/API/C ABI/相机检查和每suite独立进程；正常episode AST不变。无独立smoke任务；训练完成后自动贴PNG测试，不等待人工外观批准。核验记录实际RGB更新及0/250/500/750/1000差异，不能用“Adam计数完成”代替补丁真的移动。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_official_long8_1000_v1.sh
```

`--preflight`/`--dry-run`只CPU，不查队列、不占GPU、不提交。全部文件集中`research_runs/wrist_dino_official_long8_1000_v1/`：补丁`training/<method>/outer_1000.png`，数据`data/`，日志`logs/`，正式结果`results/LONG8_SUMMARY.csv`与`RESULTS.md`，提交/核验`reports/`。没有分散写入旧outputs。

估计正式启动后：官方索引/缓存0.5–1.5h，8卡并行训练5–7h，8卡测试2–4h，CPU核验/统计10–30min；整链约8–13h另加排队。历史2000单卡6.70–7.29h作估算依据，本轮官方数据/C1/C2尚无GPU实测，正式日志据实际吞吐更新。当前仅实现与CPU检查完成，未产生本轮训练/攻击结果。

### 45.5 实现完成与核验

32项CPU/mock通过，19 Python AST、1 config及7 shell语法通过，真实本地数据/模型/种子preflight通过。实际官方214步episode上，专家标签归一化与官方函数误差0、双视角原生resize误差0；真实checkpoint image processor和官方crop下，固定PNG部署/训练前向差0、固定/随机EOT保持Front且RGB STE梯度有限非零。这不是实际策略动作梯度或闭环ASR；真实VLA接口检查嵌在正式训练，不新建smoke作业。

全部八路模拟循环、完整Adam/scheduler/RNG恢复、DINO restore、几何/注意力梯度、全秩RGB步长与单腕PNG/配对统计已核验。输入指纹52个模型JSON/代码、24个权重stat、32个官方shard及400个seed保持；原源0444/SHA不变。无GPU执行、仿真、Slurm查询/提交、重置卡或旧结果删除。本轮signature及详细记录见`reports/CPU_PREFLIGHT.json`和`reports/IMPLEMENTATION_REVIEW.md`，源码差异见`SOURCE_PROVENANCE.md`。用户可运行45.4手动入口，训练后自动测试。

### 45.6 完成状态及本次重训依据（2026-10-12）

八路1000轮已完成，40个suite-task测试分片全部COMPLETE，3600唯一闭环回合；完整数据见该批`results/LONG8_ROLLOUTS.parquet`及`LONG8_SUMMARY.csv`，不重跑。四个本次保留方法的Object/Spatial/Goal/Long task-equal CASR依次为：UADA 8/19.56/19.75/100%，UADA+DINO+频率24.11/29.67/46.89/100%，TMA-XYZ 8/15.11/44.33/100%，TMA-XYZ+DINO 8/20.56/50.56/100%。Long为来源内，不把100%扩张为全部套件或跨模型结论。

本批实际LR在500/750/1000轮约为0.001035/0.000307/5.14e-9，保存结束后为0；500–750及750–1000 RGB/PNG均发生实际变化，不能称未训练，也不能仅凭外观认定已收敛。先前25固定/25随机、额外亮度/对比度是助手的适配选择，不是论文要求。本次按用户要求撤回这些选择，但不据此倒推它们是旧外观/效果不佳的唯一原因。

## 46. 2026-10-12：四路官方Long从零2000轮，恢复作者原始几何流程

用户授权更新文档、在独立副本中实现并完成CPU检查，直到提供手动提交命令；助手不sbatch、不取消作业、不使用重置卡、不删除旧结果。研究仅限有权限的本地LIBERO。跨模型另行安排，不抢占本次四路训练资源，也不随此入口启动。

### 46.1 冻结方法、共同预算和必要适配

四路：`uada_original`、`uada_dino_freq`、`tma_xyz`、`tma_xyz_dino`。都用官方Long train双视角/当前语言与proprio；UADA由专家未来8步归一化动作确定X维相反边界，TMA保持归一化XYZ=(0,0,0)的8步MSE。不增加教师、区域反事实、WHEN、注意力、几何方向辅助或额外采样臂。

每路从seed42随机3×50×50 RGB、空Adam开始，2000outer×50inner=100000更新；B1、峰值0.002、weight_decay0、预热20outer、余弦总周期2000，每outer一次scheduler。保存0/250/500/750/1000/1500/2000，正式比较1000和2000，不按测试结果选最优。原生OFT连续动作头、双视角仅腕部贴图、native uint8/PIL前向及STE反向是明确保留的适配，模型权重不更新。

### 46.2 用户批准的变换恢复及差异账本

全部50次inner都随机取合法贴图位置；每次调用未改的作者旋转/剪切采样：20%恒等矩阵，否则旋转[-30,30]度、双轴剪切[-0.2,0.2]，按作者S@R、affine_grid/grid_sample与border/align_corners=False插值。恒等分支仍随机位置。禁用额外亮度、对比度、缩放、固定位置比例、重抽/可见性筛选，不改变源函数。几何可以使补丁部分出画，不加入论文之外的完整可见门。

DINO采样仍只用当前补丁的clean-DINO恢复所造成的目标损失回升评分、正分前半权重2/其余1、四候选取一；为遵守本次原变换流程，评分输入也采用同一种原几何随机变换，四候选使用同seed以配对位置/矩阵，评分不再固定(26,137)。训练用每outer/inner冻结seed，四方法相同。频率组保持原全秩RGB步预条件，不新增外观loss或事后模糊。原数据不改，重新为2000×4候选生成缓存，不加载任何旧学习补丁/优化器/回放库。

论文与本轮不可混称完全相同：作者示例B8，本轮沿用批准的B1；动作token损失适配连续MSE；8步目标与单腕威胁模型保留。本次只恢复几何流程、扩大轮次，不擅自更改这些已确认设置。相对第45节，训练总周期与EOT同时改变，因此旧1000与新1000不是单因素轮次对照；新四组内部和同方法1000/2000才是本轮匹配比较。UADA增强是采样+频率组合，本轮四组不单独分离两者贡献。

### 46.3 自动测试、资源与期限

CPU准备→4090四组训练array0–3%4→CPU资产核验→4090测试array0–9%8→CPU逐suite统计，GPU阶段afterok串行。fresh clean400＋4方法×2快照×400=3600唯一回合，O/S/G/Long全部测试；每suite10任务×init40–49，同reset seed、50×50单腕PNG位置(26,137)、原生8步部署、相同OFT超时预算。测试不加载训练hook/DINO恢复或教师，不改模型参数。

正常测试位置和预算沿用第45节，并不冒称论文suite-native位置/500回合/FR协议。报告每套件SR/FR、task-equal clean-success CASR、同轮次两项增强−基线差值，以及四方法2000−1000配对差值/区间；不作跨suite平均主指标。Long为来源内，其余三suite为像素补丁迁移，既有40–49是开发复验，不称新盲测或保证未见初始化。

训练每路一张4090，最多4张；测试每任务一张4090，最多8张；没有双卡DDP或额外GPU smoke。手动入口等待既有本人GPU作业避免超额，重复提交/来源漂移保护与完整RGB/Adam/scheduler/RNG恢复保留。所有输出集中`research_runs/wrist_dino_official_long4_2000_v1/`，不写旧outputs。

上一批四路线1000实测3.31/3.88/3.37/3.75小时，2000粗估6.6/7.8/6.7/7.5小时，恢复原EOT后预留并行7–9小时；官方4000候选缓存实测87秒，本批8000候选CPU准备预留3–10分钟，含额外检查/统计CPU共10–30分钟。相同3600回合上一批总GPU时间10.80小时、8卡理想1.35小时，本批测试保守2–4小时。整链预计9.5–13.5小时另加排队，10月13日中午组会前收尾；实时速度以正式日志为准，不保证攻击或外观改善。代码与CPU检查已完成，新批GPU未提交。

### 46.4 实现完成、检查和手动命令

独立副本已实现四组从零2000、原位置/几何EOT、无额外颜色扰动、1000/2000八张PNG自动测试及逐suite两档统计。37项CPU/mock通过，18份Python AST、config JSON、7份shell语法和真实本地379演示/32shard/52模型JSON代码/24权重stat/400seed预检通过。原变换与源逐RGB值、梯度一致；基线与DINO+频率模拟中断恢复一致；实际正常episode函数与成功4090评测AST保持。旧八路签名a04fd8d98715137788479cbb14eaccec4a032e71500012ee6a40c0f5f1b6110a保持，原Exploring源0444/SHA不改；没有GPU训练/仿真、Slurm查询/提交/取消、重置卡或旧结果删除。

手动命令：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_official_long4_2000_v1.sh
```

包含自动测试，不需再单独提交测试；`--preflight`仅CPU。全部新输出位于`research_runs/wrist_dino_official_long4_2000_v1/`，最终逐suite表`results/RESULTS.md`与`LONG4_SUMMARY.csv`。来源改动和核验见该目录SOURCE_PROVENANCE.md、reports/IMPLEMENTATION_REVIEW.md及CPU_PREFLIGHT.json。CPU检查不当作闭环结果，用户提交后的原生GPU接口检查嵌在正式训练，不新建独立smoke。

## 47. 2026-10-12：源四路进行中，跨模型只准备接口

用户授权查看剩余时间并准备跨模型接口，待第46节结果核验后再迁移测试；不自动启动新GPU阶段。独立目录`cross_model_patch_interface_20261012/`的PLAN/README/config与code保留原四张2000轮补丁，不挑目标模型最优补丁、不重训。优先VLA-Adapter-Pro（本地DINOv2-register+SigLIP）及π0.5-LIBERO（非DINO架构对照），CogVLA暂不进入。本次不做新的π0.5因果验证。

补丁在每个模型原生相机方向/resize之后、processor之前贴到Wrist uint8 224 RGB，位置(26,137)、50×50，Front/语言/proprio保持。Adapter保留作者Pro和8步动作/gripper流程；π0.5保留原10×32显式采样噪声、10×7环境动作、5步重规划、原LiberoOutputs，不应用OFT gripper翻转。同task/state/decision跨条件同噪声。同一目标模型fresh clean＋四补丁，四suite各10task×init40–49，共2000回合/模型；各suite及增强−基线分开报告，不能借OFT clean分母，不把非DINO迁移写成DINO因果机制证据。

接口及episode kernel/资产检查/冻结清单已实现，11项CPU/mock通过；没有目标GPU加载/仿真/新科学结果。Adapter本地四suite权重、头和配置齐，但缺作者运行源码，两次取源遇代理失败/外网超时；准确标MISSING_AUTHOR_RUNTIME，不用OFT代码冒充。π0.5的原JAX checkpoint/归一化/tokenizer/代码/解释器静态存在，GPU接口待正式执行核验。仅当源结果完整核验后再冻结最终PNG；完整环境/Slurm评测脚本待源结果和作者代码核验后交付，不自动提交，目标阶段不与OFT评测叠加GPU。

截至北京时间约15:22，第46节四路在410–460/2000附近，真实更新持续发生；最慢训练仍约6小时，随后自动测试2–4小时加排队。训练139049，核验139050、测试139051、统计139052由原afterok推进。助手当前controller DNS不可达，只依据磁盘日志/状态判断，不把依赖等待称卡死。具体速度按新日志更新；主线仍先完成OFT四路，而非扩展新方法。

## 48. 2026-10-12：作者源码已下载；跨模型补丁改为用户看结果后选择

用户明确跨模型不一定用第46节四张，先看OFT效果再决定，当前只需预留接口。此节覆盖第47节“固定四张2000、2000回合/模型”的默认输入，不改第46节四组训练及自动测试。

Adapter作者源码完整，commit `23fa0c9c159e2aa04341cdd3e924f44061311060`。新版独立目录`cross_model_patch_interface_20261012_v2/`只读引用v1/vendor，旧目录保留。`patch_selection.json`默认为approved=false、patches=[]；不预选、不自动冻结、不按目标模型结果挑补丁。未来可选择本批/旧批与不同轮次的N张，记录id/label/原PNG/来源说明及可选完成结果SHA；条件为fresh clean＋N，单目标400×(1+N)回合，未确定N不承诺固定测试量或时间。

保留单腕50×50 RGB、位置(26,137)、各目标独立fresh clean、四suite10任务×init40–49，以及Adapter原8步/π0.5原10步预测和5步重规划。目标模型不训练、不更改补丁或位置。空/未确认选择拒绝freeze；确认后才冻结PNG、选择清单、来源结果、reset种子、接口和模型metadata/运行代码。当前仍未提供完整GPU/Slurm评测器、不启动跨模型、不与OFT当前链自动相连。

实际检查：四suite权重和原生架构配置静态齐全；作者配置/API字段AST通过。modeling_prismatic与四suitecheckpoint配套代码有实际逻辑差异，原只读loader会拒绝；configuration逐字节一致，processing只有docstring层面差异，去掉docstring后AST一致。不能把下载完整等同加载兼容，也不自动覆盖checkpoint。Torch/Transformers、TensorFlow和作者策略加载helper在CPU分别导入通过；完整LIBERO入口在登录节点MuJoCo扩展导入SIGILL，根因与计算节点是否相同未验证。包版本差异记录，不擅自升级/降级共享环境。

v2共23项CPU/mock通过（最终1.26秒），动态1/2/5张、待确认状态、重复/尺寸/哈希/完成状态、冻结后变更、原生输入/动作等检查完成。原模型代码/metadata和权重stat检查前后保持；未载入权重、执行GPU/仿真、Slurm提交/取消、安装升级或重置卡。具体证据在v2/COMPATIBILITY_REVIEW.md及reports/COMPATIBILITY.json。旧网页ZIP为16:05历史快照；其源码缺失/默认四张迁移状态以本节及用户选择为准。

## 49. 2026-10-13：OFT四路闭环完成，跨模型入口准备完成但待选补丁

第46节四组从零2000训练全部完成，1000/2000正常RGB评测3600回合、40任务分片完整，无重复；STATUS与RAW/SUMMARY哈希吻合。fresh clean O/S/G/L为99/90/96/95成功/100。训练时间UADA6.90h、增强7.50h、TMA6.78h、TMA增强7.37h，outer2000×50；沿作者首outer零LR，实际非零RGB更新99950/100000。此处不重跑旧实验。

|方法|轮次|Object CASR%|Spatial CASR%|Goal CASR%|Long CASR%|
|---|---:|---:|---:|---:|---:|
|UADA|1000|19.22|25.78|56.00|98.89|
|UADA+DINO+频率|1000|22.67|29.00|36.19|98.89|
|TMA|1000|8.11|29.22|41.31|100.00|
|TMA+DINO|1000|12.00|23.78|33.56|95.00|
|UADA|2000|16.00|26.89|48.42|100.00|
|UADA+DINO+频率|2000|21.56|31.00|34.31|98.89|
|TMA|2000|7.22|33.33|40.44|100.00|
|TMA+DINO|2000|10.00|25.78|37.78|97.00|

无跨suite平均。2000轮UADA增强−基线O/S/G/L为+5.56/+4.11/−14.11/−1.11pp，TMA为+2.78/−7.56/−2.67/−3pp。不能称四suite一致增强，也不能把多训1000视为普遍改进。Long为来源内且接近上限；Object/Spatial增量的已有配对区间跨零，Goal UADA退化区间为−26.44至−2.33pp。无全套件统一最佳：Object UADA增强1000、Spatial TMA2000、Goal UADA1000。来源完整表见第46节目录results/RESULTS.md。

跨模型继续第48节用户选择约定，当前approved=false、patches=[]。已询问是否选四张2000、四张1000或八张；用户要求先看效果，尚未确认任何选择。暂推荐若赶组会先比UADA1000基线/增强两张，理由是强Goal基线、强Object增强且同轮次，不把推荐写成批准。π0.5只是非DINO迁移对照，不作为DINO机制证明。

v2已新增GPU原生策略评测器、回合级恢复、按模型/suite分片与统计、3份Slurm脚本和手动submit_crossmodel_test.sh；两模型串行最多8×4090，afterany避免一模型工程失败阻断另一模型，未完整目标不形成整套科学结果。GPU接口检查嵌在正式作业、不加smoke，空清单和队列查询失败拒绝提交，助手未sbatch/cancel/重置卡。

Adapter在独立runtime作者副本中以checkpoint发布的三份HF代码替换不匹配版本，原vendor/模型目录/权重不改；仅去掉未用HTTP客户端缺失依赖，原动作、相机、归一化及夹爪AST保持。静态逻辑对齐与helper CPU导入通过；登录节点MuJoCo导入SIGILL仍须计算节点正式核验。环境生命周期继承本轮已成功4090评测，不导入OFT模型到新目标。32项CPU/mock通过，涵盖执行、恢复、完整模拟网格、统计、资源依赖和误提交保护；不是新科学结果。

确认两张预计测试3–6h，四张4–8h，均排队另计，新目标4090未实测。补丁选择后再冻结原图/来源/seed/代码/资产并给最终手动命令。本次没有新的训练，不改变来源训练/测试集、动作标签、模型参数或已完成OFT签名。

## 50. 2026-10-13：用户确认四张最终2000轮迁移，已冻结待手动提交

用户明确选择第46/49节本轮四张最终2000补丁，覆盖第49节待选状态和助手两张1000建议。UADA与TMA基线均为本轮从随机RGB零起点训练，使用官方Long专家图像及动作标签，在只读Exploring来源副本上做OFT连续动作适配；不是旧批UADA2000，也不是Exploring官方发布PNG。相同运行的1000是中途checkpoint，2000是最终checkpoint。

v2/patch_selection.json已approved=true，顺序固定uada_original、uada_dino_freq、tma_xyz、tma_xyz_dino；对应增强比较基线提前固定。四张outer_2000.png已与原AUTO_VALIDATION的PNG/PT哈希、训练COMPLETE状态、来源3600结果及签名核对，来源签名a6532436ca2a70adc34b81d7043bfa44220b3117bae32629581e2c8edca33178保持。原图、来源训练/测试、模型参数不改。

目标VLA-Adapter-Pro与π0.5，仅零样本测试、不重训。每目标fresh clean＋四PNG×四suite×10task×init40–49=2000回合，两目标4000。Adapter→π0.5串行最多8张4090，最后CPU自动逐suite统计，原生GPU/渲染核验嵌在正式任务，不增加smoke。估计两个目标各2–4h，共4–8h，排队另计，CPU冻结/汇总5–15分钟；新目标4090吞吐未实测。

32项CPU/mock再次通过，四张选择及两目标本地资产预检通过，400配对reset seed核对。FROZEN_TRANSFER已生成并由真实读取器重新核验：137份环境源/任务初态、16份接口/脚本文件，以及目标运行代码和模型资产metadata/stat；冻结SHA c0f66290e7930c87af7e2b1150b72277e1811cc98b12a77c19d87f431eb4ea97。模型权重仅存在/size/mtime检查，不声称全文件hash或真实GPU推理已通过。旧登录节点MuJoCo SIGILL仍由正式计算节点任务核验。

用户手动命令：

```bash
cd /home/caijiasheng25/robopatch_defense/cross_model_patch_interface_20261012_v2
bash submit_crossmodel_test.sh
```

包含两目标测试及自动统计；所有输出在此独立目录evaluation/<model>/<suite>，最终results/RESULTS.md。手动--preflight不提交，恢复--resume只补缺失回合。当前助手仅准备、冻结，没有GPU模型加载/仿真、sbatch/cancel、安装依赖或使用重置卡；SUBMISSION.json尚不存在，不虚报作业id。

## 51. 2026-10-13：跨模型首轮工程失败与独立v3修复

用户提交v2后：Adapter139491的10task在第一Object回合DINO patch embedding卷积报cuDNN初始化失败；π0.5 139492的10task在Torch2.7读取旧LIBERO numpy初态时被weights_only默认安全规则拦截。均无完成科学回合，无*.parquet；统计139493输出INCOMPLETE。这不是补丁失败率或成功率，更不能把工程错误算攻击成功。

用户授权检查/修复/重给手动命令。独立cross_model_patch_interface_20261013_v3复制自v2，不改旧日志/代码/冻结。Adapter沿用已成功OFT的禁cuDNN CUDA卷积，TF仅负责图像处理、不占GPU；目标模型仍Torch CUDA，原权重/层/动作头不改。新增局部安全初态加载，只对SHA冻结的原.pruned_init NumPy float64数组加入最小safe_globals，weights_only=True保留，不全局放开pickle。LIBERO源和数据不改。

36项CPU/mock通过，两目标Python环境各40份真实初态均与原解码逐数值相等；Torch/TF后端配置及作者策略helper CPU导入通过。旧v2两模型各10分片全FAILED、已完成0回合；新v3四PNG、seed/environment资产核对与旧冻结一致，另建新签名，不引用错误行。新GPU修复验证仅正式作业执行，不增smoke，助手不sbatch/cancel/安装/重置卡。

实验不变：四张最终2000正常RGB零样本迁移到Adapter-Pro、π0.5，目标各fresh clean＋四PNG×400初态=2000回合，总4000。Adapter→π0.5串行最多8×4090，预估各2–4h、共4–8h另计排队。手动命令：

```bash
cd /home/caijiasheng25/robopatch_defense/cross_model_patch_interface_20261013_v3
bash submit_crossmodel_test.sh
```

当前新目录不加--resume，原版无成功回合；v3将来中断才--resume。自动统计在v3/results/RESULTS.md，旧版结果/日志保持。队列查询失败拒绝提交。

具体贴图：原始256图按目标原方向/resize变224，只在Wrist原RGB覆50×50，原点左上、x26/y137，列26..75/行137..186，每次策略查询都用同PNG和位置。Front/语言/proprio不改。后续目标原生processor照常（Adapter90%面积中心裁剪、π0.5原padding），并非特征替换或最后pixel_values定点覆图，不搜位置、不跟踪物体、不是仿真桌面上的实体补丁。完整修复证据见v3/reports/REPAIR_REVIEW.md。

## 52. 2026-10-13：Adapter Pro调用分支修复，π0.5保持运行

用户提交v3后，Adapter139520的10task在首回合Pro头None.reshape；cuDNN、初态与EGL问题已越过。π0.5 139521正常进入episode，检查时287条完整回合/7份部分或完整分片，未出现Traceback。不能因长告警而判断两个目标都失败，也不能把已完成π0.5回合重跑。

根因是旧checkpoint里的save_version=vla-adapter标签选择了OFT式单层回归分支，漏传多层特征和proprio，而作者Pro头要求这些。独立v4仅改wrapper选择已发布checkpoint的v1 Pro分支，不补零、不改头或原模型代码；该分支与作者当前Pro方法在重命名后逐AST一致，四suite均验证。40项CPU/mock通过，新增回归实际运行原predict_action/原head方法及mask/拼接（小fake视觉/语言输出，无真实模型加载），旧分支复现同错误，新分支输出有限8×7并保留512任务token、64query、25层/机器人状态。新GPU推理待正式作业。

新目录cross_model_patch_interface_20261013_v4。只新增Adapter阵列及CPU汇总，最多8×4090，等待现有本人GPU结束后启动。π0.5通过只读external_evaluation引用v3冻结manifest/原数据和作业139521，不重训、不重测、不取消或改在跑v3。核对四PNG、reset/noise/视角/动作/模型与原运行源码一致，汇总保留π0.5原v3行签名，完整分片才报告整模型CASR。

用户手动命令：

```bash
cd /home/caijiasheng25/robopatch_defense/cross_model_patch_interface_20261013_v4
bash submit_crossmodel_test.sh
```

默认both为汇总两个目标，但不再提交π0.5阵列；只补测未成功的Adapter2000回合。新增Adapter估2–4h，现有π0.5剩余和排队另计。四图、模型参数、训练/测试集、贴图协议不改。旧日志/冻结和继续运行保留；助手未sbatch/cancel/装包/重置卡。修复证据在v4/reports/PRO_BRANCH_REVIEW.md，最终合并表在v4/results/RESULTS.md。

## 53. 2026-10-13：DINO特征三方向及生成器低成本筛选，已实现待手动提交

用户批准生成器也一起实现。新增目录`roboticAttack-main/research_runs/wrist_dino_feature_screen250_v1/`，合同PLAN.md。四组为同预算UADA、dSVA-inspired原生倒数第二block Key、FSAA-inspired去中心化实际dense、AdvEncoder-cos公开结构通用生成器。新增路线不继续DINO选样、不加附带UADA/教师/任务区背景区、不依据外观淘汰。

前三组优化3×50×50 RGB，从seed42同一初图开始；生成器独立随机网络+固定100维噪声，训练网络而非噪声，导出静态50×50PNG。Key冻结干净CLS→空间注意力权重并破坏整个加权Key网格；dense使用1024条Long train记录固定均值去掉共同特征偏置，不把它直接等同物体/背景分离；生成器保留代码cos目标`5*cos+RGB_MSE−HFC_MSE`，高频项是攻击项，不是降噪。论文原对象与本轮DINOv2/静态通用图案/原Exploring EOT差异逐项列明，不假称完整复现；作者原LICENSE下载故障在来源声明中标记，对外发布前补齐。

四组共享官方Long 379专家演示池上的250条普通记录顺序，250outer×50inner=12500更新，B1。像素lr0.002、warmup20、余弦总周期2000（250时不降零）；生成器Adam0.0002/betas0.5,0.999固定lr。原作者全inner随机位置、旋转±30/剪切±0.2/20%恒等矩阵、无额外颜色变换。保存0/125/250，每10outer完整保存Adam/scheduler/网络/BN缓冲/固定噪声/RNG；源模型全部冻结。

训练后自动fresh clean+四最终PNG，四suite各10任务×init40/41，共400回合。只用正常OFT的真实uint8/PIL前向/8步部署、Wrist50×50位置26,137，Front/proprio不动；测试不使用训练hook或生成器。既有开发初态，仅作方向筛选；逐suite SR/FR/CASR、候选−同预算UADA配对差和正向task数，不跨suite平均。不改变第50节四张2000的跨模型选择，也不取消/重跑第52节作业。

27项CPU/toy通过（10.89秒）；21份Python AST、JSON、7份shell及真实本地资产预检通过。原Exploring三源0444/SHA不改，旧四组签名`a6532436ca2a70adc34b81d7043bfa44220b3117bae32629581e2c8edca33178`保持。比较旧预检，52模型JSON/代码hash、24权重stat、32shard stat及数据metadata hash均相等；不声称大权重全字节hash。新签名`590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620`，证据在新目录reports/IMPLEMENTATION_REVIEW.md和CPU_PREFLIGHT.json。CPU检查不当作攻击效果；真实GPU接口检查嵌在正式训练，没有独立smoke。

CPU准备→4卡4090训练→CPU资产检查→4卡4090评测→CPU统计；GPU阶段afterok串行，并afterany等待现有本人GPU作业。预留CPU准备3–10分钟、最慢训练1–1.5h、评测1–2h，总2–4h加等待/排队；新三方法吞吐未实测。助手没有sbatch、GPU/仿真、作业取消、安装或重置卡。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_feature_screen250_v1.sh
```

一条命令包含自动测试和统计；全部输出集中在新目录，四图`training/<method>/outer_0250.png`，表`results/RESULTS.md`。`--preflight`仅本地CPU资产检查，不查询/提交Slurm。

## 54. 2026-10-13：生成器首次前向工程失败，仅修复启动顺序与依赖

用户提交第53节后，数据139713完成；Key/dense139714_1/2各250轮完成，耗时541.39/585.27秒（dense训练均值21.26秒）。UADA139714_0正常推进，检查中至240/250；生成器139714_3首次ConvTranspose2d前向报CUDNN_STATUS_NOT_INITIALIZED，无初图/checkpoint、没有训练或闭环科学结果。原139715/16/17等待整个array的afterok，失败子任务使该链不能完成。不能把工程故障记为攻击成功/失败。只读队列取得，sacct数据库连接拒绝，未虚报历史会计ExitCode。

根因是本轮初始化顺序遗漏：原load_wrapper已禁4090 cuDNN，但先运行了G的初始化前向，后加载OFT。独立嵌套目录`roboticAttack-main/research_runs/wrist_dino_feature_screen250_v1/repair_cudnn4090_v2/`只增加启动副本，在原训练函数之前执行同一禁cuDNN策略，并记录初始阶段报错。正在使用的原代码/签名不改；G结构/seed/固定噪声/loss/LR/数据/EOT/预算全部不改。

原三组与1267记录缓存只读复用，不训练另三张、不重新采集。只提交单生成器1张4090，可与正常UADA1卡并行（共2卡）；验证afterok依赖新G及仍活跃的原UDA子任务139714_0，不依赖失败的139714整个array。随后沿用原正常OFT RGB评测（4090最多4、400回合）和逐suite统计。等待其他本人GPU，不取消旧任务，旧后续若正在运行则拒绝重复。

38项CPU/toy通过（11新增+27原有，10.42秒），4新Python AST、6shell及真实资产预检通过；针对回归模拟原启动错误/正确提前禁cuDNN、真实外部小生成器全参数反向、早期报错记录和替换依赖。原签名`590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620`保持；新增启动源码/Slurm/入口的独立repair_signature为`5cae09ef184a823f60ea3975b7c87c421dcbd65714e755a511b369d7373ca0f5`，GPU启动前冻结RUNTIME_REPAIR，验证与最终RESULT_LINEAGE关联，不伪称新启动源码包含在原哈希中。

GPU实际修复效果待正式作业，不新增smoke。生成器预留20–60分钟、测试1–2h，总约1.5–3h另计排队；本轮没有助手sbatch/cancel/装包/重置卡/实际GPU或仿真。故障日志和旧输出保留，原始数据/标签、模型参数、第52节跨模型任务不改。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_feature_screen250_cudnnfix_v2.sh
```

新receipt/生成器修复日志在repair_cudnn4090_v2；实际四图仍在上级training/<method>/outer_0250.png，完整四suite表仍为上级results/RESULTS.md。旧SUBMISSION_RECEIPT保留历史，新作业查修复目录receipt。不要重提旧四路；失败诊断和新检查详见修复目录reports/REPAIR_REVIEW.md。

第54节检查收尾时UADA已COMPLETE250，耗时3024.21秒（50.40分钟）；前三组都已完成，后续不需等待原UDA即可只补生成器。当前仍无闭环成功率，勿用不同记录/不同目标的训练loss当作跨suite结果或方法排名。

## 54. 2026-10-13：生成器首次前向cuDNN遗漏修复，只补单组并重接自动测试

用户要求看结果、发现问题直接修复并重给命令。原139713数据准备完成64.37秒，官方379演示/98816窗口、1267缓存；Key139714_1与dense139714_2分别250轮/12500更新完成541.39/585.27秒，原生GPU接口、真实PNG和checkpoint齐。UADA139714_0实际仍RUNNING，最新检查到240/250、2908.06秒，没有Traceback。生成器139714_3在第一次ConvTranspose2d报`CUDNN_STATUS_NOT_INITIALIZED`，没有初图或优化checkpoint，尚未训练，更没有闭环成功率。

实现遗漏是原4090禁cuDNN只在load_wrapper，而第一次G前向先于它。独立`wrist_dino_feature_screen250_v1/repair_cudnn4090_v2/`启动副本先禁cuDNN，再调用未修改的原train。原code/slurm/config/手动入口、源数据、模型权重、前三组/日志不改。源科学签名保持`590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620`；新启动代码另签名`5cae09ef184a823f60ea3975b7c87c421dcbd65714e755a511b369d7373ca0f5`，正式运行冻结RUNTIME_REPAIR.json并关联最后RESULT_LINEAGE，不伪称原签名包含新源码。

原139715/16/17只读squeue为PENDING/Dependency，后续afterok整个139714将被G子任务失败阻断。不使用该旧链，不重提四组训练array。新单G（1×4090）可与正常UDA并行（合2卡），等待新G及UDA子任务139714_0（若仍活跃）成功后，CPU核对→原正常OFT四suite400回合（最多4×4090）→统计。其他本人GPU先等待；旧后续若已运行/重复新修复/UDA异常则拒绝提交，不取消旧任务。sacct数据库拒绝连接，不伪报历史Slurm退出码。

新11项针对故障CPU回归通过；与原27项合计38 passed，10.42秒。4份新AST、6份shell、真实本地资产/缓存hash及原三组状态预检通过。修复速度/真实GPUcuDNN须正式任务验证，未另加smoke，未修改研究损失、EOT、数据、LR、250×50或模型参数。详细证据见修复目录reports/REPAIR_REVIEW.md。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_feature_screen250_cudnnfix_v2.sh
```

GPU生成器预留20–60分钟，测试1–2小时，总1.5–3小时加排队；截至检查原UDA仅剩约2分钟，可直接提交新链，不要再用原四组入口。新作业id/日志在修复目录reports/logs，四张图及最终RESULTS.md仍在原实验training/results，未自动提交、取消、删数据或使用重置卡。

## 55. 2026-10-13：三种不同DINO目标继续低成本筛选，已实现待手动提交

### 55.1 上一批闭环已完成，不重提生成器修复

实际v1结果STATUS为COMPLETE、400回合；原始数据/trace、清洁分母、配对seed、PNG/运行时来源和汇总核对通过。四组同250×50更新、官方Long训练记录，不是旧2000轮对照。逐suite CASR如下：

|方法|Object|Spatial|Goal|Long|
|---|---:|---:|---:|---:|
|UADA250|5|25|5|85|
|DINO-Key|5|20|40|85|
|中心化dense|5|15|30|65|
|AdvEncoder-cos生成器|5|10|45|90|

Key/生成器Goal相对UADA提升35/40pp，分别4/6个task改善、无task退步；Spatial都低于UADA、Object持平。Long来源内。保留Key和生成器，center不扩。实测训练UADA50.40分钟、Key9.02、center9.75、生成器10.03；速度差来自不同反向计算/参数化，不宣称通用端到端5倍加速。两初态、单训练seed为方向筛选。

### 55.2 当前唯一新批次

用户批准上一回复三方向。合同为`roboticAttack-main/research_runs/wrist_dino_objectives_screen250_v2/PLAN.md`，新增独立目录：纯DINO注意力`dino_attention`（重做：不再是10%动作辅助）、signed多层FDA-inspired响应反置`dino_multilayer`、dense区域相似关系反序`dino_affinity`（新假设）。不继续采样、不叠动作/教师，不用任务/背景区域标注，不用外观门。目标公式、block8/16/22、topk8及与原论文差异先记录再实现。

三组同seed42随机7500RGB、空AdamW从零开始，同v1冻结250条官方Long记录，250×50/B1、lr0.002/warmup20/cosine2000、作者全随机位置/旋转/剪切/20%identity，无额外颜色扰动。EOT、uint8/PIL/STE、模型加载/正常评测八个文件逐字复制v1；原Exploring三源仍0444/SHA不改。源数据、标签、模型权重不改；唯一学习参数是外部RGB。新增只读hook不替换输出，测评不加载训练目标。

原v1的clean/UADA/Key/G共320原始回合只读复用（保留原signature/修复来源），新增三图案四suite×10task×2state=240；合并560回合/6方法，不混center或旧1000/2000/官方PNG。仍是已查看的40/41开发初态。逐suite CASR与新候选−UADA/Key/G配对差，不作跨suite平均、不把代理变化当闭环结果。最多保留一两路，扩轮/新初态/物理载体需另行决定。

### 55.3 实现核验和手动入口

新25项CPU/toy通过（11.72秒），原27项独立回归通过（10.06秒）；16份Python AST、2 JSON、7 shell及真实只读资产preflight通过。覆盖原生Q/K归一化/scale/完整prefix分母、目标符号、排除覆盖/自配对、signed强弱分组、hook不改输出且清理、三路实际RGB更新、CPU连续/中断恢复一致、最终断点已提交但status丢失的无模型恢复、PNG一致性、240+320合并且不改来源身份、手动依赖链/重复提交/失效等待处理。

CPU/toy不是实际OFT训练/攻击结果。无真实模型forward、GPU/仿真、Slurm查询/提交/取消、下载/安装、旧文件删除或重置卡。输入来源/复用hash在APPROVED_REUSE和CPU_PREFLIGHT；大权重只沿用已冻结size/mtime检查，不冒称全字节hash。新signature=`35abc0c93dad8534d5f3876c87d8d4c934d55223ba408090473e8aedf3b27542`；旧v1科学及repair签名保持。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_objectives_screen250_v2.sh
```

CPU复用→最多3×4090训练→CPU资产检查→最多4×4090自动测试→CPU统计，GPU阶段串行。提交时等待本人其它可运行GPU作业；永远不能运行的DependencyNeverSatisfied只报告、忽略等待，不取消。已完成训练/回合复核后跳过，未完成恢复完整断点；没有独立GPU smoke。

预留运行2–3小时另加排队：CPU5–10分钟，三组训练0.5–1h，240回合测试1–2h；新目标速度未实测。新PNG在training/<method>/outer_0250.png，完整表在results/RESULTS.md；所有新文件在同一目录。实现细节及核验见SOURCE_PROVENANCE.md和reports/IMPLEMENTATION_REVIEW.md。

## 56. 2026-10-13：Value与生成器对比学习最后补充筛选

### 56.1 第55节已完成结果

v2三组完整250×50，各实际RGB移动12450步（共同首outer warmup lr0），末尾lr0.0019347，训练8.38/9.26/8.90min；40个测试分片完整。新增240回合+320匹配v1原记录=560回合，原数据独立复算、配对seed、PNG/hash、15129条trace和44组差值一致。注意力、多层、关系的Object/Spatial/Goal/Long CASR分别0/10/5/40、0/15/40/80、5/35/35/80。关系相对UADA的Spatial+10pp（3task改善/1退步），Goal+30pp（5改善/0退步）；Object仍5%。关系末25轮反序query平均约0.025%，不能把弱目标变化说成充分关系翻转。完成判断来自日志/产物，Slurm历史库当前连接失败，不冒称sacct已验证退出码。

### 56.2 唯一新增批次与明确适配

用户当前“继续执行、一次性做完”授权到实现/检查/手动命令。先冻结附件`roboticAttack-main/research_runs/wrist_dino_value_contrast_screen250_v3/PLAN.md`：A=纯DINO内部Value（block8/16，空间V完整网格CosDist，VEV-UAP-inspired）；B=现有生成器cos的图像B8对照；C=同生成器、同B8完整跨图像负InfoNCE。C参考AdvEncoder默认nce，不是WHEN的空间token NCE。两条真正补充方向加一必要对照，三组从零250×50，不加载旧补丁或教师，不再扩采样/几何/注意力矩阵。

B/C同一生成器latent batch1产一张图案，训练图像batch8，两个概念不混淆。用完整8×8矩阵和精确两遍微批梯度缓存，不将B1 NCE相加冒充批内负样本；CPU对直接B8梯度验证。B/C12500更新/100000受扰样本，A12500/12500，因此只主张B/C等计算，不与旧B1冒称等FLOPs。生成器保留作者高频项，不承诺降噪。原作者B256/全图UAP等设置与局部OFT适配明确分开。

### 56.3 自动闭环与时间

官方Long原379专家数据只读；复用同250记录池，A原顺序，B/C八条循环、完全配对。作者原随机位置/旋转±30/剪切±0.2/20%identity不变，无额外颜色、外观门、模型更新或测试入训练。新增三图案四suite×10任务×40/41共240回合，复用clean/UADA/Key/G-cos-B1/关系共400旧行，合计640。旧行保留原来源身份，正常部署无训练hook/生成器，逐suite比较，不跨suite平均。

CPU准备→≤3张4090训练→CPU核验→≤4张4090自动测试→CPU统计，阶段afterok串行、不自动提交、不取消本人/他人已有作业。无独立GPU smoke。预留运行2–4h加排队（准备5–10min、最慢B8训练1.5–3h、测试30–45min），B8计算尚未实测，原240回合实测约1.11GPU小时用于估计测试。完成后停止追加路线，从结果最多保留两条；2000/新初态/实机须另行决定。

### 56.4 实现完成与手动命令

14项新CPU/toy通过18.81s、25项旧v2独立回归通过11.98s；18份AST/2JSON/7shell及真实只读来源/模型metadata/原数据/旧结果预检通过。覆盖native V/完整批内负InfoNCE/与直接B8梯度一致/三组RGB学习与完整恢复/最终收据无模型恢复/原部署复制/400旧行保留身份/640网格/配对目标与预算标记。未加载真实OFT或启动GPU，工程检查不当新科学效果。签名`de070ba86a7da4f9547138b8086ba49d47743004614a03773fd2a410b82f2c4d`；来源与证据见新目录SOURCE_PROVENANCE与reports/IMPLEMENTATION_REVIEW。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_value_contrast_screen250_v3.sh
```

同命令已包含自动测试/统计、按断点恢复和完成分片跳过；`--preflight`/`--dry-run`不提交。新图案在`training/<方法>/outer_0250.png`，四suite表在`results/RESULTS.md`。未自动sbatch/取消作业/删除旧数据/使用重置卡。温度0.1是明确记录的筛选值，作者主入口未传temperature实参，utils默认未取回，不伪称作者全配置一致。

## 57. 2026-10-14：用户选定8方法×4个batch，从零32张500轮

### 57.1 唯一当前矩阵与规模

用户选择ID1/2/3/4/6/7/9/12，1/12是基线；分别为UADA、DINO-Key、去中心化dense、生成器cos、多层强弱响应、dense空间关系、生成器负InfoNCE、TMA-XYZ。每个B1/2/4/8，共32张，从零500outer×50inner=25000优化器调用，不续训旧250图。9-B1明确为一类CE=0/特征梯度0的退化对照，保留原RGB/HFC项，不偷换cos或记忆库。

批准合同见`roboticAttack-main/research_runs/wrist_dino_batch500_32_v1/PLAN.md`。此次最新授权6张4090训练，8张4090测试，覆盖旧偏好5卡或前一讨论4卡上限。GPU阶段串行、总峰值8，每arm单GPU，6个最长先排worker承担32组，训练6对/测试10对Slurm日志，不独立GPU smoke。

### 57.2 保持不变与具体适配

官方Long train379专家轨迹的上一轮冻结250窗口池循环取样，原未来8步专家动作、原processor对齐/native RGB/PIL+STE，全部模型参数冻结，只优化RGB/外部生成器。μ复用同编码器1024条Long train记录的固定统计，不是旧补丁暖启动，不增加目标套件、教师、回放或DINO选样。

作者原随机位置/±30旋转/±0.2剪切/20%恒等保持，不加亮度/对比度、固定贴位比例、TV或外观淘汰门。像素lr0.002/warmup20/余弦2000；G固定0.0002/betas(.5,.999)/原5*cos或5*negativeNCE+RGB−HFC，NCE温度0.1，B变化不调LR。可分离损失除B累积再step；NCE完整BxB矩阵+精确两遍微批缓存，不把8个独立B1 step称batch。B8处理样本次数多，不宣称等FLOPs。

13份原生运行时/几何/Key/center/多层/关系/正常评测/生成器文件逐字复制到新目录，旧源码/结果/模型/数据不改。新副本只增加32组loop、B平均/调度及NCE-B1明确退化；来源详见新SOURCE_PROVENANCE与REUSE。原三Exploring源仍只读。

### 57.3 自动测试、交付与耗时

每suite10task×init40–49=100回合/补丁，clean+32条件，共13200。已有开发网格，不称新盲测；四suite逐suite OFT checkpoint及归一化统计、原8步执行/步数上限/Wrist50×50@(26,137)均保持，Front/语言/proprio不变。无训练hook/激活替换/G推理；继承4090 EGL生命周期与首G卷积前禁cuDNN。给出逐suite CASR、相同B对UADA/TMA、同方法不同B、NCE对cos、每任务及GPU耗时，不跨suite平均。Long来源内，其余补丁跨suite，不称跨模型。

保存0/125/250/375/500，正式仅测500；每10outer完整Adam/调度/RNG/G断点、每episode测试断点，重提复核完成部分跳过。默认入口只由用户执行，自动训练→测试→汇总；助手只完成文档/副本/CPU检查，无sbatch/取消/重置卡/实际模型forward。

已有日志外推总约81.8 GPUh训练，6卡最长先排理想13.7h，预留16–22h训练、6–10h测试，总22–32h另加排队。基线B8尚未实测，不写成保证期限。此轮不追加候选或自动扩1000/2000/跨模型/实机。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_batch500_32_v1.sh
```

新目录下`training/m<ID>_<objective>_b<B>/outer_0500.png`；最终表`results/RESULTS.md`，详细BATCH500表及Parquet原始数据同目录；作业ID在reports/SUBMISSION_RECEIPT.json，日志在logs。实施核对报告另写reports/IMPLEMENTATION_REVIEW.md；工程通过不等于新攻击有效。

实施收尾：58项CPU/toy回归通过18.85秒，21份AST及7份shell语法通过，真实只读资产preflight通过。1267缓存记录、全部16000个实际batch构成、双视角uint8/专家8步标签、1024训练均值来源已核对；13份原生文件副本逐字一致。新旧数据/标签/模型/源码未改，没有助手真实GPU/仿真、Slurm查询/提交/取消、下载或重置卡。CPU检查不是实际攻击结果；详见新目录reports/IMPLEMENTATION_REVIEW.md及CPU_PREFLIGHT.json。
