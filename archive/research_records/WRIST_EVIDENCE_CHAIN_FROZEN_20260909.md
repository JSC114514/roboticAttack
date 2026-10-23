# OpenVLA-OFT Wrist 动作偏差：从现象到 DINO 上游视觉几何因果分解

冻结日期：2026-09-09（Asia/Shanghai）  
研究对象：OpenVLA-OFT，LIBERO-Spatial，Front/Wrist 双视角，连续 `[8,7]` Action chunk  
文档用途：交给 Web GPT 收录并冻结当前证据链  
统计惯例：除特别说明外，正式统计采用 task→episode hierarchical bootstrap、10,000 draws、percentile 95% CI；state、donor、view、branch 均是 repeated measures，不作为独立样本。

## 1. 一句话结论

目前最有证据支持的解释是：**Wrist 视角的局部相机几何使相同来源的视觉变化在 DINO 分支产生更大的表征位移；这种上游 DINO 差异通过共享、分布式的后续路由传到 Action，因此 Wrist 的瞬时动作偏差通常大于 Front。** 已得到因果支持的几何成分是局部距离/表观尺度（range）与横向视差（lateral），不是所测试的俯仰角（angular/pitch）。

这是一条“部分根因”证据链，不是完整根因。它尚未证明训练过程中何时形成、是否由 OFT 训练创造、是否跨模型成立、是否足以解释对抗攻击成功率或闭环任务失败。

## 2. 起始现象：Wrist 的动作变化更大

早期攻击与固定状态实验反复观察到：对同一 clean state，只改变 Front 或 Wrist 图像时，Wrist 图像变化引起的标准化完整 Action-chunk 偏差更大。E0 在 WHEN、UADA、UPA 三类来源扰动的 Discovery/Validation 中广泛复现；匹配的通用视觉控制也可出现 Wrist bias，因此该现象不是“对抗补丁独占机制”。闭环实验中 Wrist 劣势方向一致，但任务→episode 区间跨 0，不能宣称已得到 cluster-resolved 的闭环成功率差异。

最早的正式 ActionDeviation V1 进一步证明差异已经存在于下一条实际执行的 step-0 prediction：WHEN 的 Wrist−Front 为 `+0.364169 [0.182191,0.586282]`，UADA 为 `+0.279627 [0.175536,0.385972]`，UPA 为 `+0.299684 [0.185587,0.416173]`；M5、Black、Random 的对应区间跨 0。这里的 `attack_rebaseline_v1/RESULTS.md` 只表示更大的整套 rebaseline formal run 未启动；其子实验 `action_deviation_v1` 是完整、可采纳的正式结果，不能因父目录状态而删除。

在后来更干净的 600-state、120-episode、10-task、3-donor source-replacement 数据上，自然匹配输入产生：

- Front Action dependence：`2.502356`；
- Wrist Action dependence：`3.703311`；
- Wrist−Front gap：`1.200955 [0.902245, 1.458493]`；
- 10/10 tasks 为正。

所以需要解释的核心事实是稳定的**瞬时 Action sensitivity gap**，而不是已经确认的闭环失败率 gap。

## 3. 证据链概览

| 环节 | 主要结果 | 当前科学含义 |
|---|---|---|
| E0 | 三类来源扰动广泛复现 Wrist>Front；闭环仅一致趋势 | 现象成立于离线瞬时 Action；不是攻击独占，也尚非闭环确认 |
| E1 | 三类来源的 Wrist−Front input sensitivity 均为正；来源相对匹配控制有特异性 | 输入差异具有强解释力，但不是自然中介证明 |
| E2 | held-out view-conditioned Action-delta decodability；Projector/L0 repair 高度恢复 | 信息在早期输入/投影处可用且可因果恢复，随后快速路由 |
| LFCA/EFT/VCAG | 局部线性、精确有限传递或五站点方向几何的确认 Gate 未通过 | 排除若干简单局部机制，不等价于“没有机制” |
| E3 | 累积修复支持分布式充分性；局部化不支持 | 不存在足以解释现象的单一窄模块 |
| E4-1～E4-4 | Z1/Z3 可解码、可干预、可中介；独立 route linkage 未支持 | 功能因子真实，但不是各自独占路线 |
| E4-5 | `SHARED_DISTRIBUTED_ROUTING` | Z1/Z3 共享结构化、分布式路线 |
| SEM0–SEM4 | 34→27→12→2；仅 A2-dx/A2-dy 通过自然因果中介 | 细粒度语义中，平移 x/y 有小但稳定的自然中介效应 |
| W1 | DINO/fused/Projector/L0 的 Wrist 表征位移显著更大；SigLIP 不成立 | 将差异定位到上游 DINO 主导的输入传播 |
| W2/W3 | 同向量 site gain 与 propagation retention 未支持 | 当前实现不能证明后端 Wrist-specific decision gain；有重要方法限制 |
| P1 | 等范数后 gap 缩小但仍存在；强 direction×site interaction | 幅度解释一部分，剩余差异依赖来源匹配的方向×位置 |
| P2 | DINO gap contribution 显著；SigLIP 与交互 gap 未确认 | DINO 承载绝大多数额外 Wrist gap |
| VF1 | range、lateral 完整链为正；angular 不支持 | 给出 DINO 上游相机几何的具体因果来源 |

## 4. 从输入到模型内部：E1、E2 与早期路由

E1 的三类 `G_ISS` 和区域指标均显示 Wrist−Front 为正；来源效应与输入敏感度的关系在 WHEN/UADA 中达到 cluster-resolved，来源减匹配控制的特异性在三类来源中均为正。它说明 Wrist 输入变化更容易形成大动作效应，但相关/解释关系本身不是自然因果中介。

E2-A 在 episode-disjoint Validation 中复现了 view-conditioned Action-delta decodability：WHEN/UADA/UPA 的广泛 pooled R² gap 再现，主要由 translation/gripper 驱动，rotation 更异质。它证明表征中含有可读出的 view-conditioned action-delta 信息，但 `probe decodability != mechanism`。

E2-B 的修复实验进一步表明：Projector whole-view repair 完全恢复，L0 view-origin token-position repair 对 Front/Wrist 和三类来源均恢复约 94%–96%，之后随着层数增加而衰减。早期是强因果中介边界，但 L0 后没有持续的 `R_W >> R_F`。因此更合适的描述是“早期视觉/Projector 中介，随后快速、共享路由”，不是“下游有一个 Wrist 专属电路”。

## 5. 哪些简单内部机制没有得到确认

这些负结果是有效约束，必须保留：

- LFCA：Discovery 产生 Projector suggestive candidate，但 held-out Validation 的 pooled `I_LCA=0.1132`，95% CI `[-0.0804, 0.3113]`，确认 Gate 未通过；Random/M5 也有正区间，削弱攻击特异性。
- Exact Finite Transfer：Discovery 曾选 L8 candidate，但 Validation 的主 Gate 失败，正式 verdict 为 `V1-NOT-VALIDATED`。
- VCAG：五个预注册站点都没有通过 fidelity eligibility，正式为 `M1-HYPOTHESIS-NOT-SUPPORTED`，按预注册停止，未运行 Validation/M2/M3。
- E3-2S：distributed repair 8/8，localized repair 0/8；支持分布式充分性，不支持单点局部充分性。原本需要 random controls 的更强 E3-2 verdict 因不可行而禁止使用。

这些结果拒绝的是各自预注册的简单局部解释，不证明所有内部机制都不存在。

## 6. 功能因子与共享分布式路由：E4

E4-1 显示 Z1/Z2/Z3 在双视角可解码；E4-2 中 Z1、Z3 具有功能因果性，Z2 不成立，控制 ZC 为 null；E4-3 支持 Z1/Z2/Z3 clean mediation，并支持 Z1/Z3 conjunction。E4-4 没有支持独立的 route linkage，因此不能把这些因子写成各自独占的功能路线。

E4-5 的正式 verdict 为 `E4_5_SHARED_DISTRIBUTED_ROUTING`：

- ranking specificity PASS：pooled `DeltaRankCov8=0.405092 [0.376186, 0.433138]`；
- 但 factor Top64 相对 generic Top64 的 `DeltaFCov64=-0.012331 [-0.022562, -0.003153]`，primary FAIL；
- Z1/Z3 的 Top64 coverage 约 0.66，未达到 0.80 strong-sufficiency Gate；
- own-route advantage 均未支持，而 cross-factor coverage 仍约 0.66–0.67。

因此 Z1/Z3 的 ranking 有真实结构，但它们嵌入的是共享的结构化分布式 local-manipulation route，不是两个独立专用回路。

E4-6 因 phase label 数据覆盖为 0/10 tasks 而锁止，没有科学结果。E4-7 后来用事件感知 acquisition 得到 9 个可用任务，但 `EM=0.288653 [-0.176872, 0.986256]`，正式 verdict 为 `E4_7_EVENT_MODULATION_NOT_SUPPORTED`；任务之间事件 profile 异质，不能据此声称完全没有事件依赖。

## 7. 细粒度语义：SEM0–SEM4

SEM1 从 600 个 Discovery states 冻结 34 个候选；SEM2 的 61,200 rows 中 27/34 表现为 Wrist-sensitive；SEM3 在 63,000 forwards、16,200 rows 中筛出 12/27 amplify-specific 候选。具有 Action-valid 语义证据的包括 A2-dx、A2-dy、A2-dz、approach score 和 Häon fast。

SEM4 在 48,600 rows 的自然中介检验后，仅两项通过：

- `LIT_A2_DX`：CSM `0.000197 [0.000009, 0.000387]`，10/10 tasks 正向；
- `LIT_A2_DY`：CSM `0.000251 [0.000014, 0.000478]`，8/10 tasks 正向。

效应很小，但稳定，说明 x/y 平移相关语义确实被自然因果使用。其他候选可见或可操纵，不代表自然路径采用它们。早期 SEM3 的派生 RMS 指标已从完整 raw 数据无损更正为 L2；布尔 verdict 未改变，旧错误版本仅作为 invalid audit 保存。

## 8. W1–W3：差异首先出现在哪里

W1 在 600 states、120 episodes、10 tasks、3 donors 上比较同源输入替换造成的归一化表征位移，Wrist−Front 结果为：

| Site | 差值与 95% CI | 正向任务 |
|---|---:|---:|
| DINO | `0.261153 [0.220157, 0.300175]` | 10/10 |
| SigLIP | `-0.011909 [-0.050760, 0.022335]` | 5/10 |
| fused vision | `0.221662 [0.186543, 0.255122]` | 10/10 |
| Projector | `0.099417 [0.064297, 0.131535]` | 9/10 |
| L0 | `0.097544 [0.062993, 0.129127]` | 9/10 |

差异在 DINO 已经很强，SigLIP 没有建立同方向差异，融合后仍保留并进入 Projector/L0。W1 支持“输入易感性/上游表征位移”，不是单独的 Action 中介证明。

W2 对冻结的 A2-dx/dy 做 same-vector site test，Action decision-gain gap 均跨 0；W3 的 L24 restoration-recovery gap 也跨 0，均为 NOT SUPPORTED。但这两项不能强力排除后端 gain，因为 11,008-D 跨层神经元坐标不天然同义，Projector/A2 维度不匹配触发了 L0 SwiGLU post-gate fallback，且原方向在 L30/L31 grounding；因此相同向量在 L0 的语义同一性没有建立。L31 source-token restore 又发生在末层 attention 之后，只能作为 terminal sanity control。

## 9. P1–P2：幅度、位置/方向与视觉分支

P1 的主尺度结果：

- natural gap：`1.200955 [0.902245, 1.458493]`；
- 等范数后 gap：`0.904192 [0.613895, 1.166818]`；
- gap reduction：`0.296763 [0.215747, 0.371744]`，约为原 gap 的 24.7%，只是描述比例，不是 mediation proportion；
- site effect：`0.844990 [0.640765, 1.050028]`；
- origin effect：`0.059201 [-0.094324, 0.200949]`，未确认；
- direction×site interaction：`2.783562 [2.471747, 3.109682]`；
- Front-natural direction 的 Wrist−Front site effect：`-0.546791 [-0.773209, -0.303947]`；
- Wrist-natural direction 的 Wrist−Front site effect：`2.236771 [1.964428, 2.526472]`。

所以差异不是“任意扰动放到 Wrist 都会放大”。输入变化的范数解释了一部分，但大部分剩余差异来自**来源匹配方向 × 注入位置**的强交互。

P2 对视觉塔做分支恢复：

- DINO gap contribution：`1.221561 [0.974587, 1.451339]`，10/10 tasks；
- SigLIP gap contribution：`0.065209 [-0.045648, 0.171787]`，未确认；
- DINO×SigLIP interaction gap：`-0.085814 [-0.238374, 0.069727]`，未确认；
- restore-both Action closure exact。

自然 Action distance 为 Front `2.502356`、Wrist `3.703311`；恢复 DINO 后剩余 Front `1.027244`、Wrist `1.006639`，gap 近零；恢复 SigLIP 后仍为 Front `2.070467`、Wrist `3.206213`。这说明 DINO 承载了几乎全部额外 Wrist gap。分支混合可能 off-manifold，因此应说“intervention 下 DINO 承载主要差异”，不应写成唯一完整根因。

## 10. VF1：DINO 上游视觉几何的因果分解

VF1 只改变 camera extrinsics，保持 physical simulator state exact，并测试三种预注册几何因素：G1 range、G2 lateral、G3 angular。600-state V3 的原预注册 engineering QA **仍是 FAILED**：唯一失败项是 36/600 states 的 `identity_activation_error>2`，最大 5.5；阈值没有修改，V3 绝不重新标为 QA PASS。

其余控制通过：identity image ≤1；identity Action L2 为 600/600 ≤0.1，最大约 0.067815；physical simulator state exact；hook write exact；camera pose restore PASS；DINO+SigLIP restore-both Action closure exact。Web GPT 在没有查看任何 VF1 scientific contrasts 的前提下完成 outcome-blind adjudication，授权 primary 使用全部 600 states，并仅增加排除上述 36 states 的 564-state sensitivity；sensitivity 不进入 primary Gate。

Primary medium-magnitude 结果：

| Factor | FDG | FAG | DGR | SGR | DGR−SGR | 数值结论 |
|---|---:|---:|---:|---:|---:|---|
| G1 RANGE | `0.559963 [0.538023,0.582478]` | `1.971451 [1.727608,2.206208]` | `1.508446 [1.295633,1.728010]` | `0.132302 [0.005542,0.256708]` | `1.37614 [1.13678,1.62380]` | 完整 DINO-mediated chain |
| G2 LATERAL | `0.291687 [0.264550,0.319390]` | `1.381408 [1.116373,1.609879]` | `1.231363 [1.014898,1.440330]` | `-0.025653 [-0.109189,0.054872]` | `1.25702 [1.03621,1.47401]` | 完整 DINO-mediated chain |
| G3 ANGULAR | `-0.167170 [-0.185851,-0.148760]` | `-0.331518 [-0.542948,-0.122791]` | `0.038149 [-0.104655,0.189408]` | `-0.224975 [-0.296772,-0.149122]` | `0.263125 [0.137508,0.395554]` | primary factor 不支持 |

G1/G2 的 small/large robustness 中 FDG/FAG/DGR 也均为正且 CI lower>0。G3 的三个幅度 FDG 都为负，FAG/DGR 非单调，不能用 large 结果挽救 medium primary。564-state sensitivity 与 600-state primary 的方向和结论一致，没有实质冲突，因此最终 claim 不标记为 rendering-noise sensitive。

严格状态表述为：

- 原预注册 Gate：`ENGINEERING_QA_FAILED`；
- outcome-blind adjudicated numerical conclusion：G1/G2 为 `DINO_MEDIATED_UPSTREAM_VISUAL_FACTOR`，G3 为 `VISUAL_FACTOR_NOT_SUPPORTED`；
- overall：`DINO_MEDIATED_UPSTREAM_FACTOR_SUPPORTED_UNDER_ADJUDICATION`。

## 11. 当前对“为什么 Wrist 比 Front 偏差更大”的解释

证据共同支持以下分解：

1. 相同来源的图像变化在 Wrist 端并不只是范数更大；范数匹配只能消掉约四分之一 natural gap。
2. Wrist 的近距离、局部、随末端运动变化的成像几何，使物体/末端的表观尺度变化与横向视差更强。
3. 这些几何变化首先在 DINO 分支产生更大的 feature displacement；SigLIP 没有显示相同的稳定差异。
4. DINO 分支差异对 Action gap 有因果贡献；restore DINO 后 Wrist−Front gap 基本闭合。
5. dx/dy 语义有小而稳定的自然中介效应，与 range/lateral 所强调的局部平移几何一致。
6. 进入 Projector/L0 后，信号沿共享的结构化分布式路线传播；现有证据不支持一个独立的 Wrist-specific late module。

这比“腕部画面更复杂”或“腕部扰动更大”更具体：**关键不是一般复杂度，而是 Wrist 成像几何诱发的、DINO 敏感的局部尺度/平移模式，以及这些模式与所在视角的方向匹配。**

## 12. 训练形成证据与当前暂停边界

较早的 `wrist_formation_f0_f3` 审计发现没有可比的同一训练 run 的 early/middle checkpoints，也没有可比的 base/final 双视角连续动作 policy。F1 因而是 `FORMATION_INCONCLUSIVE_NOT_ELIGIBLE`，0 forwards；不得把 base VLA 加 final auxiliary modules 拼成伪训练时刻。

对 final checkpoint 的七组 LoRA 做过 necessity rewind：没有任何组通过完整 candidate Gate，F3 未进入。这只能说明所测单个 LoRA group 没有成为 final-model Wrist gap 的特异、充分必要候选；不能说明 asymmetry 何时形成，也不能证明 OFT 是否创造了它。

当前新的 W4 Training Formation Analysis 继续暂停：不下载或处理 W4 训练数据，不运行 base→OFT checkpoint drift。VF1 V4 GPU rerun也未授权。

## 13. 可以冻结的 claim 与禁止 claim

可以冻结：

- OpenVLA-OFT 在 LIBERO-Spatial 的离线瞬时 Action readout 上具有稳定 Wrist>Front sensitivity gap。
- 该 gap 不是攻击独占；通用、来源匹配的视觉替换也能产生。
- 输入幅度解释部分差异，但强 direction×site interaction 解释了剩余结构。
- 上游差异主要出现在 DINO 而非 SigLIP，并在干预下对 Action gap 有主要贡献。
- range/表观尺度与 lateral/parallax 是当前测试中得到支持的具体上游几何因素；angular/pitch 未支持。
- 后续传递更符合共享、结构化、分布式路由；A2-dx/dy 有小而稳定的自然中介效应。

禁止冻结为事实：

- “已经找到完整或唯一根因”；
- “Wrist 有单一专属神经元、头、层或独立路线”；
- “所有相机角度变化都会增强 Wrist 偏差”；
- “该机制已解释闭环失败或攻击成功率”；
- “该机制跨模型成立”；
- “OFT 训练创造了该机制”或“已定位形成时刻”；
- “VF1 原始工程 QA PASS”。

## 14. 仍未解决的问题

- range 与 lateral 各自能解释 natural Wrist gap 的多少，二者是否可加、交互或共享同一自然方向；
- VF1 固定 camera-factor 干预与真实 wrist motion-induced image statistics 的生态对应程度；
- DINO 内部具体由哪些空间频率、尺度、遮挡边界、物体/末端局部 token 组合承载；
- 这些瞬时 Action 差异是否在闭环中形成可重复的任务成功率影响；
- 是否跨模型/跨套件成立；
- 在缺少可比训练轨迹的前提下，形成机制仍不可识别。

## 15. 冻结状态

`CURRENT_SCIENTIFIC_STATUS = PARTIAL_CAUSAL_EXPLANATION_SUPPORTED_UNDER_VF1_OUTCOME_BLIND_QA_ADJUDICATION`

`VF1_PRIMARY = ALL_600_STATES`

`VF1_SENSITIVITY = EXCLUDE_CONTROL_DEFINED_36_STATES_ONLY_NON_GATING`

`VF1_ORIGINAL_ENGINEERING_QA = FAILED_DO_NOT_RELABEL`

`PRIMARY_SENSITIVITY_SUBSTANTIVE_CONFLICT = FALSE`

`W4 = PAUSED`

`V4 = NOT_AUTHORIZED`

`RESET_CARD_USED = NO`

`STOP_TYPE = FREEZE_AND_RETURN_TO_USER_WEB_GPT`
