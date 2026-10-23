# OpenVLA-OFT MGWP-v2：动作敏感有符号几何机制补丁实验方案

日期：2026-09-16  
方案简称：MGWP-v2（Action-sensitive Signed Geometric Wrist Patch）  
研究用途：本方案仅用于科学研究、模型鲁棒性评估与机制验证，不用于真实系统攻击或部署。  
文档状态：**实验方案，尚未授权执行；不自动提交作业。**

---

## 1. 研究背景

前期实验已经形成以下证据链：

1. OpenVLA-OFT 对 Wrist camera 的输入变化比 Front camera 更敏感。
2. Wrist–Front 动作差异主要经 DINO 视觉编码器向动作头传播。
3. RANGE 与 LATERAL 是相对稳定的 Wrist 几何脆弱方向，ANGULAR 证据不稳定。
4. 当前 MGWP-v1 已将几何方向用于补丁优化，并在 LIBERO-Object 闭环实验中得到最高点估计：
   - MGWP-v1：任务等权 CASR 22.56%；
   - DINO-Norm：18.11%；
   - Persistent：13.22%；
   - WHEN：12.67%；
   - Exploring：7.22%；
   - Random：10.44%。
5. MGWP-v1 显著高于 Random 和 Exploring，但尚不能证明高于 DINO-Norm、Persistent 或 WHEN。
6. MGWP-v1 的机制方向得分在第500步仍在上升，但动作偏差在最后50步略有下降。

因此，当前最重要的问题不是简单延长训练，而是：

> 能否把“Wrist 几何变化经 DINO 传播到动作”的机制进一步转化为一个方向固定、动作敏感、能够跨状态持续累积的优化目标，从而提高闭环任务失败率？

---

## 2. 当前 MGWP-v1 的局限

### 2.1 训练步数可能不足，但不是唯一问题

- 训练状态数：300。
- 当前更新步数：500。
- 每次更新使用一个状态，相当于约1.67次训练集遍历。
- 第401–450步到第451–500步，MGWP-v1 的机制得分仍由851.9升至873.6。
- 同期动作偏差由8.73降至8.37，持续动作偏差由7.38降至6.95。

这说明机制目标尚未饱和，但继续训练可能只会放大 DINO 偏移，不一定提高动作破坏或闭环 CASR。

### 2.2 当前机制目标没有固定语义方向

MGWP-v1 使用：

```text
SmoothMax(q_RANGE, q_LATERAL)
```

它允许不同状态选择不同的易优化方向，不能保证补丁在所有状态中都产生同一种几何误判。

### 2.3 当前目标没有显式区分“动作敏感”的 DINO 方向

落入 RANGE/LATERAL 子空间不等于一定能影响动作头。当前目标没有明确使用 DINO 几何方向到动作输出的局部传播强度。

### 2.4 当前梯度权重在训练后期失衡

MGWP-v1 在第一步根据梯度范数设置固定 `lambda_aux=0.1168`，之后不再调整。最后50步平均梯度约为：

```text
主动作梯度：6.22
机制梯度：66.14
缩放后的机制梯度：66.14 × 0.1168 ≈ 7.73
```

后期机制梯度约为主梯度的124%，而原目标是约30%。这可能是机制得分继续增加、动作偏差反而下降的重要原因。

---

## 3. 研究问题与假设

### 3.1 主研究问题

在相同 Wrist-only、补丁尺寸、位置、训练数据和训练预算下，MGWP-v2 是否比：

- Persistent；
- DINO-Norm；
- MGWP-v1；
- Exploring；
- WHEN；

产生更高的闭环 CASR？

### 3.2 机制问题

如果 MGWP-v2 的闭环效果更强，这种提升是否同时满足：

1. 补丁诱导的 DINO 变化更稳定地沿固定有符号几何方向传播；
2. 该方向对动作头具有较高传播增益；
3. 恢复干净 DINO 表征后，动作偏差明显下降；
4. 有符号几何偏移能够预测同方向的动作漂移。

### 3.3 实验假设

- H1：动态梯度平衡可以避免机制项在训练后期压过动作目标。
- H2：动作敏感的有符号几何目标比无方向 DINO-Norm 产生更一致的动作漂移。
- H3：一致的动作漂移比单纯增大动作 L2 更容易转化为闭环任务失败。
- H4：MGWP-v2 的优势应在统一补丁位置和统一训练预算下仍然存在。

---

## 4. 威胁模型与实验边界

### 4.1 模型与任务

- 模型：OpenVLA-OFT。
- 主任务集：LIBERO-Object，10个任务。
- 攻击视角：Wrist-only。
- Front image：保持干净。
- 语言指令、模型权重、机器人状态、控制器和动作后处理：保持不变。

### 4.2 补丁能力

- Universal RGB patch。
- 尺寸：`50 × 50 × 3`。
- 测试时不按 episode 重新优化。
- Primary 使用 continuous exposure，即每个闭环 step 均在 Wrist image 上应用相同补丁。
- 当前只评价仿真环境，不宣称物理世界有效性。

### 4.3 攻击者知识

本实验采用白盒科研评估设置：

- 可访问模型权重和梯度；
- 知道 Wrist-DINO-action 机制结论；
- 知道 RANGE/LATERAL 几何方向；
- 可使用训练划分上的干净动作、DINO 表征和中间层干预结果；
- 不使用最终测试划分的任务结果进行方向、超参数或检查点选择。

---

## 5. MGWP-v2 方法

## 5.1 固定补丁位置

Primary comparison 固定使用前期机制定位得到的位置：

```text
patch_x = 26
patch_y = 137
```

Random、Persistent、DINO-Norm、MGWP-v1、MGWP-v2、Exploring 和 WHEN 在公平主比较中均使用相同位置和尺寸。

Exploring/WHEN 的官方位置 `[0, 0]` 仅作为历史兼容结果，不与机制位置的结果混为同一因果比较。

## 5.2 离线估计动作敏感几何方向

已有 DINO 几何子空间：

```text
U = [u_RANGE, u_LATERAL]
```

对训练状态，在 DINO 表征处分别实施小幅正负 RANGE/LATERAL steering，通过动作头得到局部动作响应：

```text
B_i[:, 0] = [a(z_i + eps*u_RANGE) - a(z_i - eps*u_RANGE)] / (2*eps)
B_i[:, 1] = [a(z_i + eps*u_LATERAL) - a(z_i - eps*u_LATERAL)] / (2*eps)
```

其中 `B_i` 表示二维几何坐标到归一化动作空间的局部传播。

为优先寻找跨状态一致的动作传播方向，计算：

```text
B_bar = mean_i(B_i)
c_star = top_eigenvector(B_bar.T @ B_bar)
v_star = normalize(U @ c_star)
b_star = normalize(B_bar @ c_star)
```

- `v_star`：固定的 DINO 有符号几何方向。
- `b_star`：该几何方向对应的固定动作漂移方向。

方向符号采用预先固定的约定：令 `c_star` 中绝对值最大的分量为正。不得根据最终 CASR 翻转方向。

### 实现约束

- 上述方向在补丁训练前离线计算并冻结。
- 不在补丁训练循环中重复计算完整 Jacobian。
- 不对 Jacobian 进行二阶反向传播。
- 方向估计只使用训练划分。

这样可以把动作敏感信息用于损失，同时保持补丁训练为普通一阶反向传播。

## 5.3 主动作目标

令归一化动作差为：

```text
Delta_a = (a_patch - a_clean) / action_scale
```

动作偏差：

```text
D_action = mean_h ||Delta_a[h]||_2
```

动作块内持续偏差：

```text
D_persistent = ||mean_h Delta_a[h, 1:6]||_2
```

主损失：

```text
L_main = -(D_action + beta * D_persistent)
beta = 0.5
```

## 5.4 有符号 DINO 机制损失

补丁产生的 DINO 变化：

```text
Delta_z_i = z_DINO(x_i with patch) - z_DINO(x_i clean)
```

有符号机制投影：

```text
s_i = <Delta_z_i, v_star> / sigma_v
```

`sigma_v` 由训练划分中的自然 RANGE/LATERAL 干预幅度确定，使 `s_i=1` 约等于一个标准几何干预单位。

使用平滑 margin 损失：

```text
L_mech = Softplus((m_z - s_i) / T_z)
m_z = 1
```

它要求每个状态都沿固定方向达到至少一个标准干预单位，而不是无限追求表征数值增大。

## 5.5 有符号动作漂移损失

定义动作方向投影：

```text
r_i = mean_h <Delta_a_i[h, 1:6], b_star> / sigma_a
```

`sigma_a` 使用训练划分中 `v_star` steering 对动作产生的典型响应进行归一化。

损失：

```text
L_drift = Softplus((m_a - r_i) / T_a)
m_a = 1
```

这一项要求不同状态产生同方向动作漂移。它替代需要 batch>1 的跨状态方差损失，因此不需要同时保存多个大模型计算图。

## 5.6 完整损失

机制辅助损失：

```text
L_aux = L_mech + eta * L_drift
eta = 1
```

完整优化不直接使用固定标量相加，而是分别计算：

```text
g_main = grad(L_main, patch)
g_aux  = grad(L_aux, patch)
```

如果二者冲突：

```text
if dot(g_main, g_aux) < 0:
    g_aux = g_aux - dot(g_main, g_aux) / ||g_main||^2 * g_main
```

每步动态调整机制梯度比例：

```text
lambda_t = rho * ||g_main|| / (||g_aux|| + eps)
g_total = g_main + lambda_t * g_aux
```

Primary：

```text
rho = 0.30
```

Open-loop sensitivity：

```text
rho in {0.15, 0.50}
```

`rho=0.15/0.50` 不根据最终闭环结果选择，不自动进入主闭环矩阵。

---

## 6. 可训练性与资源约束

### 6.1 现有证据

MGWP-v1 已验证：

- patch→DINO→loss 梯度链路有效；
- 500步无 NaN/Inf；
- 主梯度和辅助梯度均非零；
- RTX 3090 峰值已分配显存约20.36 GiB；
- 最终 patch 只有约10.5%的像素接近 `[0,1]` 边界，没有严重饱和。

### 6.2 MGWP-v2 的工程约束

- 正式训练使用 batch size 1。
- 不在线构建完整 Jacobian。
- 不使用需要 batch>1 计算图的显式方差项。
- 动作敏感方向离线估计并保存。
- 正式训练仍为两次一阶 patch gradient，与 MGWP-v1 同量级。

因此预期3090能够运行，但正式作业前必须按服务器规范执行一次完整最坏路径峰值显存测试。该测试只用于确定显存申请，不用于估算正式运行时间。

资源选择规则：

1. 首选单张 RTX 3090。
2. 峰值显存保留合理余量；如果实际峰值过于接近24 GiB或发生 OOM，再切换4090。
3. 不使用 A800，除非3090和4090均无法满足显存要求。
4. 不通过增加 GPU 数量掩盖单卡实现问题。

---

## 7. 数据划分

### 7.1 训练集

复用当前冻结的300个 clean-success training states：

```text
10 tasks
5 trajectories / task
6 states / trajectory
```

覆盖 pre-contact、contact/gripper-close 和 post-contact 阶段。

### 7.2 开发验证集

此前已经用于 Stage 1/Stage 2 的 `state_index 30–39` 降级为 development set：

- 可用于训练长度曲线和 open-loop sensitivity；
- 不再作为 MGWP-v2 的全新最终测试集；
- 先前22.56% CASR 只作为历史基线，不作为新方案的独立确认结果。

### 7.3 最终测试集

计划使用 `state_index 40–49` 作为新的最终闭环测试集，实施前先验证这些初始状态：

- 在 LIBERO-Object 中实际存在；
- 未进入300个训练状态；
- 未用于方向估计、超参数调整或检查点选择。

如果索引不可用，只能在查看任何新方法结果前冻结另一组未使用状态，不得根据结果更换测试集。

### 7.4 配对与运行顺序

- 每个方法使用完全相同的 task和initial state。
- 每个 state 先运行 Clean。
- 攻击条件顺序按固定随机种子在 state 内随机化。
- Clean-success episode 构成 CASR 分母。
- 同一 episode 的多帧和多动作步是重复测量，不作为独立样本。

---

## 8. 方法矩阵

### 8.1 训练方法

| 方法 | 优化目标 | 作用 |
|---|---|---|
| Persistent | `D_action + D_persistent` | 输出空间基线 |
| DINO-Norm | 主目标 + `||Delta_z||` | 通用 DINO 表征扰动 |
| MGWP-v1 | 主目标 + RANGE/LATERAL SmoothMax | 无固定动作敏感方向的机制基线 |
| MGWP-v2 | 主目标 + signed mechanism + signed drift | 完整新方法 |

所有可训练方法：

- 使用相同训练状态；
- 使用相同 patch 初始化种子；
- 使用相同尺寸和位置；
- 使用相同优化器、学习率和最大训练步数；
- 保存相同检查点。

### 8.2 外部方法

| 方法 | Primary 位置 | 备注 |
|---|---|---|
| Exploring | `[26,137]` | 公平位置对照 |
| WHEN | `[26,137]` | 公平位置对照 |
| Exploring-official-position | `[0,0]` | 仅作历史兼容/辅助结果 |
| WHEN-official-position | `[0,0]` | 仅作历史兼容/辅助结果 |

### 8.3 最小闭环主矩阵

```text
Clean
Random
Exploring-same-location
WHEN-same-location
Persistent
DINO-Norm
MGWP-v1
MGWP-v2
```

即：

```text
10 tasks × 10 states × 8 conditions = 800 episodes
```

官方原位置结果不进入上述主矩阵；如需在新测试集复测，单独标记为 secondary。

---

## 9. 训练长度与超参数方案

### 9.1 训练长度

所有可训练方法保存：

```text
step 500
step 750
step 1000
step 1500
```

Primary checkpoint 预先固定为 `step 1500`，不根据最终闭环 CASR 改选。其余检查点用于训练长度消融。

如果出现 NaN、Inf、梯度为零或工程 OOM，可以修复后从最近检查点恢复；不得因为某个较早检查点结果更好而事后改写 primary checkpoint。

### 9.2 固定参数

```text
optimizer = AdamW
learning_rate = 2e-3
weight_decay = 0
beta = 0.5
rho_primary = 0.30
m_z = 1
m_a = 1
eta = 1
patch clamp = [0,1]
```

`T_z`、`T_a` 使用归一化后固定温度，实施时通过训练数据的数值范围设置一次并写入配置；不得用最终测试 CASR 调整。

### 9.3 少量敏感性分析

只在 open-loop development set 比较：

```text
rho = 0.15 / 0.30 / 0.50
```

主结论仍使用预先指定的 `rho=0.30`。敏感性结果用于说明方法是否严重依赖梯度权重，不用于挑选最好结果进入主表。

---

## 10. 执行阶段

### Stage A：离线方向估计

目标：

- 构建训练状态上的 RANGE/LATERAL → Action 局部传播矩阵；
- 计算并冻结 `c_star`、`v_star` 和 `b_star`；
- 保存方向、归一化尺度和训练输入 hash。

输出：

```text
artifacts/mgwp_v2/ACTION_SENSITIVE_DIRECTION.pt
reports/MGWP_V2_DIRECTION_REPORT.json
```

### Stage B：3090峰值显存测试

覆盖：

- 模型与 EGL 加载；
- patch forward；
- `L_main` 与 `L_aux` 两次梯度；
- PCGrad 投影；
- 动态梯度缩放；
- optimizer step。

只报告实际峰值显存和是否有足够余量，不用短测试推算正式运行时间。

### Stage C：训练与检查点

训练：

```text
Persistent
DINO-Norm
MGWP-v1
MGWP-v2
```

已有前三种方法可从step 500继续，但必须确认 optimizer state、patch和配置 hash一致。MGWP-v2从与其他方法相同的初始patch开始。

### Stage D：开发集 open-loop 分析

在固定 development states 上评估所有检查点：

- Action L2；
- Persistent drift；
- signed DINO alignment；
- signed action drift；
- DINO restore reduction；
- 梯度比例和像素饱和率。

此阶段只解释训练动态，不替换预先固定的主检查点。

### Stage E：新测试集闭环主实验

在完全相同的 final test states 上运行8个条件，计算 CASR 与配对差异。

### Stage F：机制闭环解释

对最终固定补丁进行：

- DINO restore；
- signed alignment–action drift关联；
- 成功/失败episode的机制指标对比；
- per-task和failure phase分析。

完成后停止，不自动扩展到 Spatial、Goal、Long、物理环境或其他模型。

---

## 11. 评估指标

### 11.1 Primary

```text
CASR = clean成功但攻击后失败的episode数 / clean成功episode数
```

正式报告任务等权 CASR，并同时给出 pooled event count，避免两种口径混淆。

### 11.2 Secondary

- Attacked Success Rate；
- Raw Failure Rate；
- Mean episode steps；
- Time-to-success / timeout；
- Failure phase；
- Normalized Action L2；
- Persistent Action Drift；
- Translation / Rotation / Gripper deviation。

### 11.3 机制指标

- Signed DINO alignment：`<Delta_z, v_star>`；
- Signed action drift：`<Delta_a, b_star>`；
- DINO representation displacement；
- DINO restore reduction；
- signed alignment 对 action drift 的预测关系；
- signed alignment 对闭环 failure 的任务分层关系。

---

## 12. 统计分析

### 12.1 配对结构

- 同一 task和initial state 在所有条件下配对。
- 统计层级：task → episode。
- 不把trajectory step、frame、action dimension当独立样本。

### 12.2 Bootstrap

```text
draws = 10,000
seed = 20260907
95% percentile CI
```

主要对比：

```text
MGWP-v2 - MGWP-v1
MGWP-v2 - DINO-Norm
MGWP-v2 - Persistent
MGWP-v2 - Exploring-same-location
MGWP-v2 - WHEN-same-location
MGWP-v2 - Random
```

同时报告：

- point estimate；
- 95% CI；
- bootstrap差值为正的比例；
- positive / negative / tied tasks；
- 每个任务的原始失败计数。

不把“点估计最高”自动写成“显著优于”。

---

## 13. 结果解释边界

### 情况 A：MGWP-v2 高于 MGWP-v1 和 DINO-Norm

如果 CASR 提升、signed alignment增强、DINO restore可以明显关闭动作偏差，则支持：

> 动作敏感的有符号 Wrist-DINO 几何机制可以用于设计更有效的闭环鲁棒性压力测试。

### 情况 B：MGWP-v2 高于 MGWP-v1，但不高于 DINO-Norm

说明固定方向和动态梯度平衡改善了旧MGWP，但特定几何方向尚未优于一般 DINO 表征扰动。

### 情况 C：机制指标提高，但 CASR不提高

说明机制方向可被稳定操纵，但闭环控制仍能恢复这种误差；不能宣称机制利用提高了任务破坏。

### 情况 D：CASR提高，但restore和alignment不支持

只能说明新补丁更强，不能说明提升主要来自 Wrist-DINO 几何机制。

### 情况 E：1500步不优于500步

说明延长训练不能解决问题；应报告训练长度负结果，不再仅靠增加步数继续优化。

---

## 14. 必要工程检查

保留少量必要检查，不建立复杂审核流程：

- CUDA必需，正式模型训练禁止CPU fallback；
- 输入、方向、patch和配置hash；
- 训练/开发/测试索引无重叠；
- 梯度、损失和输出无NaN/Inf；
- 每个检查点完整可恢复；
- 同一state各方法唯一配对；
- 结果行数和任务覆盖完整；
- 日志按stage/job聚合，不逐state产生小文件。

建议输出结构：

```text
outputs/mgwp_v2/<stage>/task_00.parquet ... task_09.parquet
artifacts/mgwp_v2/patches/<method>/step_XXXX.pt
results/MGWP_V2_OPEN_LOOP_RAW.parquet
results/MGWP_V2_OPEN_LOOP_SUMMARY.csv
results/MGWP_V2_CLOSED_LOOP_RAW.parquet
results/MGWP_V2_CLOSED_LOOP_SUMMARY.csv
results/MGWP_V2_CLOSED_LOOP_CONTRASTS.csv
reports/MGWP_V2_STATUS.json
```

---

## 15. 与 Exploring / WHEN 的核心差异

| 维度 | Exploring / WHEN | MGWP-v2 |
|---|---|---|
| 设计起点 | 通用动作或特征破坏 | 已验证的 Wrist-DINO-action 机制 |
| 中间目标 | 输出偏差或通用feature displacement | 动作敏感的固定有符号几何方向 |
| 状态一致性 | 不要求相同几何语义 | 每个状态均要求同向机制投影 |
| 时间行为 | 通用偏差 | 固定动作方向上的持续漂移 |
| 因果验证 | 以攻击效果为主 | 同时使用DINO restore检验传播路径 |
| 公平比较 | 常使用各自默认位置 | 在相同位置、尺寸和数据上比较 |

本方案的核心贡献不应表述为“增加了更多损失项”，而应表述为：

> 将已经识别的 Wrist几何→DINO→Action 因果路径，转化为一个固定语义、动作敏感、跨状态一致的补丁优化方向，并检验这种机制约束是否能提高闭环失败。

---

## 16. 当前停止点

本文档只完成实验设计。当前不执行以下操作：

- 不修改现有代码；
- 不训练 MGWP-v2；
- 不提交 Slurm 作业；
- 不运行新的闭环实验；
- 不扩展其他suite或模型。

下一步必须由用户明确授权后，才进入代码实现、峰值显存测试和提交脚本生成。
