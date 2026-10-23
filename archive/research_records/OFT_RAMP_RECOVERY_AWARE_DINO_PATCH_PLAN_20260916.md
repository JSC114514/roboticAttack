# OpenVLA-OFT RAMP：恢复感知 DINO 机制补丁实验方案

日期：2026-09-16  
方案简称：RAMP（Recovery-Aware Mechanism Patch）  
研究用途：仅用于科学研究、OpenVLA-OFT 鲁棒性评估与闭环恢复机制分析，不用于真实系统攻击或部署。  
文档状态：**仅完成实验设计，当前不修改代码、不训练、不提交作业。**

---

## 1. 现有证据与问题定位

### 1.1 已确认结果

在 LIBERO-Object、Wrist-only、相同补丁位置 `[26,137]`、新测试状态 `40–49` 上：

| 方法 | 任务等权 CASR | Open-loop Action L2 | DINO restore reduction |
|---|---:|---:|---:|
| Random | 11.00% | 5.07 | 60.11% |
| Exploring | 9.00% | 6.74 | 68.49% |
| WHEN | 7.11% | 6.52 | 68.18% |
| Persistent | 38.00% | 7.06 | 71.52% |
| DINO-Norm | **54.22%** | 8.14 | 74.39% |
| MGWP-v1 | **52.44%** | 7.83 | 74.30% |
| MGWP-v2 | 40.00% | **8.17** | **75.22%** |

### 1.2 关键负结果

MGWP-v2 的 Action L2 和 DINO restore 均不弱，但闭环 CASR 明显低于 DINO-Norm 和 MGWP-v1。其预先固定的 signed action 在失败回合中反而弱于成功回合。

因此：

> 全局最敏感方向不等于最容易造成闭环失败的方向。继续扩大固定方向、增加相同损失权重或单纯延长训练，不是当前最合理路线。

### 1.3 新问题

下一阶段不再研究“如何制造更大的单状态动作偏差”，而研究：

> 如何利用 Wrist-DINO 脆弱路径，产生跨连续决策状态不被闭环控制抵消的累计漂移，从而越过任务恢复边界？

---

## 2. 核心思路

RAMP 将已有机制分三层使用：

1. **位置层**：继续使用 Wrist-only 和已验证的 DINO 敏感位置 `[26,137]`。
2. **表征层**：以当前最强的 DINO-Norm 为主目标，保持 Wrist-DINO 扰动强度。
3. **闭环转换层**：在同一轨迹的多个阶段上优化累计动作漂移，减少后续观测对早期误差的纠正。

RANGE/LATERAL 不再提供一个固定有符号方向，而只作为无符号几何脆弱子空间约束，允许不同状态选择不同方向。

---

## 3. 研究问题与假设

### 3.1 主问题

在相同初始补丁、额外训练数据、反向传播预算和位置下，加入跨状态累计漂移是否能比单纯继续优化 DINO-Norm 获得更高的闭环 CASR？

### 3.2 消融问题

1. 提升是否只来自额外训练，而不是 trajectory objective？
2. 无符号 RANGE/LATERAL 子空间约束是否提供额外收益？
3. RAMP 是否真的降低了策略的短期恢复能力？
4. Recovery residual 是否比单状态 Action L2 更能预测任务失败？

### 3.3 假设

- H1：RAMP-T 的跨状态累计漂移高于 DINO-Continue，并带来更高 CASR。
- H2：RAMP-TG 在保持 DINO 扰动的同时，比 RAMP-T 更集中于已验证几何子空间。
- H3：RAMP 的 `Residual@K` 高于 DINO-Continue，即扰动在移除补丁后仍更难恢复。
- H4：累计漂移和 `Residual@K` 对闭环失败的解释力高于单状态 Action L2。

---

## 4. 威胁模型与实验边界

- 模型：OpenVLA-OFT，权重不变。
- Primary suite：LIBERO-Object，10个任务。
- 输入：只修改 Wrist image；Front、语言、proprioception 不变。
- Patch：universal RGB，`50×50×3`，固定位置 `[26,137]`。
- Primary exposure：闭环中每个决策步连续出现。
- 测试时不根据任务、状态或 episode 重新优化补丁。
- 训练为白盒科研评估，可访问模型梯度和 DINO 中间表征。
- 当前不评价物理世界、不扩展其他模型或 suite。

---

## 5. 训练数据组织

继续使用现有冻结的300个 clean-success training states：

```text
10 tasks
5 trajectories / task
6 ordered states / trajectory
共50条轨迹、300个状态
```

每条轨迹的6个状态按已有 proxy 排列：

```text
2 pre-contact proxy
2 contact / gripper-close proxy
2 post-contact proxy
```

这些标签是轨迹阶段 proxy，不表述为真实接触传感标签。

RAMP 的独立训练单位由“单状态”改为“一条6-state轨迹”。每个 epoch 对50条轨迹按固定随机种子打乱，避免任务顺序与训练时间混淆。

---

## 6. 攻击方法

### 6.1 单状态基础量

对轨迹 (k) 的第 (t) 个状态：

\[
\Delta a_{k,t}
=
\frac{a^{patch}_{k,t}-a^{clean}_{k,t}}{s_a}
\]

动作块平均漂移：

\[
d_{k,t}
=
\frac{1}{H}\sum_{h=1}^{H}
\Delta a_{k,t,h,1:6}
\]

DINO 表征扰动：

\[
\Delta z_{k,t}
=
z_{DINO}(x^{patch}_{k,t})
-
z_{DINO}(x^{clean}_{k,t})
\]

### 6.2 DINO-Norm 基础目标

保留当前最强方法的主要目标：

\[
D_{DINO}
=
\frac{1}{6}\sum_{t=1}^{6}
\|\Delta z_{k,t}\|_2
\]

并保留平均动作偏差和动作块内 Persistent drift：

\[
D_{state}
=
\frac{1}{6}\sum_t
\left(
D_{action,t}+0.5D_{persistent,t}
\right)
\]

### 6.3 跨状态累计漂移

阶段权重预先固定为：

```text
pre-contact: 1.0, 1.0
contact:     1.5, 1.5
post-contact:2.0, 2.0
```

先归一化使权重均值为1，再定义：

\[
D_{trajectory}
=
\left|
\sum_{t=1}^{6}w_t d_{k,t}
\right|_2
\]

该项与原 Persistent 的区别：

- Persistent 只减少一个状态内部 action chunk 的抵消；
- `D_trajectory` 减少一条轨迹多个决策状态之间的抵消。

方向由每条轨迹当前产生的累计漂移自适应决定，不使用跨所有状态固定的全局方向。

### 6.4 无符号几何子空间约束

令：

\[
q_R=\langle\Delta z,u_{RANGE}\rangle,
\qquad
q_L=\langle\Delta z,u_{LATERAL}\rangle
\]

使用：

\[
D_{geometry}
=
\frac{1}{6}\sum_t
\sqrt{q_{R,t}^2+q_{L,t}^2+\epsilon}
\]

该项只要求扰动进入已验证的几何脆弱子空间，不固定正负号，也不要求不同状态选择相同 RANGE/LATERAL 比例。

### 6.5 三个训练条件

#### DINO-Continue

计算等预算控制组：

\[
J_{DC}=D_{state}+\lambda_DD_{DINO}
\]

`lambda_D` 沿用现有 DINO-Norm 训练的冻结值，不重新根据结果调整。

#### RAMP-T

\[
J_{RT}=J_{DC}+\lambda_TD_{trajectory}
\]

#### RAMP-TG

\[
J_{RTG}=J_{DC}+\lambda_TD_{trajectory}+\lambda_GD_{geometry}
\]

Primary 方法预先指定为 `RAMP-TG`；`RAMP-T` 用于判断 geometry constraint 的增量贡献。

### 6.6 梯度组合

以 DINO-Continue 基础目标为主梯度：

```text
g_base = grad(-J_DC)
g_traj = grad(-D_trajectory)
g_geom = grad(-D_geometry)
```

如果新增目标与基础目标冲突，投影去掉抵消 `g_base` 的分量。

固定梯度比例：

```text
||scaled(g_traj)|| / ||g_base|| = 0.5
||scaled(g_geom)|| / ||g_base|| = 0.2
```

不使用最终 CASR 调整比例。记录实际梯度比例、冲突率和像素饱和率。

---

## 7. 低显存两遍式训练

不能同时保留6个 OpenVLA 计算图，否则3090可能 OOM。每个 trajectory update 使用两遍：

### Pass 1：无梯度确定轨迹方向

依次对6个状态 forward，计算：

\[
S_k=\sum_tw_td_{k,t}
\]

冻结：

\[
b_k=\operatorname{stopgrad}
\left(
\frac{S_k}{\|S_k\|_2+\epsilon}
\right)
\]

### Pass 2：逐状态反向传播

依次重新 forward 每个状态，并最大化：

\[
\sum_tw_t\langle d_{k,t},b_k\rangle
\]

每个状态 backward 后只累计 patch gradient，立即释放模型激活，不同时保存6个计算图。

该梯度等价于当前点上对累计范数 `||S_k||` 的一阶梯度，同时保持 batch size 1 的显存量级。

---

## 8. 训练预算与公平性

三种方法均从同一个现有 `DINO-Norm step 1500` patch 开始，使用新建且相同配置的 optimizer。

额外预算：

```text
100 trajectory updates
6 states / update
= 600 state-level backward exposures / method
```

保存：

```text
trajectory update 0 / 50 / 100
```

公平约束：

- 三个方法相同初始 patch；
- 相同轨迹顺序；
- 相同 optimizer、学习率和更新次数；
- 每次 update 均处理完整6-state轨迹；
- DINO-Continue 也使用相同两遍/逐状态执行结构，排除计算流程差异；
- 不把 trajectory update 数与旧的单状态 step 数直接混为同一单位。

---

## 9. 数据划分

### 9.1 Training

- 现有300 states、50 trajectories。
- 初始状态来源与旧训练库一致。

### 9.2 Development

此前已经查看过结果的状态全部作为 development：

```text
state_index 30–49
```

用途：

- open-loop；
- 小型 recovery probe；
- 判断是否值得进入新的最终测试。

这些状态不再产生独立确认性结论。

### 9.3 Fresh final test

计划使用：

```text
state_index 0–9
```

执行前必须确认：

- 10个任务均存在这些初始状态；
- 不在300-state训练库中；
- 不用于权重、检查点或方法选择。

如不可用，必须在查看任何 RAMP 结果前冻结另一组未使用状态。

---

## 10. 实验阶段

### Stage R0：峰值显存测试

在单张3090上覆盖：

- 完整6-state Pass 1；
- 单状态 Pass 2；
- base / trajectory / geometry gradients；
- 冲突投影、梯度累计和 optimizer step。

根据实测主机 MaxRSS×1.25申请正式内存；不直接沿用大内存配置。3090失败后才考虑4090，不使用A800作为默认资源。

### Stage R1：三种方法计算等预算微调

训练：

```text
DINO-Continue
RAMP-T
RAMP-TG
```

输出训练曲线、实际梯度比例、冲突率、累计漂移和patch饱和率。

### Stage R2：Development open-loop

在30–49上比较：

- Action L2；
- Persistent drift；
- trajectory cumulative drift；
- DINO displacement；
- unsigned geometry energy；
- DINO restore reduction。

### Stage R3：小型恢复实验

在 development 中每个任务固定选择3个 clean-success initial states，共约30个配对状态。

对每个状态：

1. 运行 clean reference，记录 contact/post-contact proxy 的 simulator state；
2. 从完全相同 simulator state 分支；
3. 一个 action chunk 使用 patch；
4. 随后移除 patch，运行3个 clean-policy decision blocks；
5. 与 clean reference 的对应状态比较。

定义初始偏离：

\[
E_0=D(s^{attack}_{t+1},s^{clean}_{t+1})
\]

定义恢复后残差：

\[
E_K=D(s^{attack\rightarrow clean}_{t+K},s^{clean}_{t+K})
\]

恢复残差：

\[
Residual@K=\frac{E_K}{E_0+\epsilon}
\]

同时报告：

- state/eef position residual；
- orientation residual；
- gripper mismatch；
- action residual；
- recovered-within-threshold rate。

该实验只检验恢复机制，不作为攻击成功率的替代指标。

### Stage R4：Fresh final closed-loop

R1–R3整理后先停止并由用户决定是否启动。若启动，在0–9状态运行：

```text
Clean
Random
Exploring-same-location
WHEN-same-location
DINO-Norm-1500 frozen
MGWP-v1-1500 frozen
DINO-Continue
RAMP-T
RAMP-TG
```

规模：

```text
10 tasks × 10 states × 9 conditions = 900 episodes
```

所有攻击条件在同一 state 内使用固定随机顺序，Clean始终先运行。

完成 Object 汇总后停止，不自动扩展其他 suite、模型、EOT或物理环境。

---

## 11. 评估指标

### 11.1 Primary

\[
CASR=
\frac{\text{Clean成功但攻击后失败}}{\text{Clean成功episode}}
\]

正式报告：

- task-equal CASR；
- pooled event count；
- 95% hierarchical bootstrap CI；
- per-task CASR。

### 11.2 Secondary

- Attacked Success Rate；
- Raw Failure Rate；
- Mean episode steps；
- Failure phase；
- Action L2；
- chunk Persistent drift；
- trajectory cumulative drift。

### 11.3 Mechanism and recovery

- DINO displacement；
- unsigned RANGE/LATERAL geometry energy；
- DINO restore reduction；
- `Residual@1/2/3`；
- recovered-within-threshold rate；
- cumulative drift / residual 与 closed-loop failure 的任务分层关系。

---

## 12. 统计比较

统计单位：task → paired episode，不把frame、action step或同一episode内的多个状态当独立样本。

```text
bootstrap draws = 10,000
seed = 20260907
95% percentile CI
```

主比较：

```text
RAMP-TG - DINO-Continue
RAMP-T  - DINO-Continue
RAMP-TG - RAMP-T
RAMP-TG - DINO-Norm-1500
RAMP-TG - MGWP-v1-1500
```

解释：

- `RAMP-T - DINO-Continue`：trajectory objective 的贡献；
- `RAMP-TG - RAMP-T`：unsigned geometry 的额外贡献；
- `DINO-Continue - DINO-Norm-1500`：额外训练本身的贡献。

同时报告 effect size、CI、positive/negative/tied tasks，不只报告是否通过。

---

## 13. 结果解释边界

### 情况 A：RAMP-T / TG 高于 DINO-Continue，Residual也更高

支持：跨状态累积目标能够利用 Wrist-DINO 脆弱性突破闭环恢复。

### 情况 B：RAMP高于 DINO-1500，但不高于 DINO-Continue

说明提升来自额外训练，而不是 recovery-aware objective。

### 情况 C：累计漂移和Residual提高，但CASR不提高

说明恢复受到影响，但尚未跨过任务失败边界；不能宣称攻击增强成功。

### 情况 D：CASR提高，但DINO restore不支持

只能说明补丁变强，不能说明提升主要经 Wrist-DINO 路径传播。

### 情况 E：RAMP-T优于RAMP-TG

说明几何子空间适合解释脆弱性，但限制了最有害 DINO 扰动；主方法应收窄为 recovery-aware DINO，而不是 geometry-guided attack。

---

## 14. 与 Exploring / WHEN 的差异

| 维度 | Exploring | WHEN | RAMP |
|---|---|---|---|
| 核心目标 | 单状态动作偏差 | 通用特征扰动/迁移 | DINO机制＋跨状态累计漂移 |
| 时间结构 | 主要为单状态/动作块 | 主要为通用feature objective | 同一轨迹多个阶段联合优化 |
| 恢复机制 | 不显式建模 | 不显式建模 | 单独测量并抑制闭环恢复 |
| 机制证据 | 输出效果为主 | feature效果为主 | DINO restore＋recovery residual |
| 方向 | 通用攻击方向 | 通用feature方向 | 每条轨迹自适应，不固定全局方向 |

预期贡献表述：

> RAMP does not merely maximize instantaneous action or feature deviation. It exploits the Wrist-DINO vulnerability while explicitly optimizing trajectory-level drift that persists across manipulation phases and resists subsequent closed-loop correction.

---

## 15. 工程与资源规范

- 正式模型 forward/backward只在GPU运行，禁止CPU fallback。
- 首选RTX 3090，单任务1张GPU、4 CPU。
- 正式任务前先跑完整峰值显存与主机内存测试。
- GPU array最多并发5张卡。
- 训练每10个trajectory update保存恢复点，每25个update打印一次摘要。
- 输出按method/task聚合，不逐state生成小文件。
- 原始重复测量保存Parquet，汇总保存CSV/JSON。
- 已完成的MGWP-v2结果只读保留，不覆盖、不删除。

建议目录：

```text
oft_ramp_recovery_aware/
  configs/
  scripts/
  slurm/
  artifacts/patches/<method>/
  outputs/train/
  outputs/open_loop/
  outputs/recovery/
  outputs/closed_loop/
  results/
  reports/
  logs/
```

---

## 16. 执行停止点

若后续授权执行，默认只运行：

```text
R0 峰值测试
R1 三方法微调
R2 development open-loop
R3 小型 recovery probe
```

完成后停止并汇报，不自动进入 R4 fresh final closed-loop。

R4 必须由用户查看 R1–R3 结果后另行授权。
