# OFT-DINO-REGISTER500-V1 — 2026-10-04 实施方案

本轮目标不是继续扩大上一轮 state/axis 采样，而是把 DINO 利用点从“路径/采样”推进到 DINOv2 内部 token 聚合。训练仍然只产生单腕部 RGB patch；部署和测试仍然是正常 OFT 前向，不安装 hook、不恢复特征、不改 Front/状态/语言/模型。

## 1. 为什么换方向

已完成的 state-axis 结果显示：`uniform_axis0` 在 Goal 上强，`dino_remaining` 没有稳定胜过一般动作难例采样。这个结果不能说明 DINO 机制不存在，但说明“按 DINO 恢复贡献选状态/坐标”没有抓到最能提升跨套件攻击性的点。继续在采样权重上小修小补，容易变成只调训练分布，不像一个利用 DINO 特性的攻击方法。

本轮转向 DINOv2 register token。当前 OFT checkpoint 的视觉配置为 `vit_large_patch14_reg4_dinov2.lvd142m`，中间层有 1 个 CLS + 4 个 register prefix tokens；下游最终只拿去掉 prefix 后的 dense patch tokens，但 register 在中间层参与自注意力聚合。因此合理假设是：如果补丁能让 register 在中层更稳定地读取补丁区域，它可能更早、更广地污染 dense visual representation，再由动作头放大为错误动作。

这个假设比“DINO路径有效”更具体，也比“让所有特征变坏”更像一个可报告的方法。

## 2. 训练方法

全部从 seed42 随机 RGB 从头训练 500 outer × 50 inner，AdamW/lr/warmup/EOT/贴图位置/动作目标沿用 Exploring-OFT UADA 派生代码。学习率调度仍按 2000 outer 的余弦总长计算，使 500 轮补丁和已有 UADA500/uniform500 同预算可比。

数据仍只用 Object clean300 + Object replay258；前 250 轮 clean，之后按原 uniform axis0 replay 日程训练，不混入 Spatial/Goal/Long。动作主损失仍是第 0 维连续动作 opposite-boundary MSE。

新增的 DINO 项不是替代主损失，而是训练期内部引导：在 DINO featurizer 的中层 block 8 和 16，读取 attention 的 Q/K，最大化目标 query tokens 对腕部 patch 区域 key tokens 的 attention mass。实现时不改模型输出，只对 RGB patch 求梯度。

为避免任意 loss 权重，采用梯度比例：

```text
grad = action_grad + ratio * ||action_grad|| / ||dino_grad|| * dino_grad
```

三个新训练臂：

| 条件 | DINO 目标 | 梯度比例 | 作用 |
|---|---:|---:|---|
| register_g10 | 4 个 register tokens 读 patch tokens | 10% | 温和机制引导 |
| register_g30 | 4 个 register tokens 读 patch tokens | 30% | 更强机制引导 |
| spatial_g30 | 普通非 patch dense tokens 读 patch tokens | 30% | 非 register 的注意力控制 |

同场自动评估两张参照补丁：

- `original_uada_500`：旧 Exploring→OFT UADA 500 轮基线。
- `uniform_axis0_500`：上一轮最新同管线 uniform 500 轮对照。

## 3. 评估与结论边界

训练完成后自动做四 suite 正常 OFT 测试：Object、Spatial、Goal、Long，各 10 task × 10 state，共 5 条件 × 400 = 2000 回合。只统计 clean-success 初态上的 CASR，任务配对 bootstrap 与此前一致。

主看 Spatial 是否相对 `uniform_axis0_500` 和 `original_uada_500` 提升，同时 Goal 不能明显崩掉。Object 作为源套件，Long 完整报告但不是主结论。若 register_g10/g30 至少一个在 Spatial 有实际提升，再扩 1000/2000 并补同轮次对比；若 register 不赢 spatial control 或 uniform，则本假设作为明确负结果停止，不继续无边界调参。

## 4. 工程约束

源 `VLAAttacker/white_patch/UADA.py` 保持只读、SHA256 不变。新代码在独立文件中实现：

- `configs/exploring_oft_dino/object_dino_register500_v1.json`
- `scripts/dino_register_common.py`
- `VLAAttacker/white_patch/UADA_OFT_DINO_REGISTER.py`
- `VLAAttacker/UADA_OFT_DINO_REGISTER_wrapper.py`
- `scripts/run_exploring_oft_dino_register_cross_suite.py`
- `scripts/analyze_exploring_oft_dino_register_cross_suite.py`
- `slurm/exploring_oft_dino/127_register_prepare_cpu.sbatch` 到 `131_register_stats_cpu.sbatch`
- `submit_exploring_oft_dino_register500_v1.sh`

用户手动提交；助手不自动提交、不使用额度重置卡。

## 5. 预计耗时

三张 4090 并行训练 3 个新补丁，预计 2.5–4 小时；8 张 3090 并行评估 10 个 task shard，预计 1.5–2.5 小时；CPU 准备/验证/统计约 10–30 分钟。整链约 4.5–7 小时，排队另算。

## 6. 当前状态

IMPLEMENTED_AND_PREFLIGHT_PASS，尚未提交 Slurm 作业，尚无新训练或 CASR 结果。`py_compile`、helper `--preflight`、127–131 号 sbatch 语法检查已通过；CPU 预检核对了 300 条 Object clean、258 条 Object replay、两张 500 轮参照补丁、四 suite clean-success 分母、只读 Exploring `UADA.py` SHA256 与 source manifest。预检未加载 GPU 模型、未初始化 CUDA、未提交队列。

2026-10-04 代码审核发现训练目标中的 `torch.flatnonzero` 在实际 PyTorch 2.5.1 环境不存在，已替换为等价的 `torch.nonzero(..., as_tuple=True)[0]`。审核覆盖了注意力公式、两种 query 选择与梯度的小型 CPU 张量检查，并核对 251–500 outer 共 6250 个多样化几何变换的 token footprint；这些检查不代表真实 GPU 训练或攻击效果已通过。DINO 增量应主要与同管线 `uniform_axis0_500` 比较；相对旧 UADA 的差异同时包含 replay、贴图位置与 EOT 的变化。

手动提交命令：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_register500_v1.sh
```

只检查不提交：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_register500_v1.sh --preflight
```
