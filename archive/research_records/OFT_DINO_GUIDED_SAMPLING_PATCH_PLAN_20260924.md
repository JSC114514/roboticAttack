# OFT Wrist-DINO 引导采样补丁：500 轮配对实验

状态（2026-09-25）：CPU 选点与离线审计已完成；用户提交的两组 500 轮训练均已完成，500 轮补丁图像已由用户明确审核“通过”；四套件闭环测试**尚未提交、未运行**。所有 Slurm 作业由用户亲自提交。原 Exploring `UADA.py` 保持只读和原始 SHA256；新方法在独立副本/适配器中实现。

## 问题与调整

先前 Object-Decoy r03/r10 没有把补丁稳定诱导成目标物体，且更强的 DINO 附加目标挤占动作主目标。现在不再增加 DINO 辅助损失。保留 Exploring UADA 的连续动作第 0 维 opposite-boundary MSE、AdamW、每 outer 50 次随机几何内循环、余弦日程与 RGB clamp；DINO 只用于离线确定训练样本的抽样概率。Front、语言、本体状态、模型保持不变；补丁仅贴 Wrist。

CPU 选点脚本：`roboticAttack-main/scripts/plan_dino_guided_placement.py`。在 Object 训练 bank 的 10 任务×4 条 discovery 轨迹（240 帧）上，用审计过的目标/背景掩码拟合每任务 DINO 原型，并从 676 个固定 50×50 Wrist 位置中选出 DINO 分数最高的位置 `(91,140)`。另用相同 discovery 集的目标物体位置先验选出 `(84,140)`。每任务第 5 条轨迹共 60 帧仅作离线审计：DINO 位置平均覆盖目标 57.58%，先验位置 59.07%。因此**不能宣称 DINO 比静态位置先验更会选位置**；两位置也都高度遮挡目标。此结果限制了新位置相对旧 `(26,137)` 的解释。

在固定 `(91,140)` 位置，预留轨迹内按 DINO 补丁区域目标概率排序，前三帧的目标掩码覆盖率比后三帧平均高 7.09 个百分点，10 个任务中 8 个同向。该信号有一定动态信息，但只有 10 条独立轨迹，不等于攻击收益。我们据此只检验 DINO 引导**训练画面抽样**是否比同位置普通 UADA 更有效。

训练抽样权重在各任务内的有效样本数为 22.83–28.97/30，并未坍缩为只重复极少数帧；这是抽样覆盖检查，不是攻击效果证据。

## 两个新训练组

两组都从相同 seed 的随机 RGB 补丁开始、500 outer × 50 inner、同一 Object bank、同一任务轮换、同一 `(91,140)` 固定训练/评估位置、同一仿射 EOT、同一动作损失、同一学习率和结构门。唯一差异：

- `exploring_oft_site_uada`：每个 outer 按原 UADA 已冻结的记录置乱序列选择训练状态。
- `exploring_oft_site_dino_sample`：任务顺序保持一致；在该任务 30 个训练状态内，按干净 Wrist-DINO 的补丁区域目标概率质量加权抽样。原型仅在 discovery 轨迹拟合；不使用闭环测试帧、不查询未来动作或任务结果。无 DINO 反传、无辅助权重。

原有 UADA500 补丁 `(26,137)` 作为历史性能参照；它与新组的差异同时包含位置、EOT 和采样设置，不用于单独归因 DINO。训练 200/500 checkpoint 都保存，但 200 是 500 轮日程中的中间点，不能冒充独立的 200 轮训练。补丁图像必须由用户审核，不能只凭结构数值放行。

## 闭环评估与主结论边界

用户审核通过且训练补丁 SHA256 冻结后，在 Object、Spatial、Goal、Long 四套件使用同一 40 任务×10 状态、同一 frozen clean、同一 rollout seed 配对测试四条件：旧 UADA500、新同位置 UADA、新同位置 DINO 引导、同位置纯灰色 50×50 遮挡。总计 1600 次新增攻击 rollout，clean rollout 复用。

主对比是**DINO 引导减同位置 UADA**的分套件任务等权 CASR；旧 UADA500 只作性能参照；纯灰遮挡用于判断是否只是挡住物体就能造成失败。Object 是训练套件，Spatial/Goal 是零样本迁移，Long 照测但不作为唯一主结论。不合并四套件报一个总体数字。若两新补丁都与灰遮挡接近，不能宣称新训练方法成功。若 DINO 引导不胜同位置 UADA，但同等 CASR 使用更少训练步数，才可讨论成本收益，并必须把 DINO 离线准备时间计入成本。

## 文件与人工执行顺序

训练入口：`roboticAttack-main/VLAAttacker/UADA_OFT_DINO_SAMPLING_wrapper.py`；攻击副本：`roboticAttack-main/VLAAttacker/white_patch/UADA_OFT_DINO_SAMPLING.py`；配置：`roboticAttack-main/configs/exploring_oft_dino/dino_sampling500_v1.json`；Slurm：`roboticAttack-main/slurm/exploring_oft_dino/44_train_dino_sampling500_3090.sbatch`。提交 helper：`roboticAttack-main/submit_exploring_oft_dino_sampling500.sh`。先在 helper 上 `CHECK_ONLY=1`；训练需用户手动运行不带该变量的 helper，或直接 `sbatch` 对应脚本。

两组训练作业 `136245_0/1` 均完成 500/500，训练表各 500 行且无缺失，100/200/500 结构门全部通过；各耗时约 2 小时 45 分钟。普通组最终补丁 SHA256 为 `74f69451158eb0ff2633d29aa56a3edf737508b59e661f73b83bb2681e9a7891`，DINO 引导组为 `ff0048562cf8107cecd432c5b26649d205de552e428dbbe940d6c85084a1c94c`。用户已于 2026-09-25 确认两张 `outer_0500.png` 外观通过；评估配置已冻结为 `PASS_USER_DINO_SAMPLING500` 和上述哈希。闭环测试仍只能由用户手动运行 `submit_exploring_oft_dino_sampling500_cross_suite.sh`；审核通过不等于已经测得攻击成功率。

2026-09-25 评估 `CHECK_ONLY=1` 预检通过：四条件补丁哈希匹配、四套件 frozen clean 成功数为 Object 99、Spatial 90、Goal 98、Long 95；预计新增 1600 次攻击 rollout、其中 1528 行进入 clean-success 条件 CASR。更新后的 5 项单元测试通过。预检**没有提交**任何 Slurm 作业。

程序级验证：`roboticAttack-main/tests/test_exploring_oft_dino_sampling.py`、`py_compile`、`bash -n` 和训练 dry-run。GPU 训练已由用户提交完成；闭环 rollout 尚未执行。
