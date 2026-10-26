# OFT-DINO-STATE-AXIS2000-V1 — 2026-10-03 冻结实施方案

本附件是当前唯一新增执行方案。先写方案、后实现；旧训练、结果和只读 Exploring UADA.py 不改。用户手动提交；助手不提交、不使用额度重置卡。跨模型和实机不在本轮。

## 1. 根据完整旧结果作出的选择

sampling-clear 四组已完整闭环：4800 攻击回合、4584 clean-success 有效行。2000 轮 A 动态软约束的 Object/Spatial/Goal/Long CASR 为 100/8.333/64.667/68.389%；旧 diverse 同轮次为 100/21.667/51.667/76.889%。硬外观组 B/C/D 约 84–87% 更新触发修正，攻击性明显下降；稳定评分 C 也未胜过打乱 D。不继续硬降噪或上述四组扩轮次。

保留源 Object 受扰连续片段、Exploring 派生动作损失、全 RGB 与 EOT。新问题是：在不同状态选用不同平移动作目标，且把训练投入既受 Wrist-DINO 影响、又尚未到达错误边界的状态，能否改善 Spatial，同时保住 Goal？不宣称自动获得纹理或跨套件收益；允许细碎噪声，保存 PNG 给用户判断，不用外观门阻挡自动测试。

## 2. 三个从零独立训练组

|代码名|选样与目标|作用|
|---|---|---|
|uniform_axis0 (R0)|任务/阶段内均匀选中心，固定动作坐标 0|相同数据/EOT/无外观约束的参照|
|action_remaining (R1)|联合选中心与 XYZ 坐标，按当前归一化剩余边界误差选样|一般动作难例/坐标覆盖对照|
|dino_remaining (M)|联合选中心与 XYZ 坐标，按 DINO 恢复贡献 × 剩余误差选样|检验 DINO 信息的额外作用|

主要比较 M−R1，其次 R1−R0；后者同时改变坐标覆盖与难例采样，不能单独归因其一。另同场重测既有 original_uada 与 diverse_reference 的 1000/2000 检查点，不重训旧补丁。没有 SigLIP 新训练臂、额外梯度放大或特征损失。

三个组各自 seed42 原始随机 3×50×50 RGB、空 Adam 从 0 到 2000 outer×50 inner=100000 更新；不是续训或加载共同训练前缀。AdamW lr=.002、wd=0、warmup20、余弦总2000，同旧 Exploring→OFT 副本。唯一像素约束 [0,1]；无 TV/HF/平滑/硬投影。Front、状态、语言和模型不更新，只贴 Wrist。

## 3. 数据、EOT 和联合采样的完整定义

复用冻结 Object clean300 与 WINDOW_BANK258 三决策块片段，仅复用数据，不用采集补丁权重或旧评分。训练初态10–39；评估40–49；不混入 Spatial/Goal/Long。前250轮 clean/坐标0；之后奇数 local 用 clean/坐标0，偶数 local 用 replay。沿用原 base-index 的 task 配额与 early/middle/late 轮换。

每次 replay 外层只选一个中心及一个坐标 j∈{0,1,2}，50内层不换坐标，3帧循环；各帧的目标来自本帧 clean 动作，以各坐标 q01/q99 的较远端为错误边界。实际训练仍用该单坐标的原 MSE，不新增 loss。各组 EOT 完全相同：每轮后半内层采用原 affine；250后仅该半段再用合法随机位置0–174及亮度/对比度 .8–1.2，前半保留固定位置无 affine。

250/500/1000/1500后用各组当前量化 RGB（与PNG一致）刷新中心评分；不影响训练 RNG。中心 clean 动作给定 t，r_j=q99_j−q01_j，8块预测上的评分 e_j(A)=mean_h[((A_hj−t_hj)/r_j)^2]。每中心两视图：view0 固定位置无 affine；view1 合法位置、亮度/对比度及 affine；seed=data_seed×1000000+refresh_outer×10000+record_index×2+view。两种视图正好匹配上述训练的两种 EOT 条件。

R1 优先值为两个视图 e_patch,j 的算术平均。M 每视图 c_j=max(0,e_restore,j−e_patch,j)，优先值为两个视图 c_j×e_patch,j 的算术平均（不是先取平均再相乘，更不是取最小值）。restore 只恢复 clean Wrist-DINO token；缓存完全相同的受扰像素，Front、当前受扰 Wrist-SigLIP 不变并断言相等。只用于 no_grad 评分，实际训练和部署不恢复特征。R0 同样记录 clean+两正常视图的误差作为诊断，但不用它加权；R0/R1每中心3次 forward，M每中心5次；成本全部单独计数。

每 task×phase 桶中候选为全部中心×XYZ。R1/M 的概率 p_k=.5/N+.5*s_k/sum(s)；s均为0时 p均匀。R0只在该桶中心中均匀选，坐标0。固定伪随机种子和桶配额；记录完整视图、分数、选样坐标、refresh边界，可在 CPU 重构每次选样。

## 4. 自动正常模型测试和报告

训练后 CPU 检查完整预算、Adam/RNG恢复、初始RGB、PNG、源码、动态评分及真实像素移动，然后自动测试。1000/2000各5条件，四suite各10任务×10状态：4000回合（2400新、1600参照），3820资格行，冻结clean400。正常官方 OFT predict_action、单腕部量化补丁；无评分 hook、恢复或额外扰动。各suite正确checkpoint/unnorm；共享初态、seed与随机条件顺序。

首先单独报告 Spatial 的 M−R1，并展示 Goal 是否退化和 Spatial/Goal 等权平均；Object 是源suite，Long完整测试但次要。任务→clean-success 初态层次配对 bootstrap10000、seed20260907，各条件和两个检查点共享抽样。单训练seed、已使用过的开发评估集，不称未见最终测试或已证明普适机制。同时报告不同坐标频次、每轮实际RGB更新和500/1000/2000外观，不用补丁相似程度代替学习证据。

## 5. 工程、资源与时限

独立 Exploring 派生副本、新目录、新配置与来源快照；旧代码/产物不覆盖。逐outer原子 resume 保存 Adam、scheduler、所有 RNG、评分表和历史；失败重试从最近完整outer继续，已执行但未提交的成本单列保留。仅最终低学习率的可证明 float32 舍入允许整轮像素不动，不能以此宽免早期停训。

用户 helper 串联 CPU准备→3×4090训练→CPU验证→3090 task-array(0–9,%8)→CPU统计。阶段 afterok 隔离；最多3训练卡（≤5）、最多8测试卡；依赖失败终止下游，helper保存每次提交回执并阻止重复链。统计结束不自动扩轮次。无新采集，预计训练6–8小时、测试2.5–3小时、CPU准备/验证/统计10–30分钟，整链约9–12小时，排队另算。实际 GPU 接口与收益须运行后判断。

方法来源：Exploring [论文](https://arxiv.org/html/2411.13587v3)与[官方代码](https://github.com/William-wAng618/roboticAttack)供原始RGB/动作/EOT骨架；WHEN [论文](https://arxiv.org/html/2511.21192v3)仅启发跨条件覆盖，不复制其完整方法。论文训练出结构不代表损失本身保证低噪声，本轮主动移除后来加的外观主导约束。

## 6. 执行状态

IMPLEMENTED_AND_PREFLIGHT_PASS，尚无新训练/CASR结果；用户手动提交后才开始GPU链。已新增独立配置、训练器、wrapper、正常OFT评估、配对统计、122–126号Slurm脚本和helper，不修改只读Exploring `UADA.py`。

关键新增/修改文件：

- `configs/exploring_oft_dino/object_dino_state_axis2000_v1.json`
- `VLAAttacker/white_patch/UADA_OFT_DINO_STATE_AXIS.py`
- `VLAAttacker/UADA_OFT_DINO_STATE_AXIS_wrapper.py`
- `scripts/dino_state_axis_common.py`
- `scripts/run_exploring_oft_dino_state_axis_cross_suite.py`
- `scripts/analyze_exploring_oft_dino_state_axis_cross_suite.py`
- `slurm/exploring_oft_dino/122_state_axis_prepare_cpu.sbatch` 到 `126_state_axis_stats_cpu.sbatch`
- `submit_exploring_oft_dino_state_axis2000_v1.sh`

预检已完成：`py_compile`、5个sbatch语法、helper `--preflight` 均通过。dry-run核对300条Object clean、258条Object replay、三组方法、2000 outer/100000更新、从零随机RGB、无历史patch初始化、无共享训练前缀；`model_loaded=false`、`cuda_initialized=false`、`sbatch_submitted=false`。

手动提交命令（在 `roboticAttack-main` 目录）：

```bash
bash submit_exploring_oft_dino_state_axis2000_v1.sh
```

只检查不提交：

```bash
bash submit_exploring_oft_dino_state_axis2000_v1.sh --preflight
```

预计耗时：CPU准备/验证/统计约10–30分钟；三组4090并行训练约6–8小时；8张3090并行测试约2.5–3小时；整链约9–12小时，另加排队。训练产物在 `research_runs/wrist_dino_state_axis_v1/training/<method>/outer_1000.png` 和 `outer_2000.png`；最终结果在 `research_runs/wrist_dino_state_axis_v1/evaluation/four_suite/`。

## 7. 2026-10-03 提交前代码复核与工程修复

审核发现并修复一个确定的自动测试阻塞：原训练日志仅在已评分 replay 行写入 `selected_*`，clean/前250轮行缺键，Parquet 会补成 NaN，导致完整训练后的 CPU 验证失败。现在每行统一四个有限数值字段，并以 `selected_score_available` / `selected_restored_error_available` 标明适用性；不适用值仅作零占位，评分统计排除占位行。CPU验证逐轮重构选样，同时核对真实评分及标志。该修复不改变训练目标、候选、数据、评分公式或预算。

另将最终状态元数据与本方案对齐：首要指标为 Spatial，首要比较为 M−R1，Spatial/Goal 平均为联合指标；三个CPU阶段的线程数与申请的2核一致。原只读UADA及历史结果未改动。

新增 `tests/test_dino_state_axis.py`，18项CPU行为回归通过：三组从零训练与RGB更新、无外观约束、单轴与三帧/EOT日程、PNG一致性、训练表序列化/评分重构、中途失败后patch/Adam/RNG/评分一致恢复、两种贡献符号下的真实评分公式与归一化、相同受扰像素/DINO恢复/hook清理、实际贴图梯度及位置恢复、联合选样与均匀底座、模拟提交依赖和失败回执。6个Python AST及helper/5个sbatch语法检查通过。CPU/mock通过不表示真实GPU启动或CASR收益已通过；GPU模型/EGL接口在正式作业启动时继续检查。
