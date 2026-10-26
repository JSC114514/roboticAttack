# BATCH8-WHEN-DINO2000-V1（2026-10-05）
用户已确认本方案：先更新方案、实现与CPU预检，交付手动提交命令。助手不提交GPU，不使用重置卡。跨模型和实机不在本轮。

## 训练组与目的
新增四组从 seed42 同一随机3×50×50 RGB（7500参数）开始，空Adam，0至2000 outer×50 inner=100000更新，每更新平均8个不同记录的梯度（微批1、累积8）。不加载旧学习补丁/前缀。统一lr .002、wd0、warmup20、cosine总2000，[0,1] clamp；不加TV/HF/平滑。

- original_uada_b8：复制Exploring→OFT适配基线，只用clean300、动作第0维错误边界MSE、原随机贴图/affine。
- 原适配UADA记录打乱seed保留20260920；其他采样组保留20260929+seed42，避免补batch时无意改变原组的数据seed。
- uniform_replay_b8：复用STATE-AXIS uniform_axis0数据与日程，250前clean，250后clean/replay交替、任务/阶段覆盖、三帧轮换、固定坐标0；后半inner使用affine，250后该半段附加合法随机位置与亮度/对比度。
- dino_sampling_b8：保留dino_remaining的状态/XYZ联合选择，p=.5/N+.5*s/sum(s)，s=两视图平均max(0,e_restore-e_patch)*e_patch。250/500/1000/1500后按当前量化RGB刷新，恢复只在no_grad评分中，正常训练不替换特征。
- when_dino_b8：沿用uniform的源数据/EOT，在WHEN作者feature_space.py副本的特征L1+反向BiContrastive核心上改为实际腕部DINO dense输出。最小化 -(mean L1+10*token InfoNCE)，tau .07。仅未被实际贴图覆盖的dense token参与，两项均按每图归一化后跨8状态平均。不引入跨状态负样本，梯度累积等价于8状态损失均值。目标不含CLS/register、动作、PAD/PSM或全图扰动。这是WHEN特征核心适配，不是完整WHEN复现。

复用已有original_uada、uniform_axis0、dino_remaining的1000/2000权重，只重测，不重训。均匀对DINO采样比较同时含XYZ选择变化，不宣称单独DINO增量。OFT-UADA是我们训练的适配基线，不是作者发布补丁。

## 数据、预算和学习证据
Object clean300/replay258及既有训练10–39、评估40–49划分不变。每outer选择8条互异记录，50 inner共同更新；replay每条3帧循环。任务/阶段配额沿用原记录顺序扩展成batch，并拒绝重复记录。每条样本分别随机EOT，每8条平均一次Adam更新。
B1=100000受扰样本计算；B8=800000。报告两种预算和GPU时间，不把同轮次增益当成等计算成本增益。
保存0/250/500/1000/1500/2000的PT/PNG；每outer记录损失、实际RGB移动、8条记录/坐标、样本计算数量和耗时。首个有非零学习率outer必须有真实RGB更新；末尾float32舍入仅有Adam增量证据时允许。
恢复原子保存patch、Adam、scheduler、全部RNG、选样评分及历史；中断重跑从最近提交outer恢复，未提交耗时另记。所有方法同随机起点，非共同学习前缀。

## 自动评测和资源
手动入口串联CPU准备→4张4090训练array(0–3%4)→CPU产物检查→3090评估array(0–9%8)→CPU统计。每训练作业1卡、4CPU、22G RAM、72h上限；微批1控制显存。训练上限4≤5，测试上限8；两阶段afterok隔离。用户已有GPU/OFT链先afterany等待，队列失败不提交，重复链拒绝。失败依赖自动取消，恢复沿用同一命令。
1000/2000各7条件、每条件四suite×10任务×10初态：5600攻击回合、5348 clean-success资格行，冻结clean400。正常OFT预测、量化单腕部补丁，Front正常、无训练hook/恢复。Long必测但次要；主要Spatial、Goal与SG平均，Object源域单列。
补充比较为UADA B8−B1、uniform B8−B1、DINO采样B8−B1；推进主要比较为WHEN−uniform B8，其次WHEN−DINO采样B8，并列出DINO采样−uniform B8、WHEN−UADA B8。仅由已有评测结果派生，不增加训练组或测试回合。配对task→state bootstrap10000、seed20260907。一个训练seed，开发评测集，不冒称跨seed稳定性。
B1动作训练2000实测约6.5h；微批1累积8时B8动作训练保守45–60h，DINO评分另计，WHEN需真实GPU测速。测试预计3–4h，排队另算；不会保证10月6日中午整批完成。正式训练前20轮记录测速/显存，不额外提交诊断作业。外观仅保存供用户判断，不阻断自动测试。
所有输出集中于research_runs/wrist_dino_batch8_when_v1。代码在code/独立副本；旧结果和源代码不覆盖。来源快照、源码SHA、数据SHA和三条历史补丁SHA在CPU准备时冻结。
方法依据：WHEN https://arxiv.org/html/2511.21192v3#S3.SS3 与 https://github.com/yuyi-sd/UPA-RFAS；Exploring https://github.com/William-wAng618/roboticAttack。

## 状态
IMPLEMENTED_AND_CPU_PREFLIGHT_PASS。四组训练、自动校验、四suite评测与统计，以及142–146号Slurm脚本和手动提交入口已实现。38项CPU测试通过（新增20项与旧STATE-AXIS18项）；真实冻结资产与2000轮全选样预检通过，未加载模型、未初始化CUDA、未提交作业。梯度平均、作者特征核心副本、互异batch、随机起点、完整Adam/RNG恢复、评测覆盖和资源依赖均有行为测试。自动测试前检查运行对齐、快照SHA及最终checkpoint/resume的RGB相等性；这道校验也经四组2000轮CPU产物代理覆盖。实际显存、耗时、外观和CASR在用户提交后产生。
提交入口为 `bash submit_exploring_oft_batch8_when2000_v1.sh`；`--preflight`只读预检，不查询Slurm、不提交。检查点与PNG在 `training/<method>/`，最终结果在 `evaluation/four_suite/`，作业回执在 `reports/SUBMISSION_RECEIPT.json`。原始Exploring及WHEN作者源文件与独立只读快照SHA相同，未修改旧源码/补丁/实验结果。
