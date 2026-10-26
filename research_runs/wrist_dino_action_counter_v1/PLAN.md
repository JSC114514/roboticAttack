# Wrist-DINO 动作反制训练：500轮六组筛选（2026-10-02）

用户授权：先更新方案，再独立实现/CPU预检，提供手动提交命令。禁止助手提交GPU、使用重置卡或自动扩到1000/2000。所有新结果写本目录，旧实验不覆盖、不删除。

## 1. 依据与边界

上一轮repair1000完成。任务等权CASR（Object/Spatial/Goal/Long）：旧diverse1000为100/23.556/58.889/81.000%，clear为28/9.667/12.444/49.889%，uniform_clear为13/8.556/8.556/55.028%，causal为10/11.111/7.222/45.417%，repair为12/5.111/4.222/41.167%。四组真实更新，持续TV/HF降低噪声却使攻击退化。repair使DINO误差下降约11.1%，不等于动作反制或闭环收益；D/E还绑定clean Wrist-SigLIP替换，不能单独归罪于repair。

本轮问题：正常完整OFT上，用Wrist-DINO指导反制指定错误动作的输入方向，再训练补丁抵抗反制，能否优于随机增强并改善Spatial迁移？同时测有限外观预算的攻击代价。

- 原`VLAAttacker/white_patch/UADA.py`保持只读/冻结SHA；只新增源码副本/派生文件与独立入口，旧训练器不改。
- 单腕部RGB50×50×3，7500独立参数；Front、语言、proprio、模型权重不改，不做双摄补丁/低维参数化。
- 六组各自seed42同随机RGB、outer=0、空Adam开始；2000轮cosine日程、20轮warmup。**不加载已训练P或Adam，不复用250轮优化前缀**。
- 共用300条Object clean和258条Object replay。历史随机250轮P采集的replay只作共同源数据，不作初始化；不重新采集。
- 为隔离新增机制，冻结旧diverse训练表前500条源采样顺序，各组同record索引、同task/state/phase；前250轮clean，其后clean/replay交替。不各自动态重评分。比较是条件于这张共同源采样表的增量，不包装成新动态采样方法。
- Exploring渲染/位置(26,137)/后半inner几何保持一致；前250轮沿用原前缀EOT，251轮起后半inner沿用已有diverse随机合法位置、0.8–1.2亮度/对比度。各组同EOT seed。
- 只使用Object模型/图像训练；Spatial/Goal保持训练零样本，但已用于开发，不称盲确认。Long完整测试、单列次要。跨模型/实机延后。

## 2. 动作一致的两阶段更新

干净完整OFT动作a_c固定错误目标t：第0维在q01/q99中选择离a_c更远的边界。原UADA动作损失为8预测chunk上的`L = mean((a[...,0]-t[...,0])^2)`；8chunk不是8DoF。

内层固定P/model及一次EOT的实际RGB/support，η只能改变补丁外Wrist RGB。η0=0，两步sign-PGD上升，步长2/255，L∞≤4/255，投影合法RGB。

- 每个η重新计算真实完整OFT；仅Wrist-SigLIP输出暂时detach对η的梯度，**不替换/缓存clean或η0激活**。DINO用OFT实际消费的dense输出，不写死最后block，不假称已定位非覆盖token为主要因果中介。
- 用DINO偏导增大同一L，在η0/η1/η2中选实际完整OFT L最大者，防止偏导近似反向增强攻击。
- 此为DINO指导的近似动作反制，不是严格全梯度min-max；增大错误目标MSE不保证恢复clean动作，另记录相对clean动作距离。

外层detach η，重建正常完整OFT梯度，最小化同一L和本组外观项，只更新P并clamp[0,1]。没有DINO附加损失、梯度倍率或clean-SigLIP替换。

参考WHEN §3.4两阶段反制思想：https://arxiv.org/html/2511.21192v3#S3.SS4 。本轮改用OFT动作目标及DINO偏导，不称完整WHEN复现，不借论文跨模型结果保证跨suite提升。

## 3. 六组：一个方法、两个对照、两档外观

| ID / method | η方式 | TV预算τ |
|---|---|---:|
| N0 / none_original | 无η | 0.4 |
| N1 / none_clear | 无η | 0.3 |
| R0 / random_original | 匹配随机η | 0.4 |
| R1 / random_clear | 匹配随机η | 0.3 |
| D0 / dino_original | DINO动作反制η | 0.4 |
| D1 / dino_clear | DINO动作反制η | 0.3 |

沿用旧相邻RGB差异TV定义，`Lapp = relu((TV(P)-τ)/0.4)^2`、权重1。τ0.4与原hinge完全相同；τ0.3只改预算，不缩小分母。撤掉连续CharbonnierTV/GaussianHF目标，HF只作读数。0.3是冻结工程预算，不是理论最优/外观保证。

R组先在自身P上求同一反制η，再通过RGB可行的成对空间交换打乱排列，并在正负均可行的位置随机取符号（边界只取可行符号），保留绝对幅度/零值集合、L∞/RMS/mean|η|和补丁外support。控制的是幅度而非带符号均值；η0时随机组也0。clamp后核对，随机化失败报工程错误，不悄悄用有方向η代替随机。PGD成本计入R，不称低成本。

比较D−R（DINO方向增量）、R−N（普通增强）、同方法τ0.3−τ0.4（外观代价）及两档D−R差的交互。并列原UADA500和旧diverse500参照，不用500新P仅对1000旧P。

## 4. 执行与终点

- 每组500outer×50inner=25000更新，保存0/250/500 PT/PNG及完整Adam/scheduler/RNG断点；500时不使学习率归零。
- 启动嵌入真实前向identity、内层SigLIP仅截梯度、η支持/范围、η0候选、外层hook移除和RGB更新检查。合法BF16饱和不冒充接口失败。
- 记录MSE(η*)−MSE(0)、反制有效率、clean动作距离、η随机匹配、TV违反率、实际RGB更新/PNG变化、forward/backward/耗时。完整outer原子保存，失败不将半轮状态记完成。
- 用户手动入口串联CPU准备→4090训练array0–5%5→CPU验证→3090测试array0–9%8→CPU统计。数组顺序R0/R1/D0/D1/N0/N1，先并行四个耗时分支及N0，再利用N0释放的卡跑N1，避免D1整组等待快分支。afterok隔离GPU阶段；kill-on-invalid-dep避免无效依赖永久排队；部分提交失败不继续下游。
- 四suite×10task×10冻结初态40–49；六新P+原UADA500+旧diverse500，最多3200攻击回合，复用既有400clean资格/seed。验证旧同轮次结果身份后才可复用，缺结果则评估已训练P，不重训旧方法。
- 测试仅正常完整OFT+量化RGB补丁，**无η、无训练hook/干净特征替换**。task分片可恢复，不生成每episode小文件。训练后直接测试，PNG另供用户看。
- CASR仅clean-success，先task再suite等权。Spatial/Goal等权主终点，四suite及Spatial/Goal单独均报；task→state配对bootstrap10000，seed20260907。单训练seed仅筛选，不将episode数当训练重复。
- 本链到500统计停止。胜出的外观档N/R/D到1000/2000及独立seed需后续用户批准；D不胜R/N不继续堆轮次。

## 5. 时间与输出

500轮4090无η约1.5–1.8h/组，动作反制和严格匹配R约5–6.5h/组（通常4forward+3输入backward/更新）；六组≤5张4090分批。训练约5–6.5h，八张3090测试约1.5–2h，CPU约10–30min；整轮约7–9h、不含排队，真实短入口计时后修正估计。

输出：`training/<method>/outer_0500.png/.pt`、`training/<method>/training.parquet`、`evaluation/four_suite/`、`logs/`、`reports/`，均在本实验根。具体入口、输出导航见[README.md](README.md)，检查记录见[reports/PREFLIGHT.md](reports/PREFLIGHT.md)。文档中的命令不能授权助手实际提交。

## 6. 手动入口与实现边界

在`/home/caijiasheng25/robopatch_defense/roboticAttack-main`执行：`bash submit_exploring_oft_dino_action_counter500_v1.sh`。仅检查可加`--preflight`；该分支不加载模型、不查询队列、不提交作业。正式入口由用户执行后才提交107–111号五阶段作业。

新增训练器和objective、独立wrapper/config/common、正常OFT评估/配对统计、Slurm/helper以及三个CPU测试文件。Exploring原渲染、RGB优化器和OFT前向由冻结派生副本提供；新增文件扩展两阶段训练，不改旧训练器。完整CPU/mock检查与源文件哈希核对完成不代表真实GPU启动、外观或CASR已通过；这些检查嵌入正式运行，训练成功才允许下游评估。
