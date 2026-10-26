# OFT-DINO-REGISTER-DELAYED2000-V2 — 2026-10-04

这是用户确认的 OFT 最后一轮方法筛选与实验闭环。目标是在保留动作攻击能力的基础上检验温和 register 引导的增益，形成组会可交付的方法与同预算结果。本轮结束后给出采用/不采用及 OFT 是否达标的结论，不默认追加诊断、权重搜索、跨模型或实机任务。

## 1. 已完成的依据

REGISTER500-V1 已完成三张补丁训练、2000 闭环回合及配对统计。原始输出保留在 ../wrist_dino_register_v1。

| 500轮方法 | Object CASR | Spatial CASR | Goal CASR | Long CASR |
|---|---:|---:|---:|---:|
| 原始 UADA | 11.0% | 6.33% | 30.78% | 66.03% |
| uniform_axis0 | 95.0% | 16.33% | 53.22% | 63.75% |
| register_g10 | 51.0% | 19.56% | 71.0% | 87.0% |
| register_g30 | 60.0% | 19.33% | 32.89% | 66.53% |
| spatial_g30 | 59.22% | 9.22% | 4.11% | 50.89% |

10% register 相对 uniform 的 Goal 差值 +17.78pp，配对95%区间 +1.11～+34.45pp；Spatial +3.22pp 的区间跨零。四 suite 平均基本持平。更强的 register 注意力没有带来更强的闭环攻击，不能把注意力增大等同于机制利用成功。当前没有动作/DINO梯度夹角记录，不能断言退步由梯度冲突造成。

## 2. 固定方法

所有新方法从同一 seed42 随机 3×50×50 RGB 补丁开始，7500 个可训练参数；同一 Object clean300、Object replay258、样本次序、EOT、动作第0维 opposite-boundary MSE、AdamW、lr0.002、20轮warmup、2000轮cosine调度。每 outer 50次更新，共2000×50=100000次。没有历史补丁初始化，不增加 TV、低维参数化或纹理生成器。

前250 outer 仅优化动作目标，不安装新增 QKV 捕获、不计算辅助梯度；从251 outer 开始启用 DINO 引导。这个节点沿用当前 clean→replay 的 bootstrap 分界。三组均使用相同分界和10%梯度比例：

| 方法 | 251–2000轮辅助目标 | 梯度处理 |
|---|---|---|
| register_late_g10 | DINO block 8、16 的四个 register query 读取实际腕部补丁 token footprint | 原动作梯度 + 范数匹配10%辅助梯度 |
| spatial_late_g10 | 普通非补丁 dense query 读取同一 footprint | 同样10%，作为同权重、同时间控制 |
| register_late_projected_g10 | 与 register_late_g10 完全相同 | 仅剔除辅助梯度与当前动作梯度相反的分量 |

令动作梯度为 gm，原始辅助梯度为 ga。普通组使用 gm + s·ga，s=0.1·||gm||/max(||ga||,eps)。投影组仅在 ga·gm<0 时令 ga'=ga-(ga·gm)/||gm||²·gm，再使用 gm+s·ga'。s仍由原始ga计算，投影后不重新放大残余梯度，因此辅助贡献上限为动作梯度范数的10%。零范数以有限值方式处理并记录；禁止用投影替换动作主梯度。此处理借鉴已有 anchored-PCGrad 实现，只保护当前原始梯度方向，不保证 Adam 更新或闭环成功率单调提升。

补丁唯一部署权限是腕部 RGB；Front、语言、本体状态和模型权重不变。训练捕获 native QKV 只作读出，测试用正常 OFT 前向，无特征替换或辅助 hook。token footprint 沿用真实贴图/EOT后的覆盖映射。

## 3. 配对评估

保存0、250、500、1000、1500、2000轮的 .pt/.png、完整Adam/RNG恢复状态及每outer训练记录。1000、2000轮各评估三种新方法和已有同轮次 original_uada、uniform_axis0；旧补丁仅复用权重，正常重测，不重训。

四 suite：Object/Spatial/Goal/Long，各10 task×10初态（固定40–49）。5条件×2预算×400=4000回合。每初态、预算内随机条件顺序，各条件精确reset并使用相同冻结seed。clean-success 分母依次99、90、98、95，每预算资格行1910；两预算共3820资格行。任务→初态配对bootstrap为10000 draws、seed20260907；每个预算单独分析，不把检查点或同一补丁的多回合当独立训练种子。

主报告2000轮：Spatial、Goal、Object及三suite等权macro；1000轮报告同预算学习曲线，不以更有利的检查点替换2000主结果。Long完整测试但作辅助结果。重点比较新register与uniform、同权重spatial控制，以及投影组与普通register。优于原始UADA不能单独归因于DINO，因为还包含已有replay/EOT差异。只报告一训练seed的初步结论；本轮目标suite已用于方案开发，不能当成未经接触的最终确认集。

## 4. 训练变化与工程检查

训练内直接记录主/辅助loss、注意力mass、原始及投影后梯度夹角、冲突比例、实际辅助梯度比、像素更新次数和幅度；不另开诊断作业。所有数值有限，初始真实动作梯度及第一次辅助梯度须非零。正常学习率下完全不动报错；2000调度尾部最后两outer的float32舍入仅在既有只读Adam增量证据完整时允许，并明确记录。

CPU预检覆盖native timm QKV注意力计算、梯度符号和比例、投影上限/零范数、251轮开启时机、配对双预算分片恢复及统计覆盖。真实OFT模型验证随GPU作业开头执行。原 Exploring UADA.py 保持444与原SHA256。

## 5. 作业与时间

链路为CPU准备→3张4090并行训练→CPU验证→最多8张3090配对测试→CPU统计/收尾报告；阶段使用afterok依赖，训练与测试顺序执行。用户手动提交，助手不自动sbatch，不使用重置卡。失败可用同一命令重提，保留已完成训练臂和已提交的完整评估块，恢复Adam、调度和RNG。

按V1实测500轮98–105分钟，训练2000轮预计6.5–8小时；双预算4000回合预计2–3小时；整链约9–12小时，排队另算。目标是10月5日上午至下午取得结果，5日晚上冻结汇报结论，6日中午组会。

## 6. 文件与执行

所有本轮输出集中于 research_runs/wrist_dino_register_v2：training/<method>/、evaluation/four_suite/、logs/、reports/。最终统计同时生成 OFT_CLOSEOUT.md，列出两预算CASR、主要差值、梯度/训练变化和耗时，供选择方法与收尾。

配置：configs/exploring_oft_dino/object_dino_register_delayed2000_v2.json。
代码为新的 UADA_OFT_DINO_REGISTER_DELAYED.py 及独立 wrapper/common/evaluation/stats；由已验证 Exploring-OFT register 管线复制修改，旧训练文件和冻结结果不改。

手动提交：

~~~bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_register_delayed2000_v2.sh
~~~

只预检：同一命令追加 --preflight。当前状态：IMPLEMENTED_AND_CPU_CHECKS_PASS；未提交新作业。

实现检查：15个CPU测试用例已分别通过，包括三组真实训练循环的250→251启用、Adam/RNG完整恢复与不中断运行RGB一致性；模拟双预算四suite 4000条评估无重复、3820条资格数据及配对统计；提交依赖/并发上限与Markdown收尾报告。native timm注意力公式、腕部QKV捕获及反向梯度预检通过，原始UADA.py仍为444及原SHA256。模拟测试不加载完整OFT、不调用真实Slurm，数据不作实验结果。完整OFT/CUDA验证由用户提交后的GPU作业执行。
