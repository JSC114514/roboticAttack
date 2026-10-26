# 本轮入口：受扰采样 + 清晰高维补丁

唯一科学方案：[PLAN.md](PLAN.md)。源码由旧 Exploring→OFT transfer 副本独立扩展，旧文件/结果不修改。2026-10-03：原链A完成2000轮，B/D完成1998轮、C完成1999轮后被末尾数值检查中断；自动测试未运行，暂无新CASR。用户已授权工程修复，恢复边界和新入口见[RECOVERY.md](RECOVERY.md)。不要用下方原始从头入口替代专用恢复命令。原CPU审查记录见[reports/PREFLIGHT.md](reports/PREFLIGHT.md)。

四组均从同一seed42修正后的随机RGB和空Adam训练至2000轮，不读取旧补丁初始化：

|目录名|内容|
|---|---|
|dynamic_soft|旧动态DINO采样 + 软TV|
|dynamic_hard|旧动态DINO采样 + 硬低噪声约束|
|stable_hard|双视图稳定DINO采样 + 同硬约束|
|shuffled_hard|同稳定评分，组内打乱权重 + 同硬约束|

保留受扰片段、多样位置/光照/仿射训练；DINO用于给源Object状态打分选样，不另加特征损失，不跑上一轮反制。部署只贴腕部RGB，Front干净，模型正常推理。

## 当前恢复提交命令

工程修复和实际CPU预检已通过；A不重训，B/C/D只补1–2轮后自动测试1000/2000检查点。补训和CPU验证约5–15分钟，3090四suite测试约2–3小时，排队另算。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_sampling_clear_recovery_v1.sh
```

最多3张4090补训，后续最多8张3090测试；原科学配置、数据和训练预算不变。修复与来源审查见[RECOVERY.md](RECOVERY.md)。

## 原始从头入口（归档，当前不要执行）

预计训练7–9小时、评估2–3小时、CPU准备/验证/统计10–30分钟；整链约9–12小时另加排队。训练最多4张4090，测试最多8张3090，两个阶段不重叠。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_sampling_clear2000_v1.sh --preflight
```

仅检查文件、签名和配置，不加载模型、使用GPU、查询队列或提交。正式手动提交命令为：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_sampling_clear2000_v1.sh
```

用户执行后串联CPU准备→四组4090训练→CPU验证→3090四suite测试→CPU统计。失败阶段阻止下游；不会续训或自动开展跨模型/实机。实际GPU启动校验仍须正式作业通过。

## 文件在哪里

- 补丁：`training/<目录名>/outer_1000.png` 和 `outer_2000.png`；同时保存0/250/500/1500轮及PT。原始随机RGB与约束修正后的共同起点分别保存，不冒称训练过的图。
- 训练轨迹：各组`training.parquet`、`resume.pt`、`status.json`和分轮评分文件。查看RGB轮间差异、有效更新率和约束修正比例；首轮warmup零学习率明确记录。
- 四suite结果：`evaluation/four_suite/`，1000/2000分别报告，包含原UADA和旧diverse同轮次参照；旧补丁只评估、不重训。
- 日志：`logs/`；提交收据、身份清单和预检：`reports/`。

硬约束三组检查实际量化PNG的TV≤0.3、高斯局部高频最大偏差≤16/255；约束并不保证人眼完全无噪声，更不保证攻击成功率。评估完整包含Long但以Spatial/Goal为主。全部现有评估初态属于开发复验，不称独立最终测试。
