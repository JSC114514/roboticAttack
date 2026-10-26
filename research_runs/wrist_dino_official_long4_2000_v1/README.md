# 官方Long四路从零2000轮：唯一提交入口

[冻结方案](PLAN.md)；[源码来源与改动](SOURCE_PROVENANCE.md)；[实现核验](reports/IMPLEMENTATION_REVIEW.md)。本目录与旧八路1000结果分开，不复用旧训练补丁或优化器。

四组：UADA基线、UADA＋DINO采样＋全秩频率优化、TMA-XYZ基线、TMA-XYZ＋DINO采样。统一B1、官方Long专家数据、2000×50更新、原作者随机位置＋旋转/剪切/20%恒等变换。无亮度/对比度、固定位置半程、教师、WHEN或新增辅助loss。

提交前预计：四卡并行训练7–9小时；八卡测试2–4小时；CPU组织10–30分钟；整链约9.5–13.5小时，另加排队。训练和测试阶段不重叠，最多分别4/8张4090。每补丁一GPU，不使用双卡DDP。

用户手动提交（助手不自动提交）：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_official_long4_2000_v1.sh
```

仅CPU检查：同一命令追加`--preflight`，不查队列、不加载VLA、不申请GPU。默认提交入口会等待本人其它GPU作业，防止合计超额；不自动取消旧作业。

训练四组完成后自动测试1000和2000的八张PNG及fresh clean，不等待人工外观批准；O/S/G/Long共3600回合。每suite分别报告成功率、失败率、clean-success CASR及基线差值，不输出跨suite平均。

- 补丁与预览：`training/<method>/outer_{0000,0250,0500,0750,1000,1500,2000}.png`，浮点RGB在对应`.pt`。
- 训练状态与恢复：`training/<method>/{status.json,training.parquet,resume.pt}`。首次全新0开始；中断只恢复本批同方法完整状态。
- 正式结果：`results/{RESULTS.md,LONG4_SUMMARY.csv,LONG4_CONTRASTS.csv,LONG4_PER_TASK.csv,LONG4_ROLLOUTS.parquet,LONG4_TRACES.parquet,STATUS.json}`。
- 原始评测分片及实际动作块：`evaluation/<suite>/task_*_{rollouts,traces}.parquet`。
- 官方训练数据索引/候选缓存：`data/`；来源指纹、核验与提交回执：`reports/`；所有作业日志：`logs/`。
- 完整Exploring只读源快照：`source/`；本次可运行副本：`code/`；批次设置：`config.json`。

原论文的几何变换保持，但B1与作者示例B8、连续OFT损失与原离散token损失、既有OFT测试协议与原论文测试协议仍有明确差别，详见方案，不称未经适配的官方复现。本入口不启动跨模型或实机。
