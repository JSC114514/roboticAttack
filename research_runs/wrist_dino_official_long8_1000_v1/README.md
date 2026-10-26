# 官方Long八路1000轮唯一入口

[完整方案](PLAN.md)。A1/A2/A3=UADA基线/DINO采样/采样＋频域；B1/B2/B3=TMA-XYZ相同三组；C1/C2=TMA＋几何动作方向/TMA＋动作相关DINO注意力。

[源码来源与改动](SOURCE_PROVENANCE.md)；[实现核验记录](reports/IMPLEMENTATION_REVIEW.md)。只保存本轮材料，不合并旧outputs。

从官方Long演示随机起点训练，不用旧回放库/教师/补丁；训练及测试最多各8张4090，分阶段串行；训练完成自动测试。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_official_long8_1000_v1.sh
```

只检查不提交：上述bash命令追加`--preflight`。

- 补丁：`training/<method>/outer_1000.png`；0/250/500/750预览在同目录。
- 训练记录/恢复：`training/<method>/{training.parquet,status.json,resume.pt}`。
- 官方数据来源：`data/{DATASET_INDEX.json,DATA_STATUS.json,CANDIDATE_BANK.pt}`。
- 四套件结果：`results/{LONG8_SUMMARY.csv,LONG8_CONTRASTS.csv,LONG8_ROLLOUTS.parquet,LONG8_TRACES.parquet,RESULTS.md,STATUS.json}`。
- 全部日志：`logs/`；提交回执和检查：`reports/`。
- 源副本：`source/`；本轮修改只在`code/`，原Exploring源码不改。

预计整链8–13小时另加排队，具体按正式作业吞吐更新。不新增独立smoke，不自动提交或使用重置卡。
