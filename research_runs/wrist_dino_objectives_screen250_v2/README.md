# 三种DINO目标筛选（250轮）

纯注意力、多层强弱响应反置、区域相似关系破坏；完整合同见[PLAN.md](PLAN.md)。新三组从零训练，复用上一批250轮UADA、Key、生成器与clean原始评测，不重跑旧实验。

用户手动提交（包括自动测试及逐suite汇总）：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_objectives_screen250_v2.sh
```

仅检查，不查询/提交Slurm：

```bash
bash submit_exploring_oft_dino_objectives_screen250_v2.sh --preflight
```

CPU复用→3张4090训练→CPU核验→最多4张4090测试→统计，GPU阶段串行，提交时等待本人其他GPU作业。新增240回合，合并560回合。运行预留2–3小时，排队另计；新目标吞吐待正式日志确认。

所有新输出集中在本目录：

- 补丁：`training/{dino_attention,dino_multilayer,dino_affinity}/outer_0250.png`
- 日志：`logs/`
- 结果：`results/RESULTS.md`、`OBJECTIVES250_SUMMARY.csv`
- 原始数据：`results/OBJECTIVES250_ROLLOUTS.parquet`、`OBJECTIVES250_TRACES.parquet`，保留新旧来源
- 提交回执和核验：`reports/`

恢复用同一命令：已完成训练/回合验证后跳过，未完成者恢复原子保存点；本批有活跃任务则拒绝重复提交。未修改模型、原始专家数据、原Exploring文件或旧结果。
