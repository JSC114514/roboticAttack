# 本轮入口：Spatial迁移与补丁清晰度

实验设计：[PLAN.md](PLAN.md)。代码已实现，CPU/mock预检记录见下；尚未提交新的GPU作业，真实GPU接口检查会在正式作业启动时执行。

| 想找什么 | 本目录内位置 |
|---|---|
| 最新方法与比较设计 | `PLAN.md` |
| CPU预检与代码审核 | `reports/PREFLIGHT.md`、`reports/PREFLIGHT.json` |
| 用户提交后的去噪权重校准 | `reports/APPEARANCE_CALIBRATION.json` |
| 新四组补丁 | `training/{clear,uniform_clear,dino_causal_clear,dino_repair_clear}/outer_1000.png` |
| 训练是否有实际更新 | 各组`training.parquet`与`status.json` |
| 最终四套件汇总 | `evaluation/four_suite/` |
| 所有阶段日志 | `logs/` |

A组最佳1000与原UADA1000参照仍在`../wrist_dino_transfer_v1/`，本轮复用已有结果，不复制或删除大文件。新训练仅恢复计入总预算的随机250轮前缀，不从1000/2000补丁起跑。

预计校准5–15分钟、四组并行训练3–5小时、八卡测试2–4小时、统计数分钟，总约5–9小时另加排队；修复内循环尚无真实计时。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_repair1000_v1.sh
```

命令内部自动衔接去噪校准、四组4090训练、产物核验、四套件3090测试和统计。单独检查但不提交：`bash submit_exploring_oft_dino_repair1000_v1.sh --preflight`。收到的job ID与依赖链保存在`reports/SUBMISSION_RECEIPT.json`。

最终主要表：`evaluation/four_suite/DINO_REPAIR1000_XSUITE_PER_SUITE.csv`、`DINO_REPAIR1000_XSUITE_CONTRASTS.csv`、`DINO_REPAIR1000_XSUITE_SPATIAL_GOAL.csv`。`STATUS.json`的COMPLETE只表示工程完成，不自动表示攻击、机制或外观成功。
