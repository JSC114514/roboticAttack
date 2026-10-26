# DINO 剩余分量原幅度：4 回合收尾验证

只新增 c03/c04/c05/c09 各一个回合，三个 Long 补丁共享一个初态，Goal 占另一个初态。
保留剩余分量原始幅度，既不缩小到功能分量范数，也不放大。其余条件复用已校验记录。
逐决策重算基，记录梯度、实际特征改变量、动作和夹爪执行、状态、耗时及异常。

手动提交准备、四个 GPU worker 和自动 CPU 分析：

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_residual4_v1/run.sh --submit
```

查看进度：

```bash
bash research_runs/wrist_dino_residual4_v1/run.sh --status
```

最多 4 张 RTX4090，每卡 2 CPU/32GB；CPU 阶段 2 CPU/8GB。
四卡按 10–15 分钟预算，不含排队。计算依据保存在 reports/PREPARATION.json。
GPU 作业的 20 分钟时限是超时上限，不是预计耗时。

最终入口 results/RESULTS.md、DECISION.json、QUALITY.json、COMPARISONS.json。
四回合结束即停止，不追加样本、换层或自动训练。
两个初态均复现失败才进入真实 PNG 短验证准备；仅一个初态或零个初态有效，停止本路线。
质量故障单独报告，不算科学负结果。完整规则见 [PLAN.md](PLAN.md)。
已有 77 回合源码与结果保持原样；全部复用接口和证据均有签名/文件哈希约束。
