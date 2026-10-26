# DINO 功能分量：77 回合验证

本轮只验证训练目标，覆盖四套件，固定 10 个历史失败案例、3 张已有 B1 PNG。
共 60 个整轨迹干预回合和 17 个同期原生对照，不启动训练。

手动提交（准备 → 验证 → 分析，依赖自动串联）：

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_functional77_v1/run.sh --submit
```

最多同时占 4 张 RTX4090；每个 GPU 作业申请 2 CPU、32GB 内存。
CPU 准备和分析各申请 2 CPU、8GB。预计四卡运行 20–40 分钟，不含排队。
10 个案例分到 10 个 worker，避免把 Long 的 3 张 PNG 挤在同一个 worker。
如果已有自己的 GPU 批次在运行或排队，提交器会阻止重叠申请。

查看本地进度和调度器队列：

```bash
bash research_runs/wrist_dino_functional77_v1/run.sh --status
squeue -u "$USER"
```

只重新做 CPU 预检（无模型加载、无 GPU 申请、无任务提交）：

```bash
bash research_runs/wrist_dino_functional77_v1/run.sh --preflight
```

每个决策块都会重算功能子空间，随机对照等秩，随机和正交对照都匹配
BF16 实际特征改变量。输出包含梯度、线性预测与实际动作变化、执行动作、
夹爪阈值翻转、像素变化、状态轨迹、资源和耗时。完整向量只在 0/4/12 块保存。

结果入口是 `results/RESULTS.md`；逐案例结果是 `results/OUTCOMES.json`；
决策是 `results/DECISION.json`；质量核查和机制指标分别是 `QUALITY.json`
与 `MONITORING.json`。`CASES.json` 指向已核实的历史收据与日志。
尚未运行时这些结果文件不会生成。

通过门槛：恢复和注入两个方向上，功能分量分别相对随机、正交对照净多改变
至少 3/10 个回合，且每项优势覆盖至少两个套件；同时通过机制诊断质量检查。
同期对照改变的案例不删出十例分母，不计有效改善。通过后再准备与 UADA
同预算的两组训练；不通过则结束这个候选。不会自动训练或自动寻找其他方法。
这轮干预的表现用于选择训练目标，不作为真实 PNG 已超过 UADA 的结果。

断点续跑仍用同一条手动提交命令；只安排缺失 worker，已完成回合校验后跳过。
未完成的 attempt 保留，重新从固定初始状态执行，不用截图恢复模拟器。
若提交响应超时，查看 `reports/SUBMISSION.json` 与调度器后再处理，避免重复提交。

完整冻结方案见 [PLAN.md](PLAN.md)。
