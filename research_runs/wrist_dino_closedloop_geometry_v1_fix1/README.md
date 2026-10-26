# 腕部 DINO：闭环数字几何误导小规模试验

**2026-10-17 fix1**：修复抽屉 `SiteObjectState` 继承抽象 `get_joint_state()` 导致的 Goal 接口检查失败。原目录、失败日志和三套件已通过检查均保留。新入口校验原文件哈希后复用 Object/Spatial/Long，只提交 Goal 的 GPU 检查，再执行四套件评测和汇总。修复说明见 [REPAIR.md](REPAIR.md)。

实现方案见 [PLAN.md](PLAN.md)。不训练模型或新补丁；复用冻结 UADA 补丁，同场重测 clean / UADA 参考。

提交（由用户执行）：

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_closedloop_geometry_v1_fix1/run.sh --submit
```

一个命令提交剩余 Goal GPU 接口/状态回退检查 → 四套件比较 → CPU 汇总。最多同时 **4 张 RTX 4090、16 CPU、128 GB 主机内存**。每套件十任务、每任务初态40，四条件共160个完整回合。主比较是相同数字变换/查询预算的 `loop_geometry` 与 `uada_geometry`；静态补丁的攻击权限不同，作为单独参考。

粗估获分配资源后 **1–3小时**，排队另计；GPU检查后可查看实测推算：

```bash
bash research_runs/wrist_dino_closedloop_geometry_v1_fix1/run.sh --status
bash research_runs/wrist_dino_closedloop_geometry_v1_fix1/run.sh --estimate
```

`--check` 只做 CPU/文件预检，不提交，不加载模型或启动仿真。完成的回合自动跳过；队列不可查询时拒绝提交，避免重复。

结果：`results/RESULTS.md`、`DECISION.json`、`INTERNAL_EVIDENCE.json`。

监控：`evaluation/<suite>/task_XX/<condition>/STATUS.json`；其 `attempt_*/LIVE_PLANS.jsonl` 包含候选选择、真实任务进度、回退和正式执行一致性。所有候选的 DINO/像素/动作/任务细节见 `CANDIDATES.parquet`，正式动作轨迹见 `TRACES.parquet`，首次真实输入/原生特征/状态见 `FIRST_PLAN_INPUTS.pt`。失败证据与GPU成本位于 `reports/FAILURES/` 和 `reports/WORKER_COSTS/`。
