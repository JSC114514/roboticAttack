# 扩大原生评测

历史实验只读；不自动提交或取消Slurm。

Goal十任务初态2–5，40个起点。固定原8个条件，最多320回合；clean失败显式跳过其余条件。

评测为实际PNG的完整双分支前向、原生8步动作块、300步上限；没有特权动作恢复。按任务→起点重采样，不能把槽位或帧当独立样本。

预计两张4090约45–90分钟加CPU汇总10–20分钟，不含排队。

评测每worker1×4090/4CPU/12G；训练16G；CPU2CPU/4G。30秒监控，记录RSS、CUDA峰值、耗时与失败成本；失败立即退出，完整回合与500更新检查点续跑。

手动入口：`bash research_runs/wrist_spatial_relation_expand_v1/run.sh --submit`；可用`--preflight`、`--dry-run`、`--resume`、`--status --slurm`、`--watch --slurm`。新视频自动汇集到`research_videos/wrist_spatial_relation_expand_v1/INDEX.html`。
