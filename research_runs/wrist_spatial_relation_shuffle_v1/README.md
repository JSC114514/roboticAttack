# 关系对应对照

历史实验只读；不自动提交或取消Slurm。

复用四套件混合训练池，217个原有效样本。两种匹配方法×seed42/43，共4张新PNG、每张4000次更新。逐槽控制命令扰动长度相同、共同缩放至有效范围；缩放量完整报告。对照在同套件跨任务、跨轨迹置乱方向。不是原远边界方法换名。

评测为实际PNG的完整双分支前向、原生8步动作块、300步上限；没有特权动作恢复。按任务→起点重采样，不能把槽位或帧当独立样本。

训练两张4090、每卡串行一对候选，约75–90分钟；新PNG最多160回合约25–60分钟，CPU汇总10–20分钟。不含排队。等待扩大评测完整完成，不同时占用4卡。

评测每worker1×4090/4CPU/12G；训练16G；CPU2CPU/4G。30秒监控，记录RSS、CUDA峰值、耗时与失败成本；失败立即退出，完整回合与500更新检查点续跑。

手动入口：`bash research_runs/wrist_spatial_relation_shuffle_v1/run.sh --submit`；可用`--preflight`、`--dry-run`、`--resume`、`--status --slurm`、`--watch --slurm`。新视频自动汇集到`research_videos/wrist_spatial_relation_shuffle_v1/INDEX.html`。
