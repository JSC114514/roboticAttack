# 正确关系／置乱关系：独立评测入口

仅复用四张最终4,000次PNG，不训练、不迁移检查点、不改变既有217条混合训练观测及目标。正确关系读取原始shuffle目录；置乱关系读取fix1目录。两张正确关系PNG与旧扩大评测不同，全部重新评测。

固定Goal模型，Goal十任务、初态2—5全部40起点，包括clean失败的起点。四张PNG共160回合；每种子内正确／置乱配对。机械臂成功率按各组实际完成回合计算，CASR仅按clean成功且已配对完成的回合计算；缺失不计成功。按任务→起点重采样。它是Goal开发评测，不是四套件平均或完全独立确认。

每个条件新建环境，重复同一构造随机状态；严格核验物理状态、本体、相机RGB、时钟与历史clean起点，首个clean动作块最大误差1e-5。模型冻结且单GPU常驻；接口失败立即退出，保留成本，无自动重试。每30秒记录进度、RSS、CUDA峰值和预计剩余时间。按完整回合收据续跑。

最多两张4090，每worker4CPU、12G；无训练GPU。CPU准备／汇总2CPU、4G。45—90分钟GPU预算（不含排队），另留CPU视频转换时间。按任务分片，每worker80回合；模型每进程只加载一次。依赖等待不占GPU。

手动入口：
```bash
bash research_runs/wrist_spatial_relation_shuffle_eval_v1/run.sh --submit
bash research_runs/wrist_spatial_relation_shuffle_eval_v1/run.sh --status --slurm
bash research_runs/wrist_spatial_relation_shuffle_eval_v1/run.sh --watch --slurm
bash research_runs/wrist_spatial_relation_shuffle_eval_v1/run.sh --resume
```

--check检查实际批处理入口；--prepare为纯CPU资产核验；--dry-run只打印提交图；--report输出当前完整配对结果并导出已完成视频，不提交任务。不会自动替用户提交或取消Slurm。

视频：research_videos/wrist_spatial_relation_shuffle_eval_v1_singleview/MP4或IDE_WebM/方法/front或wrist/。单回合独立相机，无拼接、文字栏或合集，保持原生控制时间。WebM/MP4完整解码不代表已验证IDE播放。

结果在results/RESULTS.md、CASR.csv、PAIRED_CONTRASTS.csv、TASK_RESULTS.csv、EPISODES.csv；所有组完整报告。只依据本轮正确关系／置乱关系结果讨论下一步，本入口不追加训练或扩大套件。
