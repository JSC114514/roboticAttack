# 关系对照续跑修复1

原实验只读。复用matched_relation两个种子的4000次完成结果；仅将shuffled_relation两个种子的完整2000次检查点迁移到本目录后续跑至4000次。无目标、损失、训练池、初始化、样本顺序、优化器、调度及RNG改变。迁移只重绑定执行版本，原文件哈希记录在MIGRATION.json。

每GPU进程只加载一个完整GPU驻留模型、训练一个补丁；CPU/disk/meta offload检查失败即退出。两卡并行，每卡4CPU/16G，预计续训20–35分钟、不含排队，训练时限90分钟。评测每卡4CPU/12G，最多两卡；CPU汇总2CPU/4G。恢复后固定最终4000次PNG，最多156个有效新回合（40个起点中39个clean成功），不重跑扩大评测。

每30秒心跳，报告本候选实际更新数及本进程吞吐、RSS/CUDA峰值和失败成本，500次更新保存完整状态。按回合和检查点续跑，不自动提交或取消Slurm。

视频直接从前视、腕部相机帧分别保存，无拼接/叠字/长合集。MP4及IDE_WebM集中交付到research_videos/wrist_spatial_relation_shuffle_v1_fix1_singleview/。可复用的clean、初始补丁、UADA视频来自扩大评测单视角目录，不重新生成科学轨迹。

手动入口支持--preflight、--dry-run、--submit、--resume、--status --slurm、--watch --slurm、--report-local。训练未完成时汇总返回明确WAITING/PARTIAL及失败成本，不读取不存在的SELECTED.json。
