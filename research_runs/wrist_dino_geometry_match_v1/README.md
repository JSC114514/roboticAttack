# DINO完整几何目标匹配

旧实验只读：wrist_dino_relation_bias_v1。新方法直接匹配translation_plus完整block22空间特征。
损失=原投影损失+正交残差；不加入动作目标，不使用EOT。两种损失均按原token变化能量归一化。

手动入口（从项目根目录运行）：

```bash
bash research_runs/wrist_dino_geometry_match_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_match_v1/run.sh --submit-pilot
bash research_runs/wrist_dino_geometry_match_v1/run.sh --submit-night
bash research_runs/wrist_dino_geometry_match_v1/run.sh --status --slurm
bash research_runs/wrist_dino_geometry_match_v1/run.sh --watch --slurm --interval 30
```

submit-night可在pilot仍运行时手动调用：夜间训练依赖pilot报告正常结束，不会提前占卡。
任何脚本都不会替用户自动调用提交入口，不会取消任务。watch为只读，可用Ctrl-C退出。
提交入口去重；终止后按整轮/单图/评测任务边界续跑。源代码变化会拒绝复用本目录旧版本结果。
阶段失效或缺文件时写FAILED_OR_INCOMPLETE报告，绝不将缺失结果记成零成功率。

短验证：48拟合图像（24任务、初态40/41、frame_rank1），48验证图像（初态42/43）。
每图100次访问，Adam固定lr0.002。单图两种损失；两张共享补丁；理想DINO/原生RGB参考。
共享补丁在完整原生前向中核验；单图不作跨轨迹泛化声明。模型固定Spatial。

夜间：full_s42、full_s43、projection_s43各48000更新；projection_s42复用旧完成PNG。
共享960张训练池，Adam lr0.002，warmup20，cosine；主结果预定full_s42，不选测试集最优种子。
验证任务初态48、留出任务新初态0/1；所有clean重新执行。任务历史已查看，不能声称全新任务确认。
validation与heldout全条件相同：clean、initial_s42/43、projection_s42/43、full_s42/43、UADA。
四套件原生模型闭环；固定Spatial/原生suite离线诊断分开，动作命令不当物理位移。

GPU仅4090：训练4CPU/16G，纯评测4CPU/12G；CPU2CPU/4G。最多4GPU并发。
短验证训练数组3卡+oracle1卡，然后独立GPU原生核验；夜间3卡训练后4卡套件评测。
GPU在依赖等待时不分配。每worker30秒心跳、RSS/CUDA峰值、失败成本；任务异常时查成本与Slurm。

报告：results/pilot/RESULTS.md；results/night/RESULTS.md。
源缓存只读分片读取；预检固定源签名、完整分片哈希、观测标识、初态文件与本地checkpoint代码。
源缓存并非所有文件同一签名：早期已审计分片沿用e438原始签名，其他分片沿用30ca签名。
本实验通过源实验的正式审计逻辑核验它们，不重写或复制缓存。
