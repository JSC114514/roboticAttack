# DINO特征250轮低成本筛选

> 2026-10-13修复：Key/dense/UADA三组均已完成250轮；生成器首次前向出现4090 cuDNN初始化失败。当前请使用[单生成器修复入口](repair_cudnn4090_v2/README.md)中的`bash submit_exploring_oft_dino_feature_screen250_cudnnfix_v2.sh`，不要重提下面的原四组训练命令。它复用三组与数据缓存，补生成器后自动测试。原代码/结果保持。

本轮四组：同预算UADA、DINO稠密Key、去中心化dense、AdvEncoder-cos生成器。每组从零250轮×50内循环；使用官方Long训练演示、普通随机取样、单腕50×50图案和Exploring原几何变换。方法合同见[PLAN.md](PLAN.md)，源码改动见[SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。

## 用户手动提交

预计CPU准备3–10分钟，最慢训练1–1.5小时，测试1–2小时，合计预留2–4小时；排队/等待原有本人GPU任务另计。新特征/生成器吞吐尚未实测。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_feature_screen250_v1.sh
```

自动执行CPU数据准备→4卡训练→CPU资产检查→4卡测试→逐suite统计。两GPU阶段串行；不会取消别的作业。只是检查文件、暂不提交可加`--preflight`，不启动模型/仿真、不查询队列。

## 输出位置

- 本目录`training/<method>/outer_0000.png`、`outer_0125.png`、`outer_0250.png`：初始、途中和最终图案。
- `training/<method>/training.parquet`、`status.json`：实际loss、梯度、RGB变化、LR、更新次数及ETA。
- `training/advencoder_cos/resume.pt`：生成器、固定噪声、Adam、RNG；不是只保存PNG。
- `logs/`：独立Slurm日志；`reports/SUBMISSION_RECEIPT.json`：真实作业id与依赖。
- `results/RESULTS.md`、`SCREEN250_SUMMARY.csv`：四套件分别的结果。
- `SCREEN250_CONTRASTS.csv`：三候选相对本轮UADA的配对差和正向/负向任务数。
- `evaluation/<suite>/task_XX_resume.pt`：回合级恢复；完整分片才汇总。

测试clean+四张250轮PNG，每suite10任务×init40/41，共400回合，正常OFT8步闭环；不加载生成器、不使用训练hook、不修改Front或proprio。初态是已有开发状态，本轮是方向筛选，不冒充新盲测/实机验证，也不把旧2000轮UADA结果冒充同预算基线。

## 状态

代码和自动链已实现。CPU检查不会加载真实OFT权重，真实DINO接口和CUDA/EGL检查嵌在正式GPU作业内。助手不自动sbatch、不使用重置卡；模型/数据/旧结果/现有跨模型任务保持。
