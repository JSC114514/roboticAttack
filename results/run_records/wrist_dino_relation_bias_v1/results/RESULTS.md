# Spatial DINO 空间关系补丁结果

验证集锁定的方案为 `translation`，选择规则为 `offline_directional_consistency_tiebreak`。留出任务没有参与选择。

## 闭环结果

- 胜者验证集 task-equal CASR：0.0000 [0.0000, 0.0000]（干净成功起点 8 个）。
- 胜者留出集 task-equal CASR：0.1429 [0.0000, 0.4286]（干净成功起点 14 个）。
- `ROLLOUT_SUMMARY.csv` 同时给出初始随机补丁、UADA 和候选补丁的配对结果。

## 解释边界

训练损失只使用冻结 DINO block 22 的有符号反事实空间方向。闭环行只部署 50x50 PNG，并经过完整 DINO、SigLIP 和原生动作头。
渲染反事实、DINO-only 替换和分支恢复只用于机制诊断，不计作真实补丁成功。A/B 与 UADA 的训练池和更新预算不同，UADA 是部署参考。
离线 `dino_signed_q`、动作方向一致性和分支恢复结果见 `OFFLINE_SUMMARY.csv` 及逐行 parquet；表示变化、动作变化和任务失败应分别解读。

q 是单个几何方向的投影坐标，q 接近 1 不证明完整空间关系被复现。另报正交残差、到正向目标的归一化距离以及相对干净特征的变化；这些监测不参与补丁优化或胜者选择。

## 成本

各 worker 的进程 RSS HWM、CUDA allocated/reserved 峰值和耗时保存在 `reports/COSTS`；Slurm MaxRSS 由状态入口另行显示。
