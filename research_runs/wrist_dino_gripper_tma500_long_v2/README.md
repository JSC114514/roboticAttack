# Long夹爪TMA500：当前修复版v2

本地授权LIBERO仿真。唯一主方案第43.2–43.3节及本目录[PLAN.md](PLAN.md)；v1失败版只读保留，不重新提交v1命令。原Exploring源码不改，新修复只在独立副本。

## 修了什么

- clean保留官方processor输入；RGB/EOT前向经过真正的uint8量化、官方crop及processor，反向用明确的torch straight-through近似。
- 任务区/背景区都从各自可用token按固定排列取min数量；保留等token比较，不再因整块Long器件过大删整任务。
- RECOVERY_SOURCE.json锁定v1的228条原始观测；所有动作/close/DINO/SigLIP标签重新提取，缺失72条按原状态前缀继续采集。六个已完成任务不重演轨迹。
- 每task失败JSON写超差数值、槽位、DoF和阶段，并保留单个失败候选PT。按完整轨迹恢复。

A普通夹爪TMA，B任务区DINO-CF，C背景DINO-CF。500×50更新、seed42随机RGB、B1、LR0.002/20 warmup/500 cosine不变，没有TV/blur/旧补丁热启动。B/C仍max(full,region_cf)，测试只真实腕部PNG，Front干净。历史Object-UADA500是不同来源参考，不是公平Long-UADA基线。

## 手动提交

先在项目根运行可选只读预检：

```bash
bash submit_exploring_oft_gripper_tma500_long_v2.sh --preflight
```

正式提交（仅用户执行）：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_gripper_tma500_long_v2.sh
```

提交后自动完成：CPU准备→A800≤3恢复/标签→CPU组池→A800≤3三组训练→CPU核验→4090≤5四suite测试→CPU统计。阶段不重叠，总GPU峰值≤5，无3090依赖；等待用户已有GPU/OFT作业，拒绝重复/无效链，不自动取消其他作业。原0.02接口阈值不放宽。真实GPU检查失败则停链，不把CPU测试通过说成攻击成功。

粗估恢复/补采10–30分钟、训练3–6小时、测试3–5小时，总约6.5–12小时加排队。v1源采集实测约9分钟；本批训练/测试还未实测，不能保证外观或效果。

## 去哪里看

- `data/labels/`：新标签分片、逐轨迹resume、失败候选。
- `data/LONG_TRAIN_BANK.pt`、`data/TRAINING_POOL.pt`：300条Long来源库与共同资格池。
- `training/{plain_gripper_tma,task_region_dino_tma,background_region_dino_tma}/outer_0500.png`：三张补丁；同目录status.json、TRAIN_HISTORY.parquet、GPU_INTERFACE.json。
- `evaluation/GRIPPER_TMA500_SUMMARY.csv`、`GRIPPER_TMA500_CONTRASTS.csv`：四suite结果及配对差。
- `reports/SUBMISSION_RECEIPT.json`：作业链；`logs/`：各阶段stdout/stderr。

Object/Spatial/Goal是Long来源迁移，Long来源内也必测；每suite使用其自己的OFT微调模型。40–49是已查看开发网格，接触/掉落是仿真代理。500闭环后停止，不自动扩轮或执行其他研究路线。
