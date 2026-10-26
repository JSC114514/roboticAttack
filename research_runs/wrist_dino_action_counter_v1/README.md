# 本轮入口：Wrist-DINO 动作反制训练（500轮）

唯一设计见 [PLAN.md](PLAN.md)。这是独立新实验；旧结果不覆盖，原 Exploring `UADA.py` 保持只读。助手只实现和做 CPU 检查，不自动提交 GPU 作业。

## 时间与手动提交

预计4090训练约5–6.5小时，最多8张3090测试约1.5–2小时，CPU准备/核验/统计约10–30分钟；整链约7–9小时，另加排队。这是估算，正式作业启动后按实际吞吐修正；12/8小时是调度上限，不是预计运行时间。

用户在登录节点执行：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_action_counter500_v1.sh
```

只读检查、不查询队列或提交作业：

```bash
bash submit_exploring_oft_dino_action_counter500_v1.sh --preflight
```

正常入口依次提交：CPU准备 → 六组4090训练（最多5张）→ CPU检查 → 四套件3090测试（最多8张）→ CPU统计。训练和测试通过 `afterok` 分离；失败依赖自动取消，提交中途失败不再提交下游，已成功提交的作业ID保存在 `reports/SUBMISSION_RECEIPT.json`。不自动扩到1000/2000轮。重复执行会检查现有队列、已完成组及断点，不覆盖不同身份的实验。

## 本轮做什么

六组都从同一个seed42随机RGB补丁、空Adam、outer=0开始；每组500×50更新，不以旧补丁或旧优化器初始化。已有300条Object clean、258条Object replay及冻结源采样顺序只作共同训练数据。单腕部50×50×3高维RGB，Front保持干净。

| 输出method | 输入增强 | TV预算 |
|---|---|---:|
| none_original | 无反制 | 0.4 |
| none_clear | 无反制 | 0.3 |
| random_original | 幅度匹配的随机反制 | 0.4 |
| random_clear | 幅度匹配的随机反制 | 0.3 |
| dino_original | DINO指导的动作反制 | 0.4 |
| dino_clear | DINO指导的动作反制 | 0.3 |

内层固定补丁，沿Wrist-DINO梯度在补丁外生成两步、小幅RGB扰动，使原指定错误动作目标更难达到；外层仍用正常完整OFT和原UADA动作MSE训练补丁抵抗该方向。每次内层真实重算SigLIP，仅暂时截断其腕部梯度；不替换clean特征。零扰动也参与候选选择。没有附加DINO损失或人为梯度倍率。

去掉上一轮持续TV/HF去噪目标，改用超预算才惩罚的TV hinge。0.3档不保证视觉清晰，图像质量和攻击效果都需运行后看结果。DINO组是否优于随机/无反制组，是本轮要回答的问题，不预先认定增强成立。

## 测试与文件位置

训练结束直接以最终量化RGB补丁测试；测试无内层扰动、无训练hook。六张新500轮补丁与已有原UADA500、旧diverse500一起在Object/Spatial/Goal/Long测试，共3200攻击回合，复用400条冻结clean资格；不会重训旧对照。Spatial/Goal为主要开发终点，Long完整报告但列为次要。只做单训练seed筛选。

所有新产物集中在此目录：

- `training/<method>/outer_0000.png`、`outer_0250.png`、`outer_0500.png`：看初始、250轮、最终外观。
- `training/<method>/training.parquet`：损失、TV/HF读数、实际RGB更新、PNG变化、反制效果与随机幅度匹配。
- `training/<method>/resume.pt`、`RUN_MANIFEST.json`、`source_snapshot.zip`：完整优化器/调度/RNG断点及来源快照。
- `evaluation/four_suite/DINO_ACTION_COUNTER500_XSUITE_SUMMARY.csv`：各套件及主要终点结果。
- `evaluation/four_suite/DINO_ACTION_COUNTER500_XSUITE_CONTRASTS.csv`：DINO−随机、随机−无反制、外观代价等配对对比。
- `logs/`：按阶段/数组任务保存Slurm日志。
- `reports/PREFLIGHT.md`：实现检查与尚未实际运行的部分。

关键新增代码：`VLAAttacker/white_patch/UADA_OFT_DINO_ACTION_COUNTER.py`、`oft_dino_action_counter_objectives.py`、独立wrapper、`scripts/*action_counter*.py`、配置 `object_dino_action_counter500_v1.json` 和107–111号Slurm脚本。Exploring派生副本提供原渲染器、RGB优化器与OFT前向，本轮文件扩展两阶段更新；旧训练器不改。
