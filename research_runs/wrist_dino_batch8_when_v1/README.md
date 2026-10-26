# Batch8补充 + WHEN-DINO推进（2000轮）

本轮唯一方案见 [PLAN.md](PLAN.md)。四组从同一seed42随机RGB补丁、空Adam开始，不续训旧补丁；每组2000 outer×50 inner。实际batch=8，以微批1累积8条不同记录的平均梯度，再执行一次Adam更新。

| 训练目录 | 方法 |
| --- | --- |
| `training/original_uada_b8/` | Exploring→OFT适配的原始UADA，clean Object、动作第0维目标 |
| `training/uniform_replay_b8/` | 既有均匀回放与EOT方法补batch=8 |
| `training/dino_sampling_b8/` | 既有DINO反事实评分的状态/XYZ联合采样补batch=8 |
| `training/when_dino_b8/` | WHEN作者特征损失核心，针对腕部DINO的非补丁dense tokens |

WHEN-DINO不是完整WHEN复现；原始UADA是本地Exploring→OFT适配训练，不是作者发布补丁。三种既有B1方法的1000/2000补丁复用权重，只重新配对测试。

## 时间和资源

动作组B8估计45–60小时，DINO选样评分另计。WHEN特征组需实际GPU测速；前20轮自动写 `training/<method>/TIMING.json`。四组并行，总训练时间由最慢组和可用卡决定，不是四组时长相加。测试预计3–4小时；排队和旧任务等待另计。不能保证10月6日中午前整批完成。

训练为4个单卡4090作业，最多并行4卡（小于5卡上限），每卡微批1控制显存；各申请4CPU、22G主存、72小时。全部训练结束并通过CPU产物校验后，使用最多8张单卡3090测试，训练/测试阶段不叠加。若用户已有GPU/OFT任务，提交入口自动先等待它们结束；失效旧依赖或队列查询失败会停止提交并说明原因，不会取消用户任务。

## 手动提交

助手没有提交任何GPU任务。用户在登录节点执行：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_batch8_when2000_v1.sh
```

只预检可加 `--preflight`，该模式不查询队列、不提交Slurm。正常入口串联：CPU准备 → 4090训练 → CPU校验 → 3090正常OFT四suite测试 → CPU统计。重复提交正在运行的链会被拒绝；失败任务结束后，同一命令保留并续跑已完成outer，跳过已校验完整训练组，沿用现有评测分片。

## 到哪里看

- `training/<method>/outer_{0000,0250,0500,1000,1500,2000}.png`：补丁外观；同名PT用于部署。
- `training/<method>/training.parquet`：损失、梯度、实际RGB移动、8条选样记录、更新数与耗时。
- `training/<method>/RUNTIME_ALIGNMENT.json`：实际OFT与直接Wrist-DINO输入/输出一致性。
- `logs/`：训练/测试Slurm日志；`reports/SUBMISSION_RECEIPT.json`：提交的作业ID。
- `AUTO_VALIDATION.json`：四组完成后预算、快照、PNG量化、实际学习及恢复状态校验。
- `evaluation/four_suite/BATCH8_WHEN2000_XSUITE_SUMMARY.csv`：1000/2000各方法的CASR。
- `evaluation/four_suite/BATCH8_WHEN2000_XSUITE_CONTRASTS.csv`：配对方法差值。
- `evaluation/four_suite/BATCH8_WHEN2000_XSUITE_TRAINING_METRICS.csv`：RGB学习与DINO损失指标。
- `evaluation/four_suite/BATCH8_WHEN2000_XSUITE_COSTS.json`：训练/测试耗时与样本预算。
- `evaluation/four_suite/STATUS.json`：全部完成状态。

每预算7条件×4suite×10任务×10初态，两预算合计5600攻击回合；冻结clean-success有效行5348。Object源域单列；Spatial首要、Goal及SG共同考察，Long完整测试但次要。B8每组800000受扰样本计算，B1每组100000；同轮次不是同计算成本。单seed开发实验，不能把CPU/mock测试当作攻击增强结果。

## 来源和验证

`code/source_exploring.py`、`code/source_when.py`为与作者源文件SHA完全一致的只读快照；OFT适配与新训练在本目录副本上实现，不修改作者源码、旧实验或模型权重。20项新CPU行为测试及18项STATE-AXIS回归测试通过；全2000轮冻结数据选样与六张旧参照补丁预检通过。测试采用科研工作流和Python测试技能，强调真实梯度平均、恢复一致与输入/结果可追溯；尚未启动真实GPU训练。
