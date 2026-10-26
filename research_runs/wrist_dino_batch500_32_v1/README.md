# 32张500轮补丁：8方法 × batch size 1/2/4/8

2026-10-14。完整合同见[PLAN.md](PLAN.md)，参数见[config.json](config.json)。1/12本轮从零UADA/TMA基线，其余ID2/3/4/6/7/9保持既有DINO目标；不新增方法。9-B1为零InfoNCE特征梯度的退化对照。

## 提交（用户手动）

运行时间：6张4090训练16–22小时，8张4090测试6–10小时，共22–32小时，排队另计。这是旧计时外推，不是本轮实测。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_batch500_32_v1.sh
```

包含CPU准备→6个训练worker→CPU资产核对→自动8卡四套件测试→CPU汇总；两GPU阶段串行。训练6对日志，测试10对日志，不另起smoke。每组单GPU，worker最长先排多个独立子进程；同方法同初始种子、同LR/作者几何变换，只更新外部补丁/生成器。原模型/数据/标签/源码不改。

可选`--preflight`或`--dry-run`仅CPU检查，不查询/提交Slurm。默认无参数才实际提交。助手没有执行默认提交。

## 输出位置

- 补丁：`training/m<ID>_<objective>_b<B>/outer_0500.png`。
- 同目录0/125/250/375/500快照，`resume.pt`完整Adam/scheduler/RNG/生成器断点，`training.parquet`真实训练曲线。
- 四套件主表：`results/RESULTS.md`。每套件100回合/补丁，共13200配对回合；不跨套件平均。
- 详细表：`BATCH500_SUMMARY.csv`、`BATCH500_CONTRASTS.csv`、`BATCH500_PER_TASK.csv`、`BATCH500_TRAINING_COST.csv`。
- 原始闭环：`results/BATCH500_ROLLOUTS.parquet`及`BATCH500_TRACES.parquet`。
- 作业ID：`reports/SUBMISSION_RECEIPT.json`；日志：`logs/`。

例如：`training/m01_uada_original_b8/outer_0500.png`、`training/m07_dino_affinity_b4/outer_0500.png`、`training/m12_tma_xyz_b1/outer_0500.png`。

## 重提/恢复

仍用同一提交命令。入口拒绝同批活跃作业重复提交；没有未完成训练时不申请训练数组。未完成worker只调用其方法列表，完整组先核对资产后跳过，其他组恢复每10轮的完整断点。测试仅提交未完成task，并按episode断点续跑；已完成suite-task复核后跳过。失败不删除数据、不改科学设置。

同worker某组失败会继续处理其它组，worker最后非零退出，afterok禁止缺图情况下进入评测。服务器硬关停可能丢失最近未提交的≤10轮，重提会从最近完整断点继续。

原Long250窗口池和1024条均值统计只读依赖见`REUSE.json`；本轮期间不要删除这些源文件。没有训练回放/教师/目标套件选样，不修改原始专家动作。
