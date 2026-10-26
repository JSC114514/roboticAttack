# wrist_joint_checkpoint_patch_retest_v1：配置与阅读说明

## 实验目的

比较不同攻击目标、视觉梯度路径和训练配置对真实PNG部署效果的影响。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "repo_id": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
  "revision": "638918f3d1c2e43a39a8a20772bdb8b91835e4b7",
  "model_dir": "/mnt/beegfs/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
  "head_step": 300000,
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "screen_states": [2],
  "confirmation_states": [3],
  "patch_xy": [26, 137],
  "patch_shape": [50, 50],
  "num_steps_wait": 10,
  "chunk": 8,
  "seed_base": 20261020,
  "max_steps": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "initial_manifest": "research_runs/wrist_spatial_approach_pilot_v1/data/MANIFEST.json",
  "fair_manifest": "research_runs/wrist_spatial_relation_fair_v1/data/MANIFEST.json",
  "action_tolerance": 0.00001,
  "max_workers": 4,
  "bootstrap_draws": 2000,
  "bootstrap_seed": 42
}
```

## 核心实现

- [code/download_model.py](code/download_model.py)
- [code/jr_assets.py](code/jr_assets.py)
- [code/jr_cli.py](code/jr_cli.py)
- [code/jr_common.py](code/jr_common.py)
- [code/jr_loading.py](code/jr_loading.py)
- [code/jr_math.py](code/jr_math.py)
- [code/jr_prepare.py](code/jr_prepare.py)
- [code/jr_report.py](code/jr_report.py)
- [code/jr_rollout.py](code/jr_rollout.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_joint_checkpoint_patch_retest_v1/results/CASR.csv)
- [results/COMPLETE.json](../../results/run_records/wrist_joint_checkpoint_patch_retest_v1/results/COMPLETE.json)
- [results/EPISODES.csv](../../results/run_records/wrist_joint_checkpoint_patch_retest_v1/results/EPISODES.csv)
- [results/RESULTS.md](../../results/run_records/wrist_joint_checkpoint_patch_retest_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
