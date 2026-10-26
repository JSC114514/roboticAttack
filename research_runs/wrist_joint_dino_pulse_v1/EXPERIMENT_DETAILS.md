# wrist_joint_dino_pulse_v1：配置与阅读说明

## 实验目的

比较不同攻击目标、视觉梯度路径和训练配置对真实PNG部署效果的影响。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

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
  "tasks": {"libero_spatial": [0, 2, 4, 6], "libero_object": [0, 2, 4, 6], "libero_goal": [0, 3, 5, 7], "libero_10": [0, 2, 5, 8]},
  "state_index": 11,
  "seed_base": 20261020,
  "chunk": 8,
  "num_steps_wait": 10,
  "max_steps": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
  "patch_xy": [26, 137],
  "patch_shape": [50, 50],
  "patch": {"name": "historical_uada500", "path": "research_runs/wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png", "sha256": "c5f59d59af80f145a46346e6387fede6ef44d0ed8b2481d917fc0dd577a7567c"},
  "durations": [8, 32],
  "modes": ["clean", "png", "dino", "siglip"],
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "action_tolerance": 0.00001,
  "bootstrap_draws": 2000,
  "bootstrap_seed": 42,
  "max_workers": 2,
  "training": false,
  "eot": false
}
```

## 核心实现

- [code/dp_cli.py](code/dp_cli.py)
- [code/dp_common.py](code/dp_common.py)
- [code/dp_math.py](code/dp_math.py)
- [code/dp_prepare.py](code/dp_prepare.py)
- [code/dp_report.py](code/dp_report.py)
- [code/dp_rollout.py](code/dp_rollout.py)

## 结果与失败记录

- [results/COMPLETE.json](../../results/run_records/wrist_joint_dino_pulse_v1/results/COMPLETE.json)
- [results/EPISODES.csv](../../results/run_records/wrist_joint_dino_pulse_v1/results/EPISODES.csv)
- [results/RESULTS.md](../../results/run_records/wrist_joint_dino_pulse_v1/results/RESULTS.md)
- [results/SUMMARY.csv](../../results/run_records/wrist_joint_dino_pulse_v1/results/SUMMARY.csv)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
