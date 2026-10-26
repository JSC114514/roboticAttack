# wrist_dino_geometry_closedloop_v1：配置与阅读说明

## 实验目的

区分理想几何表示是否影响策略、共享RGB是否实现该表示、是否进一步造成闭环失败。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "suite": "libero_spatial",
  "tasks": [0, 1, 2, 4, 6, 7],
  "states": [44, 45],
  "workers": [[0, 1], [2, 4], [6, 7]],
  "conditions": ["clean", "initial", "projection", "full", "uada"],
  "order_seed": 20261019,
  "max_steps": 220,
  "num_steps_wait": 10,
  "patch_xy": [26, 137],
  "checkpoint_suite": "libero_spatial",
  "source_run": "research_runs/wrist_dino_geometry_match_v1",
  "source_signature": "86b9ae6c5de8cbd997c28f8e414c48cd4213a0274145f9e46de58b8d97b9bac8",
  "seed_reference": "research_runs/wrist_dino_official_long4_2000_v1/results/LONG4_ROLLOUTS.parquet",
  "patches": {
    "initial": "research_runs/wrist_dino_geometry_match_v1/data/initial_s42.png",
    "projection": "research_runs/wrist_dino_geometry_match_v1/training/pilot/projection/final.png",
    "full": "research_runs/wrist_dino_geometry_match_v1/training/pilot/full/final.png",
    "uada": "research_runs/wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png"
  },
  "keyframe_decisions": [0, 4, 12],
  "bootstrap_draws": 2000,
  "gpu": {"partition": "rtx4090", "gres": "gpu:4090:1", "cpus": 4, "memory": "12G", "time": "00:45:00"},
  "cpu": {"partition": "cpu", "cpus": 2, "memory": "4G", "time": "00:10:00"}
}
```

## 核心实现

- [code/cli.py](code/cli.py)
- [code/closed_report.py](code/closed_report.py)
- [code/protocol.py](code/protocol.py)
- [code/rollout.py](code/rollout.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_geometry_closedloop_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
