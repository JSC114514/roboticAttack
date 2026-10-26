# wrist_dino_residual4_v1：配置与阅读说明

## 实验目的

比较不同攻击目标、视觉梯度路径和训练配置对真实PNG部署效果的影响。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)
- [PLAN.md](PLAN.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "wrist_dino_residual4_v1",
  "source_parent": "wrist_dino_functional77_v1",
  "source_parent_signature": "6c8032ab20b2fac883bddc2d1b538fd018cb8465140c29d86ae36fecf46a1e87",
  "parent": "wrist_dino_spatial_b12_diag500_v1",
  "patches": [
    "m01_uada_original_b1@500",
    "m02_dino_key_b1@500",
    "m03_dino_center_b1@500"
  ],
  "random_seed": 20261016,
  "new_episode_count": 4,
  "max_gpu_concurrent": 4,
  "finite_difference_relative_steps": [
    0.002,
    0.01
  ],
  "finite_difference_error_limit": 0.5,
  "native_action_atol": 0.0001,
  "norm_match_relative_tolerance": 0.03,
  "snapshot_blocks": [
    0,
    4,
    12
  ],
  "component": "residual_full",
  "amplitude_policy": "unscaled_original_complement",
  "independent_initial_states": 2,
  "gate": {
    "minimum_physical_states_with_failure": 2,
    "automatic_training": false,
    "automatic_more_episodes": false
  },
  "time_budget_minutes_excluding_queue": [
    10,
    15
  ]
}
```

## 核心实现

- [code/residual4_analyze.py](code/residual4_analyze.py)
- [code/residual4_common.py](code/residual4_common.py)
- [code/residual4_math.py](code/residual4_math.py)
- [code/residual4_pipeline.py](code/residual4_pipeline.py)
- [code/residual4_probe.py](code/residual4_probe.py)
- [code/residual4_progress.py](code/residual4_progress.py)
- [code/residual4_run.py](code/residual4_run.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
