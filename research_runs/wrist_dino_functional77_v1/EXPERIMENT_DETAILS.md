# wrist_dino_functional77_v1：配置与阅读说明

## 实验目的

区分理想几何表示是否影响策略、共享RGB是否实现该表示、是否进一步造成闭环失败。 本版本的具体问题以原始README或方案为准。

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
  "experiment_id": "wrist_dino_functional77_v1",
  "parent": "wrist_dino_spatial_b12_diag500_v1",
  "parent_signature": "ac44e284cedc1bd5b47ca02702c2eeed20cffca1d2ecc6ae4505cea3f27c0349",
  "patches": [
    "m01_uada_original_b1@500",
    "m02_dino_key_b1@500",
    "m03_dino_center_b1@500"
  ],
  "suites": [
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10"
  ],
  "components": [
    "functional",
    "random",
    "orthogonal"
  ],
  "directions": [
    "restore",
    "inject"
  ],
  "random_seed": 20261016,
  "new_episode_count": 77,
  "max_gpu_concurrent": 4,
  "cpu_stage_cores": 2,
  "finite_difference_relative_steps": [
    0.002,
    0.01
  ],
  "finite_difference_error_limit": 0.5,
  "finite_difference_min_best_of_two_fraction_per_group": 0.9,
  "norm_match_relative_tolerance": 0.03,
  "native_action_atol": 0.0001,
  "snapshot_blocks": [
    0,
    4,
    12
  ],
  "gate": {
    "minimum_net_favorable_episodes_per_contrast": 3,
    "minimum_favorable_suites_per_contrast": 2,
    "controls": [
      "random",
      "orthogonal"
    ],
    "require_both_directions": true,
    "eligibility": "fresh_clean_success_and_fresh_png_failure",
    "denominator": 10,
    "no_automatic_training": true
  },
  "estimated_four_gpu_minutes_excluding_queue": [
    20,
    40
  ]
}
```

## 核心实现

- [code/functional77_analyze.py](code/functional77_analyze.py)
- [code/functional77_common.py](code/functional77_common.py)
- [code/functional77_math.py](code/functional77_math.py)
- [code/functional77_pipeline.py](code/functional77_pipeline.py)
- [code/functional77_probe.py](code/functional77_probe.py)
- [code/functional77_progress.py](code/functional77_progress.py)
- [code/functional77_run.py](code/functional77_run.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_functional77_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
