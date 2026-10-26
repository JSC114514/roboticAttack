# wrist_dino_closedloop_mechanism_v1_svd_fix1：配置与阅读说明

## 实验目的

区分理想几何表示是否影响策略、共享RGB是否实现该表示、是否进一步造成闭环失败。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [PLAN.md](PLAN.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "wrist_dino_closedloop_mechanism_v1_svd_fix1",
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
  "calibration_tasks": [
    0,
    1
  ],
  "calibration_states": [
    40,
    41
  ],
  "screen_state": 40,
  "screen_blocks": [
    0,
    4,
    12
  ],
  "main_tasks": [
    2,
    3,
    4,
    5,
    6
  ],
  "confirmation_tasks": [
    7,
    8,
    9
  ],
  "main_state": 41,
  "early_window": [
    0,
    4
  ],
  "late_window": [
    12,
    16
  ],
  "spatial_candidates": [
    "footprint",
    "outside_top_left",
    "outside_top_right",
    "outside_bottom_left",
    "outside_bottom_right"
  ],
  "functional_candidates": [
    "translation",
    "rotation",
    "gripper",
    "all"
  ],
  "strengths": [
    0.0,
    0.5,
    1.0
  ],
  "finite_difference_relative_steps": [
    0.002,
    0.01
  ],
  "finite_difference_error_limit": 0.5,
  "finite_difference_min_valid_fraction": 0.5,
  "norm_match_relative_tolerance": 0.03,
  "native_action_atol": 0.0001,
  "baseline_trace_atol": 1e-06,
  "random_seed": 20261016,
  "new_episode_count": 592,
  "max_gpu_concurrent": 4,
  "cpu_stage_cores": 2,
  "bootstrap_replicates": 2000
}
```

## 核心实现

- [code/pilot_analyze.py](code/pilot_analyze.py)
- [code/pilot_common.py](code/pilot_common.py)
- [code/pilot_math.py](code/pilot_math.py)
- [code/pilot_pipeline.py](code/pilot_pipeline.py)
- [code/pilot_probe.py](code/pilot_probe.py)
- [code/pilot_run.py](code/pilot_run.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
