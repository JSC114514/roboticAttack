# wrist_dino_geometry_match_v1：配置与阅读说明

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
  "experiment_id": "WRIST-DINO-GEOMETRY-MATCH-V1",
  "source_run": "research_runs/wrist_dino_official_long4_2000_v1",
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "uada_png": "research_runs/wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png",
  "training_checkpoint_suite": "libero_spatial",
  "suites": [
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_10"
  ],
  "task_splits": {
    "libero_spatial": {
      "train": [
        0,
        1,
        2,
        4,
        6,
        7
      ],
      "validation": [
        3,
        8
      ],
      "heldout": [
        5,
        9
      ]
    },
    "libero_object": {
      "train": [
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "validation": [
        6,
        7
      ],
      "heldout": [
        8,
        9
      ]
    },
    "libero_goal": {
      "train": [
        0,
        1,
        3,
        5,
        6,
        7
      ],
      "validation": [
        2,
        8
      ],
      "heldout": [
        4,
        9
      ]
    },
    "libero_10": {
      "train": [
        0,
        2,
        3,
        4,
        5,
        8
      ],
      "validation": [
        1,
        7
      ],
      "heldout": [
        6,
        9
      ]
    }
  },
  "trajectory_states": {
    "train": [
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47
    ],
    "validation": [
      40,
      41,
      42,
      43
    ],
    "heldout": [
      44,
      45,
      46,
      47
    ]
  },
  "frame_fractions": [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9
  ],
  "methods": [
    "translation"
  ],
  "max_displacement_m": 0.03,
  "block": 22,
  "patch_shape": [
    3,
    50,
    50
  ],
  "patch_xy": [
    26,
    137
  ],
  "seed": 42,
  "epochs": 50,
  "updates_per_epoch": 960,
  "learning_rate": 0.002,
  "warmup_updates": 20,
  "checkpoint_epochs": [
    0,
    10,
    20,
    30,
    40,
    50
  ],
  "validation_closedloop_states": [
    48
  ],
  "heldout_closedloop_states": [
    48,
    49
  ],
  "bootstrap_draws": 2000,
  "resources": {
    "train": {
      "cpus": 4,
      "host_memory_gib": 16,
      "gpus": 1,
      "gpu_type": "4090",
      "partition": "rtx4090",
      "time": "03:00:00"
    },
    "eval": {
      "cpus": 4,
      "host_memory_gib": 12,
      "gpus": 1,
      "gpu_type": "4090",
      "partition": "rtx4090",
      "time": "01:30:00"
    },
    "cpu": {
      "cpus": 2,
      "host_memory_gib": 4,
      "gpus": 0,
      "time": "00:30:00"
    }
  },
  "origin_run": "research_runs/wrist_dino_relation_bias_v1",
  "gpu_type": "4090",
  "pilot_fit_states": [
    40,
    41
  ],
  "pilot_verify_states": [
    42,
    43
  ],
  "pilot_frame_rank": 1,
  "pilot_visits": 100,
  "pilot_lr": 0.002,
  "night_candidates": {
    "full_s42": {
      "loss": "full",
      "seed": 42
    },
    "full_s43": {
      "loss": "full",
      "seed": 43
    },
    "projection_s43": {
      "loss": "projection",
      "seed": 43
    }
  },
  "primary_candidate": "full_s42",
  "fresh_heldout_states": [
    0,
    1
  ],
  "rollout_seed_base": 420000,
  "max_concurrent_gpus": 4,
  "deadline_beijing": "2026-10-20T08:00:00+08:00",
  "method_lock_deadline_beijing": "2026-10-20T00:00:00+08:00"
}
```

## 核心实现

- [code/common.py](code/common.py)
- [code/data.py](code/data.py)
- [code/engine.py](code/engine.py)
- [code/evaluate.py](code/evaluate.py)
- [code/pipeline.py](code/pipeline.py)
- [code/report.py](code/report.py)
- [code/train.py](code/train.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

- [results/pilot/COMPLETE.json](../../results/run_records/wrist_dino_geometry_match_v1/results/pilot/COMPLETE.json)
- [results/pilot/RESULTS.md](../../results/run_records/wrist_dino_geometry_match_v1/results/pilot/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
