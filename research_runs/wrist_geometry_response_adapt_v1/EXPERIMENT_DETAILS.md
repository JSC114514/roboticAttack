# wrist_geometry_response_adapt_v1：配置与阅读说明

## 实验目的

核查腕部输入变化如何影响控制，以及视角适配是否恢复操作并保留目标位置响应。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "version": 1,
  "model_dir": "/mnt/beegfs/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
  "revision": "638918f3d1c2e43a39a8a20772bdb8b91835e4b7",
  "head_step": 300000,
  "suites": [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10"
  ],
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "libero_root": "/home/caijiasheng25/Desktop/LIBERO",
  "expert_root": "/home/caijiasheng25/Desktop/roboticAttack/dataset",
  "camera_index": "research_runs/wrist_dino_relation_bias_v1/data/INDEX.json",
  "source_traces": "research_runs/wrist_dino_official_long4_2000_v1/results/LONG4_TRACES.parquet",
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
      "development": [
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
      "development": [
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
      "development": [
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
      "development": [
        1,
        7
      ],
      "heldout": [
        6,
        9
      ]
    }
  },
  "pilot_tasks": {
    "libero_spatial": [
      0,
      1
    ],
    "libero_object": [
      0,
      1
    ],
    "libero_goal": [
      0,
      1
    ],
    "libero_10": [
      0,
      2
    ]
  },
  "num_steps_wait": 10,
  "chunk": 8,
  "seed_base": 20261023,
  "max_steps": {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520
  },
  "adapter": {
    "blocks": [
      18,
      19,
      20,
      21,
      22
    ],
    "modules": [
      "qkv",
      "proj"
    ],
    "rank": 8,
    "alpha": 8,
    "dropout": 0.0
  },
  "optimizer": {
    "lr": 0.0001,
    "weight_decay": 0.0,
    "warmup": 50,
    "min_lr": 1e-05,
    "gradient_clip": 1.0
  },
  "dual": {
    "initial": 1.0,
    "ema": 0.95,
    "lr": 0.01,
    "maximum": 50.0,
    "epsilon": 1e-06
  },
  "camera_train_shifts_m": [
    -0.01,
    -0.005,
    0.005,
    0.01
  ],
  "camera_axes": [
    "x",
    "z"
  ],
  "action_tolerance": 1e-05,
  "rgb_tolerance": 0.0001,
  "bootstrap_draws": 2000,
  "bootstrap_seed": 42,
  "main_seeds": [
    42,
    43,
    44
  ],
  "resources": {
    "pilot_workers": 2,
    "main_workers": 4,
    "partition": "rtx4090",
    "gres": "gpu:4090:1",
    "cpu_cpus": 2,
    "cpu_mem_gb": 4,
    "gpu_cpus": 4,
    "train_mem_gb": 16,
    "eval_mem_gb": 12
  },
  "stages": {
    "pilot": {
      "updates": 1000,
      "train_expert_trajectories": 8,
      "val_expert_trajectories": 5,
      "pairs_per_trajectory": 8,
      "camera_fit_states": [
        40,
        41
      ],
      "camera_val_states": [
        42,
        43
      ],
      "eval_states": [
        18,
        19,
        20,
        21
      ],
      "eval_shift_m": 0.015
    },
    "main": {
      "updates": 5000,
      "train_expert_trajectories": 20,
      "val_expert_trajectories": 5,
      "pairs_per_trajectory": 8,
      "camera_fit_states": [
        40,
        41,
        44,
        45,
        46,
        47
      ],
      "camera_val_states": [
        42,
        43
      ],
      "eval_states": [
        22,
        23,
        24
      ],
      "eval_shift_m": 0.02
    }
  },
  "dependency_files": [
    "research_runs/wrist_joint_checkpoint_patch_retest_v1/code/jr_common.py",
    "research_runs/wrist_joint_checkpoint_patch_retest_v1/code/jr_loading.py",
    "research_runs/wrist_joint_checkpoint_patch_retest_v1/code/jr_rollout.py",
    "research_runs/wrist_dino_spatial_b12_diag500_v1/code/model_runtime.py",
    "VLAAttacker/multiview/openvla_oft_wrapper.py",
    "VLAAttacker/multiview/openvla_oft_experiments.py",
    "research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py",
    "research_runs/wrist_dino_spatial_b12_diag500_v1/code/common.py",
    "research_runs/wrist_joint_checkpoint_patch_retest_v1/code/jr_math.py",
    "/home/caijiasheng25/Desktop/tex3D/openvla-oft/experiments/robot/openvla_utils.py",
    "/home/caijiasheng25/Desktop/tex3D/openvla-oft/prismatic/extern/hf/modeling_prismatic.py"
  ],
  "pilot_candidates": [
    {
      "name": "bc_s42",
      "method": "bc",
      "seed": 42,
      "weight": 0.0,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "augment_s42",
      "method": "augment",
      "seed": 42,
      "weight": 0.0,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "consistency_w01_s42",
      "method": "consistency",
      "seed": 42,
      "weight": 0.1,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "consistency_w1_s42",
      "method": "consistency",
      "seed": 42,
      "weight": 1.0,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "consistency_w10_s42",
      "method": "consistency",
      "seed": 42,
      "weight": 10.0,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "constrained_t0_s42",
      "method": "constrained",
      "seed": 42,
      "weight": 1.0,
      "tau": 0.0,
      "updates": 1000
    },
    {
      "name": "constrained_t010_s42",
      "method": "constrained",
      "seed": 42,
      "weight": 1.0,
      "tau": 0.1,
      "updates": 1000
    },
    {
      "name": "constrained_t025_s42",
      "method": "constrained",
      "seed": 42,
      "weight": 1.0,
      "tau": 0.25,
      "updates": 1000
    }
  ]
}
```

## 核心实现

- [code/wr_camera.py](code/wr_camera.py)
- [code/wr_common.py](code/wr_common.py)
- [code/wr_data.py](code/wr_data.py)
- [code/wr_engine.py](code/wr_engine.py)
- [code/wr_evaluate.py](code/wr_evaluate.py)
- [code/wr_main.py](code/wr_main.py)
- [code/wr_math.py](code/wr_math.py)
- [code/wr_orchestrate.py](code/wr_orchestrate.py)
- [code/wr_report.py](code/wr_report.py)
- [code/wr_train.py](code/wr_train.py)
- [code/wr_video.py](code/wr_video.py)

## 结果与失败记录

- [results/pilot/EPISODES.csv](../../results/run_records/wrist_geometry_response_adapt_v1/results/pilot/EPISODES.csv)
- [results/pilot/RESULTS.md](../../results/run_records/wrist_geometry_response_adapt_v1/results/pilot/RESULTS.md)
- [results/pilot/SELECTION.json](../../results/run_records/wrist_geometry_response_adapt_v1/results/pilot/SELECTION.json)
- [results/pilot/SUITE.csv](../../results/run_records/wrist_geometry_response_adapt_v1/results/pilot/SUITE.csv)
- [results/pilot/TASK.csv](../../results/run_records/wrist_geometry_response_adapt_v1/results/pilot/TASK.csv)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
