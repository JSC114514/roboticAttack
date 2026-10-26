# wrist_dino_relation_bias_v1：配置与阅读说明

## 实验目的

检验利用目标—末端方向构造错误命令，是否比错配关系或普通动作端点更有效。 本版本的具体问题以原始README或方案为准。

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
  "experiment_id": "WRIST-DINO-RELATION-BIAS-V1",
  "source_run": "research_runs/wrist_dino_official_long4_2000_v1",
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "uada_png": "research_runs/wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png",
  "training_checkpoint_suite": "libero_spatial",
  "suites": ["libero_object", "libero_spatial", "libero_goal", "libero_10"],
  "task_splits": {
    "libero_spatial": {"train": [0, 1, 2, 4, 6, 7], "validation": [3, 8], "heldout": [5, 9]},
    "libero_object": {"train": [0, 1, 2, 3, 4, 5], "validation": [6, 7], "heldout": [8, 9]},
    "libero_goal": {"train": [0, 1, 3, 5, 6, 7], "validation": [2, 8], "heldout": [4, 9]},
    "libero_10": {"train": [0, 2, 3, 4, 5, 8], "validation": [1, 7], "heldout": [6, 9]}
  },
  "trajectory_states": {"train": [40, 41, 42, 43, 44, 45, 46, 47], "validation": [40, 41, 42, 43], "heldout": [44, 45, 46, 47]},
  "frame_fractions": [0.1, 0.3, 0.5, 0.7, 0.9],
  "methods": ["translation", "bearing"],
  "bearing_degrees": 30.0,
  "max_displacement_m": 0.03,
  "block": 22,
  "patch_shape": [3, 50, 50],
  "patch_xy": [26, 137],
  "seed": 42,
  "epochs": 50,
  "updates_per_epoch": 960,
  "learning_rate": 0.002,
  "warmup_updates": 20,
  "checkpoint_epochs": [0, 10, 20, 30, 40, 50],
  "validation_closedloop_states": [48],
  "heldout_closedloop_states": [48, 49],
  "validation_conditions": ["clean", "initial", "uada", "translation", "bearing"],
  "heldout_conditions": ["clean", "initial", "uada", "winner"],
  "bootstrap_draws": 2000,
  "resources": {
    "prepare": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "time": "00:30:00"},
    "cache": {"cpus": 4, "host_memory_gib": 12, "gpus": 1, "gpu_type": "a800", "partition": "a800", "qos": "a800", "time": "03:00:00"},
    "train": {"cpus": 4, "host_memory_gib": 16, "gpus": 1, "gpu_type": "a800", "partition": "a800", "qos": "a800", "time": "04:00:00"},
    "validate": {"cpus": 4, "host_memory_gib": 12, "gpus": 1, "gpu_type": "a800", "partition": "a800", "qos": "a800", "time": "02:00:00"},
    "select": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "time": "00:20:00"},
    "heldout": {"cpus": 4, "host_memory_gib": 12, "gpus": 1, "gpu_type": "a800", "partition": "a800", "qos": "a800", "time": "02:00:00"},
    "analyze": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "time": "00:30:00"}
  }
}
```

## 核心实现

- [code/analyze.py](code/analyze.py)
- [code/cache.py](code/cache.py)
- [code/cache_reuse.py](code/cache_reuse.py)
- [code/common.py](code/common.py)
- [code/engine.py](code/engine.py)
- [code/evaluate.py](code/evaluate.py)
- [code/geometry.py](code/geometry.py)
- [code/pipeline.py](code/pipeline.py)
- [code/train.py](code/train.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

- [results/COMPLETE.json](../../results/run_records/wrist_dino_relation_bias_v1/results/COMPLETE.json)
- [results/RESULTS.md](../../results/run_records/wrist_dino_relation_bias_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
