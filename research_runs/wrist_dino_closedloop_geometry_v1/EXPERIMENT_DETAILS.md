# wrist_dino_closedloop_geometry_v1：配置与阅读说明

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
  "experiment_id": "WRIST-DINO-CLOSEDLOOP-GEOMETRY-V1",
  "seed": 20261017,
  "suites": ["libero_object", "libero_spatial", "libero_goal", "libero_10"],
  "tasks": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "states": [40],
  "conditions": ["clean", "uada_static", "uada_geometry", "loop_geometry"],
  "horizon_blocks": 2,
  "commit_blocks": 2,
  "chunk_length": 8,
  "candidates": [
    {"name": "identity", "scale": 1.0, "dx": 0, "dy": 0},
    {"name": "scale_090", "scale": 0.9, "dx": 0, "dy": 0},
    {"name": "scale_110", "scale": 1.1, "dx": 0, "dy": 0},
    {"name": "left4", "scale": 1.0, "dx": -4, "dy": 0},
    {"name": "right4", "scale": 1.0, "dx": 4, "dy": 0},
    {"name": "up4", "scale": 1.0, "dx": 0, "dy": -4},
    {"name": "down4", "scale": 1.0, "dx": 0, "dy": 4}
  ],
  "progress_distance_scale_m": 0.1,
  "restore_state_atol": 1e-10,
  "replay_state_atol": 1e-6,
  "replay_action_atol": 2e-5,
  "baseline_png_sha256": "c5f59d59af80f145a46346e6387fede6ef44d0ed8b2481d917fc0dd577a7567c",
  "max_concurrent_gpus": 4,
  "cpus_per_gpu": 4,
  "host_memory_gb_per_gpu": 32,
  "automatic_second_stage": false,
  "threat_model": "digital wrist stream; privileged simulator branching for both adaptive conditions; frozen policy/front/language/proprio"
}
```

## 核心实现

- [code/branch_state.py](code/branch_state.py)
- [code/closedloop.py](code/closedloop.py)
- [code/geometry.py](code/geometry.py)
- [code/loop_core.py](code/loop_core.py)
- [code/native_predictor.py](code/native_predictor.py)
- [code/submit_run.py](code/submit_run.py)
- [code/summarize.py](code/summarize.py)
- [code/task_progress.py](code/task_progress.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
