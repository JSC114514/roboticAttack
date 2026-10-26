# wrist_dino_geometry_pilot_v1：配置与阅读说明

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
  "experiment_id": "SPATIAL-DINO-GEOMETRY-PILOT-V1",
  "atlas": "research_runs/wrist_dino_information_atlas_v2",
  "uada_png": "research_runs/wrist_dino_spatial_b12_diag500_v1/training/m01_uada_original_b1/outer_0500.png",
  "suite": "libero_spatial",
  "layers": [16, 22],
  "seeds": [42, 43],
  "episodes_per_task_split": 2,
  "direction_degrees": [15.0, 30.0],
  "distance_fractions": [0.25, 0.5],
  "max_displacement_m": 0.03,
  "updates": 1000,
  "learning_rate": 0.002,
  "warmup_updates": 20,
  "checkpoints": [0, 500, 1000],
  "resume_interval": 100,
  "patch_shape": [3, 50, 50],
  "patch_xy": [26, 137],
  "evaluation_eot_seeds": [2026101901, 2026101902, 2026101903, 2026101904],
  "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
  "bootstrap_draws": 2000,
  "random_control_seed": 42,
  "resources": {
    "prepare": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "time": "00:45:00"},
    "mechanism": {"cpus": 4, "host_memory_gib": 12, "gpus": 1, "gpu_type": "4090", "time": "02:00:00"},
    "pilot": {"cpus": 4, "host_memory_gib": 16, "gpus": 1, "gpu_type": "4090", "time": "04:00:00"},
    "analyze": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "time": "00:45:00"}
  }
}
```

## 核心实现

- [code/pilot_analysis.py](code/pilot_analysis.py)
- [code/pilot_common.py](code/pilot_common.py)
- [code/pilot_engine.py](code/pilot_engine.py)
- [code/pilot_evaluate.py](code/pilot_evaluate.py)
- [code/pilot_geometry.py](code/pilot_geometry.py)
- [code/pilot_mechanism.py](code/pilot_mechanism.py)
- [code/pilot_pipeline.py](code/pilot_pipeline.py)
- [code/pilot_pixels.py](code/pilot_pixels.py)
- [code/pilot_preprocessing.py](code/pilot_preprocessing.py)
- [code/pilot_readouts.py](code/pilot_readouts.py)
- [code/pilot_render.py](code/pilot_render.py)
- [code/pilot_train.py](code/pilot_train.py)
- [code/pilot_transform.py](code/pilot_transform.py)
- [code/pilot_worker.py](code/pilot_worker.py)

## 结果与失败记录

- [results/COMPLETE.json](../../results/run_records/wrist_dino_geometry_pilot_v1/results/COMPLETE.json)
- [results/RESULTS.md](../../results/run_records/wrist_dino_geometry_pilot_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
