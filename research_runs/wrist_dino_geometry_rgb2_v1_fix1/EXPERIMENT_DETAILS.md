# wrist_dino_geometry_rgb2_v1_fix1：配置与阅读说明

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
  "source_signature": "86b9ae6c5de8cbd997c28f8e414c48cd4213a0274145f9e46de58b8d97b9bac8",
  "repair_source_signature": "af8af43a8133605dbafc6f1537f2e2a29c775cebb29a5532816f961b2182870c",
  "alignment_repair": "legacy_rgb_or_exact_sampled_state_with_same_frame_targets",
  "scale2_signature": "28edc2820b59e612437c8312ee3e53535ad094245844139d29a4ce0df34844a6",
  "checkpoint": "libero_spatial",
  "fit_states": [40, 41], "verify_states": [42, 43],
  "frame_ranks": [0, 1, 2, 3, 4],
  "displacement_scale": 2.0, "target_sign": "plus",
  "candidates": ["projection", "full"],
  "seed": 42, "learning_rate": 0.002, "visits": 20,
  "shared_updates": 4800, "individual_count": 12, "individual_updates": 100,
  "patch_shape": [3, 50, 50], "patch_xy": [26, 137],
  "eot": false, "action_loss": false,
  "eval_tasks": [0, 1, 2, 6, 7], "eval_states": [44, 45],
  "conditions": ["clean", "initial", "uada", "projection", "full"],
  "num_steps_wait": 10, "max_steps": 220, "keyframe_decisions": [0, 4, 12],
  "bootstrap_draws": 2000, "order_seed": 20261020,
  "max_concurrent_gpus": 2,
  "resources": {
    "cache": {"memory": "12G", "time": "00:45:00"},
    "train": {"memory": "16G", "time": "00:30:00"},
    "evaluate": {"memory": "12G", "time": "00:30:00"},
    "report": {"memory": "4G", "time": "00:10:00"}
  }
}
```

## 核心实现

- [code/rgb_alignment.py](code/rgb_alignment.py)
- [code/rgb_cache.py](code/rgb_cache.py)
- [code/rgb_cli.py](code/rgb_cli.py)
- [code/rgb_common.py](code/rgb_common.py)
- [code/rgb_evaluate.py](code/rgb_evaluate.py)
- [code/rgb_report.py](code/rgb_report.py)
- [code/rgb_train.py](code/rgb_train.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_dino_geometry_rgb2_v1_fix1/results/CASR.csv)
- [results/COMPLETE.json](../../results/run_records/wrist_dino_geometry_rgb2_v1_fix1/results/COMPLETE.json)
- [results/RESULTS.md](../../results/run_records/wrist_dino_geometry_rgb2_v1_fix1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
