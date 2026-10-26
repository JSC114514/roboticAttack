# wrist_dino_geometry_action_teacher_v1：配置与阅读说明

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
  "source_signature": "4a03991f2a1d87a5a35d29164b3d8cca492172830a7003b2c92faab967ff5674",
  "checkpoint": "libero_spatial",
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "seeds": [42, 43],
  "methods": ["geometry_teacher", "clean_far_boundary"],
  "primary": "geometry_teacher_s42",
  "updates": 24000,
  "checkpoint_interval": 1000,
  "learning_rate": 0.002,
  "warmup": 20,
  "patch_shape": [3, 50, 50],
  "patch_xy": [26, 137],
  "eot": false,
  "displacement_scale": 2.0,
  "target_sign": "plus",
  "train_until": "2026-10-20T05:30:00+08:00",
  "interim_at": "2026-10-20T07:50:00+08:00",
  "eval_states": [0, 1],
  "rollout_seed_base": 20261020,
  "bootstrap_draws": 2000,
  "action_tolerance": 0.00001,
  "max_concurrent_gpus": 4
}
```

## 核心实现

- [code/at_cache.py](code/at_cache.py)
- [code/at_cli.py](code/at_cli.py)
- [code/at_common.py](code/at_common.py)
- [code/at_engine.py](code/at_engine.py)
- [code/at_evaluate.py](code/at_evaluate.py)
- [code/at_math.py](code/at_math.py)
- [code/at_report.py](code/at_report.py)
- [code/at_train.py](code/at_train.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_dino_geometry_action_teacher_v1/results/CASR.csv)
- [results/RESULTS.md](../../results/run_records/wrist_dino_geometry_action_teacher_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
