# wrist_spatial_approach_pilot_v1：配置与阅读说明

## 实验目的

检验利用目标—末端方向构造错误命令，是否比错配关系或普通动作端点更有效。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "checkpoint": "libero_goal",
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "fit_states": [40, 41],
  "verify_states": [42, 43],
  "frames_before_first_contact": 5,
  "seeds": [42, 43],
  "methods": ["radial_reverse", "translation_far_boundary"],
  "primary": "radial_reverse_s42",
  "updates": 4000,
  "checkpoint_interval": 500,
  "learning_rate": 0.002,
  "warmup": 20,
  "patch_shape": [3, 50, 50],
  "patch_xy": [26, 137],
  "eot": false,
  "retreat_goal_m": 0.005,
  "eval_states": [0, 1],
  "rollout_seed_base": 20261020,
  "action_tolerance": 0.00001,
  "bootstrap_draws": 2000,
  "max_concurrent_gpus": 2
}
```

## 核心实现

- [code/sr_cache.py](code/sr_cache.py)
- [code/sr_cli.py](code/sr_cli.py)
- [code/sr_common.py](code/sr_common.py)
- [code/sr_engine.py](code/sr_engine.py)
- [code/sr_evaluate.py](code/sr_evaluate.py)
- [code/sr_geometry.py](code/sr_geometry.py)
- [code/sr_math.py](code/sr_math.py)
- [code/sr_report.py](code/sr_report.py)
- [code/sr_train.py](code/sr_train.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_spatial_approach_pilot_v1/results/CASR.csv)
- [results/RESULTS.md](../../results/run_records/wrist_spatial_approach_pilot_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
