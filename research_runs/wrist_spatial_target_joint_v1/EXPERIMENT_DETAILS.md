# wrist_spatial_target_joint_v1：配置与阅读说明

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
  "source_signature": "67d5f01434fb00f41b14218b426ec1563dc19850603dd9b09e280d2455b312a2",
  "rules": ["near", "far", "left", "right"],
  "methods": ["joint", "fixed_harm", "realizability_only", "translation_boundary"],
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "seed": 42,
  "displacement_m": 0.02,
  "direction_epsilon_m": 1e-8,
  "feature_epsilon": 1e-12,
  "updates": 4000,
  "adaptive_budget": 3000,
  "cycle_budget": 200,
  "probe_updates": 5,
  "learning_rate": 0.002,
  "warmup": 20,
  "checkpoint_interval": 200,
  "patch_xy": [26, 137],
  "patch_shape": [3, 50, 50],
  "eot": false,
  "max_concurrent_gpus": 2,
  "eval_states": [2, 3, 4, 5],
  "rgb_nrmse_tolerance": 0.0001
}
```

## 核心实现

- [code/jt_cache.py](code/jt_cache.py)
- [code/jt_cli.py](code/jt_cli.py)
- [code/jt_common.py](code/jt_common.py)
- [code/jt_evaluate.py](code/jt_evaluate.py)
- [code/jt_math.py](code/jt_math.py)
- [code/jt_prepare.py](code/jt_prepare.py)
- [code/jt_report.py](code/jt_report.py)
- [code/jt_train.py](code/jt_train.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
