# wrist_dino_relation_readout_pilot_v1：配置与阅读说明

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
  "source_signature": "4a03991f2a1d87a5a35d29164b3d8cca492172830a7003b2c92faab967ff5674",
  "checkpoint": "libero_spatial",
  "tasks": [0, 1, 2, 4, 6, 7],
  "fit_states": [40, 41],
  "verify_states": [42, 43],
  "frame_ranks": [0, 1, 2, 3, 4],
  "gate_frame_rank": 1,
  "layer": 22,
  "seed": 42,
  "random_directions": 4,
  "updates": 2000,
  "checkpoint_interval": 100,
  "learning_rate": 0.002,
  "warmup": 20,
  "relation_scale_m": 0.03,
  "patch_shape": [3, 50, 50],
  "patch_xy": [26, 137],
  "eot": false,
  "displacement_scale": 2.0,
  "target_sign": "plus",
  "eval_tasks": [0, 1, 2, 6, 7],
  "eval_states": [44, 45],
  "conditions": ["clean", "initial", "uada", "relation"],
  "num_steps_wait": 10,
  "max_steps": 220,
  "keyframe_decisions": [0, 4, 12],
  "max_concurrent_gpus": 2,
  "bootstrap_draws": 2000,
  "interim_at": "2026-10-20T10:45:00+08:00"
}
```

## 核心实现

- [code/rr_cli.py](code/rr_cli.py)
- [code/rr_common.py](code/rr_common.py)
- [code/rr_evaluate.py](code/rr_evaluate.py)
- [code/rr_gate.py](code/rr_gate.py)
- [code/rr_math.py](code/rr_math.py)
- [code/rr_report.py](code/rr_report.py)
- [code/rr_train.py](code/rr_train.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
