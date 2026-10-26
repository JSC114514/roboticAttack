# wrist_spatial_relation_rescue_v1：配置与阅读说明

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
  "source_run": "research_runs/wrist_spatial_approach_pilot_v1",
  "source_signature": "67d5f01434fb00f41b14218b426ec1563dc19850603dd9b09e280d2455b312a2",
  "audit_run": "research_runs/wrist_spatial_failure_audit_v1",
  "audit_signature": "b3c276339a4c27c3aa109ea32c578651cf65745346ad1ccc1219f4c0aa89f1d6",
  "checkpoint": "libero_goal",
  "primary": "radial_reverse_s42",
  "methods": ["restore_radial", "restore_xyz", "restore_pose_grip"],
  "order_seed": 42,
  "patch_xy": [26, 137],
  "max_steps": 300,
  "max_concurrent_gpus": 1,
  "position_tolerance_m": 1e-9,
  "radial_tolerance_m": 1e-9,
  "direction_epsilon_m": 1e-8,
  "action_tolerance": 1e-5,
  "rgb_nrmse_tolerance": 0.0001,
  "bootstrap_draws": 5000,
  "expansion_states": [2, 3, 4, 5]
}
```

## 核心实现

- [code/rr_cli.py](code/rr_cli.py)
- [code/rr_common.py](code/rr_common.py)
- [code/rr_logger.py](code/rr_logger.py)
- [code/rr_math.py](code/rr_math.py)
- [code/rr_report.py](code/rr_report.py)
- [code/rr_run.py](code/rr_run.py)

## 结果与失败记录

- [results/COMPLETE.json](../../results/run_records/wrist_spatial_relation_rescue_v1/results/COMPLETE.json)
- [results/RESULTS.md](../../results/run_records/wrist_spatial_relation_rescue_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
