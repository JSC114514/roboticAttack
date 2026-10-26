# wrist_view_asymmetry_screen_v1：配置与阅读说明

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
  "checkpoint": "libero_goal",
  "tasks": [0, 1, 3, 5, 6, 7],
  "fit_states": [40, 41],
  "verify_states": [42, 43],
  "fit_frames": 58,
  "verify_frames": 60,
  "max_steps": 64,
  "chunk": 8,
  "patch_xy": [26, 137],
  "patch_shape": [50, 50],
  "patch_conditions": ["matched_relation_s42", "uada_clean_7dof_s42"],
  "blur_sigma": 2.0,
  "blur_kernel": 13,
  "gate_alpha": 0.5,
  "pca_components": 8,
  "ridge_alphas": [0.1, 1.0, 10.0, 100.0],
  "folds": 5,
  "seed": 20261022,
  "bootstrap_draws": 2000,
  "action_tolerance": 0.00001,
  "rgb_tolerance": 0.0001,
  "max_concurrent_gpus": 2,
  "maximum_branches": 288,
  "source_signature": "2a5734bacd01d154eceb8f014f1b5965d4725492cd0a110952845c41267992a5"
}
```

## 核心实现

- [code/va_calibrate.py](code/va_calibrate.py)
- [code/va_camera.py](code/va_camera.py)
- [code/va_cli.py](code/va_cli.py)
- [code/va_common.py](code/va_common.py)
- [code/va_math.py](code/va_math.py)
- [code/va_prepare.py](code/va_prepare.py)
- [code/va_report.py](code/va_report.py)
- [code/va_rollout.py](code/va_rollout.py)
- [code/va_stages.py](code/va_stages.py)
- [code/va_vision.py](code/va_vision.py)

## 结果与失败记录

- [results/BRANCHES.csv](../../results/run_records/wrist_view_asymmetry_screen_v1/results/BRANCHES.csv)
- [results/COMPLETE.json](../../results/run_records/wrist_view_asymmetry_screen_v1/results/COMPLETE.json)
- [results/RESULTS.md](../../results/run_records/wrist_view_asymmetry_screen_v1/results/RESULTS.md)
- [results/SUMMARY.csv](../../results/run_records/wrist_view_asymmetry_screen_v1/results/SUMMARY.csv)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
