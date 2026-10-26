# wrist_camera_task_response_diag_v1：配置与阅读说明

## 实验目的

核查腕部输入变化如何影响控制，以及视角适配是否恢复操作并保留目标位置响应。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "version": 1,
  "source_run": "research_runs/wrist_geometry_response_adapt_v1",
  "source_signature": "1c725524a1f50466b6190d5b322e41bedad95ed2a02fbfee19afa96276cef205",
  "models": [
    "base",
    "consistency_w1_s42",
    "consistency_w10_s42"
  ],
  "tasks": {
    "libero_spatial": [
      0,
      1
    ],
    "libero_object": [
      0,
      1
    ],
    "libero_goal": [
      1
    ],
    "libero_10": [
      0
    ]
  },
  "state_indices": [
    42,
    43
  ],
  "target_shifts_m": [
    -0.02,
    0,
    0.02
  ],
  "camera_shifts_m": [
    -0.02,
    0,
    0.02
  ],
  "chunk": 8,
  "bootstrap_draws": 2000,
  "bootstrap_seed": 42,
  "resources": {
    "partition": "rtx4090",
    "gres": "gpu:4090:1",
    "gpu_workers": 2,
    "gpu_cpus": 4,
    "gpu_mem_gb": 12,
    "cpu_cpus": 2,
    "cpu_mem_gb": 4
  },
  "rgb_tolerance": 0.0001,
  "action_tolerance": 1e-05,
  "support_gap_m": 0.002,
  "new_penetration_m": 0.001,
  "support_normal_z_min": 0.5
}
```

## 核心实现

- [code/diag_common.py](code/diag_common.py)
- [code/diag_evaluate.py](code/diag_evaluate.py)
- [code/diag_geometry.py](code/diag_geometry.py)
- [code/diag_prepare.py](code/diag_prepare.py)
- [code/diag_report.py](code/diag_report.py)
- [code/diag_run.py](code/diag_run.py)

## 结果与失败记录

- [reports/BRANCHES.csv](../../results/run_records/wrist_camera_task_response_diag_v1/reports/BRANCHES.csv)
- [reports/REPORT.md](../../results/run_records/wrist_camera_task_response_diag_v1/reports/REPORT.md)
- [reports/REPORT_COMPLETE.json](../../results/run_records/wrist_camera_task_response_diag_v1/reports/REPORT_COMPLETE.json)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
