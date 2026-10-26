# wrist_spatial_relation_shuffle_v1_fix1：配置与阅读说明

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
  "scope": "control",
  "checkpoint": "libero_goal",
  "patch_shape": [
    3,
    50,
    50
  ],
  "patch_xy": [
    26,
    137
  ],
  "eot": false,
  "max_concurrent_gpus": 2,
  "seeds": [
    42,
    43
  ],
  "methods": [
    "matched_relation",
    "shuffled_relation"
  ],
  "primary": "matched_relation_s42",
  "updates": 4000,
  "checkpoint_interval": 500,
  "learning_rate": 0.002,
  "warmup": 20,
  "action_tolerance": 1e-05,
  "shuffle_seed": 20261021,
  "suites": [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10"
  ],
  "video_label": "关系对应对照",
  "repair": {
    "one_candidate_per_process": true,
    "resume_step": 2000,
    "copy_completed_step": 4000,
    "singleview_video": true
  }
}
```

## 核心实现

- [code/rfix_cli.py](code/rfix_cli.py)
- [code/rfix_common.py](code/rfix_common.py)
- [code/rfix_evaluate.py](code/rfix_evaluate.py)
- [code/rfix_report.py](code/rfix_report.py)
- [code/rfix_train.py](code/rfix_train.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_spatial_relation_shuffle_v1_fix1/results/CASR.csv)
- [results/EPISODES.csv](../../results/run_records/wrist_spatial_relation_shuffle_v1_fix1/results/EPISODES.csv)
- [results/RESULTS.md](../../results/run_records/wrist_spatial_relation_shuffle_v1_fix1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
