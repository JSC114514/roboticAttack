# wrist_spatial_relation_shuffle_eval_v1：配置与阅读说明

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
  "source_signature": "67d5f01434fb00f41b14218b426ec1563dc19850603dd9b09e280d2455b312a2",
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
  "eval_states": [
    2,
    3,
    4,
    5
  ],
  "methods": [
    "matched_relation_s42",
    "shuffled_relation_s42",
    "matched_relation_s43",
    "shuffled_relation_s43"
  ],
  "max_concurrent_gpus": 2,
  "action_tolerance": 1e-05,
  "updates": 4000,
  "training_observations": 217,
  "bootstrap_draws": 2000,
  "bootstrap_seed": 42,
  "assets": [
    {
      "condition": "matched_relation_s42",
      "method": "matched_relation",
      "seed": 42,
      "root": "research_runs/wrist_spatial_relation_shuffle_v1",
      "source_signature": "8a7e5d51e904249f7a031b7f8f52ffd83187ad56d70907b6eb0cecaafc058820",
      "png": "training/matched_relation_s42/checkpoint_004000.png",
      "png_sha256": "838a0e12082ec4d1d21205dd070bed8c5e08a055260c2d5b1685a9a7313a0769"
    },
    {
      "condition": "shuffled_relation_s42",
      "method": "shuffled_relation",
      "seed": 42,
      "root": "research_runs/wrist_spatial_relation_shuffle_v1_fix1",
      "source_signature": "9885787285de6f15dfac0cc3309720459257d52a4ca5d0d6bb352661472c29e2",
      "png": "training/shuffled_relation_s42/checkpoint_004000.png",
      "png_sha256": "fae129af261da1cbe58de71e66d5f31ebdf30f590c72b49d251bdc0a6707d05f"
    },
    {
      "condition": "matched_relation_s43",
      "method": "matched_relation",
      "seed": 43,
      "root": "research_runs/wrist_spatial_relation_shuffle_v1",
      "source_signature": "8a7e5d51e904249f7a031b7f8f52ffd83187ad56d70907b6eb0cecaafc058820",
      "png": "training/matched_relation_s43/checkpoint_004000.png",
      "png_sha256": "8a224c9310454004e42b5ea49b4857bcd70ba10bc15fd80482a4482d4f1826ea"
    },
    {
      "condition": "shuffled_relation_s43",
      "method": "shuffled_relation",
      "seed": 43,
      "root": "research_runs/wrist_spatial_relation_shuffle_v1_fix1",
      "source_signature": "9885787285de6f15dfac0cc3309720459257d52a4ca5d0d6bb352661472c29e2",
      "png": "training/shuffled_relation_s43/checkpoint_004000.png",
      "png_sha256": "d221e1cc96bf8c4e2dea8f3bb8e7cee4371edc6414115584babd8435f114ed45"
    }
  ]
}
```

## 核心实现

- [code/se_cli.py](code/se_cli.py)
- [code/se_common.py](code/se_common.py)
- [code/se_evaluate.py](code/se_evaluate.py)
- [code/se_prepare.py](code/se_prepare.py)
- [code/se_report.py](code/se_report.py)

## 结果与失败记录

- [results/CASR.csv](../../results/run_records/wrist_spatial_relation_shuffle_eval_v1/results/CASR.csv)
- [results/COMPLETE.json](../../results/run_records/wrist_spatial_relation_shuffle_eval_v1/results/COMPLETE.json)
- [results/EPISODES.csv](../../results/run_records/wrist_spatial_relation_shuffle_eval_v1/results/EPISODES.csv)
- [results/RESULTS.md](../../results/run_records/wrist_spatial_relation_shuffle_eval_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
