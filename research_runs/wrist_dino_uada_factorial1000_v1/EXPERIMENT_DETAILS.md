# wrist_dino_uada_factorial1000_v1：配置与阅读说明

## 实验目的

比较不同攻击目标、视觉梯度路径和训练配置对真实PNG部署效果的影响。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [README.md](README.md)
- [PLAN.md](PLAN.md)
- 本目录未发现独立Shell入口；可能由项目级submit脚本调度，见总源码清单。

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "OFT-UADA-FACTORIAL1000-V1",
  "seed": 42,
  "outer_iterations": 1000,
  "inner_loop": 50,
  "arms": [
    {
      "name": "legacy_freq",
      "protocol": "legacy",
      "method": "uada_freq",
      "sampling": false,
      "frequency": true,
      "schedule_outer_iterations": 1000
    },
    {
      "name": "original_dino",
      "protocol": "original",
      "method": "uada_dino",
      "sampling": true,
      "frequency": false,
      "schedule_outer_iterations": 2000
    },
    {
      "name": "original_freq",
      "protocol": "original",
      "method": "uada_freq",
      "sampling": false,
      "frequency": true,
      "schedule_outer_iterations": 2000
    }
  ],
  "diagnostic_outers": [
    1,
    25,
    125,
    250,
    500,
    750,
    1000
  ],
  "diagnostic_inners": [
    0,
    49
  ],
  "bootstrap_draws": 10000,
  "bootstrap_seed": 20261017,
  "training_gpu_cap": 3,
  "evaluation_gpu_cap": 4,
  "new_episodes": 1200,
  "combined_episodes": 4000,
  "future_seed_submitted": false
}
```

## 核心实现

- [code/UADA_SOURCE_COPY.py](code/UADA_SOURCE_COPY.py)
- [code/analyze.py](code/analyze.py)
- [code/common.py](code/common.py)
- [code/deployment.py](code/deployment.py)
- [code/engine.py](code/engine.py)
- [code/engine_legacy.py](code/engine_legacy.py)
- [code/engine_original.py](code/engine_original.py)
- [code/evaluate.py](code/evaluate.py)
- [code/model_runtime.py](code/model_runtime.py)
- [code/monitor.py](code/monitor.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/readonly_model.py](code/readonly_model.py)
- [code/source_transform.py](code/source_transform.py)
- [code/train.py](code/train.py)
- [code/worker.py](code/worker.py)
- [source/legacy/code/UADA_SOURCE_COPY.py](source/legacy/code/UADA_SOURCE_COPY.py)
- [source/legacy/code/analyze.py](source/legacy/code/analyze.py)
- [source/legacy/code/common.py](source/legacy/code/common.py)
- [source/legacy/code/data.py](source/legacy/code/data.py)
- [source/legacy/code/deployment.py](source/legacy/code/deployment.py)
- [source/legacy/code/engine.py](source/legacy/code/engine.py)
- [source/legacy/code/evaluate.py](source/legacy/code/evaluate.py)
- [source/legacy/code/model_runtime.py](source/legacy/code/model_runtime.py)
- [source/legacy/code/objectives.py](source/legacy/code/objectives.py)
- [source/legacy/code/pipeline.py](source/legacy/code/pipeline.py)
- [source/legacy/code/preprocessing.py](source/legacy/code/preprocessing.py)
- [source/legacy/code/readonly_model.py](source/legacy/code/readonly_model.py)
- [source/legacy/code/source_transform.py](source/legacy/code/source_transform.py)
- [source/legacy/code/train.py](source/legacy/code/train.py)
- [source/original/code/UADA_SOURCE_COPY.py](source/original/code/UADA_SOURCE_COPY.py)
- [source/original/code/analyze.py](source/original/code/analyze.py)
- [source/original/code/common.py](source/original/code/common.py)
- [source/original/code/data.py](source/original/code/data.py)
- [source/original/code/deployment.py](source/original/code/deployment.py)
- [source/original/code/engine.py](source/original/code/engine.py)
- [source/original/code/evaluate.py](source/original/code/evaluate.py)
- [source/original/code/model_runtime.py](source/original/code/model_runtime.py)
- [source/original/code/objectives.py](source/original/code/objectives.py)
- [source/original/code/pipeline.py](source/original/code/pipeline.py)
- [source/original/code/preprocessing.py](source/original/code/preprocessing.py)
- [source/original/code/readonly_model.py](source/original/code/readonly_model.py)
- [source/original/code/source_transform.py](source/original/code/source_transform.py)
- [source/original/code/train.py](source/original/code/train.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_uada_factorial1000_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

