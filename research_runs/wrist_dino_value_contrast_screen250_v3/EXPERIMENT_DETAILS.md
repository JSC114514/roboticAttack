# wrist_dino_value_contrast_screen250_v3：配置与阅读说明

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
  "experiment_id": "OFT-DINO-VALUE-CONTRAST-SCREEN250-V3",
  "base_run": "wrist_dino_feature_screen250_v1",
  "base_signature": "590b344e7e2d4602cb19c7d8a090b6e714e1cac00ec35c11caf1e7601ebe1620",
  "reference_run": "wrist_dino_objectives_screen250_v2",
  "reference_signature": "35abc0c93dad8534d5f3876c87d8d4c934d55223ba408090473e8aedf3b27542",
  "methods": ["dino_value", "advencoder_cos_b8", "advencoder_nce_b8"],
  "comparison_methods": ["uada_original", "dino_key", "advencoder_cos", "dino_affinity"],
  "value_layers": [8, 16],
  "batch_sizes": {"dino_value": 1, "advencoder_cos_b8": 8, "advencoder_nce_b8": 8},
  "nce_temperature": 0.1,
  "gradient_cache": "exact_two_pass_microbatch1",
  "recompute_tolerance": 0.00001,
  "resources": {
    "training": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 3},
    "evaluation": {"partition": "rtx", "gpu_type": "4090", "max_concurrent": 4}
  }
}
```

## 核心实现

- [code/UADA_SOURCE_COPY.py](code/UADA_SOURCE_COPY.py)
- [code/analyze.py](code/analyze.py)
- [code/common.py](code/common.py)
- [code/data.py](code/data.py)
- [code/deployment.py](code/deployment.py)
- [code/engine.py](code/engine.py)
- [code/evaluate.py](code/evaluate.py)
- [code/feature_objectives.py](code/feature_objectives.py)
- [code/model_runtime.py](code/model_runtime.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/readonly_model.py](code/readonly_model.py)
- [code/remaining_objectives.py](code/remaining_objectives.py)
- [code/source_transform.py](code/source_transform.py)
- [code/train.py](code/train.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_value_contrast_screen250_v3/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

