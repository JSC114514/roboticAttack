# wrist_dino_key_open_pilot_v2：配置与阅读说明

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
  "experiment_id": "OFT-SPATIAL-KEY-OPEN-PILOT-V2",
  "seed": 42,
  "data_seed": 20261013,
  "training_suite": "libero_spatial",
  "methods": [
    "key_point",
    "key_direction"
  ],
  "train_observations": 20,
  "validation_observations": 20,
  "single_observations": 4,
  "single_candidates": 200,
  "universal_outer": 40,
  "inner_loop": 50,
  "learning_rate": 0.002,
  "warmup_outer": 20,
  "cosine_outer": 2000,
  "key_steps": 20,
  "key_lr": 0.02,
  "key_rms_limit": 0.2,
  "key_regularizer": 0.1,
  "open_target": 0.6,
  "backtracking": [
    1.0,
    0.5,
    0.25
  ],
  "evaluation": {
    "suites": [
      "libero_object",
      "libero_spatial",
      "libero_goal",
      "libero_10"
    ],
    "task_count": 10,
    "state_indices": [
      40,
      41
    ],
    "patch_xy": [
      26,
      137
    ],
    "order_seed": 20261013,
    "num_steps_wait": 10,
    "num_open_loop_steps": 8,
    "image_resolution": 224,
    "patch_shape": [
      3,
      50,
      50
    ],
    "dtype": "bfloat16",
    "checkpoints": {
      "libero_object": "/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-object",
      "libero_spatial": "/home/caijiasheng25/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b",
      "libero_goal": "/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-goal",
      "libero_10": "/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-10"
    },
    "unnorm_keys": {
      "libero_object": "libero_object_no_noops",
      "libero_spatial": "libero_spatial_no_noops",
      "libero_goal": "libero_goal_no_noops",
      "libero_10": "libero_10_no_noops"
    },
    "paths": {
      "legacy_mgrb_root": "/home/caijiasheng25/robopatch_defense/oft_mgrb_lite_low_cost",
      "robotic_attack_root": "/home/caijiasheng25/robopatch_defense/roboticAttack-main",
      "attribution_root": "/home/caijiasheng25/robopatch_defense/vla_internal_attribution_20260812",
      "historical_root": "/home/caijiasheng25/robopatch_defense/vla_attribution_clean_perturbed_route_e3",
      "openvla_oft_root": "/home/caijiasheng25/Desktop/tex3D/openvla-oft",
      "libero_root": "/home/caijiasheng25/Desktop/LIBERO"
    },
    "max_steps_by_suite": {
      "libero_object": 280,
      "libero_spatial": 220,
      "libero_goal": 300,
      "libero_10": 520
    },
    "seed_sources": {
      "libero_object": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/results/FINAL8_ROLLOUTS.parquet",
      "libero_spatial": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/results/FINAL8_ROLLOUTS.parquet",
      "libero_goal": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/results/FINAL8_ROLLOUTS.parquet",
      "libero_10": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_long_final8_v2/evaluation_4090_v1/results/FINAL8_ROLLOUTS.parquet"
    }
  },
  "resources": {
    "prepare": {
      "cpus": 4,
      "host_memory_gib": 4
    },
    "training": {
      "cpus": 4,
      "host_memory_gib": 16,
      "gpu_type": "4090",
      "max_concurrent": 1
    },
    "validate": {
      "cpus": 2,
      "host_memory_gib": 4
    },
    "evaluation": {
      "cpus": 4,
      "host_memory_gib": 12,
      "gpu_type": "4090",
      "max_concurrent": 2
    },
    "statistics": {
      "cpus": 2,
      "host_memory_gib": 4
    }
  }
}
```

## 核心实现

- [code/analyze.py](code/analyze.py)
- [code/common.py](code/common.py)
- [code/data.py](code/data.py)
- [code/deployment.py](code/deployment.py)
- [code/engine_native.py](code/engine_native.py)
- [code/entities.py](code/entities.py)
- [code/evaluate.py](code/evaluate.py)
- [code/key_math.py](code/key_math.py)
- [code/key_runtime.py](code/key_runtime.py)
- [code/model_runtime.py](code/model_runtime.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/readonly_model.py](code/readonly_model.py)
- [code/resources.py](code/resources.py)
- [code/source_transform.py](code/source_transform.py)
- [code/train.py](code/train.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_key_open_pilot_v2/results/RESULTS.md)
- [results/SUMMARY.csv](../../results/run_records/wrist_dino_key_open_pilot_v2/results/SUMMARY.csv)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

