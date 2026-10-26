# wrist_dino_official_long4_2000_v1：配置与阅读说明

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
  "experiment_id": "OFT-OFFICIAL-LONG4-2000-V1",
  "methods": [
    "uada_original",
    "uada_dino_freq",
    "tma_xyz",
    "tma_xyz_dino"
  ],
  "seed": 42,
  "data_seed": 20261011,
  "training_suite": "libero_10",
  "dataset_directory": "/home/caijiasheng25/Desktop/roboticAttack/dataset/libero_10_no_noops/1.0.0",
  "dataset_info_sha256": "e805d8e2e38a97fe9ea6141bdee800457594738173da9f385fccf5f1ecd6b657",
  "dataset_episodes": 379,
  "outer_iterations": 2000,
  "inner_loop": 50,
  "batch_size": 1,
  "candidates_per_outer": 4,
  "learning_rate": 0.002,
  "weight_decay": 0,
  "warmup_outer_iterations": 20,
  "checkpoint_outer_iterations": [
    0,
    250,
    500,
    750,
    1000,
    1500,
    2000
  ],
  "evaluation_checkpoints": [
    1000,
    2000
  ],
  "patch_shape": [
    3,
    50,
    50
  ],
  "uada_maskidx": [
    0
  ],
  "tma_maskidx": [
    0,
    1,
    2
  ],
  "tma_normalized_target": [
    0,
    0,
    0
  ],
  "frequency_strength": 8,
  "frequency_power": 0.75,
  "resume_every_outer": 10,
  "auxiliary_gradient_ratio": 0.1,
  "geometry_scales": [
    0.9,
    1.1
  ],
  "geometry_lateral_pixels": [
    -4,
    4
  ],
  "attention_layers": [
    8,
    16
  ],
  "dino_grid": 16,
  "dino_prefix_tokens": 5,
  "eot": {
    "mode": "exploring_original",
    "fixed_inner_count": 0,
    "rotation_degrees": 30,
    "shear": 0.2,
    "identity_probability": 0.2,
    "resize_patch": false
  },
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
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49
    ],
    "patch_xy": [
      26,
      137
    ],
    "order_seed": 20261011,
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
    "training": {
      "partition": "rtx",
      "gpu_type": "4090",
      "max_concurrent": 4
    },
    "evaluation": {
      "partition": "rtx",
      "gpu_type": "4090",
      "max_concurrent": 8
    }
  },
  "original_sources": {
    "VLAAttacker/white_patch/UADA.py": "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
    "VLAAttacker/white_patch/TMA.py": "7d622093579656ce59e8126521b7bd034317544618b094e537996d8db6f725d2",
    "VLAAttacker/white_patch/appply_random_transform.py": "4091a03a5b55760d6e8ae5463679ab8b68a8ee9253735d610154e0c5f0f6d4bf"
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
- [code/model_runtime.py](code/model_runtime.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/readonly_model.py](code/readonly_model.py)
- [code/source_transform.py](code/source_transform.py)
- [code/train.py](code/train.py)
- [source/TMA.py](source/TMA.py)
- [source/UADA.py](source/UADA.py)
- [source/appply_random_transform.py](source/appply_random_transform.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_official_long4_2000_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

- [submit_exploring_oft_official_long4_2000_v1.sh](../../submit_exploring_oft_official_long4_2000_v1.sh)
