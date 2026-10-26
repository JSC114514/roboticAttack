# wrist_dino_information_atlas_v1：配置与阅读说明

## 实验目的

区分理想几何表示是否影响策略、共享RGB是否实现该表示、是否进一步造成闭环失败。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)
- [PLAN.md](PLAN.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "SPATIAL-DINO-INFORMATION-ATLAS-V1",
  "suite": "libero_spatial",
  "seed": 42,
  "state_manifest": "/mnt/beegfs/home/caijiasheng25/robopatch_defense/vla_attribution_clean_perturbed_route_e3/e3_2q_e4/manifests/E4_1R_STATE_MANIFEST.json",
  "task_map": "/mnt/beegfs/home/caijiasheng25/robopatch_defense/vla_attribution_clean_perturbed_route_e3/e3_2q_e4/manifests/E4_0_TASK_LABEL_MAP.json",
  "old_labels": {
    "discovery": "/mnt/beegfs/home/caijiasheng25/robopatch_defense/vla_attribution_clean_perturbed_route_e3/semantic_causal_attribution_sem0_sem4/manifests/SEMANTIC_LABELS_DISCOVERY.parquet",
    "validation": "/mnt/beegfs/home/caijiasheng25/robopatch_defense/vla_attribution_clean_perturbed_route_e3/semantic_causal_attribution_sem0_sem4/manifests/SEMANTIC_LABELS_VALIDATION.parquet"
  },
  "runtime_config": "/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_key_open_pilot_v2/config.json",
  "replay": {
    "image_size": 224,
    "warmup_steps": 10,
    "rgb_normalized_rms_limit": 0.0001,
    "state_exact": true,
    "proprio_atol": 1e-06
  },
  "blocks": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23
  ],
  "detail_blocks": [
    8,
    16,
    22
  ],
  "consumed_block": 22,
  "grid": 4,
  "views": [
    "front",
    "wrist"
  ],
  "probe": {
    "ridge_alphas": [
      0.01,
      0.1,
      1.0,
      10.0,
      100.0,
      1000.0
    ],
    "folds": 5,
    "bootstrap_draws": 2000,
    "shuffle_repeats": 5,
    "local_tokens_per_image": 32,
    "mlp": {
      "width": 64,
      "lr": 0.001,
      "weight_decay": 0.0001,
      "max_epochs": 200,
      "patience": 20,
      "seeds": [
        42,
        43,
        44
      ]
    }
  },
  "parts": {
    "images": 80,
    "episodes_per_task_split": 2,
    "progress_target": 0.5,
    "classes": [
      "unlabelled",
      "bowl_rim",
      "bowl_body",
      "plate_surface",
      "left_finger",
      "right_finger",
      "other"
    ]
  },
  "resources": {
    "capture": {
      "cpus": 4,
      "host_memory_gib": 12,
      "gpus": 1,
      "gpu_type": "4090",
      "time": "02:00:00"
    },
    "analyze": {
      "cpus": 2,
      "host_memory_gib": 4,
      "gpus": 0,
      "time": "04:00:00"
    }
  },
  "class_names": [
    "background",
    "akita_black_bowl",
    "plate",
    "glazed_rim_porcelain_ramekin",
    "cookies",
    "wooden_cabinet",
    "flat_stove",
    "main_table",
    "robot_arm",
    "gripper",
    "other_object"
  ]
}
```

## 核心实现

- [code/atlas_analysis.py](code/atlas_analysis.py)
- [code/atlas_capture.py](code/atlas_capture.py)
- [code/atlas_common.py](code/atlas_common.py)
- [code/atlas_features.py](code/atlas_features.py)
- [code/atlas_labels.py](code/atlas_labels.py)
- [code/atlas_local.py](code/atlas_local.py)
- [code/atlas_math.py](code/atlas_math.py)
- [code/atlas_parts.py](code/atlas_parts.py)
- [code/atlas_pipeline.py](code/atlas_pipeline.py)
- [code/atlas_probes.py](code/atlas_probes.py)
- [code/atlas_readonly.py](code/atlas_readonly.py)
- [code/atlas_report.py](code/atlas_report.py)
- [code/atlas_runtime.py](code/atlas_runtime.py)
- [code/atlas_worker.py](code/atlas_worker.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
