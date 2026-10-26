# wrist_dino_long_final8_v2：配置与阅读说明

## 实验目的

比较不同攻击目标、视觉梯度路径和训练配置对真实PNG部署效果的影响。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [README.md](README.md)
- 本目录未发现独立Shell入口；可能由项目级submit脚本调度，见总源码清单。

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "OFT-WRIST-DINO-LONG-FINAL8-V1",
  "methods": [
    "uada_original",
    "uada_dino",
    "uada_dino_freq",
    "tma_xyz",
    "tma_xyz_dino",
    "tma_xyz_dino_freq",
    "tma_xyz_dino_teacher",
    "uada_dino_teacher"
  ],
  "seed": 42,
  "data_seed": 20261009,
  "training_suite": "libero_10",
  "training_bank": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/data/LONG_TRAIN_BANK.pt",
  "training_bank_sha256": "8a9f7db04c1dc9797c8c60f9cedaa7e2b06e95e317827defcf3f1c23f1aa997a",
  "outer_iterations": 2000,
  "inner_loop": 50,
  "batch_size": 1,
  "learning_rate": 0.002,
  "weight_decay": 0,
  "warmup_outer_iterations": 20,
  "bootstrap_outer": 250,
  "checkpoint_outer_iterations": [
    0,
    250,
    500,
    1000,
    1500,
    2000
  ],
  "evaluation_checkpoints": [
    500,
    1000,
    2000
  ],
  "refresh_after_outer": [
    500,
    1000,
    1500
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
  "tv_threshold": 0.4,
  "tv_weight": 1,
  "teacher_updates": 100,
  "teacher_shards_per_family": 4,
  "teacher_lr": 0.002,
  "distill_gradient_ratio": 0.1,
  "distill_rms_floor": 0.0001,
  "frequency_strength": 8,
  "frequency_power": 0.75,
  "resume_every_outer": 10,
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
    "order_seed": 20261009,
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
      "libero_object": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/evaluation/GRIPPER_TMA500_ROLLOUTS.parquet",
      "libero_spatial": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/evaluation/GRIPPER_TMA500_ROLLOUTS.parquet",
      "libero_goal": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/evaluation/GRIPPER_TMA500_ROLLOUTS.parquet",
      "libero_10": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/evaluation/GRIPPER_TMA500_ROLLOUTS.parquet"
    }
  },
  "training_source": {
    "suite": "libero_10",
    "task_count": 10,
    "successful_trajectories_per_task": 5,
    "records_per_trajectory": 6,
    "state_indices": [
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
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39
    ],
    "max_steps": 520,
    "seed": 20261008,
    "selection": "uniform_success_episode_decisions",
    "bddl_directory": "/home/caijiasheng25/Desktop/LIBERO/libero/libero/bddl_files/libero_10",
    "init_states_directory": "/home/caijiasheng25/Desktop/LIBERO/libero/libero/init_files/libero_10",
    "libero_config_file": "/home/caijiasheng25/.libero/config.yaml"
  },
  "resources": {
    "training": {
      "partition": "rtx",
      "gpu_type": "4090",
      "max_concurrent": 8
    },
    "evaluation": {
      "partition": "rtx",
      "gpu_type": "3090",
      "max_concurrent": 8
    }
  },
  "original_sources": {
    "VLAAttacker/white_patch/UADA.py": "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e",
    "VLAAttacker/white_patch/TMA.py": "7d622093579656ce59e8126521b7bd034317544618b094e537996d8db6f725d2",
    "VLAAttacker/white_patch/appply_random_transform.py": "4091a03a5b55760d6e8ae5463679ab8b68a8ee9253735d610154e0c5f0f6d4bf",
    "VLAAttacker/white_patch/UADA_OFT_SOURCE_COPY.py": "c54b1fc05bdb51d0c3be66cdcdce24ab7045c581267469a24f2932b694fb52f4",
    "research_runs/wrist_dino_gripper_tma500_long_v2/code/preprocessing.py": "db56d19b6f8a61474a47ebb3cf01b4a8d91b418b0288576439adb7d80668aa5c",
    "research_runs/wrist_dino_gripper_tma500_long_v2/evaluation_fix_v2_3090/code/model_runtime_fixed.py": "366dac921490480aeba76925e6c26a9d5a372e8c59cfbc5d6c7a90240f8d49c3"
  }
}
```

## 核心实现

- [code/UADA_SOURCE_COPY.py](code/UADA_SOURCE_COPY.py)
- [code/analyze.py](code/analyze.py)
- [code/bank.py](code/bank.py)
- [code/collect.py](code/collect.py)
- [code/common.py](code/common.py)
- [code/engine.py](code/engine.py)
- [code/evaluate.py](code/evaluate.py)
- [code/model_runtime.py](code/model_runtime.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/repair.py](code/repair.py)
- [code/source_transform.py](code/source_transform.py)
- [code/teacher.py](code/teacher.py)
- [code/train.py](code/train.py)
- [evaluation_4090_v1/code/analyze.py](evaluation_4090_v1/code/analyze.py)
- [evaluation_4090_v1/code/eval_common.py](evaluation_4090_v1/code/eval_common.py)
- [evaluation_4090_v1/code/evaluate.py](evaluation_4090_v1/code/evaluate.py)
- [evaluation_4090_v1/code/model_runtime_4090.py](evaluation_4090_v1/code/model_runtime_4090.py)
- [evaluation_4090_v1/code/pipeline.py](evaluation_4090_v1/code/pipeline.py)
- [recovery/restart.py](recovery/restart.py)
- [recovery/test_restart.py](recovery/test_restart.py)
- [source/TMA.py](source/TMA.py)
- [source/UADA.py](source/UADA.py)
- [source/appply_random_transform.py](source/appply_random_transform.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

- [submit_exploring_oft_long_final8_2000_repair_v2.sh](../../submit_exploring_oft_long_final8_2000_repair_v2.sh)
- [submit_exploring_oft_long_final8_2000_resume_v2.sh](../../submit_exploring_oft_long_final8_2000_resume_v2.sh)
- [submit_exploring_oft_long_final8_eval4090_v1.sh](../../submit_exploring_oft_long_final8_eval4090_v1.sh)
