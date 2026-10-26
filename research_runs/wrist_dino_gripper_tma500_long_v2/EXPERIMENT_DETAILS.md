# wrist_dino_gripper_tma500_long_v2：配置与阅读说明

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
  "experiment_id": "OFT-WRIST-DINO-GRIPPER-TMA500-LONG-V2",
  "methods": [
    "plain_gripper_tma",
    "task_region_dino_tma",
    "background_region_dino_tma"
  ],
  "seed": 42,
  "data_seed": 20261008,
  "outer_iterations": 500,
  "inner_loop": 50,
  "batch_size": 1,
  "learning_rate": 0.002,
  "weight_decay": 0,
  "warmup_outer_iterations": 20,
  "geometry": true,
  "resize_patch": false,
  "patch_shape": [
    3,
    50,
    50
  ],
  "checkpoint_outer_iterations": [
    0,
    100,
    250,
    500
  ],
  "close_threshold": 0.5,
  "gripper_target": 1,
  "region_token_threshold": 0,
  "patch_exclusion_threshold": 0,
  "max_eot_attempts": 32,
  "resources": {
    "labels": {
      "partition": "a800",
      "qos": "a800",
      "gpu_type": "a800",
      "device_name": "A800",
      "max_concurrent": 3
    },
    "training": {
      "partition": "a800",
      "qos": "a800",
      "gpu_type": "a800",
      "device_name": "A800",
      "max_concurrent": 3
    },
    "evaluation": {
      "partition": "rtx",
      "qos": null,
      "gpu_type": "4090",
      "device_name": "4090",
      "max_concurrent": 5
    }
  },
  "training_bank": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v2/data/LONG_TRAIN_BANK.pt",
  "eval_reference": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/configs/exploring_oft_dino/scale500_cross_suite.json",
  "eval_reference_sha256": "80931a4e2b9e5dc1d2f04d25d6781082790c1a5651b84b816f9adaf3ca34caff",
  "reference_patch": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_transfer_v1/training/original/exploring_oft_uada/outer_0500.pt",
  "reference_manifest": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_transfer_v1/training/original/RUN_MANIFEST.json",
  "reference_history": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_transfer_v1/training/original/exploring_oft_uada/training.parquet",
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
    "order_seed": 20261008,
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
    "frozen_object_clean": "/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware/results/FINAL_100_ROLLOUTS.parquet",
    "frozen_cross_suite_clean": "/home/caijiasheng25/robopatch_defense/oft_ramp_recovery_aware/results/CROSS_SUITE_ZERO_SHOT_ROLLOUTS.parquet"
  },
  "event_metrics": {
    "stable_grasp_steps": 3,
    "lift_height_m": 0.02,
    "suspected_drop_height_m": 0.02,
    "contact_loss_confirm_steps": 3
  },
  "recovery_source_root": "/home/caijiasheng25/robopatch_defense/roboticAttack-main/research_runs/wrist_dino_gripper_tma500_long_v1",
  "training_suite": "libero_10",
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
  "reference_training_bank_sha256": "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8",
  "reference_training_suite": "libero_object"
}
```

## 核心实现

- [code/TMA_OFT.py](code/TMA_OFT.py)
- [code/alignment.py](code/alignment.py)
- [code/analyze.py](code/analyze.py)
- [code/collect_labels.py](code/collect_labels.py)
- [code/engine.py](code/engine.py)
- [code/entities.py](code/entities.py)
- [code/evaluate.py](code/evaluate.py)
- [code/gripper_common.py](code/gripper_common.py)
- [code/long_source.py](code/long_source.py)
- [code/manage.py](code/manage.py)
- [code/model_runtime_copy.py](code/model_runtime_copy.py)
- [code/objectives.py](code/objectives.py)
- [code/pipeline.py](code/pipeline.py)
- [code/preprocessing.py](code/preprocessing.py)
- [code/recovery.py](code/recovery.py)
- [code/source_transform.py](code/source_transform.py)
- [code/train.py](code/train.py)
- [evaluation_fix_v1/code/analyze_fixed.py](evaluation_fix_v1/code/analyze_fixed.py)
- [evaluation_fix_v1/code/eval_common.py](evaluation_fix_v1/code/eval_common.py)
- [evaluation_fix_v1/code/evaluate_fixed.py](evaluation_fix_v1/code/evaluate_fixed.py)
- [evaluation_fix_v1/code/model_runtime_fixed.py](evaluation_fix_v1/code/model_runtime_fixed.py)
- [evaluation_fix_v1/code/pipeline_fixed.py](evaluation_fix_v1/code/pipeline_fixed.py)
- [evaluation_fix_v2_3090/code/analyze_fixed.py](evaluation_fix_v2_3090/code/analyze_fixed.py)
- [evaluation_fix_v2_3090/code/eval_common.py](evaluation_fix_v2_3090/code/eval_common.py)
- [evaluation_fix_v2_3090/code/evaluate_fixed.py](evaluation_fix_v2_3090/code/evaluate_fixed.py)
- [evaluation_fix_v2_3090/code/model_runtime_fixed.py](evaluation_fix_v2_3090/code/model_runtime_fixed.py)
- [evaluation_fix_v2_3090/code/pipeline_fixed.py](evaluation_fix_v2_3090/code/pipeline_fixed.py)
- [source/TMA.py](source/TMA.py)
- [source/appply_random_transform.py](source/appply_random_transform.py)
- [source/model_runtime.py](source/model_runtime.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

- [submit_exploring_oft_gripper_tma500_long_v2.sh](../../submit_exploring_oft_gripper_tma500_long_v2.sh)
- [submit_exploring_oft_gripper_tma500_long_v2_evalfix1.sh](../../submit_exploring_oft_gripper_tma500_long_v2_evalfix1.sh)
- [submit_exploring_oft_gripper_tma500_long_v2_evalfix2_3090.sh](../../submit_exploring_oft_gripper_tma500_long_v2_evalfix2_3090.sh)
