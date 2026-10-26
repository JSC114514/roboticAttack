# wrist_dino_standoff_mechanism_v1：配置与阅读说明

## 实验目的

区分理想几何表示是否影响策略、共享RGB是否实现该表示、是否进一步造成闭环失败。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [run.sh](run.sh)
- [README.md](README.md)

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

### config.json

```json
{
  "experiment_id": "WRIST-DINO-STANDOFF-MECHANISM-V1",
  "source_run": "research_runs/wrist_dino_official_long4_2000_v1",
  "runtime_run": "research_runs/wrist_dino_spatial_b12_diag500_v1",
  "checkpoint_suite": "libero_spatial",
  "tasks": [
    {"suite":"libero_spatial","task_id":0,"entity":"akita_black_bowl_1","kind":"object","states":[40,41,42,43]},
    {"suite":"libero_object","task_id":0,"entity":"alphabet_soup_1","kind":"object","states":[40,41,42,43]},
    {"suite":"libero_goal","task_id":0,"entity":"wooden_cabinet_1","kind":"middle_handle","states":[40,41,42,43],"handle_collision_centers":[[-0.00278,-0.10161,0.11041],[0.02916,-0.08348,0.11041],[-0.03560,-0.08348,0.11041]]},
    {"suite":"libero_10","task_id":5,"entity":"black_book_1","kind":"object","states":[42,43]}
  ],
  "distance_m": 0.05,
  "bins": {"far":{"minimum":0.06,"maximum":0.12,"preferred":0.08},"near":{"minimum":0.01,"maximum":0.04,"preferred":0.03}},
  "dwell_band_m": [0.04,0.06],
  "max_steps": 128,
  "action_chunk_slots": 8,
  "block": 22,
  "conditions": ["clean","dino_near","dino_far"],
  "random_directions": 4,
  "seed": 42,
  "rgb_nrmse_limit": 0.0001,
  "penetration_tolerance_m": 0.00001,
  "bootstrap_draws": 2000,
  "max_concurrent_gpus": 2,
  "resources": {"capture":{"cpus":4,"memory_gib":12,"gpus":1,"minutes":45},"mechanism":{"cpus":4,"memory_gib":12,"gpus":1,"minutes":60},"report":{"cpus":2,"memory_gib":4,"gpus":0,"minutes":10}}
}
```

## 核心实现

- [code/sd_cli.py](code/sd_cli.py)
- [code/sd_common.py](code/sd_common.py)
- [code/sd_geometry.py](code/sd_geometry.py)
- [code/sd_math.py](code/sd_math.py)
- [code/sd_mechanism.py](code/sd_mechanism.py)
- [code/sd_model.py](code/sd_model.py)
- [code/sd_replay.py](code/sd_replay.py)
- [code/sd_report.py](code/sd_report.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
