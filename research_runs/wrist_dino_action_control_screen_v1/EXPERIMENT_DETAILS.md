# wrist_dino_action_control_screen_v1：配置与阅读说明

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
  "experiment_id": "OFT-WRIST-INTERNAL-ACTION-CONTROL-SCREEN-V1",
  "seed": 42,
  "training_suite": "libero_spatial",
  "source": {
    "experiment": "wrist_dino_key_open_pilot_v2",
    "artifact": "research_runs/wrist_dino_key_open_pilot_v2/data/DUAL_CLOSE.pt",
    "sha256": "23b3a6b421afbd2c72d4e55a264606702f61721c9ce73d0fd6a929714bb17ac1",
    "signature": "22c2211fb6f094709e6b881b7f1d4c967c75a1617878c991b44fc4385dc9d4f5",
    "train_records": 20,
    "validation_records": 20
  },
  "layers": [8, 16, 22],
  "endpoints": ["wrong_open", "uada_opposite"],
  "intervention": {
    "steps": 12,
    "learning_rate": 0.02,
    "relative_rms_limit": 0.2,
    "shared_rms_candidates": [0.05, 0.1, 0.2],
    "regularizer": 0.02,
    "open_margin": 0.6,
    "prefix_tokens": 5,
    "dense_tokens": 256,
    "open_loop_steps": 8,
    "action_dim": 7
  },
  "candidates": [
    "q_spatial_b08", "k_spatial_b08", "v_spatial_b08",
    "q_spatial_b16", "k_spatial_b16", "v_spatial_b16",
    "q_spatial_b22", "k_spatial_b22", "v_spatial_b22",
    "attn_spatial_b08", "attn_spatial_b16", "attn_spatial_b22",
    "mlp_spatial_b08", "mlp_spatial_b16", "mlp_spatial_b22",
    "block_spatial_b08", "block_cls_b08", "block_register_b08",
    "block_spatial_b16", "block_cls_b16", "block_register_b16",
    "block_spatial_b22", "block_cls_b22", "block_register_b22",
    "dino_wrist_output", "siglip_wrist_output", "projector_wrist_output"
  ],
  "resources": {
    "screen": {"cpus": 4, "host_memory_gib": 16, "gpu_type": "4090", "gpus": 1, "walltime": "04:00:00"},
    "analyze": {"cpus": 2, "host_memory_gib": 4, "gpus": 0, "walltime": "00:20:00"}
  }
}
```

## 核心实现

- [code/analyze.py](code/analyze.py)
- [code/common.py](code/common.py)
- [code/engine.py](code/engine.py)
- [code/pipeline.py](code/pipeline.py)
- [code/resources.py](code/resources.py)
- [code/screen.py](code/screen.py)
- [code/screen_math.py](code/screen_math.py)
- [code/sites.py](code/sites.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

- [results/RESULTS.md](../../results/run_records/wrist_dino_action_control_screen_v1/results/RESULTS.md)

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

