# wrist_dino_action_readout_v1：配置与阅读说明

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
  "experiment_id": "OFT-DINO-ACTION-READOUT-V1",
  "seed": 42,
  "head_seed": 2026101642,
  "teacher_seed": 20261016,
  "split_seed": 202610164,
  "outer_iterations": 500,
  "inner_loop": 50,
  "arms": ["B_native_energy", "C_dino_readout"],
  "evaluation_states": [40, 41],
  "response_eta": 0.00784313725490196,
  "diagnostic_outers": [1, 25, 125, 250, 375, 400, 401, 500],
  "baseline_png_sha256": "c5f59d59af80f145a46346e6387fede6ef44d0ed8b2481d917fc0dd577a7567c",
  "confidence_subjective_joint_success": 0.25,
  "second_seed_automatic": false
}
```

## 核心实现

- [code/fit_head.py](code/fit_head.py)
- [code/native_bridge.py](code/native_bridge.py)
- [code/readout.py](code/readout.py)
- [code/rollout.py](code/rollout.py)
- [code/run_core.py](code/run_core.py)
- [code/submit_run.py](code/submit_run.py)
- [code/summarize.py](code/summarize.py)
- [code/teacher.py](code/teacher.py)
- [code/train_patch.py](code/train_patch.py)
- [code/worker.py](code/worker.py)

## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

使用本目录入口与原始方案；未发现以完整运行ID引用本目录的项目级提交器。
