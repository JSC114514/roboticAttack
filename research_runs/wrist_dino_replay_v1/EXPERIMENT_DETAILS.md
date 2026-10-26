# wrist_dino_replay_v1：配置与阅读说明

## 实验目的

检验训练观测选择与受扰状态覆盖能否改进共享补丁；与动作损失本身的改变分开比较。 本版本的具体问题以原始README或方案为准。

## 原始方案与执行入口

- [PLAN.md](PLAN.md)
- 本目录未发现独立Shell入口；可能由项目级submit脚本调度，见总源码清单。

这些是历史配置，不在本次整理中重新运行。完整GPU执行仍需模型、初态、缓存和来源收据。

## 原始配置逐项保留

下面按原文件展示模型、数据划分、补丁、优化与资源字段；配置未声明的值需继续查看实现，不填入推测值。

本版本没有独立配置文件；参数由源码常量、命令行或上游入口提供。下方列出原始代码入口。

## 核心实现


## 结果与失败记录

本次未提取到小型结果表；不据目录名判断成功或失败。查看原始README、历史结果索引及对应运行输出。

方法负结果、工程错误和缺失结果分别记录；fix版本不是新的独立科学重复。任务成功率与CASR分母见[指标说明](../../docs/REPRODUCIBILITY.md)。


## 关联项目级入口与配置

以下关联由源码中出现的完整运行ID与配置路径确定。

- [configs/exploring_oft_dino/object_dino_replay_refine200_cross_suite_v1.json](../../configs/exploring_oft_dino/object_dino_replay_refine200_cross_suite_v1.json)
- [configs/exploring_oft_dino/object_dino_replay_refine200_v1.json](../../configs/exploring_oft_dino/object_dino_replay_refine200_v1.json)
- [configs/exploring_oft_dino/object_dino_window_refine200_v1.json](../../configs/exploring_oft_dino/object_dino_window_refine200_v1.json)
- [configs/exploring_oft_dino/scale500_cross_suite.json](../../configs/exploring_oft_dino/scale500_cross_suite.json)
- [scripts/dino_replay_common.py](../../scripts/dino_replay_common.py)
