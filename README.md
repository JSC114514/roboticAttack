# RoboticAttack：从腕部视觉脆弱性到可靠控制

本仓库整理OpenVLA / OpenVLA-OFT在本地LIBERO中的视觉安全研究，包括原始攻击基线、腕部／前视不对称性与DINO因果分析、共享补丁探索，以及目前的腕部视角适配实验。

**当前方向：腕部相机观察位置变化时，如何维持正常机器人操作，并保留对真实目标变化的响应。**

[当前工作与最新结果](docs/CURRENT_DIRECTION.md) · [完整研究脉络](docs/RESEARCH_STORY.md) · [全部实验配置](docs/ALL_EXPERIMENTS.md) · [代码阅读指南](docs/CODE_GUIDE.md)

## 为什么转向这个问题

历史配对干预发现腕部输入比前视更影响控制，DINO分支参与这部分效应的传播。随后尝试把几何表示变化转成共享RGB补丁：表示可读、内部干预有效，没有直接转成更强的真实补丁；空间关系动作目标产生过闭环失败，但没有稳定超过同条件普通动作基线。因此，目前将研究重点放在腕部视觉控制可靠性，旧攻击保留为历史成果和压力测试工具。

普通视角一致性训练在当前开发评测中，将相机偏移条件的任务成功数从124/128提升到128/128，正常条件保持32/32。最新相机×真实目标配对诊断完成279个有效8步分支，尚未发现一致性训练造成跨任务一致的空间响应损害。当前保留权重1配置，不继续为尚未成立的问题增加约束。

## 按问题找代码

| 研究线 | 内容 | 阅读入口 |
|---|---|---|
| Exploring | 原始UADA/TMA/UPA，OFT连续动作与双视角适配 | [基线说明](docs/methods/exploring.md) |
| WHEN / UPA-RFAS | 特征、注意力和语义目标，源模型训练与迁移 | [WHEN说明](docs/methods/when.md) |
| DINO采样与回放 | 观测评分、训练池选择、受扰状态覆盖 | [采样说明](docs/methods/dino_sampling.md) |
| 空间关系补丁 | 目标—末端方向、正确／置乱关系及公平动作基线 | [关系目标说明](docs/methods/spatial_relation.md) |
| 腕部视觉适配 | 同状态相机配对、专家监督、一致性与约束对照 | [当前方法与数据](docs/methods/wrist_reliability.md) |
| 相机／目标响应诊断 | 分别移动相机与真实物体，比较动作及实际8步运动 | [最新诊断](research_runs/wrist_camera_task_response_diag_v1/EXPERIMENT_DETAILS.md) |

## 目录与使用顺序

1. [安装与数据](docs/SETUP_AND_DATA.md)：先区分模型checkpoint、补丁训练数据和LIBERO测试起点。
2. [全部运行索引](docs/ALL_EXPERIMENTS.md)：每个版本附原始配置、入口、代码和可用结果记录。
3. [实验阶段索引](docs/EXPERIMENT_INDEX.md)：按研究问题阅读，避免从几十个版本名猜主线。
4. [结果索引](results/README.md)：任务成功率、CASR及短窗口过程指标分开解释。
5. [失败与部分成功路线](archive/negative_results/README.md)：保留无增益、工程失败及未完成的区别。
6. [复现说明](docs/REPRODUCIBILITY.md)：外部模型、数据、签名与发布注释版的处理。

代码位于`VLAAttacker/`、`scripts/`、`research_runs/`、`research_tools/`；文档在`docs/`，精选结果在`results/`。模型、数据集、特征缓存、优化器和视频不入Git。

## 检查与手动上传

在仓库根目录执行，无需GPU：

```bash
python scripts/release/check_release.py
python -m unittest discover -s tests_release -v
```

[本次更新文件清单与上传说明](docs/UPLOAD.md)。实验仍由用户手动提交Slurm，本仓库整理脚本不会自动启动训练。

## 来源

Exploring源于William-wAng618/roboticAttack，模型基础为OpenVLA / OpenVLA-OFT，WHEN对应UPA-RFAS。许可见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)分别记录原始源码与发布注释副本的哈希。
