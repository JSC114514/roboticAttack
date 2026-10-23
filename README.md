# RoboticAttack：腕部视角脆弱性、DINO采样与空间关系补丁

本仓库整理基于 OpenVLA / OpenVLA-OFT、在本地 LIBERO 仿真中完成的视觉安全研究。研究从前视与腕部视角的不对称性出发，分析 DINO 分支的策略响应，再探索机制引导补丁、训练观测采样、受扰轨迹回放与空间关系动作目标。

**先读 [研究脉络](docs/RESEARCH_STORY.md)，再按下面四个入口找代码。** 历史实验的失败尝试、模型条件和后续纠正均保留，不将不同设置下的最好数字拼成同条件结论。

| 代码线 | 内容 | 入口文档 |
|---|---|---|
| Exploring 基线 | 原始 UADA/TMA/UPA；连续动作头与双视角 OFT 适配 | [Exploring](docs/methods/exploring.md) |
| WHEN / UPA-RFAS | 特征、注意力、语义目标及鲁棒优化；源模型训练与 OFT 部署 | [WHEN](docs/methods/when.md) |
| DINO 采样与回放 | 评分影响训练观测选择，动作攻击损失仍为 UADA 式目标 | [DINO sampling](docs/methods/dino_sampling.md) |
| 空间关系攻击 | 按目标—末端方向构造错误平移命令，优化共享 RGB 补丁 | [Spatial relation](docs/methods/spatial_relation.md) |

## 如何阅读与使用

1. [安装、模型和数据](docs/SETUP_AND_DATA.md)：先区分 VLA、补丁、套件和 checkpoint。
2. [实验索引](docs/EXPERIMENT_INDEX.md)：方法、来源、状态及结果一一对应。
3. [结果索引](results/README.md)：分清任务成功率与 CASR，区分 Goal 原生模型和联合模型。
4. [失败路线](archive/negative_results/README.md)：为什么从直接 DINO 目标转向采样与关系动作目标。
5. [发布与复现边界](docs/REPRODUCIBILITY.md)：源码快照、外部数据及固定哈希的处理。

```text
VLAAttacker/                  原始基线、OFT适配、WHEN和DINO训练实现
  spatial_relation/           可独立导入的关系目标与公平对照数学实现
scripts/                      数据准备、训练及评测Python入口
configs/                      原实验配置；路径和哈希属于历史环境
research_tools/spatial_followup/  关系对应/置乱对照
research_runs/                精选历史源码依赖；不包含缓存、日志、优化器
prismatic/                   OpenVLA模型与数据模块
third_party/openvla-oft/      OFT模型与LIBERO接口的必要源码
src/hooks/                    因果捕获/替换工具
results/                      精选汇总表、最终PNG和来源
archive/                      历史方案、机制证据、失败路线及早期核心代码
docs/                         当前导航与研究说明
```

## 本次发布包含什么

这是研究源码与证据归档。四条方法的核心代码、必要历史依赖、配置、关键对照和结果均集中到本仓库。历史运行目录保留相对结构，以便追溯动态导入；首页不以几十个实验版本作为使用入口。

模型权重、LIBERO/RLDS数据、数GB的候选bank、特征缓存、优化器状态和视频不入Git。复现历史整链还需要恢复对应外部资产及来源信息；修改路径后的新实验应建立新清单，不能沿用旧完成收据。

发布检查（不加载模型、不运行仿真）：

```bash
python scripts/release/check_release.py
python -m unittest discover -s tests_release -v
```

本次上传没有启动训练或Slurm作业。GPU实验仍由用户手动提交，资源参考训练16G、评测12G、每worker 4CPU和1张GPU，并实测RSS/CUDA峰值。

## 来源

Exploring 原始代码来自 William-wAng618/roboticAttack；模型基础为 OpenVLA / OpenVLA-OFT；WHEN对应UPA-RFAS代码线。上游代码与本地改动的边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。逐文件原始来源及 SHA256 见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。
