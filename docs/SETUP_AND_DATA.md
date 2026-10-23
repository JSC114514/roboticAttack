# 模型、数据与环境

## 四种概念分别记录

- VLA checkpoint：提供冻结视觉编码器、语言模型和动作头。
- 补丁训练池：模型所见的图像、语言、本体状态及参考动作，可以来自多个套件。
- 补丁：唯一被优化的50×50 RGB参数，部署为PNG。
- 闭环评测：在指定checkpoint和LIBERO初态下重新运行策略，不是离线动作误差。

四套件是Spatial、Object、Goal、Long（代码名libero_10）。混合池并不表示使用联合checkpoint；Goal策略在混合观测上训练补丁，也不等于重新混合训练VLA。

## 环境

本仓库保留OpenVLA的prismatic与OFT的third_party源码，两个版本按各自适配器加载，不应同时在sys.path中任意覆盖。原依赖约束见根pyproject.toml；历史OpenVLA栈包含torch2.2、transformers4.40.1、timm0.9.10。OFT及集群实际运行环境另见各历史配置，本发布没有升级依赖或承诺一套新版本覆盖所有历史分支。

建议在独立环境中先检查PyTorch/CUDA及LIBERO/robosuite，再按对应模型说明安装依赖；不要将本仓库可编辑安装覆盖正在运行的实验环境。CPU发布检查只需Python标准库，目标数学测试另需numpy。

## 模型资产

OFT套件模型分别对应moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}。联合模型为moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10，归档固定revision638918f3d1c2e43a39a8a20772bdb8b91835e4b7；使用其300000步动作头与本体投影器，并按当前套件选dataset_statistics中的归一化键。

WHEN源训练还需要对应OpenVLA checkpoint，不能拿OFT动作头替代原token训练接口。

## 数据

Exploring/WHEN原始路径使用LIBERO RLDS数据及专家动作。后续Object采样系列使用轨迹观测bank；空间关系系列使用四套件24任务、48条拟合轨迹中的237条接触前观测，其中217条有效目标用于训练；初态40/41拟合，42/43验证，按整条轨迹隔离。具体版本以各config和manifest为准。

共享bank记录通常包含suite/task/trajectory/state标识、双视角RGB、语言、proprio、动作和几何标签。仿真实体、相机时钟、原种子与逐步动作是标签对齐和重放依据。数据张量未上传；生成脚本与协议已保留。

## 迁移到新机器

将checkpoint、RLDS、LIBERO、bank和输出目录填入自己的配置。configs与research_runs中的原配置是冻结实验记录，可能有原机器绝对路径和资产SHA；它们不会在缺失数据时自动生成等价样本。新运行需要新签名和新输出，不能直接继承旧完成状态。详见[复现说明](REPRODUCIBILITY.md)。
