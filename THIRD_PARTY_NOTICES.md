# 上游代码与本地改动

- Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics：William-wAng618/roboticAttack。原始UADA/TMA/UPA、patch变换及数据加载器保留原文件；OFT适配与新增方法另设模块。
- OpenVLA：openvla/openvla。根LICENSE及prismatic来源保留；作者署名以原文件和许可证为准。
- OpenVLA-OFT：moojink/openvla-oft。必要模型、动作头、projector及LIBERO辅助代码位于third_party/openvla-oft，其LICENSE独立保留。
- WHEN / UPA-RFAS：feature_space.py及相关recipe来自项目原有WHEN代码线。本地训练入口、数字补丁/OFT部署和诊断工具分别保存；不将上游目标声明为本地创新。
- LIBERO：作为外部仿真与数据依赖，不随本仓库重新分发权重或数据集。

本地新增研究包括双视角适配、DINO诊断/采样/回放、空间关系目标、配对对照及实验监控。SOURCE_MANIFEST.json记录本次选取文件的工作区相对来源与SHA256。归档片段和原配置保留原署名及注释，本文件不替代各上游许可证。
