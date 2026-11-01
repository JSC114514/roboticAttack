# RoboticAttack：腕部视觉风险与DINO错误传播

研究双视角VLA在局部视觉补丁下的风险分布，以及视觉分支如何影响闭环任务。当前分支`research/wrist-dino-study`独立整理近期完成的仿真实验，原`main`保持历史版本。

**主线：视角配对发现腕部风险集中 → 同状态恢复定位传播路径 → 四格与参数消融检验机制利用 → 新起点确认与跨模型补充。**

[最新实验总入口](studies/wrist_dino/README.md) · [方法与数据](studies/wrist_dino/METHOD.md) · [完整研究脉络](docs/RESEARCH_STORY.md) · [复现说明](studies/wrist_dino/REPRODUCIBILITY.md)

## 已完成结果

| 实验范围 | 主要观察 |
|---|---|
| 联合OFT，40任务×10起点，三个种子 | 腕部单轴83.29% CASR，前视30.28%；单轴+DINO86.95%；七维基线87.37%，七维+DINO87.28% |
| 源OFT同状态恢复 | 对O03 seed42的65个新增失败，DINO恢复救回65个，SigLIP救回29个 |
| 调参后8任务、80起点确认，三个种子 | 每个种子80个起点；单轴+DINO100%，匹配单轴基线98.33%；这里合计是8任务×10初态=80物理起点 |
| 固定PNG跨模型 | Adapter有平均迁移增益；π0.5无一致增益，分支作用随模型/套件改变 |

CASR以同模型、同起点clean成功为分母。400起点默认表与80起点调参确认表分别报告。原生PNG、内部恢复与历史动作回放单列，余弦特征辅助不等同于指定空间关系控制。

## 代码与结果导航

- [联合协议与源码](studies/wrist_dino/README.md)：混合训练池、完整梯度、原生评测与统计。
- [四格消融](studies/wrist_dino/supplements/wrist_dino_factorial_v1/README.md)：单轴/七维×无/有DINO。
- [参数消融](studies/wrist_dino/supplements/wrist_dino_sensitivity_v1/README.md)：权重、学习率、长度、batch和冻结确认。
- [跨模型](studies/wrist_dino/supplements/wrist_dino_crosscheck_v1/README.md)：Adapter-Pro、π0.5、同状态分支恢复。
- [逐回合CSV](studies/wrist_dino/results/episodes/) · [最终PNG](studies/wrist_dino/patches/MANIFEST.json) · [默认正式图表](studies/wrist_dino/reports/figures/README.md)。

## 历史代码

[Exploring基线](docs/methods/exploring.md)、[WHEN](docs/methods/when.md)、[DINO采样](docs/methods/dino_sampling.md)、[早期空间关系目标](docs/methods/spatial_relation.md)、[可靠性适配](docs/methods/wrist_reliability.md)保留在既有目录。完整历史入口见[实验索引](docs/ALL_EXPERIMENTS.md)和[负结果归档](archive/negative_results/README.md)。

## CPU检查与上传

```bash
python scripts/release/check_release.py
python -m unittest discover -s tests_release -v
python studies/wrist_dino/summarize_results.py --study main --scope full
```

[本分支上传命令](docs/UPLOAD.md)。视频、权重、专家数据、特征缓存、优化器和集群提交脚本不包含在本次整理中。许可证与上游来源见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
