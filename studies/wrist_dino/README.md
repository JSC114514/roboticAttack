# 腕部风险集中、DINO传播路径与补丁优化

本目录收录2026-10-26至10-31完成的联合OFT统一实验、动作维度×DINO消融、参数筛选与三种子确认、跨模型补充验证。所有模型冻结；仅优化共享50×50 RGB补丁。

**研究链条：视角配对发现风险不对称 → 同状态分支恢复定位传播路径 → 动作维度与DINO开关比较 → 参数消融 → 冻结配置的新起点确认 → 跨模型检查适用范围。**

## 按目的阅读

| 实验 | 目的 | 配置与结果 |
|---|---|---|
| 联合OFT主实验 | 统一模型、混合训练池、EOT和完整梯度，比较前视/腕部与动作目标 | [方法与数据](METHOD.md)、[默认主结果](FINAL_RESULTS.md)、[方法来源](METHOD_PROVENANCE.md) |
| 四格消融 | 区分单轴/七维与DINO辅助的作用 | [消融说明](supplements/wrist_dino_factorial_v1/README.md)、[结果](supplements/wrist_dino_factorial_v1/RESULTS.md) |
| 参数与确认 | 单独改变权重、学习率、长度或batch，给基线同样的调参机会 | [说明](supplements/wrist_dino_sensitivity_v1/README.md)、[参数表](supplements/wrist_dino_sensitivity_v1/reports/SCREEN_RESULTS.csv)、[确认](supplements/wrist_dino_sensitivity_v1/CONFIRMATION.md) |
| 跨模型 | 固定旧PNG，检查Adapter-Pro与π0.5的原生结果和分支恢复 | [说明](supplements/wrist_dino_crosscheck_v1/README.md)、[结果](supplements/wrist_dino_crosscheck_v1/SUMMARY.md)、[恢复](supplements/wrist_dino_crosscheck_v1/MECHANISM_RESULTS.md) |

## 三类结果分开阅读

| 范围 | 条件 | 核心结果 |
|---|---|---|
| 默认配置，40任务×10初态 | 三个固定训练种子；clean383/400 | 腕部X基线83.29% CASR，前视X30.28%；X+DINO86.95%；七维基线87.37%，七维+DINO87.28% |
| 调参后确认，8任务×初态20–29 | 三个训练种子重复同80物理起点；clean71/80 | X基线98.33%，X+DINO100%；七维基线97.50%，七维+DINO97.92% |
| 固定PNG跨模型，各80起点 | 两目标分别重跑自己的clean；每目标1280回合 | Adapter的X对比12.50→18.28%、七维14.18→20.76%；π0.5没有一致增益 |

CASR以同模型、同起点clean成功为分母，再按任务与套件等权；任务成功率统计全部起点。完整400起点包含24训练任务、8开发任务与8确认任务，不全部称为未见任务。DINO辅助降低表征相似性，没有指定某一种物体空间关系。基线是clean动作参考的UADA-OFT适配评估。

源OFT的O03 seed42分支恢复中，DINO救回65/65个原新增失败，SigLIP救回29/65；这属于同状态供体机制诊断。Adapter传播路径随套件变化，不能据此把DINO称为所有模型唯一的失败通道。

## 文件与复算

- [逐回合CSV](results/episodes/)：保留研究阶段和复用关系；同一起点的多个种子不是独立物理样本。
- [交互代理CSV](results/interaction/)：每回合接触、抓持和末端路径记录；部分来自封存命令回放。
- [PNG清单](patches/MANIFEST.json)：最终与指定长度快照的PNG按文件SHA去重。PNG文件SHA与RGB像素SHA是两个不同字段。
- [来源清单](provenance/RELEASE_SOURCES.json)、[导出统计](provenance/EXPORT_SUMMARY.json)：区分原运行源码、注释副本和导出结果。
- [复现说明](REPRODUCIBILITY.md)：直接Python入口、外部资产、父实验依赖及运行目录。

无需GPU即可复算：

```bash
python studies/wrist_dino/summarize_results.py --study main --scope full
python studies/wrist_dino/summarize_results.py --study sensitivity --scope holdout
python studies/wrist_dino/summarize_results.py --study crosscheck --scope transfer --model vla_adapter_pro
```

源码快照保留科学计算AST与冻结配置。运行数据、旧完整收据、模型、视频、优化器及集群排队脚本不随发布；不把精选结果文件冒作可跳过前置检查的运行收据。
