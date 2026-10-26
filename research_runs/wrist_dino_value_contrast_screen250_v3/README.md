# 最后一批低成本筛选：DINO Value + 生成器InfoNCE

三张新图案从零250轮：`dino_value`、`advencoder_cos_b8`、`advencoder_nce_b8`。先训练，自动核验PNG，然后正常OFT四套件测试和统计。旧UADA/Key/G/关系与clean只读复用，不重跑。

预计运行2–4小时加排队：准备5–10min；最慢B8训练1.5–3h；四套件测试30–45min。训练≤3张4090，随后测试≤4张4090，阶段不重叠。B8训练尚未实测，预计时间不是承诺。资源多不等于强制占满；每作业仅一张卡，完成就释放。

## 手动提交（完整自动训练→测试→统计）

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_value_contrast_screen250_v3.sh
```

仅检查不提交：加`--preflight`或`--dry-run`。助手不自动sbatch，不使用重置卡，不取消旧作业。本人已有可运行GPU任务存在时，链式任务等待其结束；永久失效依赖只报告不等待、不取消。相同批次还在运行时拒绝重复提交，失败/中断后再次用同命令恢复，完成的训练/测试分片跳过。测试旧指标、模型/权重stat、原数据、随机初态、正常部署及PNG hash漂移时会停止合并。

## 结果集中在本目录

- `training/<方法>/outer_0250.png`：最终可查看/部署图案；0/125/250都保存。
- `training/<方法>/training.parquet`：loss、真实RGB移动、更新/样本计数、梯度缓存重算误差和用时。
- `training/<方法>/resume.pt`：完整Adam/scheduler或生成器BN buffers、固定noise与RNG。
- `evaluation/<套件>/task_00..09_*`：按任务分片，四套件都测。
- `results/RESULTS.md`：逐套件CASR与差值主表，不做跨套件平均。
- `results/VALUE_CONTRAST250_{SUMMARY,CONTRASTS,PER_TASK}.csv`与合并raw/trace Parquet。
- `reports/SUBMISSION_RECEIPT.json`：用户实际提交后的job IDs；`logs/`同目录。

只新增240回合；复用400回合且保留v1/v2各自来源签名，合并640。每方法每套件10任务×两个开发初态，用于筛选而不是最终规模结论。

## 看结果时的比较

Value对同轮次旧UADA/Key/生成器/关系。InfoNCE优先对同B8生成器cos，以隔离对比目标；B8相对旧B1有更多样本/前向，不能将全部收益写成方法或等成本提升。生成器latent batch固定1输出同一张通用图案，训练图像batch是8；测试不运行生成器。

只优化外部RGB或外部生成器；Front/语言/proprio、模型参数、官方Long数据/动作标签、旧结果和原只读Exploring源码不改。原作者随机贴位/旋转/剪切不变，没有新增TV、亮度/对比度或外观门。生成器保留原高频项，不承诺降噪。

本批完成后停止扩展路线；根据Object/Spatial/Goal分别选至多两条。Long完整保留，但属训练来源内。详细合同与来源适配见[PLAN.md](PLAN.md)、[SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)，实现核验见`reports/IMPLEMENTATION_REVIEW.md`。
