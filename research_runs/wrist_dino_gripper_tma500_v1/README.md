# 腕部区域DINO夹爪TMA：500轮单轮验证

Object版未运行，已按用户决定停用。当前请使用[Long版](../wrist_dino_gripper_tma500_long_v1/README.md)和`bash submit_exploring_oft_gripper_tma500_long_v1.sh`；以下仅为原准备记录，旧提交入口明确报错退出，不提交Object作业。

本地受控LIBERO；用户手动提交。主问题：正常应闭合/保持时，单腕部RGB补丁能否诱导错误张开；任务实体DINO区域是否有助于这种具体错误。详细冻结方案见[PLAN.md](PLAN.md)，参数见[config.json](config.json)。真实效果尚未测得。

## 提交与耗时

从项目根目录运行：

```bash
bash submit_exploring_oft_gripper_tma500_v1.sh --preflight
bash submit_exploring_oft_gripper_tma500_v1.sh
```

第一条只做CPU资产检查，不查队列、不提交。第二条由用户执行：检查自己的队列，等待已有GPU/OFT作业，提交有依赖的整链。标签最多3张A800，三组训练最多3张A800，训练后四套件测试最多5张4090；各阶段不重叠，不再依赖3090。A800沿用既有脚本的分区和`qos=a800`。若发现已有无效依赖或本轮作业仍在队列，停止并报告，不自动取消或重复提交。

估计标签1.5–3小时、训练3–5小时（三组并行，以较慢的B/C计）、测试3–5小时、CPU10–30分钟，总约8–14小时，排队及等待已有任务另算。旧4090双分支500约2.36小时仅作参考；新A800训练和5卡4090测试尚未实测。

先完成当前Object来源夹爪TMA。Long来源的原始UADA/均匀采样/DINO采样留作后续独立批次：B1，每方法从零到2000轮保存500/1000/2000，方法之间串行；配套Long状态池、模型和归一化统计需重新准备。本次没有创建或提交Long作业，详见PLAN的后续批次段。

## 三组与原方法差别

|组|训练目标|变化|
|---|---|---|
|A `plain_gripper_tma`|正常OFT预测gripper=1|Exploring TMA离散token CE适配为连续gripper MSE|
|B `task_region_dino_tma`|正常分支与任务区域CF分支都张开|最小化两者较大损失，保留物体+夹爪dense区域实际RGB扰动|
|C `background_region_dino_tma`|正常分支与等token背景CF分支都张开|与B等区域数、同RGB/EOT，换为背景，作为必要区域对照|

三组从相同seed42随机高维RGB开始，500×50=25000次Adam更新、B1、LR0.002、warmup20、cosine500。不是续训，不优化特征张量，没有TV/blur/梯度倍增。B/C额外前向与反向，因此不声称等计算或降低成本。TMA夹爪目标本身不是新颖性。

完整只读源快照在`source/`。`code/TMA_OFT.py`是目标损失、连续动作接口和状态池适配后的训练循环；`code/source_transform.py`是完整贴图副本；`code/engine.py`增加实际RGB/EOT的DINO区域CF。不是直接运行未适配的OpenVLA离散token训练器，也不是完整Exploring论文复现。原始TMA/贴图代码只读且原字节SHA256不变。

## 输出导航

```text
source/                         完整只读TMA、贴图、模型加载器快照
code/                           独立适配、标签、训练、正常评测、统计与提交逻辑
tests/                          CPU/mock检查；不代表科学效果
slurm/                          147–153七阶段及共同环境设置
logs/                           每作业/数组分片一对out/err
reports/PREPARATION.json         真实资产与代码冻结身份（用户提交后生成）
data/labels/task_00.pt ...       每任务30条标签/干净锚点，整轨迹可恢复
data/TRAINING_POOL.pt            三组共用池、close槽位、状态顺序
data/POOL.json                  可见性/close资格排除、任务覆盖
training/<method>/
  outer_0000/0100/0250/0500.pt   RGB快照（实际文件名如outer_0500.pt）
  outer_0000/0100/0250/0500.png  原始uint8补丁，无后处理
  TRAIN_HISTORY.parquet         RGB更新、开夹爪损失与比例、时间
  resume.pt                     Adam矩、scheduler、patch和outer事务
  status.json                   RUNNING/FAILED/COMPLETE
reports/AUTO_VALIDATION.json     训练/预算/PNG完整性核对；不是外观或效果PASS
evaluation/<suite>/task_*.parquet 正常模型、逐实际执行槽原始数据
evaluation/GRIPPER_TMA500_SUMMARY.csv    四suite与宏平均结果
evaluation/GRIPPER_TMA500_CONTRASTS.csv  B−A/B−C等配对差值
evaluation/STATUS.json           整批完成与停止状态
```

正式测试共2000回合：400 clean +1600受扰（旧UADA500参考及A/B/C）。Object/Spatial/Goal为主，Long完整但次要。该40–49网格已被多轮查看，是开发集复验。指标包括冻结clean-success CASR、新clean成功数、同决策状态wrong-open、接触/疑似掉落代理；代理不能当作实机掉落或已确认因果机制。没有可用测量的任务保留缺测。

500轮闭环即停；不自动扩到2000、不进入跨模型或实机。只有真实GPU接口通过、正常RGB闭环结果出现后才能判断方法是否有效。
