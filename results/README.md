# 结果索引与口径

| 目录 | 模型/评测条件 | 如何解读 |
|---|---|---|
| summary/wrist_spatial_relation_expand_v1 | Goal；10任务×初态2–5 | 较早径向关系PNG：29/39、30/39；不是后续matched同一PNG |
| summary/wrist_spatial_relation_shuffle_eval_v1 | Goal；相同40起点 | 正确/置乱最终4000次PNG：24/39对13/39；36/39对18/39 |
| summary/wrist_spatial_relation_fair_v1 | Goal；original40、expanded40分开 | 同池同预算XYZ/7DoF适配，不能与历史UADA500混称同条件 |
| summary/wrist_spatial_target_joint_v1_fix3 | Goal；40开发起点 | 三组完整几何匹配0/39，动作基线35/39 |
| summary/wrist_joint_checkpoint_patch_retest_v1 | 联合OFT；四套件40起点 | 22旧PNG迁移，26条件1040回合；clean40/40 |

每个目录保留原始CASR.csv、配对差异和报告（实际存在时）。比例列为0–1；任务成功率与CASR分别列出。原报告中的工作区视频链接与机器路径是来源信息，视频不随仓库发布。

部分summary配置包含原冻结路径与哈希，用于记录实验条件，不直接作为新机器配置。patches中的四张关系PNG按原文件哈希保存；JOINT_RETEST_ASSETS.json记录联合复测时各资产的来源、预算与哈希。

PPT“动态回放组合”表格的完整统计来源尚未在本发布中核验，因此未重新发布该表为正式结果。早期结果与限制可在archive/research_records中追溯。
