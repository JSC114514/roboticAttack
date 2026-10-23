# 实验索引

| 阶段 | 主要问题 | 代码/证据入口 | 状态 |
|---|---|---|---|
| 腕部/前视因果分析 | 额外动作差异通过什么路径传播 | archive/research_records/WRIST_EVIDENCE_CHAIN_FROZEN_20260909.md；src/hooks | 机制证据归档 |
| MGWP/RAMP/阶段条件/CID | 几何与方向约束能否增强闭环 | archive/negative_results/early_sources；历史主方案 | 源域部分有效，增强/迁移不稳定 |
| Exploring同源DINO增强 | 路径、特征、对象诱饵是否有增益 | VLAAttacker/white_patch/UADA_OFT_DINO*、OBJECT_DECOY | 负结果归档 |
| 位置/基础采样 | DINO能否选更有效的训练观测 | UADA_OFT_DINO_SAMPLING；plan_dino_guided_placement | 原版本已评测 |
| Jacobian/ILA/交替/ILPD | 能否改善方向与RGB实现 | 对应white_patch模块及configs | 无稳定主要套件增益 |
| 受扰回放/动态采样 | 补丁后访问状态是否缺失 | DINO_REPLAY、SAMPLING_CLEAR | 多版本来源分别保留 |
| 几何表示到补丁 | 完整匹配/教师/读出能否控制动作 | research_runs/wrist_dino_geometry_*；relation_readout | 多条负结果 |
| 关系条件动作目标 | 对应目标的错误方向是否有效 | wrist_spatial_approach_pilot；spatial_followup | 真PNG闭环已完成 |
| 正确/置乱与公平动作基线 | 对应关系与普通动作攻击比较 | shuffle_eval、relation_fair | 已完成，结果分组保存 |
| 联合模型固定PNG复测 | 模型覆盖变化后旧PNG如何迁移 | wrist_joint_checkpoint_patch_retest_v1 | 1040/1040回合完成 |

每条方法的详细设置由docs/methods中的四份说明导航；原路径与文件哈希由SOURCE_MANIFEST.json定位。
