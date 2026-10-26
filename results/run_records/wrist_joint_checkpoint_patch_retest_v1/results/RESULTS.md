# 联合OFT旧补丁复测

完成 1040/1040 回合；clean先完成40个起点。

表中CASR以该方法已完成、且clean成功的配对起点为分母；缺失回合单列。四套件均值按套件等权。

| 条件 | Spatial CASR | Object CASR | Goal CASR | Long CASR | 等权均值 | 完成 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 40/40 |
| matched_relation_s42 | 20.00% | 30.00% | 20.00% | 50.00% | 30.00% | 40/40 |
| shuffled_relation_s42 | 10.00% | 30.00% | 30.00% | 50.00% | 30.00% | 40/40 |
| translation_far_boundary_s42 | 10.00% | 30.00% | 30.00% | 60.00% | 32.50% | 40/40 |
| uada_clean_xyz_s42 | 10.00% | 20.00% | 20.00% | 40.00% | 22.50% | 40/40 |
| uada_clean_7dof_s42 | 20.00% | 40.00% | 30.00% | 60.00% | 37.50% | 40/40 |
| earlier_radial_s42 | 30.00% | 10.00% | 30.00% | 60.00% | 32.50% | 40/40 |
| matched_relation_s43 | 30.00% | 30.00% | 40.00% | 50.00% | 37.50% | 40/40 |
| shuffled_relation_s43 | 20.00% | 20.00% | 20.00% | 80.00% | 35.00% | 40/40 |
| translation_far_boundary_s43 | 30.00% | 20.00% | 50.00% | 70.00% | 42.50% | 40/40 |
| uada_clean_xyz_s43 | 30.00% | 40.00% | 50.00% | 80.00% | 50.00% | 40/40 |
| uada_clean_7dof_s43 | 20.00% | 40.00% | 30.00% | 80.00% | 42.50% | 40/40 |
| earlier_radial_s43 | 30.00% | 30.00% | 50.00% | 80.00% | 47.50% | 40/40 |
| geometry_select_joint | 0.00% | 10.00% | 10.00% | 0.00% | 5.00% | 40/40 |
| geometry_select_fixed_harm | 0.00% | 10.00% | 10.00% | 10.00% | 7.50% | 40/40 |
| geometry_select_realizability_only | 0.00% | 10.00% | 20.00% | 0.00% | 7.50% | 40/40 |
| geometry_select_translation_boundary | 20.00% | 40.00% | 20.00% | 30.00% | 27.50% | 40/40 |
| geometry_action_geometry_teacher_s42 | 10.00% | 10.00% | 10.00% | 20.00% | 12.50% | 40/40 |
| geometry_action_clean_far_boundary_s42 | 50.00% | 60.00% | 50.00% | 90.00% | 62.50% | 40/40 |
| geometry_action_geometry_teacher_s43 | 20.00% | 10.00% | 20.00% | 10.00% | 15.00% | 40/40 |
| geometry_action_clean_far_boundary_s43 | 40.00% | 60.00% | 60.00% | 90.00% | 62.50% | 40/40 |
| geometry_match_projection | 0.00% | 10.00% | 20.00% | 30.00% | 15.00% | 40/40 |
| geometry_match_full | 0.00% | 10.00% | 10.00% | 10.00% | 7.50% | 40/40 |
| initial_s42 | 10.00% | 10.00% | 10.00% | 40.00% | 17.50% | 40/40 |
| initial_s43 | 10.00% | 30.00% | 10.00% | 40.00% | 22.50% | 40/40 |
| historical_uada500 | 70.00% | 80.00% | 80.00% | 90.00% | 80.00% | 40/40 |

CASR.csv同时列出任务成功率、clean成功分母和任务→起点重采样区间；PAIRED_CONTRASTS.csv给出同起点差异。
本轮初态2，每任务只有一个起点，区间主要反映任务间变化。

MODEL_COMPARISON.csv仅比较来源一致、物理状态与双视角缓存一致的历史Goal起点；攻击差值使用两模型共同clean成功起点。

22张补丁按各自原训练池、更新次数、源模型解释。该实验测量固定旧PNG在联合模型上的迁移；重训优先级由配套方法与基线的差异决定。

视频：research_videos/wrist_joint_checkpoint_patch_retest_v1_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/。
接口错误、资源峰值与失败成本见reports/GATES、reports/COSTS；第二批初态3尚未提交。
