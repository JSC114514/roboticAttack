# 同条件UADA适配对照
状态：COMPLETE；完成1040/1040；缺失0，无效0。
统一Goal模型、217条混合观测、相同初始化／采样、4,000更新、Adam、无EOT、50×50固定腕部PNG。
UADA对照为OFT连续动作头、clean参考适配版；不冒称专家参考原始UADA。XYZ主对照与关系方法维度匹配，7DoF补充完整报告。
物理命令空间远边界同池对照保留；历史Spatial UADA500只作条件不一致的背景，见原报告，不参加主比较。
原40、增加40及合并80分别报告，不按测试选种子。seed42为关系方法主结果，seed43检查重复性。

## original40
|条件|完成回合|任务成功率|诱发失败/clean成功配对|CASR|
|---|---:|---:|---:|---:|
|clean|40|97.50%|0/39|0.00%|
|initial_s42|40|97.50%|1/39|2.56%|
|initial_s43|40|97.50%|1/39|2.56%|
|matched_relation_s42|40|37.50%|24/39|61.54%|
|shuffled_relation_s42|40|65.00%|13/39|33.33%|
|matched_relation_s43|40|7.50%|36/39|92.31%|
|shuffled_relation_s43|40|55.00%|18/39|46.15%|
|translation_far_boundary_s42|40|12.50%|34/39|87.18%|
|translation_far_boundary_s43|40|2.50%|38/39|97.44%|
|uada_clean_xyz_s42|40|25.00%|29/39|74.36%|
|uada_clean_7dof_s42|40|2.50%|38/39|97.44%|
|uada_clean_xyz_s43|40|15.00%|33/39|84.62%|
|uada_clean_7dof_s43|40|20.00%|31/39|79.49%|

## expanded40
|条件|完成回合|任务成功率|诱发失败/clean成功配对|CASR|
|---|---:|---:|---:|---:|
|clean|40|97.50%|0/39|0.00%|
|initial_s42|40|95.00%|1/39|2.56%|
|initial_s43|40|97.50%|0/39|0.00%|
|matched_relation_s42|40|37.50%|24/39|61.54%|
|shuffled_relation_s42|40|65.00%|14/39|35.90%|
|matched_relation_s43|40|7.50%|36/39|92.31%|
|shuffled_relation_s43|40|45.00%|21/39|53.85%|
|translation_far_boundary_s42|40|12.50%|34/39|87.18%|
|translation_far_boundary_s43|40|0.00%|39/39|100.00%|
|uada_clean_xyz_s42|40|27.50%|28/39|71.79%|
|uada_clean_7dof_s42|40|5.00%|37/39|94.87%|
|uada_clean_xyz_s43|40|27.50%|28/39|71.79%|
|uada_clean_7dof_s43|40|12.50%|34/39|87.18%|

## all80
|条件|完成回合|任务成功率|诱发失败/clean成功配对|CASR|
|---|---:|---:|---:|---:|
|clean|80|97.50%|0/78|0.00%|
|initial_s42|80|96.25%|2/78|2.56%|
|initial_s43|80|97.50%|1/78|1.28%|
|matched_relation_s42|80|37.50%|48/78|61.54%|
|shuffled_relation_s42|80|65.00%|27/78|34.62%|
|matched_relation_s43|80|7.50%|72/78|92.31%|
|shuffled_relation_s43|80|50.00%|39/78|50.00%|
|translation_far_boundary_s42|80|12.50%|68/78|87.18%|
|translation_far_boundary_s43|80|1.25%|77/78|98.72%|
|uada_clean_xyz_s42|80|26.25%|57/78|73.08%|
|uada_clean_7dof_s42|80|3.75%|75/78|96.15%|
|uada_clean_xyz_s43|80|21.25%|61/78|78.21%|
|uada_clean_7dof_s43|80|16.25%|65/78|83.33%|

配对差异及任务→起点重采样区间见PAIRED_CONTRASTS.csv；多个槽位／视频帧不当独立样本。
过程指标使用前32控制步且首次接触前的有效步，分别记录实际末端径向推进、命令径向分量、目标移动、跟踪误差及接触力；它们是描述性证据，不等于因果恢复实验。
中心距离不是表面间距或精确抓取误差。可见性、接触、抓持和失败过程见EPISODES.csv及原始STEPS/EVENTS。
视频位于research_videos/wrist_spatial_relation_fair_v1_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/。
