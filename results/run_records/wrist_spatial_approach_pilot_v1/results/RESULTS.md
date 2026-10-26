# 混合数据／冻结Goal：空间接近阻断

状态：部分完成；完成206回合；clean起点80/80，其中成功18。

数据覆盖四套件；主模型固定Goal。CASR仅以同起点clean成功、该条件已完成的配对为分母；缺失回合不计成功。

原Spatial UADA500是迁移部署参考；同池平移远边界是预算/梯度/DoF匹配对照，不冒称原专家参考UADA。

控制目标的5mm是OSC期望位移，不能写成实际末端退后5mm；实际运动见TRACE与STEPS。

本轮无DINO专用损失或分支限制，不据此声称DINO独有因果机制。



|套件|条件|完成配对/clean成功|失败|CASR|

|---|---|---:|---:|---:|

|all|initial_s42|18/18|0|0.0%|

|all|initial_s43|18/18|0|0.0%|

|all|uada500|18/18|11|61.1%|

|all|radial_reverse_s42|18/18|13|72.2%|

|all|translation_far_boundary_s42|18/18|14|77.8%|

|all|radial_reverse_s43|18/18|11|61.1%|

|all|translation_far_boundary_s43|18/18|18|100.0%|

|libero_spatial|initial_s42|0/0|0|—|

|libero_spatial|initial_s43|0/0|0|—|

|libero_spatial|uada500|0/0|0|—|

|libero_spatial|radial_reverse_s42|0/0|0|—|

|libero_spatial|translation_far_boundary_s42|0/0|0|—|

|libero_spatial|radial_reverse_s43|0/0|0|—|

|libero_spatial|translation_far_boundary_s43|0/0|0|—|

|libero_object|initial_s42|0/0|0|—|

|libero_object|initial_s43|0/0|0|—|

|libero_object|uada500|0/0|0|—|

|libero_object|radial_reverse_s42|0/0|0|—|

|libero_object|translation_far_boundary_s42|0/0|0|—|

|libero_object|radial_reverse_s43|0/0|0|—|

|libero_object|translation_far_boundary_s43|0/0|0|—|

|libero_goal|initial_s42|18/18|0|0.0%|

|libero_goal|initial_s43|18/18|0|0.0%|

|libero_goal|uada500|18/18|11|61.1%|

|libero_goal|radial_reverse_s42|18/18|13|72.2%|

|libero_goal|translation_far_boundary_s42|18/18|14|77.8%|

|libero_goal|radial_reverse_s43|18/18|11|61.1%|

|libero_goal|translation_far_boundary_s43|18/18|18|100.0%|

|libero_10|initial_s42|0/0|0|—|

|libero_10|initial_s43|0/0|0|—|

|libero_10|uada500|0/0|0|—|

|libero_10|radial_reverse_s42|0/0|0|—|

|libero_10|translation_far_boundary_s42|0/0|0|—|

|libero_10|radial_reverse_s43|0/0|0|—|

|libero_10|translation_far_boundary_s43|0/0|0|—|



## 配对差异（方法失败率减对照失败率）

{'treatment': 'radial_reverse_s42', 'comparator': 'initial_s42', 'paired_starts': 18, 'failure_rate_difference': 0.7222222222222222, 'task_start_bootstrap_95ci': None}

{'treatment': 'radial_reverse_s42', 'comparator': 'translation_far_boundary_s42', 'paired_starts': 18, 'failure_rate_difference': -0.05555555555555555, 'task_start_bootstrap_95ci': None}

{'treatment': 'radial_reverse_s42', 'comparator': 'uada500', 'paired_starts': 18, 'failure_rate_difference': 0.1111111111111111, 'task_start_bootstrap_95ci': None}

{'treatment': 'radial_reverse_s43', 'comparator': 'initial_s43', 'paired_starts': 18, 'failure_rate_difference': 0.6111111111111112, 'task_start_bootstrap_95ci': None}

{'treatment': 'radial_reverse_s43', 'comparator': 'translation_far_boundary_s43', 'paired_starts': 18, 'failure_rate_difference': -0.3888888888888889, 'task_start_bootstrap_95ci': None}

{'treatment': 'radial_reverse_s43', 'comparator': 'uada500', 'paired_starts': 18, 'failure_rate_difference': 0.0, 'task_start_bootstrap_95ci': None}



## 覆盖及失败

所有工程失败和重试成本见STATUS.json与reports/COSTS；有效训练样本和缺失原因见COVERAGE.json。

多个相邻帧/动作槽位不作为独立重复；两个种子均完整报告，不按测试表现挑种子。
