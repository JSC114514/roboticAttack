# 空间接近实验：失败过程审计

原训练和206回合已完成；此目录独立汇总，未修改历史结果或争用历史report.lock。

固定Goal模型在80个起点中仅18个clean成功，全部属于Goal；本结果不证明四套件通用。

视频回放保存的原生动作，不加载VLA；仅通过起点、逐步物理/接触、终止及历史关键帧核验的视频可进入配对展示。回放不是新增独立实验。

本轮视频范围：预先指定seed42，18起点×5条件。已核验90/90段，完成18/18个比较视频。

seed43保留全部历史统计，不以seed42视频替代重复性结果。

|条件|新增失败/18|CASR|

|---|---:|---:|

|initial_s42|0/18|0.00%|

|initial_s43|0/18|0.00%|

|radial_reverse_s42|13/18|72.22%|

|radial_reverse_s43|11/18|61.11%|

|translation_far_boundary_s42|14/18|77.78%|

|translation_far_boundary_s43|18/18|100.00%|

|uada500|11/18|61.11%|



## 配对差异与不确定性（按任务→起点重采样）

- radial_reverse_s42 − initial_s42：+72.22 pp，探索性95%区间[47.06, 94.44] pp。

- radial_reverse_s42 − uada500：+11.11 pp，探索性95%区间[-17.65, 41.18] pp。

- radial_reverse_s42 − translation_far_boundary_s42：-5.56 pp，探索性95%区间[-26.67, 12.50] pp。

- radial_reverse_s43 − initial_s43：+61.11 pp，探索性95%区间[33.33, 88.24] pp。

- radial_reverse_s43 − uada500：+0.00 pp，探索性95%区间[-40.00, 43.77] pp。

- radial_reverse_s43 − translation_far_boundary_s43：-38.89 pp，探索性95%区间[-66.67, -11.76] pp。



## 已保存的实际运动诊断

- clean：同一clean接触前、至多64步窗口，实际径向前进减少的中位数0.00 mm；失败阶段{'success': 18}。

- radial_reverse_s42：同一clean接触前、至多64步窗口，实际径向前进减少的中位数62.49 mm；失败阶段{'after_target_contact': 7, 'no_target_contact': 6, 'success': 5}。

- initial_s42：同一clean接触前、至多64步窗口，实际径向前进减少的中位数-3.16 mm；失败阶段{'success': 18}。

- uada500：同一clean接触前、至多64步窗口，实际径向前进减少的中位数48.73 mm；失败阶段{'no_target_contact': 7, 'success': 7, 'after_target_contact': 4}。

- translation_far_boundary_s42：同一clean接触前、至多64步窗口，实际径向前进减少的中位数75.37 mm；失败阶段{'after_target_contact': 8, 'no_target_contact': 6, 'success': 4}。



## 解释边界与下一步

径向运动是实际末端位移沿当时目标方向投影之和，不是直线净位移，也不是动作命令对应的保证移动距离。

FOV/补丁遮挡使用原生腕部采样时刻的实例分割，类别掩码最近邻映射到224并按官方0.9面积中心裁剪采样。它是可见性估计，不是精确重建JPEG边界；初始帧未强制刷新传感器，故可见性缺失。

EVENT_TIMELINES.csv列出首次接触/抓持、首次在策略视野内不可见、首次叠加后完全不可见、仅因补丁覆盖而完全不可见及完整覆盖率。掩码为零可由遮挡或离开视野造成，不能直接断言相机转离目标；相机姿态另存于EVENTS.parquet供核对。不删除这些回合，不依据时间先后宣称因果。

RECORDED_DIAGNOSTICS.csv和视频同时检查平移、姿态、夹爪；固定实体中心不是精确抓取点，多阶段任务仅解释第一交互对象。

当前对照强度未匹配：远边界使用各平移维最远有效端点，关系法要求径向退离；失败率差异不能单独归因于是否使用空间信息。

先人工核对全部视频/事件，选择证据支持的距离、侧向对齐或姿态关系，再单独设计恢复该关系的救援验证。本入口不自动追加训练、救援或新checkpoint评测。
