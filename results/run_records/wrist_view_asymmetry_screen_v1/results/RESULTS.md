# 腕部不对称性：两路线机制筛查

状态：COMPLETE；完成288/288，缺失0，无效0。

Goal任务0/1/3/5/6/7，初态42/43；最多64控制步。接触率不是完整任务CASR。

|条件|完整分支|首次接触|抓持|窗口内任务完成|

|---|---:|---:|---:|---:|

|clean|12/12|12|5|4|

|a_front_blur_raw|12/12|12|6|1|

|a_front_blur_attenuate_damaged|12/12|10|5|0|

|a_front_blur_attenuate_other|12/12|6|2|0|

|a_front_lag_raw|12/12|12|4|1|

|a_front_lag_attenuate_damaged|12/12|10|5|0|

|a_front_lag_attenuate_other|12/12|7|3|0|

|a_wrist_blur_raw|12/12|11|5|1|

|a_wrist_blur_attenuate_damaged|12/12|5|0|0|

|a_wrist_blur_attenuate_other|12/12|10|5|0|

|a_wrist_lag_raw|12/12|7|1|0|

|a_wrist_lag_attenuate_damaged|12/12|8|5|0|

|a_wrist_lag_attenuate_other|12/12|7|1|0|

|b_clean_replay|12/12|12|5|4|

|b_relation_native|12/12|8|3|0|

|b_relation_first_chunk|12/12|11|4|0|

|b_relation_replay|12/12|8|3|0|

|b_relation_random|12/12|8|2|0|

|b_relation_random_clean|12/12|11|5|0|

|b_uada7_native|12/12|3|0|0|

|b_uada7_first_chunk|12/12|11|5|0|

|b_uada7_replay|12/12|4|1|0|

|b_uada7_random|12/12|2|0|0|

|b_uada7_random_clean|12/12|11|4|0|



非攻击路线：参见WARNING.json中的预警分数、AUROC、按任务→起点区间和误报；不能将退化类型识别当作任务风险识别。

减弱方案按起点事后取最好，仅是固定候选集内的潜力，不是可部署恢复器，也不约束所有恢复方法的上限。

补丁路线：主要解释清洁相机回放相对随机相机变化、扣除各自无PNG条件后的差异。只有持续覆盖更强，不能证明腕部运动反馈。

相机供体提前终止时，两随机条件持有终止幅度；相机因果比较只使用有真实匹配幅度的共同前缀。

未训练新补丁、适配器或VLA。未预设高分门槛；结合效应、区间、任务重复性讨论继续方向，不自动追加任务。

探索性讨论建议：优先非攻击小模块设计。依据和区间见RECOMMENDATION.json；部分结果不能作为路线最终结论。
