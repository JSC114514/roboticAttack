# 联合OFT：修复位置与任务恢复

完成144/144；复用clean 16；初态10，初态11未使用。

|条件|完成|任务成功率|clean成功起点上的失败|首次接触|抓持|
|---|---:|---:|---:|---:|---:|
|clean|16/16|93.75%|0/15|16|11|
|historical_uada500__native|16/16|31.25%|10/15|12|6|
|historical_uada500__dino|16/16|87.50%|1/15|16|11|
|historical_uada500__siglip|16/16|75.00%|4/15|14|8|
|historical_uada500__both|16/16|93.75%|0/15|16|11|
|uada_clean_7dof_s42__native|16/16|56.25%|6/15|13|7|
|uada_clean_7dof_s42__dino|16/16|87.50%|1/15|16|11|
|uada_clean_7dof_s42__siglip|16/16|75.00%|4/15|15|10|
|uada_clean_7dof_s42__both|16/16|93.75%|0/15|16|11|

|补丁|恢复位置|补丁新增失败中的救回|原成功被损害|任务成功率差|
|---|---|---:|---:|---:|
|historical_uada500|dino|9/10|0|56.25%|
|historical_uada500|siglip|6/10|0|43.75%|
|historical_uada500|both|10/10|0|62.50%|
|uada_clean_7dof_s42|dino|5/6|0|31.25%|
|uada_clean_7dof_s42|siglip|2/6|0|18.75%|
|uada_clean_7dof_s42|both|6/6|0|37.50%|

原生PNG与特权恢复分开解释。干净供体均来自当前受损分支同一观测，恢复结果不是已部署防护成绩。
全部16起点保留，包括clean失败；缺失不计成功。区间按套件分层、任务→起点重采样，动作槽位不充当独立样本。
双分支恢复与同状态clean动作每次决策检查1e-5；整回合结果不一致数：0。
复用clean的计时属于历史成本；COSTS.csv仅汇总本轮worker成本，失败尝试也保留。
视频：research_videos/wrist_joint_feature_rescue_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/。
下一步：Discuss repair location; no training is scheduled
