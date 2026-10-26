# DINO几何动作教师：原生PNG长实验

状态：PARTIAL_OR_FAILED。已完成 292/640 回合；完整配对起点 36/80；离线验证 0/240。
固定Spatial模型，24任务训练240观测、跨轨迹验证240观测；闭环40任务初态0/1。
主结果预指定geometry_teacher_s42；seed43仅作重复性检查。未按验证或闭环成绩选择检查点。
几何教师只替换腕部DINO，训练匹配未来8槽平移动作；部署是原生双分支PNG。
clean_far_boundary是clean参考的同池远边界基线，不是原专家参考UADA；uada500是历史Spatial UADA500 PNG。
初始补丁两个种子分别配对。已查看的历史场景不称独立确认。动作命令偏差不能当机械臂位移。
四组是否全部完成24000次候选更新：False。
实际评测检查点：geometry_teacher_s42=20000次候选/19335次实际更新, clean_far_boundary_s42=20000次候选/19335次实际更新, geometry_teacher_s43=19000次候选/18366次实际更新, clean_far_boundary_s43=19000次候选/18366次实际更新

|套件|条件|失败切换/clean成功|CASR|超出初始补丁的失败|
|---|---|---:|---:|---:|
|all|initial_s42|2/14|14.3%|0|
|all|initial_s43|3/14|21.4%|0|
|all|uada500|7/14|50.0%|5|
|all|geometry_teacher_s42|2/14|14.3%|1|
|all|clean_far_boundary_s42|14/14|100.0%|12|
|all|geometry_teacher_s43|2/14|14.3%|1|
|all|clean_far_boundary_s43|14/14|100.0%|11|
|libero_spatial|initial_s42|2/14|14.3%|0|
|libero_spatial|initial_s43|3/14|21.4%|0|
|libero_spatial|uada500|7/14|50.0%|5|
|libero_spatial|geometry_teacher_s42|2/14|14.3%|1|
|libero_spatial|clean_far_boundary_s42|14/14|100.0%|12|
|libero_spatial|geometry_teacher_s43|2/14|14.3%|1|
|libero_spatial|clean_far_boundary_s43|14/14|100.0%|11|
|libero_object|initial_s42|0/0|NA|0|
|libero_object|initial_s43|0/0|NA|0|
|libero_object|uada500|0/0|NA|0|
|libero_object|geometry_teacher_s42|0/0|NA|0|
|libero_object|clean_far_boundary_s42|0/0|NA|0|
|libero_object|geometry_teacher_s43|0/0|NA|0|
|libero_object|clean_far_boundary_s43|0/0|NA|0|
|libero_goal|initial_s42|0/0|NA|0|
|libero_goal|initial_s43|0/0|NA|0|
|libero_goal|uada500|0/0|NA|0|
|libero_goal|geometry_teacher_s42|0/0|NA|0|
|libero_goal|clean_far_boundary_s42|0/0|NA|0|
|libero_goal|geometry_teacher_s43|0/0|NA|0|
|libero_goal|clean_far_boundary_s43|0/0|NA|0|
|libero_10|initial_s42|0/0|NA|0|
|libero_10|initial_s43|0/0|NA|0|
|libero_10|uada500|0/0|NA|0|
|libero_10|geometry_teacher_s42|0/0|NA|0|
|libero_10|clean_far_boundary_s42|0/0|NA|0|
|libero_10|geometry_teacher_s43|0/0|NA|0|
|libero_10|clean_far_boundary_s43|0/0|NA|0|

所有比率仅来自该表共同完整配对起点；缺失、工程失败和正在运行回合不算攻击成功。
预算、选定检查点、未完成组见STATUS.json；成本、30秒心跳及启动检查见reports。
