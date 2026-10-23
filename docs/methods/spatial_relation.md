# 空间关系条件化动作目标

## 方法

以固定交互实体为参考，令u为末端指向目标的单位向量。将clean策略的平移控制目标c投影到有效动作盒与u·v≤−m的交集中，m为预定的反接近命令幅度。目标是阻碍正常接近；这里的米是控制器目标增量，不是机械臂实际移动距离。

真实PNG经完整原生DINO、SigLIP和动作头，损失为实际平移控制目标与构造目标的归一化MSE。模型冻结，只优化50×50腕部RGB，固定[26,137]，无EOT。数学实现可直接查看VLAAttacker/spatial_relation/targets.py。

## 对照

- matched_targets.py：按其他任务方向旋转扰动，正确/置乱两组共同缩放以保持逐槽幅度一致。
- action_baselines.py：以clean策略归一化动作为参考的XYZ/全动作UADA适配。
- targets.py的far_target：物理平移命令的远端点基线；不是移动场景目标或让它离开相机。

关系方法通过双视觉分支回传。早期DINO几何机制是提出关系目标的依据；当前损失不应写成DINO专属特征攻击。

## 数据和源码入口

固定Goal checkpoint；四套件24训练任务，接触前237条拟合观测、217条有效目标，另有跨轨迹验证。每种方法两个种子、4000更新。同种子初始化与样本顺序匹配。

原训练：research_runs/wrist_spatial_approach_pilot_v1/code/中的sr_geometry、sr_math、sr_engine、sr_train、sr_cache。
正确/置乱对照：research_tools/spatial_followup/；修复代码：research_runs/wrist_spatial_relation_shuffle_v1_fix1/code/。
公平基线：research_runs/wrist_spatial_relation_fair_v1/code/。
评测：wrist_spatial_relation_shuffle_eval_v1与wrist_joint_checkpoint_patch_retest_v1。

这些历史worker通过签名固定依赖链，故源码依赖保留原相对路径。运行整链需要原bank、模型及收据，本发布仅进行CPU目标数学核验，不用删除检查来绕过资产缺失。

## 结果

Goal模型原40起点中clean成功39；正确关系对照的两个种子CASR为61.54%/92.31%，置乱为33.33%/46.15%。联合OFT固定旧PNG迁移后，正确关系四套件均值为30%/37.5%，全动作基线37.5%/42.5%。不同PNG版本及模型分别列于[结果索引](../../results/README.md)。
