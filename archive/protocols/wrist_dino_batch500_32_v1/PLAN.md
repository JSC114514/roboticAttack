# OFT Wrist-DINO：8方法 × B1/B2/B4/B8，32张从零500轮

2026-10-14，Asia/Shanghai。用户已确认32组规模及6卡4090训练、8卡4090测试。仅有权限的本地LIBERO；助手实现/CPU检查/交付手动命令，不自动提交、取消、重置卡或运行模型。当前方案优先于旧5卡上限。

## 冻结实验矩阵

|ID|方法|目标|B|
|---|---|---|---|
|1|uada_original|专家未来8步归一化X维的相反边界MSE|1/2/4/8|
|2|dino_key|倒数第二block真实空间Key的干净注意力加权CosDist|1/2/4/8|
|3|dino_center|实际dense空间网格减固定Long训练均值后的CosDist|1/2/4/8|
|4|advencoder_cos|作者生成器副本，5*cos + RGB_MSE − HFC_MSE|1/2/4/8|
|6|dino_multilayer|block8/16/22强弱响应signed RMS log-ratio|1/2/4/8|
|7|dino_affinity|排除覆盖/自配对，clean top8/bottom8空间相似关系反序|1/2/4/8|
|9|advencoder_nce|同生成器，5*negative InfoNCE + RGB_MSE − HFC_MSE|1/2/4/8|
|12|tma_xyz|未来8步归一化XYZ=(0,0,0)的MSE，非世界坐标原点|1/2/4/8|

1和12是本轮从零自训基线，不复用不同预算旧图。9-B1保留为明确退化对照：1×1 CE为0、DINO特征梯度为0，仅剩既有RGB/HFC项；不偷换cos或加负样本库。9-B2/4/8分别有1/3/7个批内负样本，B变化同时改变负样本数。4与9生成器latent B恒为1，最终每组只导出一张通用PNG。

## 共同数据、预算与参数

沿用feature_screen250_v1只读官方Long train专家窗口缓存（379条原始轨迹中冻结的250次随机窗口抽取）及专家未来8步标签；所有方法、B共用同一窗口表，B条连续循环取样。500次外循环每次对同批状态做50次独立位置/几何EOT内更新，共25000次optimizer.step。B1/2/4/8分别处理25000/50000/100000/200000次受扰样本，不是等图像次数/FLOPs。不称500 epochs，不改原数据/标签，不把目标套件或闭环回放放入训练。去中心化均值复用同编码器在1024条冻结Long train记录上估计的只读统计，不是旧补丁暖启动。

32组从seed42、空优化器重新初始化。像素组共享同一7500维随机RGB；生成器共享网络/固定100维噪声初始化，不能声称与像素组同初图。OFT/DINO/SigLIP/动作头/proprio参数冻结；优化器只含外部RGB或生成器。

保留Exploring随机合法位置、旋转±30度、剪切±0.2、20%恒等矩阵；恒等时位置仍随机。无固定位置比例、额外亮度/对比度、新TV/平滑/梯度倍率或外观淘汰门。基线仅连续动作OFT必要适配；六候选是DINO目标，不虚称叠加UADA/TMA的增强。

像素AdamW lr峰值0.002、weight_decay=0、warmup20外循环、余弦总周期2000，每外循环step一次；不在500提前降零。生成器Adam lr=0.0002、betas=(0.5,0.999)、固定LR、特征/RGB/HFC权重5/1/1、半径8；NCE温度0.1。B变化不调整LR或损失权重。

普通可分离目标逐样本反向、梯度除B累积，仅step一次；不会把B个单独step称为batch。跨图NCE用完整B×B负样本矩阵和精确两遍微批缓存，不留8个OFT图。每次更新记录梯度、实际RGB变化、loss、样本次数、LR和GPU时间。保存0/125/250/375/500 PNG/PT，每10轮保存完整RGB/生成器/Adam/scheduler/RNG断点；只评测最终500，不自动挑checkpoint。

## 一次自动闭环测试

4套件各10task×初态40–49=100回合，每状态clean+32个PNG，共13200回合。是已有开发网格，不包装成新盲测。配对同task/state/reset seed，条件内固定种子随机排序，clean每状态只跑一次。Long来源内；Object/Spatial/Goal是补丁跨suite迁移，模型采用现有逐suite OFT checkpoint及匹配归一化统计，不称跨模型。

正常OFT8步动作块/既有步数上限，224图仅Wrist于(x26,y137)贴50×50。Front/语言/proprio不变；测试不加载训练hook、激活替换、损失或生成器。继承已修复的4090 EGL初始化与清理顺序，且首次生成器卷积前禁cuDNN。

输出逐suite SR/FR、clean-success条件task等权CASR、同B对UADA/TMA的差值、同方法对B1的差值、4-vs-9同B目标对比、每任务结果和GPU耗时；不跨suite平均。配对task→state bootstrap用10000 draws、seed20260907。不把一个训练seed的重复闭环回合当多个补丁训练复现，也不把代理loss改善说成闭环失败改善。

## 资源、恢复、目录与耗时

CPU复用核对→6个4090训练worker（每worker单GPU，最长先排，总32组）→CPU资产核对→10个task array最多8张4090，四suite独立子进程→CPU汇总。GPU阶段afterok串行，总峰值8。训练6对日志/测试10对日志；不额外GPU smoke。提交时等待用户其它可运行GPU作业结束，不取消其它工作。每method完整断点及每episode恢复；完整任务复核跳过。单组失败不丢其断点，训练worker尽量完成本worker其它组，最终非零退出阻止自动测试混入缺图。

所有新文件放research_runs/wrist_dino_batch500_32_v1。training/m<ID>_<method>_b<B>/outer_0500.png；results/RESULTS.md、BATCH500_SUMMARY.csv、BATCH500_CONTRASTS.csv、BATCH500_ROLLOUTS.parquet、BATCH500_TRACES.parquet、BATCH500_PER_TASK.csv；reports/SUBMISSION_RECEIPT.json；logs/。

按已有4090实测250轮/原TMA2000日志外推约81.8 GPUh。6worker最长优先理想约13.7h，训练预留16–22h；13200回合8卡测试6–10h，总22–32h，排队另计。基线B8时长尚未实测，不假装已有32组性能。此轮不追加候选，不自动扩1000/2000或跨模型/实机。

手动入口（本命令包括训练后自动测试和统计）：

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_dino_batch500_32_v1.sh
```

--preflight或--dry-run仅CPU输入核对，不查询Slurm、不提交GPU。使用同入口重提时完成组/回合复核跳过、未完成恢复断点；不覆盖旧实验目录。
