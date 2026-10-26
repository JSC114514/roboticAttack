# OFT Wrist DINO transfer 2000 — 2026-09-30

> 此文件为早期草案，不是提交入口。最终可执行版本、五组定义及资源链以
> [`../wrist_dino_transfer_v1/PLAN.md`](../wrist_dino_transfer_v1/PLAN.md) 和
> `submit_exploring_oft_dino_transfer2000_v1.sh` 为准；旧目录不写实验产物。

当前实施附件。用户手动提交；助手不提交、不使用重置卡、不覆盖旧结果。

## 目的和边界

最新DINO1000任务等权CASR(Object/Spatial/Goal/Long)=100/23.6/38.4/77.9%，均匀组100/23.7/29.9/81.0%，原始UADA1000为43.7/10.6/33.2/77.0%。重点改善Spatial/Goal，不再用饱和Object代替迁移进展。旧1000日程末LR近零，形状相似不等于零更新。

全部从随机RGB开始，2000轮cosine从一开始固定；不是续训旧1000补丁。四个新方法共享250轮前缀并计入2000预算，继承Adam/调度/RNG。原始Exploring→OFT UADA独立从头训练，保留随机位置、无TV、原数据顺序。

## 五组

|组|改变|直接对比|
|---|---|---|
|original_uada|原始OFT适配UADA|用户要求的实用基线，不是纯DINO消融|
|uniform|clean/三决策回放1:1，任务/阶段内均匀抽样|匹配回放对照|
|static|250轮DINO恢复评分，正分上半区2:1抽样|现有方法统一2000日程|
|dynamic|当前补丁在500/1000/1500轮结束后重新评分|避免排序过时，相对static|
|diverse|dynamic加训练补丁随机位置及光度变化|相对dynamic检验迁移增强|

五条训练轨迹，1000/2000各保存五张测试补丁；不是单独训练十次。seed42一组种子，筛选而非多seed结论。本轮为控制资源不加打乱对照，不能单凭获胜宣称DINO独占因果收益。

## 训练定义

只使用Object模型/300条源状态。250轮补丁在50条源初态采集新受扰轨迹；任务内优先仍成功轨迹，无幸存者用全部。固定三决策片段库，不读旧补丁和旧回放库。

DINO评分=max(0, 恢复干净Wrist-DINO后的反向动作边界MSE−受扰MSE)，在中心帧以当前量化RGB计算。dynamic/diverse在501/1001/1501轮开始前重评分并在任务/阶段内重排，正分上半区权重2，其他1。全零则记录退化均匀抽样，不偷偷改损失。

四新方法保留Exploring派生RGB AdamW、dim0反向动作边界MSE、50inner/outer、LR0.002、warmup20、TV hinge阈值0.4权重1；7500个RGB自由参数，模型冻结，不附加特征损失。前25inner固定腕部位置，后25inner affine EOT。diverse后25inner另随机合法位置，patch亮度和对比度各[0.8,1.2]；背景和Front不改写。测试所有方法同原位置(26,137)、原RGB、正常完整OFT。

WHEN依据：<https://arxiv.org/html/2511.21192v3>与<https://github.com/yuyi-sd/UPA-RFAS>。本轮借鉴随机位置/几何多样化思想并增加光度扰动，不是其特征L1/InfoNCE/min-max/PAD/PSM完整复现。范围为预注册工程选择。历史当地报告`docs/when_experiment_summary_20260618_20260624.md`的OFT单腕Spatial/Object/Goal ASR为15/16/22%，不能把其他模型/双视角高ASR当本设置承诺；其指标协议与当前CASR也不同。

记录每轮LR、动作loss、RGB梯度、50次更新前后MAE/变化比例、TV/饱和比例，重评分权重变化比例和成本。非法数值/断梯度停机；不为外观不同注入噪声。不承诺收敛图必须看起来不同或一定有效。

## 评估、资源、后续

自动训练→校验→测试→统计，不等人工外观审批，保存PNG供查看。1000/2000×5组×400状态=4000攻击回合，复用400 frozen clean；四套件10任务×状态40–49、同seed同位置配对。Object/Spatial/Goal分别报告，Spatial+Goal宏平均为迁移重点；Long全测并次要单列。这一网格已指导选法，是开发复验，不称未见最终集。报告CASR、SR、同轮次配对差与区间、成本；checkpoint不是独立训练重复。

A800训练，4090采集/测试。阶段afterok屏障保证单次链总GPU≤5：前缀1→采集先1分片真实EGL检查→剩余9分片%5→CPU合库→训练5→CPU校验→测试40分片%5→CPU统计。历史4090 EGL有过启动错误，不能把CPU预检称GPU已验证。新产物只写本实验根，旧产物不动。

非排队预计：前缀0.5–1h，采集准备10–30min，并行训练6–10h，5卡评估3–6h；合计约10–18h，A800实际速度待测。跨模型优先DINO编码器VLA，跨模型/实机本轮暂不实施。
