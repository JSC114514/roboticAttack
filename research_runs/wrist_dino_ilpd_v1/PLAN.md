# Wrist-DINO ILPD：训练时衰减中间扰动，测试只用RGB补丁

状态：设计已获用户批准，正在实现与CPU预检；未自动提交。唯一主方案第35节附件。仅LIBERO仿真研究。

## 1. 为什么改方法

交替训练136551与四套件测试136560/136561已完成。固定交替相对同预算UADA追加200，在Object/Spatial/Goal的任务等权CASR差值为+1.889/+2.889/−2.000个百分点，配对区间均跨零；Long提高+12.361个百分点，但为次要结果。滚动交替明显退化且用户认为噪声多，结束该候选，不扩轮次。这不是原DINO因果脆弱性假设被推翻，而是现有利用方法没有主套件增强证据。

本次不继续放大固定/滚动特征投影。借鉴ILPD（NeurIPS 2023，https://arxiv.org/abs/2304.13410；公式见https://arxiv.org/html/2304.13410v3#S3.SS2），在训练时让下游看到衰减后的中间扰动，同时仍优化原连续动作UADA目标。论文主要研究分类迁移攻击，本次为机器人通用RGB补丁改造，不声称论文已证明OFT有效。

## 2. 四个同预算条件

共同起点：冻结、用户接受的UADA500腕部RGB补丁。每组追加200外轮×50内步，累计700外轮/35000更新；同300个Object训练状态、seed、状态顺序和EOT。全部7500个RGB参数可训练，Front输入不贴补丁，模型参数冻结。

|数组|方法目录|训练前向|
|---|---|---|
|0|uada_bounded|原完整OFT前向|
|1|dino_gradscale_bounded|DINO输出数值不变，仅反向梯度乘0.5|
|2|dino_ilpd_bounded|腕部DINO输出变成Z0+0.5×(Zp−Z0)|
|3|siglip_ilpd_bounded|腕部SigLIP输出作同样衰减，DINO不改|

Z0来自同状态clean完整OFT前向并detach；Zp来自贴RGB补丁后的完整OFT前向。仅第二次（Wrist）编码器调用允许变换，第一次Front及另一个编码器保持原样。不归一化dense token，不排除局部token，不添加特征损失。主损失仍为8步连续动作第0维到clean动作相反分位边界的MSE；其余动作维不新增目标。只缩梯度的对照与ILPD都有0.5局部导数，但ILPD改变下游前向输入，二者在非线性模型中的动作梯度通常不同。

α=0.5为本次预先固定的单个候选，参考ILPD扰动衰减区间，不是最优系数。SigLIP组仅辨别通用中间扰动衰减是否同样有效，不是双摄像头攻击。

## 3. 外观与公平比较

沿用Exploring独立副本的RGB/EOT、AdamW(lr=0.002,wd=0)、20外轮warmup和200外轮cosine。四组均沿用TV超阈值惩罚，阈值为起点TV；每次更新后额外投影到逐像素、逐通道相对起点±16/255且[0,1]的盒内。该约束限制外观漂移，不是低维参数化，也不保证视觉上无噪声。半径是工程候选，不来自ILPD最优性结论。不加平滑、不改导出图。

历史纯UADA续训没有此硬约束，因此不能代替本次新同约束UADA对照。训练后人工看四张最终PNG；不自动认定外观通过，不自动接测试，不根据训练loss或Long挑中间checkpoint。冻结UADA2000仍是后续正式攻击基线，本次700轮仅为方法开发。

## 4. 实现与检查

新文件在源代码副本上实现；原UADA.py保持0444及冻结SHA256。复用已审核的ILA V2实际RGB输入、原Exploring EOT与完整OFT调用，不修改历史训练器或模型权重文件。临时输出hook只存在于训练前向，finally清理；测试不得带特征hook。

CPU/mock检查包括公式与detach、梯度缩放对照、非线性动作梯度差异、Front/另一编码器不变、错调用数量报错、异常hook清理、7500参数与投影、四组共同起点/状态/EOT/动作更新数、完整外轮断点恢复逐元素一致、配置/源码/已完成保护、CHECK_ONLY不提交。CPU不加载真实VLA。

用户提交后在GPU启动中一次完成真实像素/特征接口对齐、α=1前向与RGB梯度对照、α=0.5实际腕部输出公式、Front和非目标编码器不变及非零动作到RGB梯度；失败记录并停止，不自动放宽阈值。训练记录原始/混合特征扰动RMS、MSE、TV、RGB梯度、盒边界比例、最大RGB变化、调用次数和累计成本。检查点0/25/50/100/200；完整外轮每5轮保存状态、优化器、调度器、RNG，半轮失败回退重做；COMPLETE不覆盖。

## 5. 路径与后续闭环

统一根目录research_runs/wrist_dino_ilpd_v1/：training/<method>/保存PNG/PT/status/parquet/manifest/源码快照/断点，logs/保存Slurm日志，reports/保存预检记录，evaluation/预留正式测试。旧outputs目录不移动、不删除，导航另见research_runs/README.md及outputs/README_RESEARCH_INDEX.md。

本次只准备四组3090训练（最多4卡并行，单臂4CPU/10G，调度上限4小时）；所有GPU任务由用户提交。完成且外观获得用户接受后，再一次准备Object/Spatial/Goal/Long同初态配对闭环测试。主对比DINO-ILPD−同约束UADA，同时比较DINO-ILPD−梯度对照与DINO-ILPD−SigLIP-ILPD；Object/Spatial/Goal为主，Long完整但次要。最终以正常完整模型、只贴RGB下的CASR判断增强，训练时衰减模型的损失不能代替CASR。若只在代理或Long获益，结束本候选；主套件和外观兼顾后再讨论2000轮与独立验证，不追加诊断链。
