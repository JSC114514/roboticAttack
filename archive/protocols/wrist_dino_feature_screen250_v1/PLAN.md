# OFT DINO特征三路线 + 生成器：250轮闭环筛选

日期：2026-10-13，Asia/Shanghai。用户批准实现并交付手动提交入口；助手不sbatch、不取消现有任务、不使用重置卡。只在有权限的本地LIBERO进行。

当前运行修复：原生成器139714_3首次前向cuDNN初始化报错，原Key/dense/UADA均完成250。使用`repair_cudnn4090_v2/README.md`的新手动入口，仅提前应用原4090禁cuDNN策略、补生成器并替换失败array依赖；科学设置不改，前三组和数据复用，不重提原四路。

## 1. 冻结范围

四组一起做：`uada_original`（同预算连续动作UADA）、`dino_key`（dSVA-inspired稠密Key）、`dino_center`（FSAA-inspired去中心化dense）、`advencoder_cos`（作者公开cos分支的通用生成器）。停止DINO选样；四组共用普通随机官方Long演示顺序，没有回放、教师、区域划分、附加UADA损失或外观门槛。

仅优化外部RGB或外部生成器，OFT及其DINO/SigLIP/动作头/本体投影器均冻结。前三个像素参数组从seed42同一随机3×50×50开始；生成器从相同seed的随机网络、固定100维噪声开始，不能声称同一初图。模型、训练原始数据、专家标签、旧结果和跨模型四张2000冻结选择不改。

## 2. 方法合同与来源区别

- UADA：使用Exploring来源副本，专家未来8步归一化动作的X维相反边界MSE。
- Key：源OFT真实DINO的倒数第二block，原生q_norm/k_norm/scale；干净CLS到256空间token的平均多头注意力冻结为空间权重。最小化加权后整个Key网格的余弦；排除CLS/4register作为被破坏token。不是register吸引或LLM注意力引导。论文DINO、逐图生成扰动→本轮DINOv2、单张静态RGB，是明确适配；作者公开库当前未提供核心实现。
- Center：只读实际被OFT消费的256个空间dense token；1024条预先固定Long训练记录估计均值μ并冻结，最小化vec(Fadv−μ)与vec(Fclean−μ)余弦。统计数据只来自Long train，不来自测试任务。原论文逐图PGD→本轮静态通用图案/EOT，不称完整论文复现。
- Generator：保留作者DCGAN+batch-collapse生成器结构（B1）、固定噪声、cos分支，loss=5*cos+RGB_MSE−HFC_MSE，FFT低频圆半径8；学习率0.0002、Adam betas=(0.5,0.999)，无调度。网络输出64×64按可微双线性映射50×50后限制有效RGB；高频滤波为作者数学运算的设备安全实现。输出PNG后测试不运行生成器。原ResNet编码器→OFT真实DINOv2 dense网格，原固定贴位→本轮统一原作者位置/几何EOT；因此是适配而非原论文数字复现。使用cos是为了避免B1 InfoNCE负样本退化；保留高频攻击项，不称降噪。

参考：FSAA https://arxiv.org/abs/2503.03842 / https://github.com/BrianPulfer/fsaa；dSVA https://arxiv.org/abs/2506.21046；AdvEncoder https://arxiv.org/abs/2307.12280 / https://github.com/CGCL-codes/AdvEncoder。

## 3. 共同预算与数据

官方Long train 379专家轨迹；所有组250outer×50inner=12500次优化更新、B1，同一250条记录表。独立准备新缓存（含1024均值记录），不借用不匹配旧250基线，也不重跑旧2000实验。μ的额外只读前向单独记时。

像素组lr峰值0.002、warmup20outer、总余弦周期2000，250轮不提前降到零；生成器按其参数化采用作者0.0002固定lr。每次inner全部随机合法位置，原旋转±30度、剪切±0.2、20%恒等矩阵，恒等时位置仍随机；无新增亮度/对比度。保存0/125/250，正式筛选只测试250；每10outer保存RGB/网络/Adam/scheduler/RNG以恢复。外观不淘汰，数值有限和真实非零学习为工程检查。

## 4. 自动闭环评测

四套件各10task×init40/41（既有开发初态，不称新盲测），fresh clean + 四张最终PNG，共400唯一回合。同task/state沿用已有清洁重置seed，条件顺序随机。原生OFT8步执行、现有超时/4090 EGL流程保持；224图仅Wrist贴50×50于(x26,y137)，Front/语言/proprio不变。不用训练hook、表征替换或生成器推理，PNG与真实学习结果逐值核对。

逐suite输出SR/FR、clean-success任务等权CASR、三候选−UADA配对差和正向task数；两初态是探索筛选，不形成高精度显著性声明，不以跨suite平均排名。Long来源内，不靠近100%挑胜者。目标为Spatial/Goal增益跨至少3task；10pp是追加资源目标而非承诺或显著性门槛。至多保留两路，另行批准新初态、扩轮次、载体/跨机器人测试，不自动串联。

## 5. 资源和入口

CPU准备→四训练array%4（4090每路1卡）→CPU资产验证→10任务评测array%4（4090每task1卡，四suite子进程）→CPU汇总。两GPU阶段afterok串行，等待已有本人GPU作业结束，不改/取消它们。最多4卡，无独立GPU smoke，无自动提交。

预估：准备3–10分钟，最慢训练约1–1.5h，4卡小评测约1–2h，合计预留2–4h，排队另计；新特征/生成器速度未实测，以正式进度ETA更新。

用户命令：`cd /home/caijiasheng25/robopatch_defense/roboticAttack-main` 后 `bash submit_exploring_oft_dino_feature_screen250_v1.sh`。最终四套件表 `results/RESULTS.md`，PNG在 `training/<method>/outer_0250.png`。

## 6. 实现状态

独立实现已完成，27项CPU/toy检查通过（10.89秒），21份Python AST、JSON和7份shell语法通过；真实本地资产预检通过。源码与改动见SOURCE_PROVENANCE.md，工程证据见reports/IMPLEMENTATION_REVIEW.md和CPU_PREFLIGHT.json。尚未提交、未加载真实OFT权重、未执行GPU或仿真，也没有新的攻击结果。正式GPU接口检查嵌在真实训练内，不单独安排smoke。

通过本轮实验设计skill固定共同数据/预算/配对，Python测试skill覆盖实际反向与恢复，而不是用额外诊断取代闭环实验。原Exploring三文件0444及SHA不变；原Long四路签名保持，原模型metadata/hash及权重stat、原数据metadata/hash及shard stat与上一批预检相同。原AdvEncoder LICENSE受429/网络问题未取回，已明确保留来源说明；对外发布源码包前补齐原LICENSE，不伪称下载了原文件。
