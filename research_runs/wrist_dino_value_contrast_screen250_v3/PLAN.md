# OFT Wrist-DINO最后补充筛选：Value与跨图像对比学习

日期：2026-10-13，Asia/Shanghai。用户授权继续实现到手动提交命令；助手不自动sbatch、不取消/删除旧作业、不使用重置卡。仅在有权限的本地LIBERO仿真训练外部图案，模型参数/原数据/专家标签不改。

## 1. 已完成证据与本批范围

v2三新目标及240新增回合完成，和320匹配旧回合合并。注意力、多层、关系目标Object/Spatial/Goal/Long CASR为0/10/5/40、0/15/40/80、5/35/35/80；同250轮自训UADA为5/25/5/85，Key为5/20/40/85，生成器cos为5/10/45/90。关系目标Spatial有3个任务改善、1个退步，Object仍未改善。本批只补两条未完整覆盖的方向及一个必要对照，共三张新图案，不重做采样、WHEN、几何教师或注意力扫描。

## 2. 三组与方法来源

### A：dino_value，内部Value表征扰动

只读实际DINO零基block8/16的原生QKV线性层输出，取V而不是Q/K或最后dense。移除CLS/四register，仅256空间位置；所有空间token都纳入，不增加覆盖排除或任务区域标注。干净V按每源记录冻结。每层最小化clean/attacked完整空间V网格余弦，两层等权；即最大化VEV-UAP使用的CosDist。不替换/重写V，不优化模型参数，只有RGB补丁经正常前向导致V变化。

VEV-UAP（原Doubly-UAP，arXiv:2412.08108v3，2026-06修订）针对CLIP/EVA/Qwen视觉塔中层Value，本批明确适配到OFT DINOv2、50×50局部静态图案、作者EOT；不是原论文全图Linf UAP的完整复现。固定8/16是与现有注意力候选对齐的筛选选择，不冒称已定位DINO最优层。

### B/C：advencoder_cos_b8 / advencoder_nce_b8

沿用成功v1的同一作者生成器结构、一个固定100维噪声及64→50输出，不把图像batch扩大错误地改变生成器的latent batch-collapse结构。两组同seed42从零、同初图、同8条记录/几何种子/优化器/质量与HFC项，唯一差异为特征目标：

- cos：八条clean/attacked dense网格分别展平、归一化，匹配余弦取平均。
- nce：clean为query，attacked为对应positive；8×8跨图像余弦logits，温度0.1、对角为匹配标签，最小化负InfoNCE。0.1为本批冻结筛选值，不冒称已核实作者utils默认实参；作者主入口有temperature参数0.5，但调用InfoNCE()时不传该参数。其他七张图像形成真实批内负样本；不是每个B1分别算NCE后累加，不是WHEN的空间token InfoNCE。
- 总损失均为`5*feature_loss + mean(RGB_MSE) - mean(HFC_MSE)`，径向高通半径8，与已实现v1相同。作者公开gan_pat_attack.py默认nce且其代码为负InfoNCE；本批B8是资源适配，不能称原作者B256全配置复现。保留作者高频项，本批不承诺低噪声外观。

B8完整对比矩阵通过两遍微批1梯度缓存精确反传：第一遍正常前向收集8个受扰表征；对完整batch计算损失与表征cotangent；第二遍相同RGB和EOT逐图重算，将表征cotangent和质量/HFC梯度累加到单张RGB；最后一次反传到生成器并一次Adam更新。第一遍使用独立requires-grad RGB leaf并立即释放每张图的encoder graph，保证两遍原生attention的grad-enabled dispatch一致，不保留八张计算图。模型eval，无随机dropout；不近似/丢掉跨样本项，记录重算前向误差并检查严重漂移。CPU/toy与直接B8反传逐项比较。这是显存实现，不是新增科学目标。

## 3. 数据、预算和公平性

只读复用v1官方Long专家缓存。A沿用原250条普通随机记录顺序；B/C使用同250记录池的顺序循环，每outer取8个不同uid（索引`8*(outer-1)+j`模250），不读测试观测或状态，不修改原缓存、不额外采集。源官方379轨迹不变。

三组250outer×50inner=12500优化器更新。A B1：12500个受扰样本更新，RGB lr0.002/warmup20/cosine总2000/wd0；B/C图像B8：100000个受扰样本更新、两遍计算，生成器Adam lr0.0002/betas0.5/0.999、固定lr。B/C对比同计算；B8与旧B1只可比较效果/总实测成本，不能把改善全归因于目标或宣称等计算。

全组Front干净、只改Wrist RGB50×50。全部inner作者随机贴位、旋转±30、剪切±0.2、20%恒等，无额外亮度/对比度、TV、频率预条件、外观淘汰或热启动。源uint8/PIL前向+STE反向不变。存0/125/250 PNG/PT，每10outer原子保存参数/生成器BN buffers/Adam/scheduler/RNG/history。最后完整断点已提交而status丢失时无模型恢复收据，不多训练。

## 4. 自动闭环与明确停止

新增3PNG×4suite×10task×初态40/41=240回合，固定正常OFT、原8步动作块/限步、Wrist位置(26,137)、原4090真实EGL流程；测试不加载目标hook、生成器或缓存梯度。旧v2中的clean/UADA250/Key250/G-cos-B1/关系250共400原始回合及trace只读复用，各行原signature/source_training_signature/source_run不改。合并640回合、7攻击方法+clean，全部来源与资产hash核对后合并。

逐suite CASR/SR/FR、各新组对旧UADA/Key/G/关系的配对差，以及nce_b8−cos_b8。重点看Object是否突破、Spatial是否超过目前关系35%、Goal是否保持40–45%的候选水平；Long训练来源内完整报告，不按跨suite平均排名。单seed、每task两个已查看开发初态仅筛选。完成本批后最多保留两条路线；不自动继续追加路线、2000轮、新测试初态或实机。

## 5. 资源、时间和来源

独立目录本目录；原Exploring源和v1/v2均只读，不重构原文件。8个正常runtime/EOT文件逐字复制v2，生成器及feature_objectives逐字复制v1；新增common/remaining_objectives/train，数据仅组织批次，正常evaluate逐字复制v2。

CPU准备→3×4090训练→CPU资产检查→最多4×4090自动评测→CPU统计，阶段afterok串行；等待本人已有可运行GPU任务，不取消其任务。没有单独GPU smoke；Python测试技能只用于目标梯度/跨样本耦合/恢复/正常部署与来源检查，真实接口一次检查嵌正式训练。

上一批3路B1训练8–10min，240回合合计约1.11GPU小时（4卡理想16.7min）；B8梯度缓存有额外前向，尚未实测。预留准备5–10min，最慢训练1.5–3h，测试30–45min，合计2–4h加排队。调度训练上限6h不等于预计训练时间。

来源：VEV-UAP https://arxiv.org/html/2412.08108v3；AdvEncoder https://arxiv.org/abs/2307.12280；作者训练代码 https://github.com/CGCL-codes/AdvEncoder/blob/master/gan_pat_attack.py。作者默认参数和本批必要适配分别记账，不把分类/VQA成功率当机器人成功率。

状态：方案已冻结；实现及CPU核验完成状态另见reports/IMPLEMENTATION_REVIEW.md。无新GPU结果，正式任务由用户执行提交入口。
