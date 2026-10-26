# OFT Wrist-DINO 三种新目标：250轮低成本筛选

日期：2026-10-13，Asia/Shanghai。用户批准实现到手动提交命令；助手不自动提交、取消作业或使用重置卡。仅有权限的本地LIBERO仿真，不涉及实机操作。

## 1. 当前证据与新目标

v1生成器修复及400回合已完成。250轮UADA/Key/center/生成器的Object、Spatial、Goal、Long CASR分别为5/25/5/85、5/20/40/85、5/15/30/65、5/10/45/90%。保留Key和生成器，center不扩轮次。新三目标优先查找Object/Spatial多任务增益，保留Goal已有进展，Long来源内完整测试但不据其选胜者。

这是两个新适配目标和一个明确重做的注意力目标，不是三条已经证明独立的因果机制。只改训练目标，不继续采样，不叠加UADA/TMA、教师、任务区/背景区或外观门。

## 2. 可执行目标

### A：dino_attention，纯DINO注意力误导（重做）

读取实际DINO零基block8/16的原生Q/K，保留q_norm、k_norm、scale。query只用补丁影响范围之外的空间tokens；key分母保留完整CLS、4register及256空间tokens，目标key仅用补丁范围内的空间tokens。

最小化`logsumexp(全部QK logits)-logsumexp(补丁key logits)`，即负log注意力质量；多头、query、层等权平均。FP32 logsumexp避免下溢，记录真实patch-attention mass。无动作敏感query筛选，无10%辅助梯度，无动作目标。

若合法EOT使补丁完全离开有效画面，保持该变换和optimizer计数，不重采样；该步注意力目标记为连通的零贡献，显式记录可见token和active标志。不会把无补丁key当作方法负结果或让整批中止。

旧TMA注意力版是动作主目标+10% QK辅助，只挑动作敏感query；本轮注意力为唯一目标，查询覆盖所有未被补丁影响的空间token。Attention-Fool提供点积注意力攻击依据；竞争性log-mass、多query/DINOv2/静态通用图案是明确适配，不称完整复现。

### B：dino_multilayer，多层强弱响应反置（FDA-inspired）

读取零基block8/16/22原生空间输出。每源窗口干净前向一次并detach；每token按clean响应绝对值排序，前半通道为strong、后半为weak，划分冻结。保留符号，不把signed DINO当CNN ReLU。

每inner排除补丁影响tokens，各层最小化`log(RMS(z_adv,strong)+eps)-log(RMS(z_adv,weak)+eps)`，三层等权。RMS用固定clean层RMS尺度归一化，eps=1e-6仅为数值稳定项。压低原强响应、抬高原弱响应；不是末层整体余弦，不事后挑层。记录分层strong/weak RMS及原强响应占比。

这是FDA思想对signed DINOv2/局部通用图案的适配；原论文并未证明此设置能导致机器人失败。

### C：dino_affinity，区域相似关系破坏（新假设）

读取OFT实际消费的256 dense tokens并逐token L2归一化，clean detach。每次EOT排除补丁影响tokens和自配对后，用clean余弦关系确定各query的8个最高相似、8个最低相似邻居。

最小化`mean(S_adv,clean-high)-mean(S_adv,clean-low)`，削弱原高相似、增强原低相似关系。topk8是固定筛选设置，不是论文证明的最优值。不利用范数作弊，不用模拟器物体mask，不编辑物体，不指定固定右移，不构造几何教师。记录高低相似差值和反序比例；最终成败看闭环。

依据为DINO密集描述子的局部语义/对应能力；对其关系反序影响OFT是本项目新假设，不冒称已有机器人攻击成果。

## 3. 不变的数据、训练和权限边界

- 只读复用v1官方Long train缓存和同250条记录顺序。379原始专家演示、标签不改，测试不进训练。
- 三组seed42同一随机3×50×50 RGB、空AdamW从零开始；不加载旧学习补丁。DINO/SigLIP/动作头/本体投影冻结，只优化7500外部RGB参数。
- B1，250outer×50inner=12500更新；lr0.002、wd0、warmup20outer、cosine总周期2000，250时不降零。无TV/HF、倍增梯度、投影或额外颜色扰动。
- 所有inner作者随机位置、旋转±30、剪切±0.2、20%恒等矩阵；224图、50×50图案，真实uint8/PIL前向、显式STE反向，Front不变。
- 复用相同JPEG/crop保守token footprint，排除覆盖token只限制目标，不改变贴图或遮挡预算。
- 保存0/125/250 PNG/PT，每10outer原子保存RGB、Adam、scheduler、Python/NumPy/torch/CUDA RNG和训练表。中断只重做未提交尾轮。
- 训练hook只捕获，不替换输出；评测不加载训练器、不运行hook或生成器。

## 4. 复用与新增闭环

只新增4suite×10task×init40/41×3图案=240回合。每task一个4090作业，四suite独立子进程释放EGL/CUDA；保持已跑通的cuDNN/EGL预加载/映射/清理。正常OFT8步部署、位置(26,137)，Front/proprio/语言、限步和官方动作后处理不变。

只读复用v1的clean、UADA250、Key250、生成器250共320回合；不混入center或旧1000/2000/官方PNG。核对原签名、模型配置/权重stat、部署函数和完整模型运行时副本、初态/seed、PNG、raw/traces和修复运行时。漂移明确中止，不直接合并不匹配结果；旧行保留原signature/source_training_signature并注明source_run/reused。

合并560回合：clean+6方法。逐suite SR/FR、clean-success任务等权CASR、正负/持平task数。三新方法分别对比UADA、Key、生成器，不做跨suite平均。代理改善不当作闭环成功，代理未达标不禁止测试。两初态、单训练seed、已查看开发网格，仅作筛选；最多保留一两路后另行决定扩轮/新初态/载体迁移，不自动执行。

## 5. 来源、资源、时间

沿用Exploring派生副本训练结构、AdamW/scheduler和原几何，不修改原源码、v1、模型、数据或旧结果。逐字复制v1的engine、preprocessing、source_transform、model_runtime、readonly_model、deployment、objectives和UADA_SOURCE_COPY；新目标在new_objectives.py，新训练分支在train.py。evaluate.py保留原episode函数，只改为三个新增PNG。来源差异与hash见SOURCE_PROVENANCE.md和CPU_PREFLIGHT.json。

CPU复用准备→3×4090训练→CPU检查→最多4×4090自动测试→CPU统计；GPU阶段afterok串行，提交时等待本人其他GPU作业、不取消它们。没有独立GPU smoke，只做CPU/toy及正式训练内一次接口检查；正式任务仅用户执行脚本时提交。

CPU复用/检查5–10min、三组并行训练0.5–1h、240回合测试1–2h；预留约2–3h另计排队，新目标吞吐尚无GPU实测。

## 6. 参考与明确适配

- Attention-Fool / Give Me Your Attention，CVPR2022：https://openaccess.thecvf.com/content/CVPR2022/papers/Lovisotto_Give_Me_Your_Attention_Dot-Product_Attention_Considered_Harmful_for_Adversarial_CVPR_2022_paper.pdf
- FDA，ICCV2019：https://arxiv.org/abs/1909.04385 ，作者代码 https://github.com/BardOfCodes/fda
- Deep ViT Features as Dense Visual Descriptors：https://arxiv.org/abs/2112.05814

本轮是参考思想后的适配/假设，不承诺物理或机器人迁移成功率，不声称完整原论文复现。授权到实现、检查、手动命令，不自动提交GPU。
