# Spatial动作定向Key：分层诊断＋方向对照

用户已选择Spatial、分层诊断加方向对照、两张最终补丁各四套件80回合。
只在授权冻结OFT与本地受控LIBERO运行。助手实现及CPU核验，用户手动提交；不自动提交、取消、扩轮或追加方法。

## 固定算法

- 输入仅旧Spatial250观测池，保留其中150条含专家闭合的观测（1077槽）。不读中心参考1024条作为训练数据。先由原生Spatial模型取专家和clean均g<0.5的槽位；g=0.5不归入闭合或错误张开。
- 按task0..9轮转、每task保留旧银行原顺序，取最多20条训练观测。再取最多20条留出观测，排除所有训练episode；不额外采集、不按目标成败筛选。样本不足如实报告。留出不参与更新、门控或候选选择。
- 每观测只求一次干净观测下的Key目标：倒数第二层post-k_norm空间Key（CLS+4 registers以外），当前层Q/V、prefix、Front和SigLIP不修改。通过完整原生动作头优化Z=(K-K0)/每头clean Key RMS；Adam lr0.02，最多20步，每头Z RMS投影≤0.2，损失max_slot ReLU(0.6-g)+0.1 mean(Z²)。缓存动作损失最小且实际BF16 Key变化在RMS约束内的迭代，平局选变化小者。记录量化后的真实Key与动作。无改善保留零目标，部分改善保留监督；难样本始终保留在名单和分母。
- 两臂共享目标、seed42随机RGB初始化、空AdamW、观测顺序、EOT及预算。归一化变化T=(Ktarget-K0)/RMS，X=(KRGB-K0)/RMS；固定token/head权重w正比于mean_channel(T²)。点匹配为加权MSE(X,T)；方向匹配为ReLU(1-<X,T>_w/<T,T>_w)²，不惩罚正交变化。RGB梯度只来自对应Key损失。
- 每候选预览一次AdamW，按1、1/2、1/4缩放未截断proposal delta后clamp。固定同一观测/EOT，用无Key替换的完整原生RGB前向检查：对应Key损失严格下降、最差双闭合槽位张开hinge不增加才接受。三次均拒绝则像素和全部Adam状态回退。门控仅是当前观测/EOT局部条件，非全池单调保证。
- 原生uint8/PIL前向和已有显式STE反向保留。方向臂改变监督及对应下降门槛；比较完整候选策略，不单独宣称某一个组件的因果贡献。

## 分层预算与停止

1. 最多40观测各20步Key求解，含留出诊断，最多800步；若训练集合没有非零有效目标则停止，不提交评测。
2. 训练名单预先取最多4条，每观测两臂各200候选；固定部署位置、恒定RGB lr0.002，用于单观测像素可行性，不作通用补丁结果。各自重新seed42初始化，不向通用训练传递补丁或Adam。
3. 两臂各40外轮×50候选=2000，训练名单循环；RGB lr0.002、warmup20外轮、cosine2000外轮、旧原随机几何EOT。调度按候选外轮推进，接受/拒绝/无目标分别计数。总RGB候选最多5600，非正式500轮实验。每50候选保存断点，失败attempt保留。
4. 初始及最终PNG对训练和留出观测各作固定部署位置和4个固定EOT种子42..45检查；不据此挑checkpoint。只评最终两张通用PNG。两臂都未改变初始化PNG则停止；相同PNG只测一次并显式共享。
5. 用户另行手动提交评测：四套件各10任务、初态40/41，两臂共160新回合。正常OFT仅贴腕部RGB50，xy=(26,137)，Front/语言/proprio干净，绝无Key替换。复用已配对clean80、Spatial-UADA500 80、Long-UADA1000 80，合计400条件行。固定原reset seeds、原suite模型/统计/步数上限。工程异常中止，不计攻击胜利。

## 监控与验收

每候选保留梯度、Key和原生动作门控、步长尝试、RGB变化、量化变化、接受计数和实际Adam步数。Key目标保留每步RMS、动作和成败。
真实回合只用read-only head hook测原生raw动作；在同一实际状态额外预测clean，wrong-open只计clean<0.5且patch>0.5的实际执行槽位，并报告持续长度。保存接触、抬升持物、接触丢失及疑似掉落的仿真代理，不将代理当成真实因果。
逐suite CASR、等权均值、配对新增/丢失失败和task→state配对bootstrap10000次。主对照Spatial-UADA500，Long-UADA1000作强参考。单seed、开发初态、短训结果不等于正式500轮稳定收益。内部干预成效不等于RGB闭环效果。

CPU测试覆盖Key-only替换、阈值、方向正交自由度、RMS投影、Adam回退/缩步、断点记录、episode划分及执行槽位。原生GPU身份/动作接口、Key零替换、真实梯度、PNG与renderer核验在用户GPU任务内完成。CPU通过不等于GPU/效果通过。

## 资源与手动执行

CPU准备4CPU/4G（Spatial银行只读核查实测RSS HWM1.636GiB）；训练1x4090/4CPU/16G；CPU核验2CPU/4G；评测每worker1x4090/4CPU/12G、四suite分片最多并发2；统计2CPU/4G。每suite只加载一次模型，顺序评两PNG，不把两臂拆成8次模型加载。
训练和评测分两次用户手动提交，训练等待现有GPU任务；无自动评测、提交、取消或扩实验。CPU preflight/status/dry-run不查询调度器、不加载模型。
时间估计训练约1–2小时、评测约0.5–1.5小时加排队，尚无本流程GPU实测；时间上限训练3小时、评测每suite3小时不是ETA。记录process RSS HWM、CUDA allocated/reserved峰值、wall/CPU时间和失败attempt；另有time -v日志。SIGKILL导致缺失的Python成本不得伪称已完整计量。
