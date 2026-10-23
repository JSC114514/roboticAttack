# 官方Long四路从零2000轮：恢复Exploring原几何流程

2026-10-12用户批准。此文件是唯一主方案第46节的实施附件，覆盖旧八路1000的新执行入口；旧代码、源数据、模型、补丁及结果保留。助手仅实现/CPU检查并交付命令，不自动提交/取消Slurm、不使用重置卡。仅用户有权限的本地LIBERO仿真；本入口不启动跨模型或实机。

## 1. 四个方法

|编号|method|训练处理|同轮次比较|
|---|---|---|---|
|A1|uada_original|归一化专家动作决定X维相反边界，跨8步连续MSE|UADA-OFT自训基线|
|A2|uada_dino_freq|A1＋当前补丁DINO恢复贡献采样＋全秩频率步预条件|A2−A1，组合收益|
|B1|tma_xyz|归一化XYZ=(0,0,0)，跨8步连续MSE|TMA-OFT自训基线|
|B2|tma_xyz_dino|B1＋同目标DINO恢复贡献采样|B2−B1，采样收益|

不训练其它四种方法。无旧patch/教师/共享训练前缀、无区域或注意力/几何方向辅助loss、无额外uniform replay臂。原始基线本身需要随机读训练样本，不把这一数据读取另称新方法。A2的两项不能凭四组分离贡献。

## 2. 数据与动作目标保持

原始Long RLDS目录`/home/caijiasheng25/Desktop/roboticAttack/dataset/libero_10_no_noops/1.0.0`，32个TFRecord、379个官方train演示episode，完整有效8步窗口池。重新抽2000×4候选，data_seed=20261011；四组共用候选表，基线取每轮第一个候选，增强组四候选选一，有效batch=1。四候选不是四样本一起反传。只解码缓存预算所需窗口，不改原文件，不从仿真测试采训练数据，不用旧300状态或policy/replay标签。

输入是当前Front/Wrist、语言、8维proprio；未来8步专家动作只进损失，不进prompt或模型输入。保持官方gripper=1−clip(raw,0,1)、Long checkpoint q01/q99/mask归一化、JPEG/lanczos3/中心crop与native processor。UADA专家非负选−1、负选+1，mask=[0]；TMA固定归一化XYZ零，mask=[0,1,2]，不把归一化零称物理停止。其它动作维度不直接监督。

## 3. 从零和共同优化

- seed42随机3×50×50全RGB，7500变量；四组都空Adam开始，不读取任何旧学习补丁或优化器。
- 2000outer×50inner=100000 AdamW更新；B1；峰值LR0.002、wd0、预热20outer、cosine2000、每outer一次scheduler，非根据loss自适应、无提前停止/学习率重启。
- 保存outer 0/250/500/750/1000/1500/2000；每10outer保存完整RGB/初始图/Adam两矩与计数/scheduler/Python、NumPy、Torch、CUDA RNG/频率尺度/累计历史和时间。只续本新批同方法的完整断点。
- Front、语言、proprio和全部模型参数冻结；唯一优化变量是腕部RGB。保持只读checkpoint loader、native uint8/PIL前向和显式STE反向，不改共享模型或安装包。
- 无额外TV/HF辅助损失、平滑投影、低维参数化、颜色初始化、像素起点盒或人工外观门。记录真实RGB更新、PNG/PT一致性、相邻快照变化，不能把Adam计数或外观相似当作收敛证据。

## 4. 恢复作者随机位置和旋转/剪切

每一个inner均重新随机合法位置x,y∈[0,174]，随后调用作者原`combined_transform_matrix`：20%恒等矩阵；否则旋转[-30°,30°]和双轴剪切[-0.2,0.2]、矩阵S@R。调用未改`apply_affine_transform`：affine_grid/grid_sample、padding_mode=border、align_corners=False。恒等变换也不是固定贴图位置。

取消前25固定位置/后25随机安排，取消亮度/对比度扰动。无缩放、额外仿射、可见性重抽、预设锚点或新的变换比例。沿作者画布和<-20 mask规则合成图像；native RGB clamp/量化后走真实OFT processor，保持部署一致的STE适配。几何可以导致部分出画，不新增完整可见性门。

四组对同一outer/inner使用相同随机seed；差别只来自选中的训练状态/当前学习补丁，以及A2的频率处理。完整Exploring本地只读原源快照在`source/`；变换可运行副本`code/source_transform.py`完整复制，旋转/剪切/插值函数不改。CPU测试与原函数逐值比较位置、矩阵、合成像素和RGB梯度。

## 5. DINO采样与频率优化不另改方法

每outer只在四个官方train候选评分。当前量化补丁P下目标损失为Lp；恢复同状态clean Wrist-DINO、保持同一attacked Wrist-SigLIP时为Lrestore，s=max(0,Lrestore−Lp)。正分前半候选权重2、其余1，按固定seed取一个；全零正常退回等权。不让评分参与RGB反传，不使用目标suite或测试数据。

评分也使用本次原随机位置/几何EOT，四候选同seed配对，代替旧批固定(26,137)评分输入；这是本次恢复变换带来的明确改变，不隐瞒为完全只扩大轮次。训练每inner仍重新采不同EOT。仅评分临时恢复DINO，实际RGB训练/部署都正常完整模型。

A2保持上一批freq gain=(1+8*(fx²+fy²))^-0.75，全频率>0、7500 RGB自由度；首次非零Adam提议用clamp后L2步长对齐未预条件提议，之后固定尺度并保存，Adam原矩不替换。没有事后模糊PNG，B2不加入freq。

## 6. 原设置、必要适配与本次变化

|设置|来源/理由|本次处理|
|---|---|---|
|随机RGB、AdamW、20轮warmup、0.002峰值、50inner、2000outer、原几何函数|作者公开优化流程|沿用|
|连续8步MSE、双视角仅Wrist、native uint8/PIL+STE|OFT输出接口/已批准单腕威胁模型的适配|保持，不称原token损失复现|
|batch=1，作者示例batch=8|此前明确批准的时间/计算预算|保持B1；不是模型必需限制|
|25固定/25随机，额外brightness/contrast|旧批助手选择，非论文规定|按用户批准撤回|
|DINO采样、A2频率步预条件|本次选定机制增强处理|保持，采样评分EOT同步恢复|
|统一测试位置、10初态/任务、OFT超时预算与CASR|既有统一OFT比较协议，非原论文500回合/FR|保持，不擅自改变测试口径|

不在执行中擅自调整学习率/数据/攻击轴/辅助权重/随机种子。新增改动必须先明确原设置、具体问题、修改和预期影响。新1000是2000日程中点，旧1000是余弦终点；且EOT同时改变，因此不能将旧新差异单独归因为扩轮。

## 7. 自动闭环和统计

训练四路完成后自动测试1000、2000，不等人工看图，不开独立smoke。fresh clean＋4×2张PNG，O/S/G/Long各10任务×init40–49，共3600唯一回合。保存完整rollout/实际动作块、PNG SHA与来源，异常退出不当作科学失败率。正常推理无训练hook/DINO恢复/教师；每次只在Wrist原始RGB固定(26,137)贴50×50PNG，Front/状态不变，预测并执行8步动作块，成功或OFT超时停止。

初态/重置seed、条件内顺序随机化、suite checkpoint/归一化、settling10、Spatial/Object/Goal/Long最大220/280/300/520步保持旧协议。Long是来源内，其它三suite为补丁跨suite迁移；40–49为已查看开发网格，官方train是否包含同初始化不能据编号宣称未见。单训练seed不等于跨seed稳定。

报告每套件clean/attack SR、FR、task-equal clean-success CASR、逐任务结果。同轮次A2−A1/B2−B1，以及每方法2000−1000的task→state配对bootstrap10000、seed20260907。不输出跨suite平均作为主结论，不增加未经批准的消融/多seed搜索。

## 8. 资源、时间和提交

CPU数据准备→4090训练array0–3%4→CPU资产核验→4090测试array0–9%8→CPU统计；每任务一GPU，阶段afterok串行，不是两卡训练一张。正式提交由用户执行，入口等待本人已有GPU任务、检查重复提交、保存完整回执；调度不可达就明确失败、不猜资源。没有自动取消旧任务或重置卡。失败后同签名重提续本批断点，不能加载旧1000/2000学习补丁。

手动入口：`bash submit_exploring_oft_official_long4_2000_v1.sh`；`--preflight`/`--dry-run`仅CPU，不查询Slurm、不运行VLA或渲染、不提交。全部新代码/源快照/数据/日志/补丁/结果集中本目录；旧批保持。

上一批1000训练实测3.31/3.88/3.37/3.75h，本批2000并行预留7–9h；CPU候选缓存上一批87秒，本批准备预计3–10min，含核验统计总CPU10–30min；同3600回合测试八卡预留2–4h。整链约9.5–13.5h加排队，以正式日志更新。目标在2026-10-13中午组会前完成，不保证提升或纹理外观。本文件先于代码冻结，当前尚无新批GPU结果。

## 9. 实现交付状态

代码与脚本已实现，37项CPU/mock、18份Python AST、7份shell语法及真实本地资产/根入口preflight通过。作者原变换像素/梯度与2000日程、双档自动评测及完整状态恢复已检查；旧批签名不变。详细记录见reports/IMPLEMENTATION_REVIEW.md及CPU_PREFLIGHT.json。助手未自动提交，用户运行第8节命令启动正式链。
