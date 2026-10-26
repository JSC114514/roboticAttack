# 官方Long数据八路1000轮：补齐数据来源与DINO机制推进

用户于2026-10-11确认本方案并授权实现、CPU检查至手动提交命令。助手禁止自动提交Slurm、使用重置卡、取消其它任务或删除旧结果。实验只在用户授权的本地LIBERO仿真运行，跨模型及实机不在本轮范围。

## 1. 八个方法及比较关系

| ID | method | 实际处理 | 主要对照 |
|---|---|---|---|
| A1 | uada_original | 专家标签决定第0维相反归一化边界，跨8步MSE | 新数据UADA-OFT基线 |
| A2 | uada_dino | A1＋当前补丁的DINO恢复贡献选源样本 | A2−A1 |
| A3 | uada_dino_freq | A2＋全秩频域RGB更新预条件 | A3−A2 |
| B1 | tma_xyz | 固定归一化XYZ=(0,0,0)，跨8步MSE | 新数据TMA-OFT基线 |
| B2 | tma_xyz_dino | B1＋相同DINO贡献采样 | B2−B1 |
| B3 | tma_xyz_dino_freq | B2＋相同频域更新 | B3−B2 |
| C1 | tma_xyz_geometry | B1＋尺度/横向平移的动作相关DINO方向 | C1−B1 |
| C2 | tma_xyz_attention | B1＋动作敏感dense query对补丁key的Q/K分数 | C2−B1 |

C1/C2不叠加采样或频域优化，两条机制先独立比较。WHEN-DINO、教师蒸馏、旧任务/背景夹爪TMA、滚动ILA、额外uniform replay臂均不做。基线必须正常随机抽取官方训练样本，这不等于额外的uniform回放实验。

八路使用相同原始数据、随机初始RGB、候选抽样表、EOT种子、优化器和轮次。A1/B1是Exploring源代码副本上的OFT连续动作适配，不能称原离散token损失和论文实验逐项未经修改复现。原UADA/TMA/RandomPatchTransform保持只读，独立副本和溯源hash保存在本目录。

## 2. 官方数据与标签

数据：`/home/caijiasheng25/Desktop/roboticAttack/dataset/libero_10_no_noops/1.0.0/`，32个TFRecord、379个官方train演示episode。无官方validation split，不创造伪validation，也不从LIBERO测试环境采集训练样本。

CPU准备遍历完整train，建立全部有效8步动作窗口索引。只使用t到t+7均存在的窗口，动作不跨episode、不用最后标签重复填充；状态和双视角图像取窗口起点。沿用官方LIBERO动作转换（gripper=1−clip(raw_g,0,1)），按Long OFT checkpoint的q01/q99和mask归一化专家动作。UADA目标的−1/+1边界由该归一化动作空间决定，禁止用模型预测的经验分位数代替。

图像为官方RGB/JPEG、状态为官方8维proprio、任务文本为官方language_instruction。原始JPEG及动作不改写；实验缓存保留episode、时间索引、语言、来源文件路径和数据hash。训练输入只包含当前双视角、任务文本、当前proprio；未来专家动作仅作为损失目标，不写进模型prompt或native predict_action输入。

在完整有效窗口池中用固定seed抽1000×4候选窗口，所有方法共享此表。只缓存本预算实际会用到的候选图像/标签，不人为限制为300状态，也不按旧实验成功/失败筛数据。基线/C1/C2选择每轮第一个候选；DINO采样组从同轮四候选中加权取一。有效训练batch仍为1，四候选不是batch=4。输出覆盖记录，报告实际选中episode/窗口，不声称每个官方演示均参与了本次50000次更新。

## 3. 共同训练合同

- seed42；随机3×50×50全RGB，7500变量；无旧patch/Adam/前缀初始化。
- 1000 outer×50 inner=50000真实AdamW更新；batch=1；lr=0.002，weight_decay=0，warmup20 outer，cosine总周期1000；scheduler每outer推进一次。
- Front、语言、proprio和模型参数冻结，唯一优化参数是腕部RGB补丁。
- 新运行器只读核对共享checkpoint auto_map/模型代码，用作业内存保护阻止官方磁盘auto-sync；不改共享模型或安装包。
- 每outer前25 inner固定部署xy=(26,137)，后25 inner采用源RandomPatchTransform的随机合法位置、旋转±30°/剪切±0.2，并采用0.8–1.2亮度/对比度；所有八组完全相同。该共同EOT是适配配置，不宣称与所有历史基线相同。
- 前向使用实际官方resize/中心裁剪/uint8/PIL processor；RGB反向使用明确STE。保持已修复的4090原生OFT/EGL接口，不改安装包/模型文件。
- 不新增TV/HF损失、低维参数化、起点像素盒约束或外观否决门。频域组只预条件RGB更新，全部频率gain>0，不事后模糊PNG。
- 保存0/250/500/750/1000 PNG/PT；每10 outer保存完整patch、Adam、scheduler、RNG、累计时间和方法状态。中断恢复最近完整保存点，不将半轮记为完成。
- 正常作业内嵌必要接口/shape/有限值检查；无独立smoke/模型诊断作业。训练完成后自动测试，图像供用户查看，外观与攻击成败分别报告。
- 自动记录真实移动次数及相邻0/250/500/750/1000快照的RGB/PNG差异；阻止完全无RGB更新的产物进入评测，不把计数器当收敛证据，也不要求外观阈值。

## 4. DINO贡献采样与频域更新

每outer用当前量化RGB补丁在同轮四候选、同固定位置分别前向。目标t由该候选专家动作构造（UADA）或固定XYZ构造（TMA）。记正常贴图错误目标MSE为Lp，将同状态clean Wrist-DINO恢复、保持同一受扰Wrist-SigLIP时为Lrestore，分数=max(0,Lrestore−Lp)。正分数组中排名前半的候选权重2，其余1，再用冻结seed抽取一个；分数全部0时自然退回等权。目标和对照相同，不用测试数据评分。实际两支捕获/恢复identity和SigLIP未变随正常启动检查。

频域gain=(1+8*(fx²+fy²))^(−0.75)，全频率严格正。沿用上一批全秩更新预条件，在首次非零Adam提议处使clamp后的RGB步长范数匹配，再冻结尺度，保留Adam原矩状态。A3与A2、B3与B2唯一差别是该更新处理。

## 5. C1：几何—动作方向

每outer在当前源窗口现场计算clean动作、clean DINO，并以同一固定TMA目标求其对Wrist-DINO的梯度。几何参考为同一腕部RGB的尺度0.9/1.1、横向±4px；Front不变，经相同native processor提取DINO变化。它们是二维图像几何代理，不称精确物理相机干预；部署不施加这些变换。

按−<动作损失梯度,几何DINO变化>/参考RMS选择局部最有利于错误XYZ目标的参考方向；没有正方向时只执行动作主目标并记录。选择只来自train源窗口，不复用旧MGWP artifact或物理动作标准差。动作单位统一为归一化XYZ，不混入姿态和夹爪的旧尺度。

辅助项使用逐token归一化DINO变化在选中几何方向上的加权投影；权重由clean动作对dense特征的梯度范数确定，排除本次真实EOT覆盖token，避免仅将局部遮挡当成几何错觉。模型只读、参考detach，真实梯度仍从当前RGB补丁经DINO传播。记录参考类型、预测有利分数、实际投影、方向余弦和动作目标MSE，不把代理投影当成闭环效果。

## 6. C2：动作相关DINO注意力

沿用同一clean动作损失梯度范数作为dense query权重。读取真实DINO第8/16层native Q/K，保留原q_norm/k_norm及attention scale，在softmax前最大化动作敏感、未被补丁覆盖的dense queries与真实补丁key的平均匹配分数。排除CLS/register作为query；prefix数量由真实模型核对，不直接优化高范数register。按每层clean Q/K RMS做detach尺度归一化，避免层量纲压过动作目标；不重写模型attention、QKV或推理特征。

目标依据为Attention-Fool（CVPR2022，https://arxiv.org/abs/2203.13639），它不是WHEN-DINO。动作敏感query选择是本轮OFT适配，不宣称这些token已经是唯一因果中介。

C1/C2辅助RGB梯度每步范数匹配为动作主梯度的10%，参考旧温和register500配置，不当成理论最优。零方向/零辅助梯度记录并跳过辅助，不丢弃源样本；无冲突投影或强度搜索。记录主辅梯度夹角及实际比例。两组共同完整动作主梯度保持不变。

## 7. 自动闭环与交付

仅测试1000轮：fresh clean＋8张PNG，Object/Spatial/Goal/Long各10任务×10初态40–49，共3600episode。每状态条件顺序确定性随机，同初态、同reset seed、相同suite checkpoint/归一化、最大步数及8步执行。沿用已完成Final8的clean reset seed，重新执行本批clean；不复用旧失败标签。

Long为训练来源套件，其他三套件为补丁跨suite迁移；40–49已用于开发，官方train演示是否覆盖相同初始化不能据状态编号断言未见。只报告本地受控仿真，不宣称实机结果。

测试只在正常OFT上贴量化RGB PNG，无采样、几何代理、特征替换、attention hook或输入修复。使用已修复4090 EGL运行器：实际CUDA卡/NVIDIA renderer、实际环境context/API绑定、只重试RandomizationError、每suite独立子进程、每完整episode恢复。

结果逐suite报告clean和攻击任务成功率、任务等权clean-success条件ASR、A2−A1/A3−A2/B2−B1/B3−B2/C1−B1/C2−B1以及对应基线差值。task→state配对bootstrap10000、seed20260907；没有跨suite平均主指标。单训练seed不当成跨seed稳定结论。原始rollout、动作chunk轨迹、来源/patch SHA、训练更新、耗时、预览图都集中本目录。

## 8. 资源、时间和执行入口

CPU数据准备→4090训练array0–7%8→CPU自动产物核验→4090测试array0–9%8→CPU统计。GPU阶段afterok串行，每子任务1卡，不重叠占16卡。提交时若已有本人GPU任务，等待其完成；重复提交保护/提交回执/失败依赖取消保留，不自动取消任何旧任务。

历史Long2000单4090训练约6.70–7.29h/路，1000基线初估3.5–4.5h；采样和双目标新组预留4.5–7h，八卡并行训练墙钟约5–7h。3600回合4090八卡初估2–4h；官方数据索引/解码缓存约0.5–1.5h，统计约10–30min。整链预算约8–13h，不含排队；数据准备和C1/C2尚无本轮实测，正式日志按真实吞吐更新预计剩余时间。少于8张可用卡时训练/测试会分波调度。

手动入口：`bash submit_exploring_oft_official_long8_1000_v1.sh`。`--preflight`仅CPU输入和静态检查，不查队列、不加载模型、不提交。目录README提供完整cd命令和输出导航。当前为实施阶段，尚无本轮训练或攻击结果。
