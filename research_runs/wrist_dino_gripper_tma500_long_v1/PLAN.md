# Long来源 Wrist-DINO 区域夹爪TMA：500轮（2026-10-08）

用户最新决定：Object批次未开始，直接切换Long（LIBERO-10）；不再提交Object版。只做本地受控LIBERO，助手实现/CPU预检后给手动命令，不自动提交、不用重置卡。三组500 outer×50 inner、B1，单腕部50×50全RGB、干净Front；不扩跨模型/实机或2000轮采样。

## 方法保持

A plain_gripper_tma：干净OFT要求close/hold的8槽位中，第6索引gripper由raw<0.5诱导到raw=1（张开），正常RGB分支MSE。
B task_region_dino_tma：最小化max(L_full,L_region_cf)，任务区域只保留同一真实RGB/EOT引起的Wrist-DINO扰动。
C background_region_dino_tma：与B等token数量的非目标实体区域，作为机制对照。
CF=D_clean+M*(D_actual_RGB-D_clean)，未选DINO、Wrist-SigLIP、Front均干净；没有自由latent优化。B/C顺序释放语言模型图，再经实际DINO donor链式回传，与既有Object实现相同。

三组同seed42随机7500变量RGB、空AdamW、25000更新，LR0.002、warmup20 outer、500 cosine，每inner做Adam/[0,1] clamp，几何EOT/随机位置/配对区域一致。不加TV/blur/硬外观预算，不续训旧补丁。保存0/100/250/500的原始PT与量化PNG。完整Exploring TMA/贴图只读副本及独立OFT适配保留，原始源码不修改；连续目标/状态池适配不是完整论文复现。

## 真正切换Long来源

当前实现仍使用成功策略轨迹的状态池，不是完整Long RLDS数据集，也不是给Object的300条记录改名。新增A800干净OFT-Long采集：Long十任务，每任务从init0–39按固定顺序取得5条成功且至少6个决策的轨迹；每轨迹按全程等间隔选择6个不同决策，覆盖早/中/晚，合计300条新记录。训练模型和锚点均为OFT-Long，unnorm_key=libero_10_no_noops，max_steps=520；eval40–49不参与来源采集。失败、过短轨迹和尝试状态完整记录，不按补丁效果选样；40个来源状态仍不足5条时明确停链，不混其他suite补足。

记录真实当前RGB/proprio、完整sim state、原生action chunk、决策位置/阶段和精确geom标签；成功之后选帧，再用当前Long模型计算normalized clean action、DINO/SigLIP锚点。GPU核对原生动作与可微接口一致、执行夹爪方向正确。三组共用同一来源池和close槽位；正常应释放的槽位不设目标。

Long不是Object的单物体模板：任务区域取全部goal谓词的操作主体（两件搬运物、Close/Turnon等关节器件）对应的actual visual/contact geom，加夹爪。区域名如cabinet_bottom_region解析到父器件；关节器件当前按整器件geom标注，不冒充精确活动门/旋钮掩码。不包含仅作容器/目的地的参数，背景排除上述目标与全部机器人，仍可能含任务相关容器。区域映射只用于训练，部署不用分割或特征替换。

token按同一resize/center-crop映射16×16；共用过滤no-close、任务不可见、等数量背景不可配对的记录，报告按任务和阶段覆盖/排除，不用一个不可见状态否定假设。每步按与RGB颜色无关的保守affine支撑排除patch覆盖token，32次有限候选重抽。旧Object bank、mask_runtime_config不再作为训练或标签输入；新Long bank由十个采集分片组建并记录SHA256/来源身份。

## 正常四suite闭环

自动采集→组池→三组训练→CPU训练核验→正常RGB测试→统计。clean、旧UADA500参考、A/B/C各测试四suite十任务×十状态40–49，共2000回合，clean资格99/90/98/95与新clean成功率分别报告。正常模型、uint8 PNG、固定腕部xy[26,137]，无hook/latent替换；实际同状态逐执行槽计算wrong-open和接触/疑似掉落代理，正常释放/缺测分开。

Object/Spatial/Goal现在是Long来源补丁的迁移主结果；Long是必测的来源内结果，不能再称Long零样本。每suite仍使用该suite的既有OFT微调checkpoint，因此属于补丁跨suite/同架构权重迁移，不是单个冻结Long模型在四环境上的性能。40–49是既有开发网格，不称新held-out。旧UADA500仍为Object来源/2000日程500中间点，只作历史参考，不能当同Long来源严格配对UADA基线；主要方法增量仍是B−A、B−C。报告必须标注各条件训练来源。

旧DINO因果结论主要来自Object/Spatial/Goal；Long上的原视角差异较弱，不能直接将Object的因果证据搬成Long已证实。换Long理由是来源状态覆盖更丰富的子任务/阶段，跨suite收益与外观均待实测，不承诺必然增强。

## 资源、时间、停止

CPU准备→最多3张A800采集Long/区域锚点→CPU组池→最多3张A800训练→CPU核验→最多5张4090测试→CPU统计。阶段afterok串行隔离，每任务一GPU；A800 partition/qos/gres为a800/a800/gpu:a800:1，测试rtx/gpu:4090:1。等待用户已有GPU/OFT链，重复链/无效依赖拒绝，不自动取消旧作业。

预计新Long采集2–4h、三组并行训练3–5h、测试3–5h、CPU10–30min，总9–15h另加排队；新Long链未实测。500闭环后停止，不自动扩轮/运行采样版。真实GPU校准、外观和CASR只有用户手动提交后才能判断。

## 输出与版本

当前唯一实施目录research_runs/wrist_dino_gripper_tma500_long_v1/；Object版保留为未运行的旧方案，不重跑历史成果。data/labels为Long分片，data/LONG_TRAIN_BANK.pt为新来源库，data/TRAINING_POOL.pt为三组共同过滤池；training/evaluation/logs/reports集中在此处。手动入口submit_exploring_oft_gripper_tma500_long_v1.sh。后续Long来源UADA/均匀/DINO采样500/1000/2000仍另议，不随本链启动。
