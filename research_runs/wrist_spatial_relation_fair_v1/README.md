# 同条件UADA适配对照与Goal新增起点评测

新增两种对照：UADA clean-reference OFT adaptation，分别攻击XYZ与全部7DoF；seed42/43各训练4,000次。冻结Goal模型、原217条四套件混合观测、相同初始PNG及采样顺序、Adam lr0.002、warmup20、cosine、固定50×50腕部补丁[26,137]、无EOT、完整双分支梯度，与已完成关系／置乱训练对齐。XYZ是DoF匹配主对照，7DoF是能力更充分的补充。

当前观测没有专家动作标签，参考为已有Goal模型clean输出。它不是原始专家参考UADA，不使用token logits替代OFT原生连续动作头。XYZ／旋转目标位于[-1,1]，夹爪目标位于[0,1]；中点并列选下界。损失为所选全部8槽动作值的MSE。模型权重不更新。现有物理命令空间远边界对照也完整保留；归一化动作与命令坐标差别属于方法定义，不能假称所有目标完全等价。

复用当前正确／置乱最终PNG和已完成评测，不迁移优化器、不重训旧补丁。先完成Goal任务0—9、初态2—5的公平对照报告，再评测相同任务初态6—9。统一13个条件：clean、两初始PNG、两正确关系、两置乱关系、两物理平移远边界、四张新UADA适配PNG。共80起点×13条件=1,040回合；CPU核验后按实际收据复用约356回合，预计新增684回合。所有起点保留，包括clean失败的起点；CASR仅在clean成功完整配对上计算。历史Spatial UADA500仅在原40起点作为训练条件不一致的背景，不重跑扩大它。

原40／新增40／合并80分别报告。seed42为关系方法预先指定主结果，seed43检查重复性；不按评测挑种子或检查点。新增初态相对当前协议新增，不宣称整个研究历史从未见过。训练池与教师目标保持不变，本轮规模增加在配对起点。Goal模型不跨套件执行任务；旧跨套件clean均0/20，不再重跑这一配置。

资源：最多同时两张4090，每训练worker4CPU/16G，评测4CPU/12G；CPU准备、报告、视频2CPU/4G。每进程仅加载一张冻结模型；候选分进程。按候选检查点／完整回合续跑；依赖等待不占GPU。训练约1—2小时，原40起点评测约35—60分钟，新增40约1.5—3小时，合计约4—6小时加排队及CPU视频转换，首个候选与首批回合更新估时。每30秒记录进度、RSS、CUDA峰值、失败成本。

接口检查：源收据／哈希、217条有效观测、原始两关系种子的4,000次样本顺序、初始PNG、模型冻结与单GPU驻留、PNG原生预处理、可微动作与原生动作<=1e-5、有限非零RGB梯度、干净首动作、严格物理／相机／时钟配对。失败立即退出，不自动换卡、降精度、放宽阈值或反复重试。

手动入口（助手不提交或取消Slurm）：
```bash
bash research_runs/wrist_spatial_relation_fair_v1/run.sh --submit
bash research_runs/wrist_spatial_relation_fair_v1/run.sh --status --slurm
bash research_runs/wrist_spatial_relation_fair_v1/run.sh --watch --slurm
bash research_runs/wrist_spatial_relation_fair_v1/run.sh --resume
```
--prepare纯CPU核验；--check实际批处理入口检查；--dry-run仅打印依赖图；--report随时输出完整配对阶段报告。正式提交默认含原40优先评测、阶段报告、随后新增40及最终报告；不另加训练组合。

视频统一在research_videos/wrist_spatial_relation_fair_v1_singleview/{MP4,IDE_WebM}/方法/{front,wrist}/，保持原生控制时间，无文字栏、拼接或合集。已有导出通过哈希复用，新视频完整解码核验；IDE实际播放仍由用户预览确认。
