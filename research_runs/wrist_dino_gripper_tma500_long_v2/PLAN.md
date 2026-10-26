# Long来源 Wrist-DINO 区域夹爪TMA500：工程修复版 v2（2026-10-08）

当前实施附件，替代主方案第43节的失败v1执行入口。只在用户授权的本地LIBERO仿真运行；用户手动提交，助手不提交GPU作业、不用重置卡。v1代码、日志、PREPARATION和228条记录不修改、不删除。

## 不变的科学问题与预算

单腕部50×50全RGB（7500变量），Front干净；A普通gripper-TMA，B任务区DINO-CF，C等token背景DINO-CF。干净动作raw g<0.5的close/hold槽位目标为raw g=1，正常释放不设目标。A最小化正常RGB分支MSE，B/C最小化max(L_full,L_region_cf)，全分支相等时选full梯度。CF只保留同一实际RGB补丁/EOT造成的所选Wrist-DINO特征差异，其他DINO、Wrist-SigLIP、Front干净，不独立优化latent。

同seed42随机初始化、空AdamW，B1、500 outer×50 inner=25000更新，LR0.002、weight_decay0、warmup20 outer、500 cosine；每inner更新并clamp到[0,1]。原Exploring TMA训练循环/贴图副本保留；不修改只读原文件，不续训旧补丁，不加TV/blur/外观约束。保存0/100/250/500原始PT及量化PNG。

## 修复1：原生前向、可微反向

v1可微预处理漏了官方中心裁剪后的uint8转换，真实CPU图像比较发现BF16输入差异最高0.03125；四个采集任务因此在动作接口检查处退出，训练/测试未启动。这是工程失败，不是科学负结果。

v2 clean直接保留wrapper.prepare_hf_inputs的原生pixel_values，不再覆盖。RGB/EOT分支先按部署PNG同样的round/clamp转uint8，再调用实际官方prepare_images_for_vla和实际image_processor，前向值完全原生；反向使用现有torch裁剪/归一化近似的straight-through梯度。该反向是代理导数，不声称uint8量化本身可微，不用近似前向充当真实部署。Front和实际DINO donor使用同一原生前向值。0.02动作接口容差不放宽，并增加逐维差值、最大差槽位/维度、close标签一致性和失败观测记录。全部新脚本启用TF_FORCE_GPU_ALLOW_GROWTH，防止TensorFlow图像预处理预占PyTorch需要的整块显存；不改变模型/插值配置。训练起始GPU接口检查也纳入FAILED状态写入，避免异常发生在训练try块之前而没有状态文件。

## 修复2：等token配对，不因任务区域大而丢整任务

Long区域仍为全部goal操作主体的真实visual/contact geom加夹爪，Close/Turnon等器件仍取整父器件，不改成新的精细分割假设。背景排除目标和全部机器人，可能含任务相关目的地容器，不声称因果无关。

v1要求全部任务token都能由背景等量匹配，导致任务2的17条close记录全被排除。v2每个RGB/EOT排除实际patch支撑后，令n=min(可见任务token数,可用背景token数)，按每状态固定、与补丁颜色/损失无关的两份随机排列各取n个。背景足够时保留全部任务token；不足时是任务区域的等量子集，不再声称覆盖完整任务区域。B/C仍等token、同状态、同真实EOT；A也共用这些状态/EOT。只排除无close、任务完全不可见或背景为空的状态；记录任务/阶段覆盖和区域子集比例。32次EOT有限重抽保留，不扩大尝试次数或训练预算。

## 修复3：原始数据恢复，所有锚点重新提取

Long十任务、init0–39按固定顺序各5成功轨迹，轨迹全程均匀6个不同决策，共300条；训练模型OFT-Long，libero_10_no_noops，max_steps520。init40–49不入来源，不用Object/Spatial/Goal补数据；不是完整Long RLDS或完整Exploring复现。

RECOVERY_SOURCE.json冻结v1 PREPARATION、config和十个来源分片SHA。只恢复228条已保存记录的真实RGB/proprio/sim_state/native_action/geom掩码及原状态尝试顺序；旧clean_action、close_slots、DINO/SigLIP锚点全部丢弃并由当前模型重新提取。六个完成任务不重演成功轨迹；四个未完成任务在原尝试前缀后补足，72条是当前缺口，不是新增数据预算。恢复按完整轨迹提交，支持中断继续；失败候选按每task单个PT保存，不产生逐state小文件。新/旧数据分开版本化，不能仅跳过完成六任务后直接混旧标签。

## 正常四suite闭环与边界

clean、历史Object-UADA500参考、A/B/C，四suite各十任务×十初始状态40–49，总2000回合。正常OFT、uint8 PNG、腕部固定xy[26,137]，部署不分割、不hook、不替换特征。按同决策状态预测并按实际执行槽统计wrong-open；正常释放、缺测、接触/疑似掉落代理分开。冻结clean资格99/90/98/95和本批fresh-clean结果同时报告。

Object/Spatial/Goal为Long来源补丁的迁移主结果，Long为必报来源内次要结果；每suite使用该suite既有OFT微调权重，是补丁跨suite/同架构权重迁移。40–49是已查看开发网格，不称新held-out。旧UADA500来自Object且是2000日程的500中间点，只作不同源历史参考；主要方法比较B−A、B−C。Long原视角/DINO差异较弱，不把旧Object因果证据说成Long已证实，不预设换Long必然增强或外观更好。

## 资源、时间、停止

CPU准备→A800恢复/重提锚点/补采集≤3卡→CPU组池→A800训练≤3卡→CPU核验→4090测试≤5卡→CPU统计。afterok串行隔离，一任务一卡，A800显式partition/qos/gres=a800/a800/gpu:a800:1，测试rtx/gpu:4090:1。不占3090，不取消其他作业，等待用户已有GPU/OFT任务；失败依赖取消下游，避免占卡空转。

v1实际采集阶段约9分钟，不能再把此前2–4小时粗估当实测。v2预估恢复/补采10–30分钟、三组并行训练3–6小时、测试3–5小时、CPU阶段数分钟，总约6.5–12小时加排队；训练和测试尚无本批实测，日志持续更新ETA。500闭环后停，不自动扩轮、采样、跨模型或实机。

唯一新目录research_runs/wrist_dino_gripper_tma500_long_v2/，手动入口submit_exploring_oft_gripper_tma500_long_v2.sh；--preflight/--dry-run只CPU，不查询队列、不加载VLA、不渲染、不提交。CPU/mock/真实processor回归通过不等于真实GPU反传或攻击有效。
