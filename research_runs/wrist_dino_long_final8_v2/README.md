# Long Final-8 v2：采集渲染修复，保留250进度，训练后自动测试

科学方案的唯一来源是上级 `OFT_DINO_MECHANISM_CLOSED_LOOP_MASTER_PLAN.md` 第44节。这里是该节的实现索引，不另设研究分支。仅限有权限的本地LIBERO仿真；不执行跨模型/实机、不自动提交、不使用重置卡、不删除原实验。

## 当前实际结果与修复范围

2026-10-09 的 v1 已完成原始UADA250，耗时2937.279s（48分57秒），12500个真实Adam更新。采集数组138665十个任务均失败：8个在 `GL.glGetString` 段错误，2个返回空/无效vendor/renderer；没有生成replay分片，八路正式训练与闭环测试尚无结果，不据此判断方法失败。

v2只修工程问题：采集在导入模型/TF辅助代码前预加载同一个EGL/GL后端；通过该EGL的 `eglGetProcAddress` 和精确C ABI查询OpenGL字符串；仍验证真实context、真实相机图像、NVIDIA及已申请GPU，不跳过渲染/不退回CPU。补齐此前为空的四suite测试种子路径，仅取历史文件的初态/种子，本批fresh clean仍重新执行。新增采集phase/错误证据落盘。

`reports/PREFIX_ADOPTION.json`记录v1→v2来源与哈希：仅更新版本身份和配置中测试种子引用，250补丁/初始像素、Adam moments与12500 step、outer-cosine进度、完整RNG、训练记录原值保留，PNG字节相同。其它七学生仍从随机0训练，不加载250像素。v1目录、原始源码、旧日志未修改。

35项CPU/mock、19 Python文件AST、shell语法和真实本地资产检查通过；该检查没有启动模型、GL上下文或Slurm。真实4090渲染尚需重新提交后验证，不能声称CPU/mock证明驱动已成功。

## 一条提交命令

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_long_final8_2000_repair_v2.sh
```

可选 `--preflight` 或 `--dry-run` 仅CPU只读检查；不查队列、不加载模型/渲染、不提交。正常入口会查询本人队列，默认等待已有GPU作业结束，不和旧批叠加占卡。发现本批已有活动作业则拒绝重复提交；发现旧GPU任务的 `DependencyNeverSatisfied` 则明确报告，须用户处理，不悄悄加入永久等待链。

若旧失败批的下游作业仍在队列，可由用户只取消以下旧ID（不要取消无关任务）；助手未执行取消：

```bash
scancel 138665 138666 138667 138668 138669 138670 138671 138672
```

## 方法与输出

|编号|目录|方法|
|---|---|---|
|1|training/uada_original|Long干净状态、相反统计边界X维MSE|
|2|training/uada_dino|Long replay三决策窗口、动态DINO restore采样、legacy diverse EOT与soft-TV|
|3|training/uada_dino_freq|2+全频正增益RGB步预条件|
|4|training/tma_xyz|固定归一化XYZ目标；replay均匀候选，与5相同池和EOT|
|5|training/tma_xyz_dino|4+动作目标相关DINO restore采样|
|6|training/tma_xyz_dino_freq|5+与3相同频率预条件|
|7|training/tma_xyz_dino_teacher|5+实际RGB样本教师DINO dense蒸馏，无频率项|
|8|training/uada_dino_teacher|2+独立UADA RGB教师，无频率项|

每方法0→2000 outer，B1，50真实inner更新/outer，7500可学习RGB参数，LR0.002/warmup20/cosine2000，原模型冻结，单腕部。方法1的250前缀为本批采集服务，随后恢复完整Adam/scheduler/RNG到2000；其余7组从随机0开始，不加载前缀像素。总学生更新每组100000，不把教师额外预算藏进这个数字。

500/1000/2000补丁分别在各方法目录的 `outer_0500.png`、`outer_1000.png`、`outer_2000.png`，对应PT同名。`training.parquet` 记录真实RGB移动、梯度、损失、TV和已用学习率；`FIXED_SOURCE_PANEL.parquet` 用固定Long来源样本区分优化目标进展与cosine末期停止。保存PNG与PT量化一致后直接测试，无人工外观审批，无独立smoke作业。接口断言嵌入真实训练和真实环境创建。

结果统一在 `results/FINAL8_SUMMARY.csv`、`FINAL8_CONTRASTS.csv`、`FINAL8_ROLLOUTS.parquet`、`FINAL8_TRACES.parquet`、`STATUS.json`；日志全部在本目录 `logs/`，作业ID/依赖在 `reports/SUBMISSION_RECEIPT.json`。

## 调度与时间

CPU准备/核对续接250 → 1×4090采集原定task0的5条Long轨迹 → ≤8×4090采集task1–9的45条轨迹 → CPU bank → ≤8×4090教师（每family四分片）→ CPU合并 → ≤8×4090八路学生 → CPU资产核验 → ≤8×3090正式测试 → CPU统计。task0是原定数据的第一分片，不是新增smoke/pilot，不多采一条轨迹；先成功后才放行其它分片。不再提交250前缀训练。

一方法一GPU，不是8卡DDP；所有GPU阶段afterok串行，最高8卡，不让教师、训练、测试叠加到16卡。不设置4090的A800 QoS；使用既有 `rtx/gpu:4090:1` 和 `rtx/gpu:3090:1` 写法。实际调度取决于可用GPU/CPU/内存与QOS，不保证8卡立即同时启动。

|阶段|实际启动后的估计|
|---|---|
|已完成250前缀|实际48分57秒；复用，不再训练|
|剩余采集、bank|0.5–2h；task0串行再放行其余|
|8卡教师分片|3–6h|
|8卡八路学生|8–16h|
|8卡3090测试|5–8h|
|CPU统计与整理|约0.5h|

剩余合计约17–34h，建议预留两天，排队另计。这不是已实测的整批时长。教师实际状态数会随Long replay成败改变，`STATE_BANK_STATUS.json`报告精确数量；正式启动后根据真实速率更新估计。Slurm时限是保守上限，不是耗时预测。

## 代码来源、适配与解释边界

原 `VLAAttacker/white_patch/UADA.py`、`TMA.py`、`appply_random_transform.py` 保持原SHA和0444只读。本目录 `source/`保存完整快照（TMA仅补一个EOF换行）；可调用改动只发生在本目录 `code/`。`SOURCE_LINEAGE.json`记录复制来源与SHA。`UADA_SOURCE_COPY.py`复用Exploring AdamW/cosine构建；`train.py`由已有Exploring→OFT TMA/transfer副本训练循环适配，保留每inner更新、clamp、清梯度与每outer调度。不是把token-CE照搬到实际不用token解码的OFT动作头。

复用Long-v2的已验证native-RGB前向/明确STE反向；`model_runtime.py`来自已实际成功的3090 evalfix2，v1泛化阶段GPU型号，v2额外修复GL/EGL预加载与字符串查询并记录初始化phase；观察/模型加载/动作接口/重置及等待/上下文恢复函数保持相同。正常部署调用原OFT动作头，贴真实uint8 PNG；DINO restore只用于来源评分/教师资格，不进入测试。

UADA：参考是同状态无补丁连续预测（原论文为专家token标签），边界重新来自Long clean300的2400未来动作槽。TMA：目标是归一化XYZ=(0,0,0)，反归一化值记录于PREPARATION，不称物理停止；不是旧夹爪/任务区/背景区max损失。

教师每个状态固定100次RGB优化；同时改善相对于初始随机图案和无补丁动作的目标MSE，且DINO restore使目标MSE增大，才参与蒸馏。无效教师对应状态仍参加主动作训练。教师目标每次用与学生相同EOT重新计算、detach、不跨状态平均、不复制教师像素。辅助系数在Long来源首个有效样本按梯度比0.1一次校准后固定。不保证教师兼容一个通用补丁，不保证教师或学生低噪声，更不把动作MSE下降直接当CASR。

3/6对AdamW的RGB提议步做可逆全频增益预条件，首个非零更新按clamp后实际RGB步长对齐，标量随后固定；无低维参数、频率截断或测试后美化。2/3/8明确保留旧soft-TV阈值0.4/权重1；4/5/6/7没有额外TV。2−1是训练包总收益；5−4是同池同EOT的DINO采样增量。若DINO评分全部为零或教师全部无效，对应方法会退化为基准行为，必须如实报告，不能包装成DINO利用成功。

正式10000唯一回合：24攻击快照×四suite×10任务×10初态 + 本批fresh clean400。CASR由本批配对clean-success资格计算，不借用历史clean。Object/Spatial/Goal为Long来源迁移；Long必测但作为源域次要。初态40–49已经看过，属于development网格，不声称新盲测；只有一个独立训练seed，回合bootstrap不等于训练随机种子复现。
