# Wrist-DINO register内容对照：500轮

2026-10-04用户授权实施。唯一主方案第20节为依据；本附件给出实现合同，不覆盖V2。先更新文档，再实现与CPU测试；GPU任务由用户提交。

## 设计

| 组 | 方法 | 251轮起新增信号 |
|---|---|---|
| A | uniform_action | 无；共同数据分布的动作对照 |
| B | register_attention_g10 | 8/16层register读取补丁区域的负log attention mass |
| C | register_content_g10 | 第16层四register的受扰K/V写入干净分支后，诱导同一错误动作 |
| D | dense_content_g10 | 同C，换成四个非补丁普通dense token的K/V |

四组均seed42随机RGB 3×50×50，从零训练500×50更新；Object clean300/replay258；腕部only；xy=(26,137)，正常Front不变。首250轮动作目标相同。动作主损失不变：连续动作块8步的第0维，干净动作决定相反边界目标。aux ratio=.1，全部不投影，AdamW lr=.002，weight_decay=0，20轮warmup，cosine总时间轴2000，500停止；无外观正则、无低维补丁、无新目标suite训练数据。

C/D训练包含正常贴图分支和干净反事实分支。正常分支产生真实动作主损失和第16层腕部qkv donor；反事实分支仅替换四token的K/V，不替换Q。原生q_norm/k_norm、softmax、residual、后续DINO、projector与动作头不变。四register索引1–4；D从实际EOT贴图覆盖以外的256个dense token中选四个等距索引，排除CLS/register。内容分支的错误目标与主损失一致。

分段链式反向：先取得主梯度，保留DINO donor上游；将donor K/V副本作为leaf做干净分支，求leaf梯度后传回donor→RGB。这与端到端链式导数等价，CPU自动微分测试验证，不是detach后丢失梯度。训练开始检查干净donor替换的恒等性与RGB可微性，不以攻击效果作为工程通过条件。保存初始、250和500轮补丁、完整Adam/scheduler/RNG恢复、RGB变化、代理损失及实际计算耗时。

## 评测与判定

四个新补丁+冻结original_uada500；四suite各10任务×10初态40–49，2000回合。复用冻结clean-success分母Object99/Spatial90/Goal98/Long95，共1910条资格数据；任务/初态完全配对、条件顺序确定性随机、task→state bootstrap10000次seed20260907。正常模型，仅量化RGB贴图，不加载反事实训练hook。

主指标Spatial/Goal task-equal宏平均，分别报告Spatial、Goal；Object作为源套件保留指标；Long完整报告但非主要结论。主差值C−A/C−B/C−D，另与原始UADA500对照。对照识别的是该训练信号的增量，不因普通采样使用了DINO输入就改名DINO方法。单训练seed、目标suite已参与开发；不将episode数当训练重复数，不自动加权重/层搜索或扩2000轮。

## 资源与时间

新任务自身训练4张RTX4090（每组1张，22GiB主机内存），测试最多8张RTX3090（每分片1张，20GiB主机内存），分阶段不重叠。提交入口实时检查当前用户的OFT链，已存在的链以afterany串行等待；队列查询失败不提交。自身阶段之间用afterok，失败不浪费下游GPU。已有V2不取消、不改代码，不会与其8卡评测并发。

初步估算（非实测）：A/B训练约2–3小时；C/D每个更新多一个原生动作前向和分段反向，约3–5小时；工程准备/校验约5–20分钟；8卡四套件2000回合约2–4小时；统计约5–15分钟。新链总约6–10小时，排队和等待现有V2另算。GPU正式启动日志给出真实吞吐；不保证代理目标能提高闭环效果。

## 文件与提交

配置：`configs/exploring_oft_dino/object_dino_register_content500_v1.json`。新训练副本与目标模块放在本目录 `code/`，不往旧 `VLAAttacker` 源树增加文件：旧指纹会递归扫描该源树。原UADA.py只读；继承已验证的Exploring-derived管线，源指纹与原生模型接口冻结。输出统一在本目录training/、evaluation/four_suite/、reports/、logs/。正式Slurm启动时检查真实OFT内容梯度与恒等性、记录显存峰值；CPU通过不等于真实GPU峰值已验证。

预检：`bash submit_exploring_oft_dino_register_content500_v1.sh --preflight`。

手动提交：`bash submit_exploring_oft_dino_register_content500_v1.sh`。

CPU准备→4组训练→CPU完整校验→8卡分片测试→CPU配对统计。命令仅由用户运行；助手不自动sbatch，不使用重置卡。

## 实施审核

四组训练、恢复、预检、自动校验、2000回合配对评测、统计/收尾报告和137–141号Slurm脚本已实现。23项CPU测试通过；补充覆盖四组最终产物验证。干净内容替换恒等性、Front/Q隔离、原生attention不改写、分段链式梯度与直接反向一致均通过CPU检查。模拟结果不是正式攻击结果。GPU真实峰值/吞吐/效果在用户提交后的内嵌启动检查与正常闭环测试中测得。

旧V2准备指纹复查一致（828个源文件）；UADA.py仍444、原SHA不变。提交时发现已有OFT链或用户GPU作业，先afterany等待，自己的后续阶段全部afterok。因这个资源依赖显示PENDING/Dependency属于正常等待，不是卡死。未自动提交任何新任务。
