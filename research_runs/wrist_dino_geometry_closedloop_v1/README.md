# 已有几何PNG：小规模闭环核验

先验证短训练补丁是否造成闭环任务失败，再根据证据决定新方法。本入口只评测，不训练，不触发旧的预设夜间方案。

固定 Spatial checkpoint，任务0/1/2/4/6/7，初态44/45，共12组起点、60回合。五条件为重新执行clean、seed42初始随机PNG、pilot共享projection、pilot共享full、现有Spatial UADA500 B1。两张pilot补丁均4800次更新。相同初态、历史重置种子、固定位置[26,137]、50×50腕部RGB，无EOT和内部特征替换。条件运行顺序在每个起点内固定种子打乱，每次完整重置，状态及双相机初始画面必须一致。

全部初态文件、任务映射、补丁PNG及来源代码哈希核验。44/45未参与本轮pilot拟合40/41或验证42/43，但在旧长训练池中；6个任务为原训练任务。这是探索性闭环核验，不是跨任务确认性测试。UADA为既有部署参考，训练数据及预算不同。

每个worker处理两个任务、20回合；最多3张4090并发，每张4CPU、12G，申请45分钟上限。预计15–25分钟，不含排队，未经本流程GPU实测；额外同观测clean预测可能增加耗时。CPU汇总2CPU、4G、10分钟上限，afterany等待不占GPU。每30秒独立心跳、动态剩余时间、RSS HWM/CUDA allocated和reserved峰值、实际GPU型号及成功/失败尝试成本。进程指标不冒充Slurm整作业内存；sacct另报MaxRSS。

## 手动入口

```bash
bash research_runs/wrist_dino_geometry_closedloop_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_closedloop_v1/run.sh --dry-run
bash research_runs/wrist_dino_geometry_closedloop_v1/run.sh --submit
bash research_runs/wrist_dino_geometry_closedloop_v1/run.sh --status --slurm
bash research_runs/wrist_dino_geometry_closedloop_v1/run.sh --watch --slurm --interval 30
```

脚本不自动取消任务。重复提交先查询Slurm，有本实验活动作业则不重复提交；控制器不可达则停止提交。失败修复后用同一--submit命令续跑，只提交有缺失回合的worker；完成回合按哈希复用。提交响应不确定时不重试，记录SUBMISSION.json供人工核查。

results/RESULTS.md直接报告各条件失败切换、CASR和与随机失败重合。CASR分母为本轮重新执行clean成功的起点，全部60回合仍保留。CASR与随机差值及任务→初态重采样区间见CASR.csv；仅6任务/12起点，不据显著性门槛筛方法。配对失败不删除任务凑结果；报告任务失败时明确缺失/异常，禁止静默统计残缺样本。

TRACE.parquet保存每次完整8×7动作、实际执行槽位、相同当前观测的clean参考预测、末端实际前后位置及夹爪关节。clean参考只作预测，不执行、不推进仿真；动作命令不写成真实位移。关键帧仅第0/4/12块及结尾，前视和腕部拼接，每回合最多4张。诊断样本量为回合，不能把块数当独立重复。原生RGB造成失败仍不能单独证明几何信息是因果中介。

下一步根据任务失败及偏差持续情况决定目标，不预先选full或projection，不按本轮结果修改PNG，不自动启动长训练。
