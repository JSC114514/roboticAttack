# 联合OFT：修复哪里能救回任务

冻结官方四套件联合OFT，使用其300000步动作头和本体投影器。暂停新补丁训练，固定两张既有50×50腕部PNG，位置[26,137]、无EOT。比较原生PNG、仅恢复腕部DINO、仅恢复腕部SigLIP、同时恢复两分支。

## 数据与目标

沿用可靠性筛查的Spatial0/2/4/6、Object0/2/4/6、Goal0/3/5/7、Long0/2/5/8，每任务初态10，共16起点；不按响应筛选。初态11不参与本轮，留给后续模块核验。

固定历史Spatial UADA500（c5f59d59…）与同池全动作UADA适配seed42（6cf22014…），完整SHA256见config。两PNG×四条件+clean＝144个完整回合。步数上限Spatial220、Object280、Goal300、Long520，原生8步动作块和10步warmup。

这是两种既定压力测试，不比较不同训练预算的攻击方法优劣。恢复供体来自当前受损分支、当前决策的未贴图观测；不会读取另一条clean轨迹的后续状态，不额外render或推进仿真。DINO替换实际消费的block22空间输出，SigLIP替换对应空间输出；前视、语言、本体状态保持原值。

分支恢复是使用干净供体的机制诊断，不是已部署防护算法。每次双分支恢复都与同状态clean动作核对1e-5，同时核对物理数组、相机及传感器时钟未被前向改变。

## 来源复用与工程验收

CPU准备核验既有clean的代码签名、模型、初态、种子、步数、来源资产和重置核验，再复制所需小文件与视频到本实验。原文件只读。GPU评测在每个起点、任何干预之前，再核对物理数组、相机/时钟、RGB、本体与首clean动作。任何不匹配立即退出，不放宽容差或自动重试。

若16个clean全部可复用，新生成128回合；如有源记录缺失，先只补缺失clean，至多144回合。无论clean是否成功，16起点全部保留。模型每GPU进程仅加载一次，无CPU/disk offload；退出或异常时卸载所有特征/相机钩子。PNG预处理沿用已核验原生路径。

## 手动入口

```bash
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --check
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --prepare
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --dry-run --workers 2
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --submit --workers 2
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --status --slurm
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --watch --slurm
bash research_runs/wrist_joint_feature_rescue_v1/run.sh --resume --workers 2
```

只在用户执行--submit/--resume时调用sbatch；去重查询失败则拒绝盲目重复提交。依赖等待不占GPU。不预排模块训练或扩大实验。--check是CPU/mock检查，不加载真实VLA或MuJoCo；真实GPU验收在用户提交后执行。

## 资源与交付

最多两张4090，每worker4CPU/12G；CPU汇总与视频2CPU/4G。现有联合模型完整评测实测RSS峰值约5.71GiB，本轮保留12G并另测新峰值。初估45–90分钟GPU墙钟时间，不含排队；前四个起点后按实测更新。每30秒心跳、吞吐、ETA、RSS与CUDA allocated/reserved峰值，失败成本单列。

results/RESULTS.md、SUMMARY.csv、RESTORATION_EFFECTS.csv、PER_TASK.csv区分任务成功、PNG新增失败、特权恢复成功及新增损失，区间按套件分层、任务→起点重采样。ACTION_RESPONSES.csv只将实际执行槽位计入动作误差；轨迹、接触和抓持保留到STEPS。source clean计时只作为历史成本，不混成本轮新工作。

视频集中到research_videos/wrist_joint_feature_rescue_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/，单回合单视角，不拼接/叠字。内部特征恢复不可见于RGB，恢复条件由目录与元数据明确标识。CPU导出逐文件验证格式，续跑只补缺失。

本轮完成后用完整配对结果确定优先修复DINO、双分支或继续定位任务差异，不预设统一高分门槛，也不自动启动模块训练。
