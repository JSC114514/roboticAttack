# 联合OFT：DINO短时干预与移除后任务后果

本轮只实施128条有限干预机制分支，不训练新补丁、后果预测器、适配器或LoRA。主问题：恢复腕部DINO能否减少“短时PNG移除后仍然发生的任务失败”？历史实验和模型文件只读。

## 模型、数据和起点

官方四套件联合OFT，revision `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`，300000步动作头及本体投影器，各套件使用自己的归一化。固定任务Spatial0/2/4/6、Object0/2/4/6、Goal0/3/5/7、Long0/2/5/8，全部初态11，共16起点。已有轻度退化筛查背景，属于开发验证。

使用可靠性筛查中相同初态、种子、模型及原生接口的clean轨迹。首次目标接触记录为执行后控制步c，干预起点为`8*floor((c-1)/8)`：确保决策发生在第一次接触之前，恰好在第8步边界接触时不会选到接触之后。没有接触或初始已接触的记录保留、注明不可用，不替换任务。

固定历史Spatial UADA500的50×50 RGB PNG，SHA256 `c5f59d59af80f145a46346e6387fede6ef44d0ed8b2481d917fc0dd577a7567c`；腕部位置[x,y]=[26,137]，无EOT。

## 四条件与8/32步干预

每个起点按原动作重放clean前缀，不调用策略重新生成前缀。四条件分别为clean、PNG、PNG+同状态干净DINO、PNG+同状态干净SigLIP。内部替换仅作用于腕部；DINO使用实际消费的block22空间输出。供体在每次决策由当前分支未贴图的画面重新提取。

每条件干预8步或32步，随后同时关闭PNG和特征替换，使用原生clean输入继续执行。套件总预算始终为Spatial220、Object280、Goal300、Long520控制步，原生8步动作块、10步重置等待。前缀已用步数计入总预算，不能从干预起点重新获得完整预算。

每轨迹内八条件使用固定种子打乱顺序。clean的两个时长也实际执行，用于核验本流程与历史clean轨迹一致。干预期间提前成功不额外运行；记录实际干预步数、是否到达移除边界及成功阶段。

## 接口和恢复核验

CPU准备核验源记录、模型、PNG、初态、种子和文件来源。GPU启动检查严格匹配历史重置状态及首动作；前缀重放后核对原接触前动作，并对八条件的起点物理数组、RGB、本体、相机及传感器时钟作严格配对。原生官方动作及双分支恢复误差不超过1e-5。每个实际贴PNG决策继续检查同状态双分支恢复恒等。额外前向不刷新传感器或推进仿真，异常后卸载钩子。

无修改clean分支还核对历史后续动作、任务结果和终止步数。接口失败立即退出，不自动重试、换卡或放宽阈值。每GPU进程仅加载一次模型，检查无CPU/disk offload。

保存START、ONSET、REMOVAL（实际达到时）和TERMINAL状态；TRACE保留全部8×7预测及实际执行槽位，STEPS保留命令、末端、目标、接触、抓持。移除后保留实际分支状态，绝不恢复到clean参考轨迹。

## 手动提交与监控

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --check
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --prepare
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --dry-run --workers 2
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --submit --workers 2
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --status --slurm
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --watch --slurm
bash research_runs/wrist_joint_dino_pulse_v1/run.sh --resume --workers 2
```

只有用户执行--submit/--resume才调用sbatch。运行中去重；续跑跳过已完成、来源匹配的分支。DAG仅包含评测数组→CPU汇总→CPU视频导出，依赖等待不占GPU，失败后仍汇总已完成的完整配对。

最多2张4090，每worker4 CPU/12G；CPU汇总和视频2 CPU/4G。GPU墙钟暂估1–2小时，不含排队，调度时限3小时仅为任务上限；首批4起点后更新吞吐估计。每30秒心跳，记录RSS、CUDA allocated/reserved峰值、设备、耗时及失败成本。--check只执行CPU/mock和启动脚本检查，真实GPU接口检查在手动提交后执行。

## 结果与判读

results/RESULTS.md、SUMMARY.csv、DINO_EFFECTS.csv、PAIRED_OUTCOMES.csv分别报告两种时长的任务成功率、PNG新增失败、DINO/SigLIP救回和新增损害、配对区间。干预期间已成功、实际到达移除边界和移除后仍失败分别计数。缺失分支不视为成功；不可用起点单列。区间按套件分层、任务→起点重采样，不将两种时长、槽位或帧当成独立轨迹。

视频集中到`research_videos/wrist_joint_dino_pulse_v1_singleview/{MP4,IDE_WebM}/条件/{front,wrist}/`，单回合单视角、不拼接、不叠字；从原始帧按控制时间编码，CPU导出完整解码验证。分支恢复属于特权机制诊断，不是防护算法成绩。

本轮仅判断DINO介导的错误是否留下后续任务后果。拟议后续技术核心是`Q(s,A_P)-Q(s,A_恢复DINO)`任务成功预测差；本轮没有实现、训练或预排该预测器与补丁优化。结果形成后讨论是否推进DINO风险目标，或转向直接任务后果目标。
