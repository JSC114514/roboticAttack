# Long来源腕部DINO夹爪TMA500：当前唯一实施

用户决定直接换Long，Object版未运行并停用。新目录保存独立副本、完整只读Exploring源快照、Long采集数据、补丁、日志和四suite结果；原始源码及历史成果不改。详细冻结方案见[PLAN.md](PLAN.md)，配置见[config.json](config.json)。没有新补丁/CASR，当前只有CPU/mock预检。

## 提交和时间

先估计耗时：Long新采集2–4h、三组并行训练3–5h、测试3–5h、CPU10–30min，总9–15h另加排队。新Long卡组合未实测，不保证换源或换卡带来提速/效果提升。

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_exploring_oft_gripper_tma500_long_v1.sh --preflight
bash submit_exploring_oft_gripper_tma500_long_v1.sh
```

第一条只读CPU资产检查，不查队列/加载模型/渲染/提交。第二条必须由用户执行，自动串联完整七阶段：准备→最多3张A800采集Long→组池→最多3张A800训练→核验→最多5张4090正常测试→统计。单分片一GPU，阶段afterok隔离；A800显式qos=a800，rtx测试沿用账号默认QOS。等待用户已有GPU/OFT链，不自动取消旧任务。源采集/真实模型校准失败则下游不运行，保留恢复点。

## 三组与数据

三组仍为A普通gripper-TMA、B任务实体区域DINO-CF TMA、C等token背景区域DINO-CF TMA。seed42随机完整3×50×50 RGB、空AdamW、B1、500×50=25000更新、LR0.002、warmup20/500 cosine；不是续训，不加外观正则，不直接优化latent。B/C优化max(正常开夹爪MSE,区域CF开夹爪MSE)，其额外前向不称等计算。TMA目标来自Exploring，新增问题是任务实体区域实际RGB扰动是否能承载并增强具体错开夹爪。

本轮真正改为OFT-Long + libero_10_no_noops统计 + Long十任务成功来源：每任务init0–39中取5条成功且足够长的轨迹，全程均匀选6决策，300条新RGB/state/sim-state/原生动作记录。eval40–49不参与来源采集。不是完整RLDS Long训练集或完整Exploring论文复现。旧Object bank与其重放配置不再作训练输入。模型JSON、实际dataset_statistics、Long BDDL/init文件和源码身份检查；真实GPU采集后才能得到新bank。

区域规则已支持多目标和Close/Turnon器件，不能沿用Object单物体断言。取goal操作主体的整物体/器件geom+夹爪；柜门region映射父器件，器件整区不是精确门/旋钮掩码。目的容器可留在背景，不能说背景必然动作无关。共同筛掉no-close/不可见/等数量不可配对记录，报告十任务及早中晚覆盖；三方法共用池/顺序/槽位/EOT。

## 测试与参考身份

共2000回合：四suite各100 clean，以及旧Object来源UADA500/A/B/C各100。训练后正常模型uint8 PNG部署，只贴腕部固定xy[26,137]，无CF/hook/segmentation。记录CASR、新clean成功数、同一实际状态逐执行槽wrong-open、接触/疑似掉落代理，不把代理当实机因果证明。

Object/Spatial/Goal是Long来源迁移主结果；Long是完整必测来源内次要结果，不称Long零样本。每suite用其既有OFT微调权重，是补丁跨suite/同架构权重迁移。40–49仍为已查看开发网格。旧UADA500来源Object、2000总日程中间点，明确标历史不同源参考，不称公平Long-UADA基线；公平本轮增量B−A、B−C。所有结果保存patch_training_suite和comparison_role。

## 输出导航

```text
source/                         完整只读Exploring TMA/贴图/原模型加载器快照
code/                           Long收集/校验、OFT适配、训练、正常评估/统计
tests/                          CPU/mock，不代表实际攻击效果
slurm/                          147–153七阶段，Long job-name，单GPU与并发锁定
reports/PREPARATION.json         用户提交后生成的冻结资产/代码身份
data/labels/task_00.pt ...       Long每任务30条源记录+区域锚点，逐完整轨迹恢复
data/labels/task_00.json ...     尝试状态、成功率、接口/夹爪校准、耗时
data/LONG_TRAIN_BANK.pt          300条新来源记录与全部来源身份
data/TRAINING_POOL.pt            三组共同资格池和500状态顺序
data/POOL.json                  bank SHA、源50轨迹、任务/阶段覆盖及排除
training/<method>/              outer_0000/0100/0250/0500.pt及原始PNG
training/<method>/resume.pt      完整Adam/scheduler/RNG，非重初始化
reports/AUTO_VALIDATION.json     训练预算/RGB变化/PNG/来源核验
evaluation/GRIPPER_TMA500_SUMMARY.csv    按训练来源标注的四suite结果
evaluation/GRIPPER_TMA500_CONTRASTS.csv  B−A/B−C等配对差值
evaluation/<suite>/task_*.parquet       原始回合/逐槽动作与接触代理
evaluation/STATUS.json           完成并停在500，不自动扩轮或跑采样版
logs/                           每Slurm作业/分片out与err
```

后续Long来源原始UADA/均匀/DINO采样B1、500/1000/2000检查点仍另议，不随本链启动。Long覆盖不同任务阶段是换源理由，不等于Long DINO机制已经证实或一定增强。
