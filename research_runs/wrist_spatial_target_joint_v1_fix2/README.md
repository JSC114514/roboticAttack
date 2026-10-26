# 根据策略响应与共享补丁实现能力选择空间错误

## 修复说明

fix1的Slurm脚本虽然位于修复目录，但environment.sh内的JT_ROOT仍写死指向原run，导致146384/146385和汇总任务在版本校验处立即失败。模型未加载，训练未启动。
本版从被source的environment.sh自身路径推导run和项目目录，不依赖旧目录常量或终端已有环境变量。
--check、--submit和--resume均先在CPU上实际执行cache/train/evaluate/cpu四个批处理入口的专用启动探针；探针核对真实Python模块的目录和修订，失败时不申请资源。
探针在模型、TensorFlow、CUDA和仿真初始化前返回，不提交Slurm。回归测试同时重现旧JT_ROOT路径错误并确认预检会拦截。

原run保持只读。原cache0/1各完成3个观测后，在同一个环境执行clean动作块、reset并重放时指纹不一致；报告worker与报告函数还重复获取report锁。
本版每个clean/几何条件使用独立新建环境，仍从同一种子/初态/原动作前缀重放，保留严格状态、相机、时钟和RGB校验，不扩大容差。
仅复用原6个完整缓存的视觉特征和教师动作，读取前检查来源及哈希，重新渲染比较图像逐像素一致；五个物理分支全部在新环境重做，旧距离结果不直接作为新教师分数。
闭环评测同样每个PNG回合使用新环境，防止前一条件的控制器/仿真内部状态进入后续配对。
worker锁统一用worker_前缀，report函数单独持report锁，避免同进程重复获取非重入锁。
原失败日志、完成结果与成本全部保留；修复版清单记录父run成本与复用范围。未证实具体哪个内部变量导致重置差异，不将猜测写成已确认原因。

冻结Goal checkpoint，四套件24任务、fit初态40/41与verify42/43各47个跨轨迹观测。
每轨迹取最后一个目标可见、未接触、未抓持的原生决策点，保留各一条缺失轨迹，不按动作响应选择。
腕部50×50 RGB PNG，位置[26,137]，seed42，无EOT。部署是完整原生DINO+SigLIP策略。

## 方法与预算

四个几何参照为目标在水平相对坐标中靠近、远离、左偏、右偏2厘米；高度不变。
仅改变渲染场景，DINO教师仅替换腕部block22空间输出，物理目标、前视、SigLIP及本体不变。
从同一原种子、初态、已存动作前缀重放，每条件最多执行8步；距离参考是决策时刻的原目标中心，非精确抓取点。
阻碍程度为相对clean的实际距离增加/0.02，截断[-1,1]，不是命令位移。

四组为joint、fixed_harm、realizability_only、translation_boundary。共用观测有效集合、初始化和4000次单观测损失梯度预算。
全部几何目标完整特征MSE，token权重由四方向平均变化能量固定，尺度由FIT全部任务共同确定，不逐样本归一化。
前3000预算每200次中20次为四个5步试更新，保留选中分支Adam并继续180次；后1000固定最后目标。
实现程度clip(1-L(P)/L(clean),0,1)。joint按任务层级等权的平均harm×实现程度选方向，only按实现程度选方向。
固定组按FIT均值harm选一个目标。不得使用验证选择方向、检查点或种子。
动态组3775次保留更新+225次丢弃探测=4000；固定组4000次更新。探测四分支共享父状态、样本和虚拟LR序列。
探测LR均从同一父预算位置计算；探测全部实际开销计费，随后主分支LR跳到计费后的预算位置。保存选择历史与全部成本。
额外前向与GPU秒独立报告；相同梯度预算不等于相同计算耗时。

## 手动入口

在项目根目录执行：

```bash
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --check
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --dry-run
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --submit
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --status --slurm
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --watch --slurm
bash research_runs/wrist_spatial_target_joint_v1_fix2/run.sh --resume
```

仅--submit/--resume会调用sbatch；没有自动取消或自动重试。相同修订去重，有其他修订活动时拒绝重复提交。
两条GPU依赖链保证最多2张4090：cache0/1→CPU统计→train0/1→train2/3→CPU固定最终PNG→evaluate0/1→CPU汇总。
每个训练进程只训练一张补丁。单worker4CPU、缓存/评测12G、训练16G；CPU准备/汇总2CPU4G。
教师四方向全部不阻碍接近时，CPU目标阶段输出科学停止并以非零状态结束，依赖任务由Slurm不再启动，不执行scancel。
GPU阶段仅可在用户提交的单4090 allocation内运行；拒绝CPU/disk模型offload。
缓存3h、单训练2h、评测3h限时；整体暂估3–6h，不含排队与实现，按首批8条轨迹更新估计。

检查点每200预算存PNG、Adam、所有RNG、选择历史与计费计数。失败的未封存周期重算成本保留。
进度每30秒写reports/PROGRESS，记录RSS、CUDA allocated/reserved、设备、ETA和失败成本。

## 评测与输出

四张最终PNG各评Goal十任务初态2–5，全部40起点，包括clean原本失败的起点，最多160回合。
任务SR以实际完成的新回合计算；CASR只以clean成功的完整配对计算，缺失不计成功，区间任务→起点重采样。
clean、初始PNG及历史UADA只在模型/种子/初态/PNG和重置证据匹配时复用。历史UADA是Spatial训练资产，并非同条件新算法对照。
验证47观测单独报告，所选方向和共同尺度只从FIT计算。内部教师效果、PNG表示变化、动作与真实闭环分别解释。
本轮是Goal模型开发评测，不称四套件完整泛化或历史未见确认。不自动扩大到其他种子或起点。
results/RESULTS.md、CASR.csv、PAIRED_CONTRASTS.csv、OFFLINE.json、TARGET_SELECTION.json为交付。
视频原始前视/腕部分开；CPU导出research_videos/wrist_spatial_target_joint_v1_fix2_singleview，MP4与IDE_WebM均全解码核验。
