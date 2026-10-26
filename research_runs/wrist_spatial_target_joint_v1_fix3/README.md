# 根据策略响应与共享补丁实现能力选择空间错误

## 修复说明

fix2的两个cache worker在Spatial任务4、验证初态42/43退出；已完成53个状态，训练未开始。旧实验和安装的LIBERO均保持原样。

已CPU复现LIBERO重复重建时object_property_initializers累计、相同种子随机数序列偏移的问题。原采集每任务复用环境，新代码每条件新建环境，必须同时复现原采集重置历史。
本版按原96条轨迹的worker奇偶分配、任务分组和先后顺序构建历史，包含没有入选观测的轨迹。每个新环境先按此前轨迹各自的种子执行硬重置，再设置当前轨迹种子、初态并重放已存动作；不执行此前轨迹的策略或修改原采样器列表。
当前模型/控制器由hard reset重新创建；严格原采样状态、相机时钟、缓存RGB、配对起点校验仍保留，不扩大容差。完整仿真是否重现，以上述校验为准，不把CPU复现当成GPU已通过。

启动先使用一张4090、不加载VLA，核验任务4的40/41/42/43各两次独立重放，并核对Goal全部十任务初态2的历史clean起点。通过后才启动剩余缓存和训练，失败则后续依赖任务不启动。此为工程接口检查，不是增加科学短实验。
fix2已完成的53份缓存逐份验证签名、原生采样指纹与5个独立物理起点后，使用软链接复用完整内容，生成明确标注复用的新收据；不重算、不复制大文件、不改旧字节。
对齐失败时保存实际模型位置/旋转、qpos/qvel/求解器与控制器数组、前视/腕部RGB及本体误差，补齐此前只有哈希的诊断缺口。

保留动态解析环境路径、四类实际批处理入口CPU预检和report独立锁。提交时检查所有历史修订的活动任务，防止重复占用GPU。

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
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --check
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --dry-run
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --submit
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --status --slurm
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --watch --slurm
bash research_runs/wrist_spatial_target_joint_v1_fix3/run.sh --resume
```

仅--submit/--resume会调用sbatch；没有自动取消或自动重试。相同修订去重，有其他修订活动时拒绝重复提交。
启动重放核验仅1张4090；随后两条GPU依赖链保证最多2张4090：重放核验→cache0/1→CPU统计→train0/1→train2/3→CPU固定最终PNG→evaluate0/1→CPU汇总。
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
视频原始前视/腕部分开；CPU导出research_videos/wrist_spatial_target_joint_v1_fix3_singleview，MP4与IDE_WebM均全解码核验。
