# 腕部视角稳定性与真实目标响应诊断

本轮只检查：相机改变时稳定下来的模型，面对真实目标位置改变，是否仍能正确调整局部控制。使用三个已有冻结模型，不训练新模型或补丁。

[完整实验定义](EXPERIMENT_PLAN.md) · [固定配置](config.json)

## 手动入口

在项目根目录运行：

```bash
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --check
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --dry-run --workers 2
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --submit --workers 2
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --status --slurm
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --watch --slurm
bash research_runs/wrist_camera_task_response_diag_v1/run.sh --resume --workers 2
```

`--submit`与`--resume`均只补未完成的阶段／分支。`--dry-run`只显示拟提交命令，不调用sbatch。检查不加载大模型、不做GPU渲染。运行中的本实验GPU任务会阻止重复提交；旧的待运行CPU汇总不会阻止续跑。脚本不自动取消任务。

## 阶段与资源

1. prepare：从已保存动作重放12条候选轨迹，核对五个缓存状态，按物理规则选样，再构建真实目标位移×相机位置的九个条件。只用1张4090、4CPU、12G，不加载模型。
2. evaluate：三个独立模型进程，每个模型只加载一次；每个分支精确重放原动作至选定状态，再执行一个8步动作块。最多2张4090并发，每worker4CPU、12G。
3. report：CPU2核、4G，统计完整配对、生成曲线并转换WebM。GPU任务结束后立即释放GPU，不为视频转换占卡。

估计GPU阶段30–60分钟，排队另计；尚无新流程实测。额外的历史动作重放和模型加载成本会记录，首四个诊断分支后更新估时。每个GPU任务最多2小时，CPU报告30分钟。进度每30秒写入reports/PROGRESS；各进程的成功／失败成本写入reports/COSTS。

## 原生采样与真实物理变化

原生图像采样时刻可能略早于控制决策时刻。为保留这个接口，准备阶段分别捕获前视和腕部的原生相机快照，在副本中平移目标的采样几何；同时在执行环境将同一实体freejoint真实平移相同世界坐标向量。后续控制与图像全部来自已经移动目标的真实环境。这与仅渲染反事实、物理目标不动的旧实验不同。

零位移须重现原生双视角，物理位移须不改变其他关节、速度、本体输入和传感器时钟。先确定支撑／穿插／可见性有效条件，之后才调用三个模型。异常变体保留，不改幅度凑齐结果。原历史代码、缓存与结果只读。

## 如何读输出

- data/MANIFEST.json：全部12条轨迹、选样、无有效状态、无效条件及原因。有效条件不足四套件则停止模型阶段。
- evaluation/模型/轨迹/条件/：RESULT.json、原生8×7动作、实际执行日志、物理起点及视频。已完成分支有COMPLETE收据；失败尝试保留独立attempt目录。
- reports/SUMMARY.json及REPORT.md：缺失／无效／完成分母、正负方向配对响应、相对base交互与区间。首8步的task_done只作过程信息，不能当完整任务成功率。
- research_videos/wrist_camera_task_response_diag_v1_singleview/：按模型／条件／视角集中保存前视、腕部独立MP4及IDE_WebM，不拼接。

P为本分支真实目标中心距离的减少量。控制响应须同时结合实际运动、接触及视频解释，不能把动作更一致直接说成操作更正确。

## 当前检查状态

CPU预检及17项测试通过，覆盖真实CPU MuJoCo位移不变量、八步配对统计、缺失处理、资源配置及MP4/WebM编码与完整解码。复核发现的提前终止和派生坐标滞后问题已修复。记录保存在reports/。GPU图像、物理重放和原生动作门槛由手动提交后的任务启动检查完成；实现准备阶段不自动申请GPU。
