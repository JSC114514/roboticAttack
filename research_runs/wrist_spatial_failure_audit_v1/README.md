# 空间关系补丁：已有失败回合的视频与可见性审计

历史 `wrist_spatial_approach_pilot_v1` 只读。原训练/206回合已经完成；旧CPU汇总在文件锁处超时，持锁来源尚未查明。本目录使用独立、非阻塞的单写入锁重建汇总，不争用或移除旧锁，不重训、不加载VLA，也不把回放算成新的策略实验。

## 固定范围

- 保留全部206个回合和两个种子的统计；有效CASR只有18个Goal clean成功起点，其他套件clean为0，不能宣称四套件攻击效果。
- 视频预先固定主种子42：18起点 × clean / initial_s42 / uada500 / radial_reverse_s42 / translation_far_boundary_s42，共90段、16,666个控制步。先做首个clean/主补丁配对测吞吐，再完成全部；不按效果挑视频，不补seed43视频。
- 按原种子、初态和已执行动作逐步重放；不重新调用模型。严格比较初始物理数组/相机/时钟，逐步核对EEF与目标坐标(1e-9 m)、接触/抓持及终止。与历史关键帧比较RGB NRMSE≤1e-4。异常立即退出并释放GPU，不自动重试、不放宽容差。
- 可见性在原生腕部相机回调中渲染分割，检查零渲染和渲染前后物理/相机/时钟不变，异常也恢复回调。第一次step之前无已核验掩码，标缺失，不强制刷新传感器。
- 掩码最近邻映射256→224，再按官方0.9面积中心裁剪的归一化采样网格计算可见像素，分别记录原始可见、策略视野和补丁叠加后可见。这是类别掩码估计，不冒充JPEG重采样的像素真值；固定实体不改为其他可见物体。
- 视频每格左前视、右腕部(含实际PNG)，统一真实控制时钟。结束条件的最后一帧明确加“ENDED”标记；不加速、不把不同时间的动作对齐为同一时刻。
- 控制动作直接用历史TRACE；分割、文字和视频均不反馈策略。时间先后只作诊断，不作空间关系因果证明。

## 资源与预算

最多同时1张4090，4CPU、12G内存，GPU仅做EGL渲染。默认沿用已核验GPU映射和12G评测预算；无7B模型，首对实测RSS/图形显存后再讨论是否降资源。之后自动衔接2CPU、4G无GPU报告，依赖等待不占GPU。

新流程尚无GPU吞吐实测：初估回放30–90分钟、CPU合成5–15分钟，不含排队，GPU申请2小时上限。首对结束自动输出 `reports/FIRST_PAIR_TIMING.json` 修正预计剩余时间；每30秒写心跳、阶段、吞吐、RSS和全卡显存采样。PyTorch CUDA allocated/reserved=0，因为无模型CUDA张量；EGL显存单独报告，不能用0冒充GPU总显存。

逐回合及逐起点合成保存哈希回执，支持续跑；版本变化或历史输入变更拒绝复用。仅一个GPU回放worker，CPU报告使用独立非阻塞锁。重复提交先查当前用户`squeue`，不能读取队列则拒绝提交；不取消任何任务。提交响应不明确则停下人工核对，避免重复计费。模型训练、救援干预、新种子/新checkpoint评测不在本入口中。

## 手动命令

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_spatial_failure_audit_v1/run.sh --preflight
bash research_runs/wrist_spatial_failure_audit_v1/run.sh --submit
bash research_runs/wrist_spatial_failure_audit_v1/run.sh --status --slurm
bash research_runs/wrist_spatial_failure_audit_v1/run.sh --watch --slurm
# 中断后，已通过核验的回合和合成不重算：
bash research_runs/wrist_spatial_failure_audit_v1/run.sh --resume
```

`--dry-run`只准备/核验来源并打印提交命令，不提交。`--summary-only`只在CPU重建已有统计，暂不生成视频。Slurm故障时仍可用不带`--slurm`的状态查看本地心跳。

主要交付 `results/INDEX.html`(全部配对视频与距离/可见性曲线)、`results/RESULTS.md`、`CASR.csv`、`EVENT_TIMELINES.csv`、`RECORDED_DIAGNOSTICS.csv`。视频未通过物理/图像核验不会加入配对展示；旧结果保持不变，部分完成明确报告，不把缺失记为成功。回放完成后再根据失败过程设计一项有针对性的空间关系救援验证，本入口不会预排下一轮。
