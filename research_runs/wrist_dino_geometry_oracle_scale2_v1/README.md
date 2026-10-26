# DINO几何目标：2倍位移校准

仅校准原理想几何目标的强度，不训练补丁，不补任务4，不自动扩展。固定Spatial checkpoint、任务0/1/2/6/7与初态44/45，两个原方向各10回合，共20个新增回合。复用完整1×实验的10个clean回合与1×两个方向结果，旧目录只读。

唯一科学参数变化：先用原规则得到世界X正负位移（30°方位弦长，截断到30mm），再将该位移整体乘2，上限60mm。不是只改上限，不改变高度、目标选择、模型分支或动作目标。只修改渲染几何，物体真实物理状态不动；这是内部机制干预，不是RGB补丁或新算法。

保持原生sensor实际采样时刻的反事实渲染、零位移RGB核验、渲染前后物理及相机时钟一致性、末执行槽位采样和异常恢复。将CF腕部DINO block22空间输出完整替换；前视、SigLIP、语言、本体与权重保持原生。无目标/不可见不删样本，记录覆盖率；每决策保存完整8×7动作、实际执行槽位、同观测clean参考、实体切换、实际位移和可见像素。

clean按原回合哈希核实后复制并重新封装，明确reused_clean与原始来源，不冒充新执行回合。新回合在动作执行前逐项匹配这些clean的sim_state、双相机及本体状态数组；失配保存START.npz和逐项数值差异，不放宽检查或计为攻击成功。已完成回合按哈希续跑，失败attempt保留。

## 手动入口

```bash
bash research_runs/wrist_dino_geometry_oracle_scale2_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_oracle_scale2_v1/run.sh --dry-run
bash research_runs/wrist_dino_geometry_oracle_scale2_v1/run.sh --submit
bash research_runs/wrist_dino_geometry_oracle_scale2_v1/run.sh --status --slurm
bash research_runs/wrist_dino_geometry_oracle_scale2_v1/run.sh --watch --slurm --interval 30
```

只申请一个4090 worker，4CPU、12G，20回合串行。预计5–10分钟，不含排队、未经本2×流程GPU实测；申请45分钟是调度上限。1×完整30回合实测6分钟，进程RSS约5.65GiB、CUDA allocated约14.96GiB。汇总2CPU、4G，afterany等待不占GPU。

只在用户运行--submit时提交。重复活动作业拒绝提交；队列查询失败停止；提交响应不确定不自动重试，不自动取消作业或启动训练。每30秒进度心跳；RSS HWM、CUDA allocated/reserved峰值、设备、耗时及失败成本完整保留，Slurm MaxRSS另报。

results/RESULTS.md并列展示1×与2×结果、覆盖率；CASR.csv是2×，BASELINE_SCALE1_CASR.csv是只读复用1×，SCALE_COMPARISON.json报告相同起点的新失败、丢失失败及任务→初态配对差区间。10个起点是已查看开发集，不据此宣称泛化，不按统一高分或显著性门槛筛掉结果。1×与2×沿各自当前轨迹生成目标，实际实体和位移都记录；不保证强度与失败率单调。

GPU首批核实identity、原生CF表示、双分支等价、DINO精确替换和clean SigLIP。只有结果支持目标具有破坏性后，才另行讨论共享RGB短训练；本入口永不自动训练。
