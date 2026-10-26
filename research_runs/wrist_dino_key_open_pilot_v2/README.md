# Spatial Key-open pilot：v2工程修复

完整方法、参数、预算与证据边界见 [PLAN.md](PLAN.md)。算法和资源不变。v1的CPU准备因计算节点缺少/usr/bin/time在Python启动前失败；失败目录完整保留，未开始GPU训练。v2直接运行Python，RSS/CUDA峰值与耗时由Python记录，不依赖外部time程序。

项目根目录，用户手动运行：

```bash
bash submit_wrist_dino_key_open_pilot_v2.sh --preflight
bash submit_wrist_dino_key_open_pilot_v2.sh
```

第二条由用户入口一次提交完整依赖链：CPU准备→单4090分层诊断/短训→CPU核验→四suite PNG评测→CPU统计。各阶段afterok连接。作业内不调用sbatch，不自行追加作业。

随时查看本地进度、提交的作业ID和停止/失败记录；无需再执行--evaluate：

```bash
bash submit_wrist_dino_key_open_pilot_v2.sh --status
```

评测四suite分片、并发最多2张4090，正常情况下共160新回合；相同PNG去重。零有效Key目标或两臂均未改变初始PNG时，CPU核验写STOPPED结果后退出3，afterok阻止评测申请GPU；后续作业可能显示DependencyNeverSatisfied但不占卡。该退出有SCIENTIFIC_STOP记录，与工程异常分开，不自动取消任务。不要重复提交活跃链。

资源：准备4CPU/4G；训练4CPU/16G/4090×1；验证/统计2CPU/4G；评测每worker4CPU/12G/4090×1、并发2。估计训练1–2h、评测0.5–1.5h，待新GPU实测。RSS/CUDA峰值和失败成本见reports/WORKER_COSTS；Python未启动或遭SIGKILL的尝试需结合Slurm记账，不虚构峰值。

观察入口：`reports/DUAL_CLOSE.json`、`TARGET_SUMMARY.json`；`data/KEY_TARGETS.pt`；`training/single/`；`training/universal/<arm>/{status.json,final.png,probe_final.json,CANDIDATE_STEPS.parquet}`；每候选真实日志在attempts内。`results/RESULTS.md`只在正常PNG评测完成后生成。

CPU检查不启动VLA/renderer：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m pytest -q -p no:cacheprovider \
 research_runs/wrist_dino_key_open_pilot_v2/tests
```
