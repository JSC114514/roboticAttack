# Spatial Key-open pilot

完整方法、参数、预算与证据边界见 [PLAN.md](PLAN.md)。隔离实现；历史代码/数据库/补丁不修改。

项目根目录，用户手动运行：

```bash
bash submit_wrist_dino_key_open_pilot_v1.sh --preflight
bash submit_wrist_dino_key_open_pilot_v1.sh
```

第二条只提交CPU准备→单4090分层诊断/短训→CPU核验，不提交评测。

训练完成后先查看状态；只有核验允许评测时才运行：

```bash
bash submit_wrist_dino_key_open_pilot_v1.sh --status
bash submit_wrist_dino_key_open_pilot_v1.sh --evaluate
```

`--evaluate`提交四suite分片、并发最多2张4090，再做CPU统计，正常情况下共160新回合。零有效Key目标或两臂均未改变初始PNG会拒绝评测；相同PNG去重。不要重复提交活跃链。已有GPU任务作为等待依赖；不取消任务、不自动续试。

资源：准备4CPU/4G；训练4CPU/16G/4090×1；验证/统计2CPU/4G；评测每worker4CPU/12G/4090×1、并发2。估计训练1–2h、评测0.5–1.5h，待新GPU实测。RSS/CUDA峰值和失败成本见reports/WORKER_COSTS及time -v日志。

观察入口：`reports/DUAL_CLOSE.json`、`TARGET_SUMMARY.json`；`data/KEY_TARGETS.pt`；`training/single/`；`training/universal/<arm>/{status.json,final.png,probe_final.json,CANDIDATE_STEPS.parquet}`；每候选真实日志在attempts内。`results/RESULTS.md`只在正常PNG评测完成后生成。

CPU检查不启动VLA/renderer：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m pytest -q -p no:cacheprovider \
 research_runs/wrist_dino_key_open_pilot_v1/tests
```
