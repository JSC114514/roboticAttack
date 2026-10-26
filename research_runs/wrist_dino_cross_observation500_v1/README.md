# DINO cross-observation pilot

The frozen algorithm and comparison are in [PLAN.md](PLAN.md). New files are
isolated from the running factorial experiment and immutable historical runs.

From the project root, the USER submits:

```bash
bash submit_wrist_dino_cross_observation500_v1.sh
```

`--preflight`, `--dry-run`, `--status` are local and never submit/query Slurm.
The default action submits a CPU cache, one GPU training job, CPU validation,
an eight-worker evaluation array at concurrency4, and CPU statistics. GPU
training depends on all existing GPU jobs, including queued old evaluation;
the CPU cache can run earlier. Active/submission-ambiguous batches are blocked.
Retries are manual and keep completed work, checkpoints and failure evidence.

Resources: cache4CPU/12G (measured source-read peak7.43GiB); train1x4090+4CPU/16G; validate2CPU/4G;
eval per worker1x4090+4CPU/12G, max4; stats2CPU/4G. Time estimate4–6h after
dependencies/queue; new backward throughput is not measured yet.

Only `dino_cross_observation` is newly trained,500x50 actual Adam updates.
Baseline500 reuses its original saved PNG and is evaluated, not retrained.
Strong baseline1000 and clean outcomes reuse matching native4090 records.
New160 episodes, reused160, four suites,10 tasks, states40/41. No full-gradient
lookahead arm and no automatic next gap/seed/training extension.

Monitor: `training/dino_cross_observation/attempts/*/INNER_STEPS.jsonl`,
`probe_*.{json,pt}`, `training.parquet`, `GPU_INTERFACE_*.json`, `status.json`.
Evaluation: `evaluation/<condition>/<suite>/task_*_{rollouts,traces}.parquet`.
Final comparison: `results/RESULTS.md`, `SUMMARY.csv`, `CONTRASTS.csv` and
`DECISION.json`; resource evidence: `results/COSTS.json` and
`reports/WORKER_COSTS/*.json`, including failed attempts.

CPU verification:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
 /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m pytest -q \
 -p no:cacheprovider research_runs/wrist_dino_cross_observation500_v1/tests
```

Numerical/model/renderer QA must pass inside the allocated GPU job. Local tests
do not claim native CUDA execution or attack efficacy. The algorithm uses expert
successor observations and a first-order gradient approximation; it does not
differentiate through simulator futures or change the inference policy.
