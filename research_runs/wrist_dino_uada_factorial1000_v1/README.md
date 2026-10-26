# UADA / DINO / frequency: three missing factorial cells

Implemented 2026-10-17. The user manually submits GPU work. No GPU job has been
submitted by the assistant. PLAN.md is the approved implementation contract;
SOURCE_LINEAGE.json locks both historical protocols and their reused evidence.

From the project directory:

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_wrist_dino_uada_factorial1000_v1.sh --preflight
bash submit_wrist_dino_uada_factorial1000_v1.sh
```

The first command is CPU-only: no queue query, model loading or submission. The
second is the manual submission entry. `--dry-run` only prints the five stages;
`--status` only reads local progress. `--prepare` verifies and materializes the
historical comparison table on CPU without submitting any job.

Three independent seed42 random-RGB runs, empty Adam, unchanged native UADA X
opposite-expert-boundary target,1000x50 updates each:

| arm | parent protocol | DINO sampling | frequency update | cosine horizon |
|---|---|---|---|---|
| legacy_freq | Long8,25fixed/25random+RGB | no | yes | 1000 |
| original_dino | Long4,original random geometry | yes | no | 2000 |
| original_freq | Long4,original random geometry | no | yes | 2000 |

Original-protocol runs stop at1000 with the2000 horizon unchanged. Each reuses its
own exact candidate-bank prefix and scoring/training EOT. No new auxiliary loss,
training prefix, lower-dimensional patch, hard appearance constraint or old
learned initialization is introduced. Native helper files and both EOT engines
are byte-identical snapshots; reports record the parent signatures and hashes.

The manual entry runs CPU preparation -> at most3x4090 training -> CPU asset and
monitor validation -> at most4x4090 evaluation -> CPU paired statistics. GPU
workers each request4CPU/32GB. Train array0..2; eval array0..11 maps arm=index//4,
suite=index%4 in Object/Spatial/Goal/Long order. GPU phases never overlap.
Estimated6-8h plus queue, about18-24 GPU-worker hours. Slurm ceilings8h training
and6h evaluation per shard are limits, not duration predictions.

Three new patches x four suites x ten tasks x ten states =1200 new episodes.
Historical2800 rows complete the two factorial tables. Model/source/data/runtime,
PNG and reset-grid drift stops reuse; it never silently retrains baselines or
counts an engineering failure as an attack win. States40..49 are development
states; one patch seed is not evidence of stability across training seeds.

Output navigation:

- `training/<arm>/training.parquet`, `status.json`, `resume.pt`, `outer_1000.png`.
- `training/<arm>/attempts/*/INNER_STEPS.jsonl`: every real update; native loss,
  gradient/update alignment, clipping/saturation and quantized changes.
- `training/<arm>/attempts/*/probe_*.{json,pt}`: same-input/EOT actual before/after,
  PNG/fixed-placement actions and native DINO observations; RNG is restored.
- `training/<arm>/MONITORING_STEPS.parquet`: validated50000-step history.
- `evaluation/<arm>/<suite>/task_*`: episode-resumable raw outcomes and actual
  actions, EEF/gripper paths; first-state native input/DINO samples under attempts.
- `results/FACTORIAL_{SUMMARY,CONTRASTS,PER_TASK}.csv`,
  `FACTORIAL_ROLLOUTS.parquet`, `NEW_TRACES.parquet`, `RESULTS.md`.
- `results/DECISION.json`: next-experiment recommendation, never a submission.
- `reports/SUBMISSION_RECEIPT.json`, `reports/FAILURES/`, `reports/WORKER_COSTS/`.

Follow-up is conditional: a stronger-protocol DINO gain supports a separately
planned second seed; a frequency-only win is credited to frequency; no gain ends
this enhancement route. No second seed, extra checkpoint campaign or new method
is automatically scheduled. The missing controls are not presented as innovation.

CPU tests (mock policy, no victim/renderer/GPU):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B \
  -m pytest -q -p no:cacheprovider research_runs/wrist_dino_uada_factorial1000_v1/tests
```
