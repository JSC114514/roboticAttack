# Spatial B1/B2 diagnostic comparison

16 fresh patches, unchanged 500x50 training, four suites/all ten tasks/states40/41,
1,360 paired rollouts. One patch-training seed; exploratory estimates only.

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_spatial_b12_diag500_v1/run.sh --submit
```

Without `--submit`, only CPU preflight runs. Existing active jobs are rejected;
never submit this command twice while this run is active. No cancellation or
cleanup is performed. Completed training arms and evaluation shards are verified
and skipped on a later explicit retry.

Sources/configuration/monitor plan: PLAN.md and SOURCE_PROVENANCE.json.
Progress: reports/SUBMISSION_RECEIPT.json, training/*/status.json, logs/.
Results: results/RESULTS.md and BATCH500_CONTRASTS.csv.
Diagnostics: training/*/inner_steps.parquet, diagnostics.parquet and selected
diagnostic_vectors_*.pt; evaluation/*/task_*_CAPTURES.pt and task_*_REPLAY.parquet;
results/INTERNAL_DIAGNOSTICS.md and DIAGNOSTIC_STATUS.json.

Diagnostic interventions never enter formal rollout actions. An incomplete or
failed diagnostic is reported explicitly; no claim of a root cause is automatic.
