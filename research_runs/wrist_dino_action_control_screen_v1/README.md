# Internal action-control screening

The implementation screens 27 DINO/fusion sites on 40 frozen Spatial
dual-close observations. It produces no adversarial PNG and runs no simulator
episode. Internal interventions are diagnostic upper bounds, not deployable
attack results.

Run CPU checks without querying Slurm:

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_wrist_dino_action_control_screen_v1.sh --preflight
```

Submit the restartable GPU screen and dependent CPU analysis manually:

```bash
cd /home/caijiasheng25/robopatch_defense/roboticAttack-main
bash submit_wrist_dino_action_control_screen_v1.sh
```

The entry refuses duplicate active jobs, waits behind the user's active GPU
jobs with `afterany`, and uses `afterok` for its own analysis. It never cancels
or submits anything during `--preflight`, `--dry-run`, or `--status`.
Before calling `sbatch`, the entry creates the log, raw, result, lock, failure,
worker-cost and submission-archive directories that Slurm and the workers use.

Outputs are written under `raw/`, `reports/`, and `results/`. The main report is
`results/RESULTS.md`; `results/MECHANISM_SUMMARY.csv` retains every candidate,
endpoint and split. A timeout can be resumed with the same submission command;
completed site/endpoint arms are checksum-verified and skipped.

