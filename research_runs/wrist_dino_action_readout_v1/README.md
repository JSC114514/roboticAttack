# DINO action readout: first-seed implementation

Run from the project directory:

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_action_readout_v1/run.sh
```

The default command is CPU preparation/checking only. Explicit user submission:

```bash
bash research_runs/wrist_dino_action_readout_v1/run.sh --submit
```

This submits B training and teacher/readout preparation in parallel, then C,
four-suite B/C evaluation, and CPU results. A UADA and clean are reused, with
hash verification. 160 new episodes, max3 concurrent GPUs,4CPU per GPU. No second
seed, cleanup, cancellation, or native model-weight edits are scheduled.

```bash
bash research_runs/wrist_dino_action_readout_v1/run.sh --status
```

Files:
- `reports/PREPARATION.json`: native/baseline hashes, actual train/validation split.
- `reports/SUBMISSION_RECEIPT.json`: job IDs and exact dependency commands.
- `logs/`: Slurm output/error.
- `teacher/INDEX.json`: all response-supervision cache records and hashes.
- `readout/history.parquet`, `readout/DIAGNOSTICS.json`: fitting/response/shuffle evidence.
- `training/{B_native_energy,C_dino_readout}/`: RGB snapshots, inner-step log,
  diagnostics, selected gradient/action/feature probes, resume state and status.
- `evaluation/<arm>/<suite>/`: per-task two-episode outcomes, traces, native captures.
- `results/RESULTS.md`, `DECISION.json`, `INTERNAL_EVIDENCE.json`: final comparison.
- `reports/FAILURES/`: worker failure tracebacks. No failure is converted to success.
- `reports/WORKER_COSTS/`: worker elapsed time including model/cache loading and failed attempts.

First result is estimated3-4h after resource allocation. Check actual throughput
and ETA in the logs. GPU runtime/interface/rendering checks occur inside the user
submitted jobs; CPU validation does not establish GPU compatibility or efficacy.

For retry, use the same --submit command only after this pipeline has no active
jobs. Complete stages are verified/skipped, and incomplete stages resume. Do not
edit source/config after submission or mix artifacts across versions.
