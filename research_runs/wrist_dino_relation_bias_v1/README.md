# Spatial DINO relation-bias experiment

This experiment trains two independent, shared 50x50 wrist RGB patches with a
frozen Spatial OFT checkpoint. `translation` follows the DINO direction induced
by moving a task-relevant visual entity along robot-base +X. `bearing` follows
the direction induced by rotating the entity-to-EEF horizontal bearing by up to
30 degrees. Per observation, both counterfactual families use the same physical
displacement (capped at 30 mm).

The loss contains no action discrepancy term. Validation chooses one method;
the held-out task images, counterfactuals, and responses are opened only after
`results/WINNER.json` is sealed.
Only real PNG inputs are used in closed-loop rollouts. Renderer counterfactuals
and branch replacements are diagnostics and are never counted as patch results.

`run.sh --submit` is the only scheduler-writing entrypoint and must be invoked
manually. Preflight, dry-run, and status modes never submit or cancel jobs.

After the implementation audit, preparation also imports the actual readonly
loader and verifies the native model code of all four checkpoints. Counterfactuals
are captured inside the native wrist-camera sensor callback; returned pixels,
physics, camera poses and observable clocks are checked without refreshing sensors.
The first GPU cache shard checks real PNG preprocessing, DINO features, gradients,
frozen weights and dual-branch action equivalence before full training.

Final-epoch recovery seals the saved patch without repeating updates. Each suite
evaluation subprocess writes its own success/failure RSS and CUDA cost receipt;
hard termination leaves explicit unavailable metrics. Projection q, orthogonal
residual and target distance are reported separately. The loss and selection rule
are unchanged.

Manual commands, run from the project directory:

```bash
bash research_runs/wrist_dino_relation_bias_v1/run.sh --submit
bash research_runs/wrist_dino_relation_bias_v1/run.sh --status --slurm
```

Submission refuses any active `rel1_` chain, including an older source revision.
Jobs pin the full source signature; superseded preparation and submission receipts
are preserved in `reports/REVISIONS` and `reports/SUBMISSIONS`. The existing epoch,
task-shard and suite receipts remain the resume boundaries.

Current allocation: partition `a800`, QOS `a800`, one A800 and four CPUs per GPU
job. Training requests 16G host RAM; cache and evaluation request 12G. CPU stages
remain two CPUs/4G. Both patches and all GPU stages run serially, so at most one
GPU is allocated at a time. CUDA and EGL checks read the configured GPU type;
runtime costs record the actual device name. The 11 audited cache shards (400
observations) are reused without rewriting the original artifacts.
Time limits remain cache 3h, both training methods together 4h, validation 2h and
held-out evaluation 2h; A800 throughput has not yet been measured.

After incident 143960, saved actions are decoded by stacking eight seven-value
slots rather than directly casting Parquet's outer object array. CPU preflight
checks every clean source chunk for exact 8x7 shape, finite float32 values and
valid executed-slot counts before submission; GPU workers repeat this inexpensive
check before loading the policy. Regression tests include actual Parquet round-trip
replay and all 8,062 real clean chunks. Original source Parquet and incident/cost
records remain unchanged.

Replay-boundary repair after incident144219: capture the same five selected
observations and stop immediately after the last one; unused episode suffixes
are not replayed or required to match terminal timing. Success strictly before
a required observation still fails with episode/slot/step diagnostics. The first
unfinished trajectory is probed with the real camera callback before loading VLA.
Completed shards can be reused only via a current-revision audit manifest that
checks unchanged capture functions/configuration/source assets and exact receipt,
artifact and observation identities. No original shard is rewritten.
