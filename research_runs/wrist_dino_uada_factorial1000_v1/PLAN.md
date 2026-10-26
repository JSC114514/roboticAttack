# UADA / DINO contribution sampling / frequency update: missing factorial cells

Approved 2026-10-17, Asia/Shanghai. Authorized owned frozen VLA models in local
LIBERO simulation. Implement and CPU-check the approved three missing patches;
the user submits Slurm manually. No automatic second seed, job cancellation,
GPU reset, model update, or modification of any historical experiment.

## Treatments, budgets and reuse

| protocol | frozen parent | new arm | sampling | frequency | updates | cosine horizon |
|---|---|---|---|---|---|---|
| legacy | wrist_dino_official_long8_1000_v1 | legacy_freq | no | yes | 1000x50 | 1000 |
| original | wrist_dino_official_long4_2000_v1 | original_dino | yes | no | 1000x50 | 2000 |
| original | wrist_dino_official_long4_2000_v1 | original_freq | no | yes | 1000x50 | 2000 |

Reuse each parent's exact candidate bank (first 1000 outers for original),
normalized expert labels, seed42 random RGB initialization, empty AdamW,
LR0.002/warmup20, native uint8/PIL forward with the parent's explicit STE,
EOT RNG sequence and source-only DINO contribution selector. No old trained RGB
or Adam state initializes a new arm. Legacy keeps 25 fixed / 25 random with
brightness/contrast; original keeps every step's original random geometry,
and its randomized DINO scoring. No new feature/attention/TV objective.

All new arms minimize the unchanged native UADA opposite-expert-boundary X MSE
over eight action slots. DINO sampling scores are max(0,LrestoreDINO-Lpatch),
with the attacked SigLIP held identical, four candidates and the historical 2/1
weights. Frequency filters the Adam proposal, keeps every frequency positive,
matches the first nonzero clamped update norm and freezes that scale. This does
not establish monotonic descent or guarantee improved episode failure.

Historical original/DINO/combined cells and clean records are reused only after
source/configuration signatures, source snapshots, dataset/cache hashes, model
metadata/weight stat, exact episode/reset grid, runtime and PNG lineage checks.
Any mismatch is an engineering error and stops the pipeline; it never becomes
an attack outcome or silently triggers baseline retraining. Historical files
are read-only dependencies and never overwritten. A frozen lineage manifest
records every reused input. New code snapshots preserve both parent engines.

## Evaluation and analysis

Only three new PNGs, four suites x ten tasks x states40..49 = 1200 new episodes.
Deploy a static50x50 RGB PNG at wrist(26,137), unchanged front/language/proprio,
native checkpoints/normalization, wait10, action chunk8 and suite time limits.
Use the exact parent reset seed for each task/state. All states are development
states. No action/feature injection or training replacement enters deployment.

Combine new1200 and reused2800 rows into two independent five-condition tables
(clean plus four factorial cells), 4000 rows. Clean eligibility is parent-specific
and paired; task-equal CASR within suite and suite-equal mean are fixed endpoints.
Report SR/FR, every task, paired new/lost failures, six factorial contrasts and
task/state paired bootstrap intervals (10000 draws). One training seed is one
training replicate, not 1000 outers, 400 episodes, or independent second seeds.
Do not interpret protocol differences as isolated EOT effects: schedules and
sampling-score inputs differ too.

Final recommendations use the stronger original protocol: combined beating both
UADA and frequency-only supports a manually approved second-seed comparison;
frequency-only winning supports keeping that effective component; no gain stops
this enhancement route. A DINO-only win is reported explicitly. No outcome
automatically launches any further experiment, and no feature threshold decides
scientific success. Filling these controls is not a new-method novelty claim.

## Monitoring and recovery

Every update records the actual UADA loss, gradient norm, actual gradient/update
alignment, pre/post pixel range/saturation, update norm and quantized changes.
Probe outers1/25/125/250/500/750/1000 at inner0/49 with the same state/EOT before
and after actual updates, quantized PNG inputs, DINO displacement, per-axis
actions and fixed deployment inputs. Probes restore all RNG state, never replace
training gradients and never update the frozen policy. Full probe tensors and
per-attempt logs permit replay. Resume only at the parent's ten-outer checkpoint;
logs after that boundary are retained but excluded from completed history.

Formal rollouts save actual executed actions, EEF/gripper trajectories and native
DINO read-only summaries. Completed task/condition grids have source/PNG/result
hashes and are skipped when verified; interrupted tasks restart uncompleted
episodes from the same initial state with a new attempt log. Engineering errors
stop afterok dependencies and remain distinct from task failure.

## Resources and duration

CPU preparation -> train3x4090 concurrently -> CPU validation -> evaluation
12 arm/suite shards, at most4x4090 -> CPU stats. Each GPU worker4CPU/32GB.
Training Slurm ceiling8h; evaluation ceiling6h; ceilings are not ETA promises.
Historical1000 training3.31/3.76/3.88h suggests4-5h with monitoring; testing
about1.5-3h on four GPUs; total6-8h plus queue wait, roughly18-24 GPU-worker hours.
No second seed is in this budget. Sources and submission signature freeze before
the manual entry is delivered. Default preflight/status paths never call Slurm.
