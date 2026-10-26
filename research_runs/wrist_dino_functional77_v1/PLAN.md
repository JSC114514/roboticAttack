# DINO functional component: 77-episode target validation

This is a new mechanism validation run, not a new patch or a benchmark claim.
Do not change the completed B1/B2 batch or the partially completed 592-episode run.
CASES.json selects exactly ten committed PNG failures with successful clean and
whole-DINO-restoration counterparts. They cover seven physical initial states in
all four suites. The failed/uncommitted Goal task-1 shard is not used.

## Fixed design

- 10 cases × 3 components × 2 directions = 60 full-trajectory interventions.
- 7 unique fresh native clean controls + 10 fresh native real-PNG controls = 17.
- Total 77 newly executed episodes; no new training, no new PNGs.
- Components: six-endpoint-VJP functional span; equal-rank random span; orthogonal
  complement of the functional span. The random and orthogonal interventions each
  match the functional intervention's ACTUAL post-BF16 feature-change L2 within
  3%, at the same physical observation and recipient endpoint. Orthogonal rank is
  the complement dimension, not the rank of the sensitive span. Log both ranks.
- Restore: recipient has the real wrist PNG; subtract the selected part of
  D_patch − D_clean. Inject: recipient has clean RGB; add that part.
- Recompute the basis at EVERY policy decision, using both clean and patched
  forwards at that branch's current physical state. No frozen initial-window basis.
- Three readouts follow the observed clean-to-patch normalized action difference
  in translation, rotation and gripper, across the eight predicted action slots.
  Two endpoints produce six VJPs. This is not a full Jacobian or semantic-channel
  decomposition. Only the 256 consumed wrist DINO tokens are replaced; CLS and
  four registers are excluded. SigLIP, front view, instruction and proprioception
  stay at the recipient endpoint's native values.
- Fixed PNGs: UADA B1, Key B1, Center B1 at outer 500, original native preprocessing.
- Six intervention orders randomized once within each case; controls first.
  Ten case workers avoid putting all three Long-task-0 patches on one worker.

## Fresh controls and frozen decision rule

Analyze all 77 episodes. A case earns directional efficacy credit only if its
FRESH clean control succeeds and its FRESH native PNG control fails. Cases that
change from historical outcomes remain in the tables and in the fixed denominator
of ten, but contribute zero favorable/unfavorable efficacy credit. Report their
actual outcomes explicitly. Historical bitwise trace differences are an audit,
not a condition for starting or completing this run.

For restore, favorable means success. For inject, favorable means failure.
For EACH direction and EACH matched control (random and orthogonal), compute
paired favorable discordances minus unfavorable discordances on eligible cases.
Proceed to prepare two equal-budget training arms ONLY when ALL FOUR contrasts
have net ≥3/10 favorable episodes, with favorable discordances in ≥2 suites for
each contrast. This is a predeclared practical screening threshold, not a
statistical significance claim or proof that a deployable PNG beats UADA.

Quality checks must also pass: native/engine and identity parity, post-cast norm
matching, complete fresh-control grid, complete asset checksums, identical physical
initial reset within current comparisons, and valid endpoint VJPs. At block zero
in every intervention episode check actual central differences at relative steps
0.002 and 0.01. For each of the three readout groups, at least 90% of endpoint
checks must have one step with matching nonzero sign and relative error ≤0.5.
Report the remaining finite-difference failures individually. Zero feature effect
or zero rank is recorded as such, never as a successful perturbation.

Output exactly one decision: TRAINING_TARGET_PASS, STOP_TARGET_NOT_SUPPORTED,
or STOP_DIAGNOSTIC_QUALITY. Incomplete runs output INCOMPLETE and list missing
episode IDs; no efficacy gate is evaluated. Nothing automatically starts training.

## Monitoring, integrity and resources

Per decision: current endpoint gradient norms/cosines, union rank/orthogonality,
requested and post-cast component norms, gains and leakage; same-state actions,
linear VJP predictions and real intervention effects; environment actions, gripper
threshold commands and actually executed slots; RGB hashes and pixel changes;
simulator states, EEF displacement, GPU memory and timing. Native controls log
the same executed action/state evidence. Save full observation, basis and VJPs at
predeclared blocks 0/4/12 only, with scalar traces at every block. The intervention
basis is still refreshed at all other blocks. Full snapshots are diagnostic,
not simulator-resume checkpoints.

Each episode commits atomically only after its trace and all assets are hashed.
Resume skips verified complete episodes. Preserve unfinished attempts and restart
them from the frozen initial state; never overwrite or append to an old attempt.
Source, runtime-library, native-source and immutable input hashes are frozen.
Submission checks provenance before touching the scheduler; refuse duplicate
active submissions and ambiguous prior submissions. No automatic GPU fallback.

CPU prepare (2 cores/8GB) → GPU array 0–9%4 (one 4090, 2 CPU, 32GB per worker)
→ CPU analysis (2 cores/8GB). User submits manually with run.sh --submit.
Estimated four-GPU wall time is 20–40 minutes excluding queue, based on completed
native/whole-intervention timings and six-VJP cost; record measured timing.
No Codex database access, credentials, training, cancellation or auto submission.
