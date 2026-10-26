# Frozen protocol

- Source: the 400 clean rollouts and 8,062 native decision records in
  `wrist_dino_official_long4_2000_v1`; trajectories are replayed from their
  saved action chunks, seeds, and LIBERO initial states.
- Cohort: 24/8/8 tasks for train/validation/held-out. Train has 960 frames from
  192 trajectories; validation and held-out each have 160 frames from 32
  trajectories. Five decision observations are selected by fixed progress.
- Training: one translation and one bearing patch, identical random initial
  patch and 48,000 candidate updates each. Only the signed block-22 DINO
  counterfactual direction supplies gradients.
- Selection: task-equal validation closed-loop CASR; ties use offline native
  directional consistency, then the predeclared bearing tie-break. Held-out
  results cannot change the winner.
- Closed loop: clean, the common initial patch, the frozen Spatial UADA500 B1
  PNG, and candidate PNGs are normal RGB deployments through matched suite
  checkpoints. No internal feature replacement occurs in rollouts.
- Alignment: render at the original wrist camera sampling callback, verify exact
  returned native pixels, and preserve physics, camera poses and sampling clocks.
- Runtime checks: CPU preflight imports the reused readonly loader and checks all
  checkpoint logic; the first GPU cache shard verifies PNG preprocessing, feature
  equality, finite nonzero patch gradients, frozen weights and branch equivalence.
- Audit monitors: q, clean q, q change, orthogonal residual and target distance are
  reported separately. These diagnostics add no loss or selection threshold.
- Recovery and provenance: final saved epoch can be sealed without more updates;
  each suite subprocess records its own failure costs; submitted jobs pin the
  source signature and refuse overlapping relation-bias chains across revisions.
- Allocation update (2026-10-19): use one A800 in partition a800 with QOS a800, four CPUs,
  training16G/cache-and-evaluation12G. Keep single-GPU serial execution, the same
  experiment budget and existing time limits; measure A800 throughput at runtime.
- Source-action fix after incident143960: stack the eight saved slots, preserve
  their order, and reject non-8x7/non-finite chunks. Validate all 8,062 clean chunks
  and executed-slot counts on CPU before GPU policy loading; cover Parquet
  round-trip replay in the regression tests. Scientific targets and budgets stay fixed.
- Replay-boundary repair after incident144219: stop at the final selected
  observation, not at the original episode endpoint. Reject success before the
  required prefix; log exact episode/decision/slot/steps. Probe the first unfinished
  prefix before loading VLA. Reuse immutable completed shards only after auditing
  capture/input logic, source hashes, full artifact hashes and observation IDs.
