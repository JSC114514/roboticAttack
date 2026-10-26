# Frozen execution protocol

- Independent directory; original experiments and model weights read-only.
- Complete DINO geometry feature matching: projection loss plus orthogonal residual, original energy weighting.
- Wrist RGB50x50 at [26,137], block22, frozen Spatial model; no action loss, no EOT.
- Pilot fit48 and verify48 observations from24 tasks; disjoint source trajectories; frame_rank1.
- Four initial workers: individual(2 objectives x48x100), shared_projection4800, shared_full4800, oracle96.
- Paired native PNG diagnostics after pilot workers; no sample filtering based on outcomes.
- Night3 independent48000-update candidates: full42/full43/projection43. Reuse old projection42 PNG.
- Initial PNGs42/43 retained; seed42 equals historical initial exactly. Primaryfull42 declared before results.
- Full960-observation training pool; old task/row scheduling, lr0.002 warmup20 cosine.
- Validation8 tasks xstate48; heldout8 tasks xstates0/1. Rerun clean and all7 patch conditions.
- Offline actual suite and fixed Spatial checkpoints kept separate. No checkpoint/seed selection by test.
- 4090 only, one per worker; <=4 concurrency, GPU train16G/eval12G and4CPU, CPU4G/2CPU.
- Manual submission only, source pinning, no automatic cancellation or expanded sweeps.
- Per-worker30s heartbeat, progress/gradient/pixel/geometry diagnostics, RSS/CUDA peaks, failed-attempt receipts.
- Reports distinguish representations, intended actions and closed-loop failures; task/trajectory bootstrap.
- Planned overnight deadline2026-10-20 08:00 Asia/Shanghai, not a guaranteed queue/runtime completion.
