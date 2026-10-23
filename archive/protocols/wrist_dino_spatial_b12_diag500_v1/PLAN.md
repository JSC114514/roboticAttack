# Spatial B1/B2: unchanged 500x50 training, four-suite deadline pilot

2026-10-15. User requests results before 23:00 Asia/Shanghai, unchanged training,
four suites with reduced test states, and internal diagnostics. Six 4090 training
slots followed by at most eight 4090 evaluation slots; no other jobs cancelled.

Source changes jointly to the Spatial checkpoint, Spatial train demonstrations,
its action/proprio normalization and a newly estimated train-only DINO mean.
No other objectives, loss coefficients, optimizers, EOT or training budgets change.
Eight method families x B1/B2 = sixteen fresh patches, one initialization seed.
NCE-B1 remains an explicitly labelled zero-feature-gradient control.

All four suites, all ten tasks, predeclared states 40 and 41, clean plus sixteen
patches: 1,360 rollouts. This is an exploratory development comparison with wide
uncertainty, not independent evidence of stable superiority. No outcome-based
sample dropping, checkpoint selection, early stopping or silent missing results.

Every inner step writes objective, gradient and actual RGB-update scalars. At
predeclared outer 1/25/125/250/375/500, inner 0/49, isolated frozen-policy probes
measure both action objectives, RGB-gradient alignment, actual-step directional
derivatives, same-input before/after losses, and DINO/projector/LLM readouts.
Probes preserve RNG and never supply a gradient to the training optimizer.
Formal rollouts use unchanged deployment. Observations at blocks 0/4/12 of state
40 are saved for same-state clean/patch/DINO-restoration replay after that shard's
formal rollouts finish. Replay identity and action parity are checked separately.

Old experiments, models, source datasets and Codex history are untouched. Results
and diagnostics remain in this isolated directory. A deadline estimate is not a
completion guarantee; queue or interface failures are reported with real status.
