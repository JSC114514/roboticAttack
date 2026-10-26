# Wrist-DINO closed-loop digital geometry pilot — 2026-10-17

Repair fix1: original run/source signature `429f36681f37d984c55b02bb2b66ff68d84b24b6dbf8afb07845829968c343ad` is preserved read-only. Goal QA failed because LIBERO SiteObjectState inherits an unimplemented joint getter; dispatch now uses object_state_type, reading site-specific joints directly. Native runtime, candidate set, selectors, scores, horizon, formal episode grid and resources remain identical. Object/Spatial/Long binary-goal QA passes are reused after source, configuration, artifact, goal-schema and external-runtime hash checks; only Goal QA is submitted. Both old successful QA and the failed Goal attempt remain in GPU cost accounting. This is an engineering repair, not a new scientific treatment. See REPAIR.md.

Authorized: local, controlled LIBERO simulation and owned/frozen OFT checkpoints. The assistant prepares code and CPU checks; the user submits Slurm. No model or patch training; no old experiment is edited. No automatic follow-up experiment.

## Question and comparisons

Use the established wrist-DINO vulnerability as a method premise. Test whether per-state selection by actual short simulated task progress beats selection by the original action-boundary surrogate. Do not run a new causal attribution campaign or train an approximate action head.

Four fresh conditions, four suites × all ten tasks × fixed development initial state 40 = **160 formal episodes**:

|Condition|RGB intervention|Selection|
|---|---|---|
|clean|None|Normal frozen policy|
|uada_static|Frozen Spatial-source UADA@500 50×50 PNG at (26,137)|No training or selection|
|uada_geometry|Seven full-wrist geometry candidates|Minimum native X-action opposite-boundary MSE|
|loop_geometry|Same seven candidates|Minimum short simulated task progress|

Primary comparison is loop_geometry − uada_geometry. Static UADA is a separate, different-constraint reference. A digital full-wrist stream and privileged simulation access are more powerful permissions than a static patch; beating that reference alone is not evidence of a superior equal-constraint DINO method. Both adaptive conditions receive the same candidate set, full native forward, simulator branching access, horizon and maximum queries per window. Actual total queries differ with episode duration / terminal branches and are reported.

This is a mechanism-motivated application, not a new discovery of DINO causality. DINO enters through the wrist spatial intervention and the native encoder; there is no extra DINO norm loss, synthetic feature injection, gradient cutoff, or auxiliary SigLIP objective. Monitoring observes actual consumed DINO dense output and downstream layers without replacing activations. No gradients are used by the finite-candidate planner.

## Frozen intervention and search

Only the 224×224 uint8 wrist image is changed. Front image, language, proprioception and victim weights remain native. Candidates: identity; centered scales 0.9 / 1.1; horizontal ±4 pixels; vertical ±4 pixels. No combinations or tuning on test outcomes. Bilinear sampling and edge padding prevent a black-border occluder. Parameters are fixed here before GPU runs; real RGB inputs are saved.

At each planning point, clone the current full simulator / controller / gripper / solver / observable / RNG state. Simulate each candidate for up to two **closed-loop** 8-action chunks, applying the same candidate transform to each fresh wrist observation. Identity supplies the current clean native 8-slot X action. Its sign defines the opposite ±1 normalized X target, reused across this window. This is an online UADA-style surrogate, not the original offline expert-supervised training procedure.

UADA selection minimizes mean X MSE over the simulated chunks. Loop selection lexicographically minimizes: (1) any task success in the hypothetical branch, (2) mean satisfied BDDL goal fraction, (3) mean dense progress, (4) the same X MSE as an exact-tie breaker, (5) candidate index. The dense proxy is fixed: binary on/in/stack goals use 0.5×actual grasp + 0.25×exp(−EEF distance/0.1m) + 0.25×exp(−object-to-target distance/0.1m); unary open/close/turnon/turnoff use 0.5×joint proximity to the object's declared success interval + 0.5×reach proximity. It is a task-progress heuristic, not a learned value function or a guarantee of eventual failure. All terms are logged. Missing goal / articulation support stops with explicit evidence rather than silently giving zero scores.

Restore the origin and commit the selected two chunks with fresh native predictions. Compare actions and endpoint physics with the selected hypothetical branch. Then replan from the new real state. A full replay/rollback contract failure aborts the episode; it is not counted as an attack success. Tolerances are frozen in config.json: restore 1e−10; committed physics 1e−6; action 2e−5. No effect-size gate is used in engineering QA.

## Runtime, pairing and endpoint

Reuse only the frozen UADA PNG and exact historical initial-state seeds. Rerun clean and static UADA in the same pipeline; no baseline training. Native RGB preprocessing, 10 wait steps, official action conversion, eight open-loop action slots, per-suite limits and checkpoint identities are imported from the signed Spatial@500 runtime. In each task clean runs first; the other three conditions have a predeclared seeded randomized order.

Final endpoint: complete-episode LIBERO task success/failure. CASR conditions on the fresh paired clean-success initial states and weights tasks equally within suite; four-suite means weight suites equally. Report all suites, raw outcomes, clean qualification, paired new/lost failures, query counts and GPU-worker hours, including failed attempts. No branch, timestep, or repeated planning query is an independent experimental replicate. One initial state/task makes this a pilot, not a multi-seed confirmation.

## Monitoring and recovery

Each adaptive planning window records every candidate, RGB changes, native DINO relative L2/cosine and 256 token delta norms, normalized/environment action chunks, selected candidate, real simulator goal bits/positions/grasp/joints, rollback error, committed replay error and query/time counts. DINO/projector/early/middle/last language-layer summaries are observed in windows 0/4/8. Save full actual first-plan inputs, native DINO outputs, and origin snapshot for every adaptive episode. LIVE_PLANS.jsonl gives live progress; Slurm stdout prints every five windows and every episode completion.

Completed episodes have source signatures and artifact hashes and are skipped on resubmission. An interrupted episode restarts from its frozen initial state in a **new attempt folder**; prior evidence is retained and failed costs are included. There is no claim of resuming an unsaved live simulator. A per-suite lock prevents duplicate workers, and submission refuses an active matching queue name or ambiguous prior sbatch response.

## Resources and submission

GPU engineering QA: four-suite array, at most four 4090 GPUs, each 4 CPUs / 32 GB RAM, 30-minute Slurm limit. It verifies deterministic restore/commit and actual native RGB/DINO interfaces, not a scientific hypothesis. Evaluation uses the same peak resources, 8-hour per-suite Slurm limit. CPU summary: 4 CPUs / 8 GB. QA → evaluation → summary dependencies are afterok, so failed engineering does not launch formal episodes.

No new training. Preliminary budget expectation: roughly 1–3 hours after allocation, strongly dependent on native inference and simulator reset costs; not measured yet. GPU QA records timings and `run.sh --estimate` projects per-suite runtime using the full episode step limits. The 8-hour directive is a scheduler ceiling, not a runtime prediction.

No calibrated joint probability of an efficacy win is available for this new intervention. Implementation completion and improved task failure are separate outcomes; CPU tests verify the former's contracts, not the latter. The experiment directly reports its result and does not automatically launch further validation or methods.
