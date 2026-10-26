# Wrist DINO internal action-control screen v1

This is a mechanism screen on the authorized local Spatial OFT model. It does
not train or deploy an RGB patch and it does not run LIBERO episodes. Its result
is evidence about which internal site can causally move the native continuous
action head under a bounded intervention.

The source is the frozen 40-record dual-close set produced by
`wrist_dino_key_open_pilot_v2`: 20 train and 20 validation observations, two per
Spatial task in each split. Every selected slot is one where both the expert and
the clean policy require closing. The screen reuses that 18.9 MB artifact and
does not deserialize the old 319 MB candidate bank.

## Candidate mechanisms

At DINO blocks 8, 16 and 22, the screen separately intervenes on spatial Q, K,
V, attention projection output, MLP output, block spatial state, CLS state and
four-register state. It also tests the consumed wrist DINO output, consumed
wrist SigLIP output and the projected wrist visual tokens. There are 27 sites.
Front-view values and all non-selected components remain native at each hook.
The projector site is a fused-path positive control; SigLIP is a redundancy
control rather than a proposed DINO method. Block-22 CLS/register states are
structural negative controls because the consumed intermediate output removes
prefix tokens immediately afterward; they cost little and verify that the hook
does not leak into spatial tokens or the action path.

For Q/K/V, scale is the clean RMS of each attention head. Other sites use one
clean-site RMS. A normalized variable is optimized for 12 Adam steps at lr 0.02
and projected to maximum relative RMS 0.2. Model weights never enter the
optimizer. Each site is evaluated against a deterministic random direction at
the same intended RMS.

Two endpoints are kept separate:

- `wrong_open`: minimize the mean hinge to native gripper output 0.6 on all
  dual-close slots; report every slot crossing the native open threshold 0.5.
- `uada_opposite`: minimize action-axis-0 MSE over all eight slots to the
  original expert-sign-opposite normalized boundary, exactly matching the
  Spatial UADA mask and target construction.

The initial endpoint gradient is accumulated separately on train and validation
records. The negative summed train gradient is normalized once. RMS 0.05, 0.1
or 0.2 is selected by mean endpoint loss on train records, then the direction
and amplitude are applied unchanged to every validation observation. A second
fixed random direction is matched to its RMS on those validation observations.
The report also gives train/validation aggregate-gradient cosine, cancellation
ratio, per-observation optimized-versus-random controls, action-axis changes and
gripper threshold crossings. The fixed held-out intervention directly probes a
shared internal direction. These tests still do not establish that a 50x50 RGB
patch can reach it.

This differs from previous work in three ways. Old Key training maximized a
feature distance without an action endpoint. The action-readout experiment
learned an auxiliary predictor at the final DINO output. Functional77 restored
or injected components only at the final 256 DINO tokens during rollouts. This
screen uses the unchanged native action head and resolves earlier DINO sites one
at a time, with held-out observations and matched random interventions.

## Execution and decision boundary

One 4090 worker loads the Spatial model once and writes one restartable artifact
per site and endpoint. A CPU worker aggregates the 54 arms. The screen nominates
internal causal leaders and shared-direction leaders separately. No PNG route,
closed-loop evaluation, second seed or expanded budget is automatically started.
The next experiment should apply RGB reachability tests only to the few leaders
that beat their matched random controls and retain train/validation direction
agreement.

The GPU worker requests four CPUs and 16 GiB host RAM. The prior comparable
worker measured 6.11 GiB process RSS HWM and 19.21/19.25 GB CUDA
allocated/reserved peaks; this screen records the same resource fields. The CPU
analysis worker requests two CPUs and 4 GiB. Expected GPU time is roughly
2--3 hours from the prior 0.43 s/update measurement; the four-hour limit leaves
restart headroom. Queue time is additional.

