# OFT DINO action readout, version 1

Approved scope: authorized local LIBERO simulation; implement first-seed B/C
training, monitored four-suite evaluation, reuse signed UADA A. User manually
submits. No second seed is submitted automatically and no native weight is updated.

## Fixed comparison

- A: signed Spatial UADA B1 seed42, 500x50. Reuse float initialization and final
  PNG/results from `../wrist_dino_spatial_b12_diag500_v1`; signature
  `ac44e284cedc1bd5b47ca02702c2eeed20cffca1d2ecc6ae4505cea3f27c0349`.
- B: native continuous-action energy objective for outer1..400, original UADA
  MSE for outer401..500.
- C: same stages as B; in first400, mix native energy logits with frozen learned
  DINO action-readout logits. beta0.5 while the helper is closer to the clean
  anchor; beta0 when helper is closer to wrong target. Ties use beta0.5.
- B/C: fresh copies of A's exact initial float RGB; empty original AdamW,
  lr0.002/wd0/warmup20/schedule2000, scheduler steps once per outer as A.
  Exactly the same 250-window cyclic source order, B1, 500x50, native EOT seeds,
  50x50 wrist patch. This is 25,000 update attempts; first outer may have zero LR.
- Wrong target: original UADA expert-sign-opposite normalized boundary, X axis,
  8 slots. Clean anchor: actual full native policy on the same clean state.
- z(a)=-(MSE(a,a_clean),MSE(a,a_wrong))/(MSE(a_clean,a_wrong)+epsilon).
  Cross-entropy targets wrong class; mix logits, not independent scalar losses.
  Last100 uses unchanged UADA MSE. No branch gradient truncation or replacement.

## Teacher/readout

Use all1,259 cached source-train windows/405 episodes. Keep every episode touched
by the original250 patch windows in head training. Select81 different episodes
for validation using split_seed; group all windows/variants by episode.

For each window cache native front/wrist dense DINO, pooled frozen input-language
embedding, normalized8D proprio and8 X-action outputs for clean,p,p+,p-.
p cycles through old UADA, old dino_key, deterministic random RGB. Teacher direction
is minus sign of full-policy UADA gradient at p. p+/-=clip(p+/-2/255*direction).
Same underlying observation, patch transform/placement/EOT seed for all patched
variants. Clean images retain the official preprocessing. There are5,036 native
forwards and1,259 native backwards; no simulator rollouts for this data.

Readout: two-view256-token dense features, learned 2D and view positions,
width128, two decoder blocks/four heads, eight action queries conditioned by
language and proprio. Teacher labels are full-policy outputs, not expert actions.
AdamW lr1e-3, wd1e-4, batch64 windows, maximum30epochs, grad norm clip5.
Action MSE + clean-to-patched delta MSE + central finite-response MSE; divide each
term by its TRAIN-only per-slot target variance, floor1e-6, equal weights.
Select checkpoint by summed validation error. Record within-task feature shuffle.
Freeze the selected head during RGB optimization. This auxiliary source data and
compute are explicitly extra; equal patch-update budget is not equal total cost.

Inspiration: https://github.com/lafeat/lafeat (original classification setting).
Continuous-action energy anchors and finite RGB-response supervision are our
adaptation and are not established VLA results. Joint completion/expected-effect
confidence0.25 is a subjective planning estimate, not a calibrated probability.

## Monitoring and execution

Per update: objective/native MSE, beta, LR, RGB gradient norm, actual update,
clipping/saturation, quantized-channel changes. At outers1/25/125/250/375/400/401/500,
inners0/49: native/helper RGB gradient cosine/norm ratio, same-state/same-EOT native
action change and target MSE before/after, float-vs-PNG action gap, front/wrist DINO
change, saved RGB/gradient/action/feature tensors. Diagnostic passes never affect
optimizer leaves or policy parameters.

Deployment uses original wrapper.predict_action with a real uint8 PNG at(26,137).
No helper/replacement/optimizer enters execution. Read-only hooks capture DINO
and projector/LLM norms at decision0/1/4/8; save observations, actual actions,
proprioception, EEF/gripper trajectories and outcomes. Renderer uses original
NVIDIA EGL binding/cleanup fixes. One native model load per arm/suite, tasks1..10
sequential, checkpoint each finished episode. Failed workers write traceback
evidence under reports/FAILURES, and afterok dependencies prevent downstream runs.
Worker costs include preparation/model/cache loading and failed attempts; GPU
costs are summed in INTERNAL_EVIDENCE.json and exclude queue wait.

## Evaluation and decision

Four suites,10tasks, initial states40/41, paired original reset seeds. A80 and
clean80 are verified/reused; B80+C80 are new. Preserve native model checkpoint,
unnormalization, wait10, action chunk8, image resize/crop, maxstep and simulation.
Per-suite primary CASR averages task failure fractions among clean-success states.
Four-suite equal mean is predeclared for the new comparison (A47.5%). Report paired
wins/losses and every task separately. These are development initial states.

Advance only if C's equal mean exceeds A and B and at least two suite CASRs exceed
A. B>A and C<=B means objective gain without DINO contribution. Otherwise stop this
readout route. A second seed43 requires its own new matched A/B/C experiment and
manual submission; it is conditional on first-seed results, never auto-scheduled.

## Resource/submission contract

B and teacher+head start independently (2 GPUs). C depends on teacher+head.
B eval has2GPU workers and can overlap C training. C eval has3GPU workers and
depends on both C training and B eval, keeping this pipeline <=3 concurrent GPUs.
Each GPU worker has4CPU cores; teacher/head48GB host RAM, other GPU workers32GB.
Evaluation arrays have4suite shards, each reuses a loaded model across10tasks.
Summary is CPU only. Estimated first result3-4h after GPUs start, queue time extra;
new compute6-8GPUh. These are estimates, not measured throughput of this code.

Default run.sh performs CPU preparation only. --submit is the only submission
entry. It refuses an active same-signature pipeline and skips verified complete
stages. It never cancels jobs or cleans historical artifacts. Failed jobs resume
from recorded boundaries. Submitted sources/config must remain unchanged.
