# Wrist-DINO cross-observation UADA pilot, v1

Approved 2026-10-17. Local owned OpenVLA-OFT and controlled LIBERO simulation.
Only a static external wrist RGB50 patch is trained. No model parameters change.
The assistant prepares and CPU-checks this run; the user submits it manually.

## Frozen algorithm

Let L_t(P) be the existing expert-opposite normalized X MSE, averaged over the
eight action slots of the native OFT head. Do not replace it with action L2,
DINO feature displacement, a learned readout, or a terminal-return surrogate.

1. Use candidate zero from each of the parent's first500 four-candidate rows.
   No DINO contribution sampling or new source selection. The source is the
   original Long4_2000 official RLDS expert bank and seed42 initialization.
2. Pair expert observation t with t+8 in the SAME episode, each with complete
   eight-action labels. This is the next decision observation, not t+1 and not
   the actual future state reached under the attack. Fix this gap; no gap sweep.
   Keep the14 windows without complete successors and use ordinary native UADA
   on those rows, without resampling. Verify this count against the actual bank.
3. At P, compute g_t = grad L_t and its wrist-DINO path component gD_t.
   Obtain upstream VJPs for the attacked DINO/SigLIP tensors, then encoder VJPs;
   g_t=gD_t+gS_t. Check this decomposition against direct native RGB autograd
   on the allocated GPU. Both visual branches remain native in every forward.
4. Preview the exact current Transformers AdamW update with g_t, including its
   existing moments and current LR, WITHOUT changing optimizer state. Its
   clipped RGB delta norm sets the virtual-step norm. Set
   Pprime=clip(P-alpha*gD_t), matching that norm AFTER clipping by scalar search.
   Zero preview steps use Pprime=P. An unavailable DINO direction or an
   unattainable norm falls back to ordinary UADA for that update and is logged.
5. At expert t+8, compute the full native gradient q=grad L_(t+8)(Pprime).
   Feed (g_t+q)/2 to the real AdamW on P ONCE. Pprime is detached and never
   becomes the public patch. This is a first-order approximation: no Hessian,
   derivative through the virtual step, exact bilevel claim or closed-loop
   gradient. Missing successors use g_t alone.
6. Keep the parent's all-random EOT, JPEG/PIL forward plus explicit STE,
   wrong-X target, AdamW(.002, weight_decay0), 20-outer warmup and cosine2000.
   Use the same seeded EOT for both observations. Train500 outer x50 real
   updates, with fresh seed42 RGB and empty moments. No checkpoint selection:
   only final500 is evaluated. Save0/250/500 for provenance, not a search.

## Comparison and budget

One new method only. No full-gradient lookahead arm; that is reserved for a
future explicitly requested ablation. Compare with saved native UADA500 and
the stronger native UADA1000, never a selected weak baseline alone.

New evaluation: method500 and saved UADA500, each four suites x10 tasks x
states40/41 =160 episodes. Reuse80 matching clean and80 matching UADA1000
episodes from the immutable parent, checking PNG identity and reset seeds.
Report320 rows, task-equal conditional attack success per suite, exact paired
failure switches and an equal-suite descriptive mean. Primary comparison is
method500 minus UADA1000. Also show method500 minus UADA500. One training seed
and two states/task form a pilot, not a claim of seed stability or a powered test.
There are up to49300 native action-loss gradients (486 paired +14 fallback rows,
50 inner updates), plus bounded diagnostics. UADA1000 used50000 native losses;
the new DINO VJP decomposition and monitoring have extra overhead. Report measured
GPU time; do not call these equal FLOP budgets.

## Monitoring and implementation fidelity

Record every actual update: both window IDs, EOT seed, native losses, DINO/full/
query gradient norms and cosines, virtual norm/alpha/fallback, actual RGB delta,
quantized-pixel changes and dot products with current/query gradients.
At fixed1/25/125/250/500 outer and inner0/49, compare same-input native before/
after and saved-PNG roundtrip, fixed deployment placement, per-axis actions,
clean-relative DINO displacement, and query-at-P versus query-at-Pprime.
Diagnostics preserve Python/NumPy/Torch/CUDA RNG and never step the optimizer.
Native rollout hooks only observe DINO; save executed actions and EEF/gripper
trajectories. Training, failed attempts and evaluation record resource peaks.
Check CPU analytic gradients, exact Adam preview, clipping, no virtual-state
mutation, one real step, first-order formula, fallback, monitoring neutrality,
interrupt/resume, historical grid pairing, submission guards and script resources.
Native model/renderer/backward QA runs only inside the user's GPU allocation.

## Resource policy and execution

CPU cache: 4cores/12G, training: 1x4090/4cores/16G, validation: 2cores/4G,
evaluation: 1x4090/4cores/12G per worker, at most4 workers, statistics: 2cores/4G.
Cache preparation releases the parent's full2.82GiB bank before importing TF.
The actual source audit measured7.43GiB process RSS HWM while deserializing it;
12G preserves headroom, whereas the initial8G estimate was too close to that peak.
Keep at most one observation's differentiable graph live. Log host RSS HWM and
CUDA allocated/reserved peaks, not invented whole-job memory measurements.
Estimated runtime: training3.5–5h, evaluation0.5–1h, total4–6h plus queue and
any prior batch dependency. This new backward workload has not been timed yet.
CPU cache may overlap old GPU work; new GPU training waits for the user's
existing GPU jobs (including queued evaluation), preventing resource overlap.

On completion, directly report whether the pilot exceeds UADA1000, with signed
per-suite results and implementation/resource evidence. No automatic second seed,
larger time gap, training extension, extra baseline method, or job resubmission.
