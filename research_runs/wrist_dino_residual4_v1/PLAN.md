# Four-episode closure of the DINO component route

Only c03/c04/c05/c09 receive one new rollout each: inject the ORIGINAL, UNSCALED
orthogonal complement of the real-PNG DINO shift into the clean recipient.
Three records share Long task 0/state 40; one is Goal task 0/state 41. There are
TWO independent physical initial states, not four independent replicates.

At every decision use same-state native clean and PNG inputs to recompute the
six endpoint VJPs and their existing SVD span Q. Define delta=D_png-D_clean,
F=project(delta,Q), R=delta-F; consume BF16(D_clean+R). Do not shrink R to F's
norm, amplify R, add pixel perturbations or change the model. Norm gain is exactly
one. Log both requested residual norm and post-cast actual norm, residual/full
shift ratio, post-cast projection leakage, six VJPs, readout predictions, actual
actions, executed slots/gripper commands, RGB/state hashes and resource timing.
Post-cast norm error is checked against the ORIGINAL residual norm (3% tolerance),
not against the small functional norm used in the completed77 experiment.

Reuse the completed77 clean, native PNG, functional-injection and small-orthogonal
injection evidence and the coarse full-DINO injection evidence. Pin all selected
receipts/traces and source hashes. All four selected full-DINO injections failed;
all four small functional injections succeeded. Twenty old-vs-fresh native control
audits already match outcomes, steps and entire predicted action traces exactly.
New block-zero physical reset and clean action parity must match the reused
functional-injection reference. No native, functional or whole-DINO rollout is
submitted again. No additional suites or state selection are opened.

## Frozen decision after exactly four new episodes

- Incomplete -> INCOMPLETE; missing case IDs are listed, no efficacy conclusion.
- Failed quality -> STOP_DIAGNOSTIC_QUALITY. Report the specific implementation
  or numerical issue; never count it as a scientific negative result.
- Residual-only injection causes task failure in BOTH physical initial states
  (at least one Long patch AND the Goal patch) -> NEXT_PIXEL_FEASIBILITY.
  This permits preparing a bounded real-PNG feasibility proposal. It does not
  establish pixel reachability, a new attack method, or superiority over UADA.
- Failure confined to one initial state -> STOP_TWO_STATE_REQUIREMENT.
- No residual-only failures -> STOP_SINGLE_COMPONENT_ROUTE. Under these tested
  closed-loop interventions only the full shift reproduced failures; do not infer
  a semantic mechanism, nonlinear feature interaction, or all-model necessity.

Always report 0..4 retained failed patch×state records AND 0..2 physical states.
This is an enriched screening set; no suite-wide ASR, p value or generalization
claim is computed. After these four episodes analysis prints a decision and stops.
It never submits more experiments, starts training, changes the rule or searches
other ranks/layers. The completed77 STOP_TARGET_NOT_SUPPORTED result is preserved.

## Runtime and resources

CPU preparation (2 cores/8GB) -> GPU array 0-3%4 (one RTX4090, 2 CPU, 32GB each)
-> CPU analysis (2 cores/8GB). User submits manually; the assistant does not call
sbatch. GPU wall limit is 20 minutes per worker; this is a timeout, not an estimate.
Budget four-GPU wall time 10-15 minutes excluding queue; PREPARATION.json computes
an additional estimate from matching completed77 episode/block rates and model
startup overhead. No GPU/renderer/model is loaded during CPU preflight.

Frozen Probe, native runtime and numerical helpers are imported from completed77;
worker/pipeline/progress are namespace copies, so completed source files remain
unchanged. Only the component rule, four-case manifest and analysis are new.
Atomic completed-episode receipts allow verified skips; unfinished attempts are
preserved and restarted from initial state. Duplicate/ambiguous submissions are
blocked. Snapshots at blocks 0/4/12 are diagnostic, not simulator resume states.
