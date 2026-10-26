# Exploring → OFT-DINO provenance contract

The source implementation is `VLAAttacker/white_patch/UADA.py` (SHA-256 recorded in every new patch artifact). The adapter is `VLAAttacker/white_patch/UADA_OFT_DINO.py`, whose attacker class directly subclasses the source `OpenVLAAttacker` and reuses the source `RandomPatchTransform`.

Preserved training structure:

1. random trainable `3x50x50` RGB patch;
2. AdamW and cosine warmup schedule;
3. a new source observation per outer iteration;
4. repeated random-location/affine EOT inside `inner_loop`;
5. one optimizer update and `[0,1]` clamp per inner iteration;
6. opposite action-boundary MSE with the original reproduction's `maskidx=0`;
7. checkpointed patch evolution.

Necessary OFT adaptations:

1. OpenVLA token logits are replaced by the OFT continuous L1 action head;
2. the single action is replaced by the eight-step action chunk, while retaining only DoF 0;
3. Front+Wrist are passed to OFT and the patch is applied only to Wrist;
4. clean continuous action and clean Wrist-DINO are recomputed from the current checkpoint for every outer iteration;
5. `exploring_oft_dino` adds a Wrist-DINO displacement gradient at a frozen 0.3 initial gradient-norm ratio.

Time-box deviations from the July full reproduction are explicit: batch 1 instead of 8, 200 outer iterations instead of 2,000, and no 1,000-batch validation sweep. Therefore this run is a provenance-correct screening experiment, not a full Exploring reproduction.

Forbidden dependencies include learned patches and cached legacy clean outputs. In particular, DINO-Norm-1500 and every RAMP/CID-CF/H1-H4/AG patch are absent from the configuration and trainer.
