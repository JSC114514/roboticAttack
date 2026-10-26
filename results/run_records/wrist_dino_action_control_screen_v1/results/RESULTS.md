# Wrist internal action-control screen

This report is an internal causal screen on 40 frozen Spatial observations. It trained no RGB patch, ran no simulator episode, and is not CASR evidence.

## Wrong-open endpoint (validation ranking)

|Rank|Site|Family|Shared val wrong-open slots|Shared val any-open|Shared loss gain|Shared−random advantage|Per-observation upper-bound open slots|Train/val grad cos|
|---:|---|---|---:|---:|---:|---:|---:|---:|
|1|projector_wrist_output|fused projector|0.0%|0.0%|0.0012|0.0011|76.6%|0.0450|
|2|block_register_b16|register state|0.0%|0.0%|0.0004|0.0004|1.4%|0.0649|
|3|block_spatial_b08|block spatial|0.0%|0.0%|0.0003|0.0003|4.5%|-0.0200|
|4|k_spatial_b16|K|0.0%|0.0%|0.0002|0.0002|0.0%|0.0036|
|5|block_spatial_b22|block spatial|0.0%|0.0%|0.0002|0.0002|34.8%|0.0278|
|6|dino_wrist_output|consumed DINO|0.0%|0.0%|0.0002|0.0002|34.8%|0.0278|
|7|v_spatial_b08|V|0.0%|0.0%|0.0002|0.0001|0.0%|-0.0805|
|8|mlp_spatial_b22|MLP output|0.0%|0.0%|0.0001|0.0000|1.4%|0.0243|
|9|siglip_wrist_output|consumed SigLIP|0.0%|0.0%|0.0001|0.0000|0.7%|0.0059|
|10|attn_spatial_b16|attention output|0.0%|0.0%|0.0001|0.0001|0.7%|-0.0024|
|11|v_spatial_b22|V|0.0%|0.0%|0.0001|0.0001|0.0%|0.0138|
|12|mlp_spatial_b16|MLP output|0.0%|0.0%|0.0001|0.0001|0.7%|0.0005|

The native gripper threshold is strictly `g > 0.5`. The endpoint optimized mean hinge to 0.6 only on expert-and-clean dual-close slots.

## UADA opposite-action endpoint (validation ranking)

|Rank|Site|Family|Shared val loss gain|Shared−random advantage|Shared target progress|Shared action ΔL2|Per-observation upper-bound gain|Train/val grad cos|
|---:|---|---|---:|---:|---:|---:|---:|---:|
|1|block_register_b16|register state|0.0599|0.0611|0.0239|0.4820|0.5034|0.0481|
|2|block_spatial_b16|block spatial|0.0392|0.0372|0.0145|0.3782|0.8070|0.0458|
|3|projector_wrist_output|fused projector|0.0375|0.0365|0.0121|0.1785|1.2728|0.0699|
|4|k_spatial_b08|K|0.0278|0.0267|0.0100|0.4896|0.1409|0.0269|
|5|dino_wrist_output|consumed DINO|0.0254|0.0243|0.0079|0.2199|0.9562|0.0758|
|6|block_spatial_b22|block spatial|0.0252|0.0220|0.0079|0.2198|0.9568|0.0757|
|7|block_spatial_b08|block spatial|0.0220|0.0230|0.0076|0.3527|0.6788|0.0180|
|8|k_spatial_b22|K|0.0209|0.0182|0.0071|0.1683|0.2303|0.0030|
|9|attn_spatial_b08|attention output|0.0204|0.0205|0.0101|0.4972|0.3017|0.0149|
|10|attn_spatial_b22|attention output|0.0200|0.0197|0.0066|0.1254|0.6309|0.0790|
|11|mlp_spatial_b22|MLP output|0.0195|0.0184|0.0066|0.1347|0.6826|0.0752|
|12|siglip_wrist_output|consumed SigLIP|0.0149|0.0148|0.0051|0.1217|0.4184|0.0807|

The opposite target is the expert-sign-opposite normalized boundary on action axis 0 over eight slots, matching the Spatial UADA mask and target construction.

## Interpretation boundary

Candidates whose fixed train-derived direction both improves the held-out endpoint and beats a fixed matched-random direction: wrong-open `projector_wrist_output, block_register_b16, block_spatial_b08`; UADA-opposite `block_register_b16, block_spatial_b16, projector_wrist_output`.

A positive result identifies an internally controllable site. A deployable 50×50 universal RGB patch still has to reach a shared change at that site while passing a full native RGB forward. The next stage should test RGB reachability only for these leaders and retain SigLIP/projector controls.

All 27 candidates, both splits, random controls, per-axis action changes, gradient cancellation ratios and sign-agreement measures are in `MECHANISM_SUMMARY.csv`; all 2,160 observation-level rows are in `SCREEN_ROWS.parquet`.
