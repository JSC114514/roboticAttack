# Cross-Dataset Transfer Configs

These configs evaluate patches trained on LIBERO-Spatial against other LIBERO suites.

- Source suite: `libero_spatial`
- Source patch suite: `libero_spatial`
- Runtime unnorm key: target suite key from `suite_model_paths.yaml`
- Target suites: `libero_object`, `libero_goal`, `libero_10`
- Target model: target-suite OpenVLA-OFT / OpenVLA-OFT_w checkpoint
- Latest fixed source manifests:
  - M1: `outputs/multiview_attack/20260616_211510_M1/patch_manifest.json`
  - M2: `outputs/multiview_attack/20260616_211513_M2/patch_manifest.json`
  - M5: `outputs/multiview_attack/20260616_211516_M5/patch_manifest.json`

This is a cross-suite transfer protocol, not per-suite patch retraining.
