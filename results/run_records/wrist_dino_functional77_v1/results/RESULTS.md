# Functional DINO target validation

**Decision: STOP_TARGET_NOT_SUPPORTED**

77 fresh episodes: 60 interventions, 7 shared clean controls, 10 actual-PNG controls. No training submitted.

In restore columns 1 means success; in injection columns 1 means failure. F/R/O = functional/random/orthogonal.

| Case | Suite | Task/state | PNG | Clean success | PNG failure | Restore F/R/O | Inject F/R/O | Eligible |
|---|---|---|---|---:|---:|---|---|---|
| c00 | libero_object | 1/40 | m01_uada_original_b1@500 | 1 | 1 | 1/0/0 | 0/0/0 | True |
| c01 | libero_object | 1/41 | m01_uada_original_b1@500 | 1 | 1 | 1/1/1 | 0/0/0 | True |
| c02 | libero_object | 1/40 | m02_dino_key_b1@500 | 1 | 1 | 1/1/1 | 0/0/0 | True |
| c03 | libero_10 | 0/40 | m01_uada_original_b1@500 | 1 | 1 | 1/0/0 | 0/0/0 | True |
| c04 | libero_10 | 0/40 | m02_dino_key_b1@500 | 1 | 1 | 0/0/0 | 0/0/0 | True |
| c05 | libero_10 | 0/40 | m03_dino_center_b1@500 | 1 | 1 | 1/0/0 | 0/0/0 | True |
| c06 | libero_10 | 1/40 | m01_uada_original_b1@500 | 1 | 1 | 1/1/0 | 0/0/0 | True |
| c07 | libero_spatial | 1/40 | m01_uada_original_b1@500 | 1 | 1 | 1/0/0 | 0/0/0 | True |
| c08 | libero_goal | 0/40 | m02_dino_key_b1@500 | 1 | 1 | 1/0/1 | 0/0/0 | True |
| c09 | libero_goal | 0/41 | m03_dino_center_b1@500 | 1 | 1 | 1/0/0 | 0/0/0 | True |

Shared clean states are shown per case in this table but executed only seven times.

| Direction | Matched control | Favorable | Unfavorable | Net / fixed 10 | Favorable suites | Pass |
|---|---|---:|---:|---|---|---|
| restore | random | 6 | 0 | 6/10 | libero_10, libero_goal, libero_object, libero_spatial | True |
| restore | orthogonal | 6 | 0 | 6/10 | libero_10, libero_goal, libero_object, libero_spatial | True |
| inject | random | 0 | 0 | 0/10 |  | False |
| inject | orthogonal | 0 | 0 | 0/10 |  | False |

Quality pass: True. Detailed checks: QUALITY.json.
Changed fresh controls: 

The decision selects or rejects a functional training target. It is not a deployable patch comparison against UADA.
Raw receipts/traces: ../episodes/c00…c09/. Input cases and old receipt paths: ../CASES.json.
Historical drift audit: HISTORICAL_AUDIT.json. Internal monitoring: MONITORING.json.
