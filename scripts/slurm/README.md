# R/M/A Slurm 脚本

## Current OpenVLA-OFT-libero10 Gradient-Conflict Stage

The current recommended run list is paused down to:

- D0 data check:
  `python3 scripts/data/check_libero10_oft.py --dry_run`
- Model check:
  `python3 scripts/models/check_oft_libero10_model.py`
- Validation:
  `bash scripts/validate/validate_oft_libero10_gradconf.sh`
- G0-random static:
  `sbatch scripts/slurm/submit_G0_random_static_gradconf.sh`
- G1-dynamic:
  `sbatch scripts/slurm/submit_G1_dynamic_when_naive_gradconf.sh`

Physical occlusion, visibility proxy, and physical patch masking jobs remain in
the repository as legacy/paused entries and are not part of the current
recommended run list. The new G0/G1 scripts follow the existing WHEN Slurm
style and each request exactly one A800 GPU with `#SBATCH --gres=gpu:a800:1`.
They must be submitted manually.

只生成 R0/R1、M0-M7 和 A0-A1。M/A 系列已接入 OpenVLA-OFT rollout；S/B、APEX、防御、π0、patch size 消融当前不实现真实训练逻辑。

默认 dry-run：

```bash
sbatch scripts/slurm/submit_M1_oft_train_front_attack_front.sbatch
```

M 系列正式运行前先跑 OpenVLA-OFT probe：

```bash
sbatch scripts/slurm/probe_M0_oft_forward.sbatch
```

只验证数据管线、不加载 7B 模型：

```bash
PROBE_MODE=dataloader sbatch scripts/slurm/probe_M0_oft_forward.sbatch
```

只验证模型接口、使用 dummy observation：

```bash
PROBE_MODE=model sbatch scripts/slurm/probe_M0_oft_forward.sbatch
```

正式提交：

```bash
RUN_MODE=run sbatch scripts/slurm/submit_M1_oft_train_front_attack_front.sbatch
```

权重校准是可选分析，不再作为 M1/M2/M5 的前置步骤。若仍要尝试，必须用非交互 `sbatch`，A800 分区不支持 `srun --pty` 交互作业：

```bash
# smoke：每个 M 条件 2 个 batch
BATCHES_PER_CONFIG=2 OUTPUT_DIR=outputs/loss_weight_calibration_smoke \
  sbatch scripts/slurm/calibrate_oft_hybrid_loss_weights.sbatch

# 正式：M1/M2/M5 各 20 个 batch
BATCHES_PER_CONFIG=20 OUTPUT_DIR=outputs/loss_weight_calibration \
  sbatch scripts/slurm/calibrate_oft_hybrid_loss_weights.sbatch
```

校准脚本和 wrapper 都会检查 CUDA；如果 GPU 不可见会直接报错退出，不会静默跑 CPU。

M 系列波次：

```bash
bash scripts/slurm/submit_wave.sh W0 run  # R0, M0
bash scripts/slurm/submit_wave.sh W1 run  # R1, M1
bash scripts/slurm/submit_wave.sh W2 run  # M2, M5
bash scripts/slurm/submit_wave.sh W3 run  # M3, M4，依赖 M1/M2
bash scripts/slurm/submit_wave.sh W4 run  # M6, M7，依赖 M5
bash scripts/slurm/submit_wave.sh WA run  # A0, A1，A1 依赖 M1
```

辅助脚本：

```bash
bash scripts/slurm/submit_one.sh M1 dry_run
bash scripts/slurm/submit_pair.sh M1 M2 dry_run
bash scripts/slurm/submit_wave.sh W0 dry_run
bash scripts/slurm/submit_one.sh A1 dry_run
```

依赖：

- M3 依赖 M1 patch
- M4 依赖 M2 patch
- M6/M7 依赖 M5 patch
- A1 依赖 M1 patch，并作为 OpenVLA-OFT_w front-only cross-model transfer eval

M1/M2/M5 训练和 M3/M4/M6/M7 评估统一使用
`openvla_oft_pixel_values_action_head_v1`：

- `attack_loss_type=oft_clean_reference_hybrid_attack`
- `train_patch_space=pixel_values`
- `eval_patch_space=pixel_values`
- `train_patch_position=anchored_eot`
- `eval_patch_position=top_left`
- `eot_enabled=true`
- `eot_samples=2`
- `loss_weight_mode=fixed`

M1/M2/M5 默认启用 stage-1 clean-reference hybrid loss：clean forward 只作为
detached reference，优化目标为 clean-action opposite boundary、clean-adv action
divergence 和 projected visual-token feature deviation；训练时使用以 top-left 为锚点的
mixed EOT，rollout 评估固定左上角；PAD/PSM 和 robust inner 默认关闭。当前正式训练跳过梯度校准，使用固定默认权重。
