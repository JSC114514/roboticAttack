#!/usr/bin/env python3
"""生成当前 3 天内要跑的 R/M/A 系列配置和 Slurm 脚本。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
SLURM_DIR = PROJECT_ROOT / "scripts" / "slurm"

OPENVLA_CKPT = "/home/caijiasheng25/Desktop/roboticAttack/models/openvla-7b-finetuned-libero-spatial"
OFT_CKPT = (
    "/home/caijiasheng25/.cache/huggingface/hub/"
    "models--moojink--openvla-7b-oft-finetuned-libero-spatial/"
    "snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b"
)
OFT_W_CKPT = "/mnt/beegfs/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-without-wrist"
OFT_W_SUBFOLDER = "libero_spatial"
DATA_ROOT = "/home/caijiasheng25/Desktop/roboticAttack"
LIBERO_ROOT = "/home/caijiasheng25/Desktop/roboticAttack/LIBERO"
PROJECT_ROOT_TEXT = "/home/caijiasheng25/robopatch_defense/roboticAttack-main"
OPENVLA_OFT_ROOT = "/home/caijiasheng25/Desktop/tex3D/openvla-oft"
OFFICIAL_M_ATTACK_IMPLEMENTATION = "openvla_oft_pixel_values_action_head_v1"
OFFICIAL_M_ATTACK_LOSS = "oft_clean_reference_hybrid_attack"
SBATCH_CPUS_PER_TASK = "4"
SBATCH_THREAD_ENV = """# Limit CPU thread pools; RLDS/TFDS otherwise oversubscribes badly when several jobs start together.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-2}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-2}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export TF_NUM_PARALLEL_CALLS=${TF_NUM_PARALLEL_CALLS:-2}
export TF_TRAJ_TRANSFORM_THREADS=${TF_TRAJ_TRANSFORM_THREADS:-2}
export TF_TRAJ_READ_THREADS=${TF_TRAJ_READ_THREADS:-2}
"""

COMMON = {
    "attack_method": "UADA",
    "patch_size": "0.05",
    "wrapper_patch_size": "3,50,50",
    "num_episodes": "50",
    "eval_tasks": "10",
    "eval_episodes_per_task": "5",
    "seed": "0",
    "task_suite": "libero_spatial",
    "unnorm_key": "libero_spatial_no_noops",
    "num_images_in_input": "2",
    "use_proprio": "true",
    "use_l1_regression": "true",
    "num_open_loop_steps": "8",
    "train_patch_position": "top_left",
    "eval_patch_position": "top_left",
    "center_crop": "true",
    "disable_cudnn": "false",
    "save_video": "true",
    "save_images": "true",
    "save_actions": "true",
    "save_trajectories": "true",
    "output_root": "outputs/multiview_attack",
    "project_root": PROJECT_ROOT_TEXT,
    "data_root": DATA_ROOT,
    "libero_root": LIBERO_ROOT,
    "openvla_oft_root": OPENVLA_OFT_ROOT,
    "view_key_map": "front=image_primary,wrist=image_wrist",
    "maskidx": "0",
    "batch_size": "8",
    "learning_rate": "0.001",
    "max_train_iters": "2000",
    "accumulate": "1",
    "warmup": "20",
    "inner_loop": "50",
    "geometry": "true",
    "filter_grip_train_to_1": "false",
    "resize_patch": "false",
    "multiview_fusion": "stitch",
    "multiview_debug": "false",
    "probe_use_dataloader": "false",
    "probe_batch_size": "1",
    "probe_instruction": "pick up the object",
    "python_bin": "python3",
}

HYBRID_STAGE1 = {
    "batch_size": "8",
    "train_patch_position": "anchored_eot",
    "eval_patch_position": "top_left",
    "patch_seed": "0",
    "eot_seed": "0",
    "eot_force_eval_anchor_sample": "true",
    "geometry": "false",
    "inner_loop": "deprecated_unused_pixel_values",
    "legacy_inner_loop_deprecated": "true",
    "clean_forward_enabled": "true",
    "action_boundary_weight": "1.0",
    "action_divergence_weight": "0.25",
    "feature_deviation_weight": "0.1",
    "weight_source": "uncalibrated_default",
    "clean_action_deadzone": "0.05",
    "action_boundary_value": "1.0",
    "time_decay": "0.85",
    "feature_hidden_state_layer": "0",
    "action_divergence_type": "mse",
    "trajectory_curvature_weight": "0.0",
    "repulsive_infonce_weight": "0.0",
    "pad_enabled": "false",
    "pad_weight": "0.0",
    "psm_enabled": "false",
    "psm_weight": "0.0",
    "robust_inner_enabled": "false",
    "eot_enabled": "true",
    "eot_samples": "2",
    "eot_warmup_fraction": "0.35",
    "eot_fixed_anchor_probability": "0.60",
    "eot_local_jitter_probability": "0.30",
    "eot_global_random_probability": "0.10",
    "eot_local_jitter_x": "16",
    "eot_local_jitter_y": "16",
    "eot_rotation_degrees": "10.0",
    "eot_scale_min": "0.90",
    "eot_scale_max": "1.10",
    "eot_shear_degrees": "5.0",
    "eot_brightness_delta": "0.0",
    "eot_contrast_delta": "0.0",
    "eval_eot_enabled": "false",
    "multiview_eot_transform_coupling": "shared",
    "rollout_validation_enabled": "false",
    "loss_weight_mode": "fixed",
    "gradient_conflict_mode": "none",
    "calibration_target_boundary_ratio": "1.0",
    "calibration_target_divergence_ratio": "0.4",
    "calibration_target_feature_ratio": "0.2",
    "calibration_weight_min": "0.001",
    "calibration_weight_max": "2.0",
    "calibration_eps": "1.0e-12",
}

EXPERIMENTS = [
    {
        "exp_id": "R0",
        "file_stem": "R0_openvla_front_clean",
        "series": "R",
        "model_name": "OpenVLA",
        "model_checkpoint": OPENVLA_CKPT,
        "input_adapter": "openvla_single_image",
        "input_views": "front",
        "patch_train_views": "none",
        "attack_views": "none",
        "train_patch": "false",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "none",
        "baseline_exp_id": "R0",
        "conda_env": "roboticAttack_2",
        "purpose": "OpenVLA front clean baseline",
        "slurm_time": "04:00:00",
    },
    {
        "exp_id": "R1",
        "file_stem": "R1_openvla_front_attack",
        "series": "R",
        "model_name": "OpenVLA",
        "model_checkpoint": OPENVLA_CKPT,
        "input_adapter": "openvla_single_image",
        "input_views": "front",
        "patch_train_views": "front",
        "attack_views": "front",
        "train_patch": "true",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "P_openvla_front",
        "baseline_exp_id": "R0",
        "conda_env": "roboticAttack_2",
        "purpose": "OpenVLA front patch attack reproduction",
        "slurm_time": "12:00:00",
    },
    {
        "exp_id": "M0",
        "file_stem": "M0_oft_front_wrist_clean",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "none",
        "attack_views": "none",
        "train_patch": "false",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "none",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "purpose": "OpenVLA-OFT front+wrist clean baseline",
        "slurm_time": "04:00:00",
    },
    {
        "exp_id": "M1",
        "file_stem": "M1_oft_train_front_attack_front",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "front",
        "attack_views": "front",
        "train_patch": "true",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "P_oft_front_mvctx",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "train_patch_space": "pixel_values",
        "eval_patch_space": "pixel_values",
        **HYBRID_STAGE1,
        "patch_grad_clip_norm": "0.0",
        "maskidx": "0,1,2,3,4,5,6",
        "purpose": "OpenVLA-OFT multi-view context, train front patch and attack front",
        "slurm_time": "12:00:00",
    },
    {
        "exp_id": "M2",
        "file_stem": "M2_oft_train_wrist_attack_wrist",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "wrist",
        "attack_views": "wrist",
        "train_patch": "true",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "P_oft_wrist_mvctx",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "train_patch_space": "pixel_values",
        "eval_patch_space": "pixel_values",
        **HYBRID_STAGE1,
        "patch_grad_clip_norm": "0.0",
        "maskidx": "0,1,2,3,4,5,6",
        "purpose": "OpenVLA-OFT multi-view context, train wrist patch and attack wrist",
        "slurm_time": "12:00:00",
    },
    {
        "exp_id": "M3",
        "file_stem": "M3_oft_front_patch_attack_wrist",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "front",
        "attack_views": "wrist",
        "train_patch": "false",
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M1",
        "pretrained_patch_family": "P_oft_front_mvctx",
        "patch_family": "P_oft_front_mvctx",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "eval_patch_space": "pixel_values",
        "purpose": "Load M1 front patch and attack wrist",
        "slurm_time": "06:00:00",
    },
    {
        "exp_id": "M4",
        "file_stem": "M4_oft_wrist_patch_attack_front",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "wrist",
        "attack_views": "front",
        "train_patch": "false",
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M2",
        "pretrained_patch_family": "P_oft_wrist_mvctx",
        "patch_family": "P_oft_wrist_mvctx",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "eval_patch_space": "pixel_values",
        "purpose": "Load M2 wrist patch and attack front",
        "slurm_time": "06:00:00",
    },
    {
        "exp_id": "M5",
        "file_stem": "M5_oft_train_multiview_attack_front",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "front,wrist",
        "attack_views": "front",
        "train_patch": "true",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "P_oft_multiview",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "train_patch_space": "pixel_values",
        "eval_patch_space": "pixel_values",
        **HYBRID_STAGE1,
        "patch_grad_clip_norm": "0.0",
        "maskidx": "0,1,2,3,4,5,6",
        "purpose": "Train front+wrist patch and attack front",
        "slurm_time": "12:00:00",
    },
    {
        "exp_id": "M6",
        "file_stem": "M6_oft_multiview_patch_attack_wrist",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "front,wrist",
        "attack_views": "wrist",
        "train_patch": "false",
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M5",
        "pretrained_patch_family": "P_oft_multiview",
        "patch_family": "P_oft_multiview",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "eval_patch_space": "pixel_values",
        "purpose": "Load M5 multi-view patch and attack wrist",
        "slurm_time": "06:00:00",
    },
    {
        "exp_id": "M7",
        "file_stem": "M7_oft_multiview_patch_attack_both",
        "series": "M",
        "model_name": "OpenVLA-OFT",
        "model_checkpoint": OFT_CKPT,
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front,wrist",
        "patch_train_views": "front,wrist",
        "attack_views": "front,wrist",
        "train_patch": "false",
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M5",
        "pretrained_patch_family": "P_oft_multiview",
        "patch_family": "P_oft_multiview",
        "baseline_exp_id": "M0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "eval_patch_space": "pixel_values",
        "purpose": "Load M5 multi-view patch and attack both views",
        "slurm_time": "06:00:00",
    },
    {
        "exp_id": "A0",
        "file_stem": "A0_oft_w_front_clean",
        "series": "A",
        "model_name": "OpenVLA-OFT_w",
        "model_checkpoint": OFT_W_CKPT,
        "model_subfolder": OFT_W_SUBFOLDER,
        "device": "cuda",
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front",
        "num_images_in_input": "1",
        "view_key_map": "front=image_primary",
        "patch_train_views": "none",
        "attack_views": "none",
        "train_patch": "false",
        "use_pretrained_patch": "false",
        "pretrained_patch_source_exp_id": "none",
        "pretrained_patch_family": "none",
        "patch_family": "none",
        "baseline_exp_id": "A0",
        "conda_env": "tex3d",
        "purpose": "OpenVLA-OFT_w front-only clean baseline",
        "slurm_time": "04:00:00",
    },
    {
        "exp_id": "A1",
        "file_stem": "A1_oft_w_m1_front_patch_attack_front",
        "series": "A",
        "model_name": "OpenVLA-OFT_w",
        "model_checkpoint": OFT_W_CKPT,
        "model_subfolder": OFT_W_SUBFOLDER,
        "device": "cuda",
        "input_adapter": "openvla_oft_multiview_wrapper",
        "input_views": "front",
        "num_images_in_input": "1",
        "view_key_map": "front=image_primary",
        "patch_train_views": "none",
        "attack_views": "front",
        "train_patch": "false",
        "use_pretrained_patch": "true",
        "pretrained_patch_source_exp_id": "M1",
        "pretrained_patch_family": "P_oft_front_mvctx",
        "patch_family": "P_oft_front_mvctx",
        "baseline_exp_id": "A0",
        "conda_env": "tex3d",
        "attack_implementation": OFFICIAL_M_ATTACK_IMPLEMENTATION,
        "attack_loss_type": OFFICIAL_M_ATTACK_LOSS,
        "eval_patch_space": "pixel_values",
        "source_patch_experiment": "M1",
        "patch_source_experiment": "M1",
        "transfer_type": "cross_model_same_view",
        "source_model": "OpenVLA-OFT",
        "target_model": "OpenVLA-OFT_w",
        "source_view": "front",
        "target_view": "front",
        "patch_retrained": "false",
        "purpose": "OpenVLA-OFT_w front-only direct transfer eval of current official M1 front patch",
        "slurm_time": "04:00:00",
    },
]


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def yaml_text(exp: dict[str, str]) -> str:
    config = dict(COMMON)
    config.update(exp)
    lines = ["# 自动生成；R/M 主实验配置。OpenVLA-OFT 通过 multiview/openvla_oft_wrapper.py 接入。"]
    for key, value in config.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def sbatch_text(exp: dict[str, str]) -> str:
    stem = exp["file_stem"]
    exp_id = exp["exp_id"]
    return f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a800
#SBATCH --qos=a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task={SBATCH_CPUS_PER_TASK}
#SBATCH --mem=128G
#SBATCH --time={exp['slurm_time']}
#SBATCH --job-name=mv_{exp_id}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# 显式指定 A800 分区、qos 和 gpu:a800:1，避免被默认 GPU 队列分配到非 A800。

set -euo pipefail

{SBATCH_THREAD_ENV}
RUN_MODE=${{RUN_MODE:-dry_run}}

if [ "$RUN_MODE" = "dry_run" ]; then
    DRY_RUN_FLAG="--dry_run"
elif [ "$RUN_MODE" = "run" ]; then
    DRY_RUN_FLAG=""
else
    echo "ERROR: RUN_MODE must be dry_run or run"
    exit 1
fi

PROJECT_ROOT=${{PROJECT_ROOT:-${{SLURM_SUBMIT_DIR:-{PROJECT_ROOT_TEXT}}}}}
cd "$PROJECT_ROOT"
mkdir -p outputs/multiview_attack logs/slurm

CONFIG_PATH="configs/experiments/{stem}.yaml"
CONDA_ENV_NAME="$(awk -F': *' '$1 == "conda_env" {{print $2; exit}}' "$CONFIG_PATH")"
setup_conda() {{
    local candidate
    for candidate in "${{CONDA_BASE:-}}" /apps/miniconda3 "$HOME/anaconda3" "$HOME/miniconda3"; do
        if [ -n "$candidate" ] && [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            source "$candidate/etc/profile.d/conda.sh"
            return 0
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        return 0
    fi
    return 1
}}

if [ -n "$CONDA_ENV_NAME" ] && [ "$CONDA_ENV_NAME" != "TODO_CONDA_ENV" ]; then
    if ! setup_conda; then
        echo "ERROR: 找不到 conda 初始化脚本，无法激活 $CONDA_ENV_NAME" >&2
        exit 2
    fi
    conda activate "$CONDA_ENV_NAME"
    echo "[env] activated conda env: $CONDA_ENV_NAME"
fi

GPU_ID=${{GPU_ID:-0}}
export GPU_ID
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-$GPU_ID}}
export CUDA_DEVICE_ORDER=${{CUDA_DEVICE_ORDER:-PCI_BUS_ID}}
echo "[gpu] GPU_ID=$GPU_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

python scripts/experiments/run_experiment.py \\
  --config "$CONFIG_PATH" \\
  --gpu_id "$GPU_ID" \\
  ${{DRY_RUN_FLAG}}
"""


def helper_scripts() -> dict[str, str]:
    return {
        "check_env.sh": """#!/bin/bash
set -euo pipefail

echo "[check] cwd=$(pwd)"
test -d VLAAttacker
test -d scripts/experiments
command -v python || true
command -v conda || true
command -v nvidia-smi || true
command -v sbatch || true
command -v squeue || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi -L | grep -i A800 || echo "[warn] 当前节点未看到 A800，提交后请以计算节点为准。"
fi
test -f configs/experiments/M1_oft_train_front_attack_front.yaml
test -f scripts/slurm/submit_M1_oft_train_front_attack_front.sbatch
mkdir -p outputs/multiview_attack logs/slurm
echo "[check] ok"
""",
        "submit_one.sh": """#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法: bash scripts/slurm/submit_one.sh M1 dry_run|run" >&2
  exit 2
fi

EXP_ID="$1"
MODE="$2"
case "$MODE" in
  dry_run|run) ;;
  *) echo "MODE must be dry_run or run" >&2; exit 2 ;;
esac

case "$EXP_ID" in
  R0|R1|M0|M1|M2|M3|M4|M5|M6|M7|A0|A1) ;;
  *) echo "当前只支持 R0/R1/M0-M7/A0-A1；S/B 系列已跳过。" >&2; exit 2 ;;
esac

check_dep() {
  local source_exp="$1"
  if [ "$MODE" = "run" ]; then
    python scripts/experiments/find_latest_patch.py --source_exp_id "$source_exp" --require_existing_patch >/dev/null
  fi
}

case "$EXP_ID" in
  M3) check_dep M1 ;;
  M4) check_dep M2 ;;
  M6|M7) check_dep M5 ;;
  A1) check_dep M1 ;;
esac

matches=(scripts/slurm/submit_"${EXP_ID}"_*.sbatch)
if [ ! -e "${matches[0]}" ]; then
  echo "找不到实验 ${EXP_ID} 对应 sbatch。" >&2
  exit 2
fi
if [ "${#matches[@]}" -ne 1 ]; then
  echo "实验 ${EXP_ID} 匹配到多个 sbatch。" >&2
  printf '%s\\n' "${matches[@]}" >&2
  exit 2
fi

RUN_MODE="$MODE" sbatch "${matches[0]}"
""",
        "submit_pair.sh": """#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "用法: bash scripts/slurm/submit_pair.sh M1 M2 dry_run|run" >&2
  exit 2
fi

EXP_A="$1"
EXP_B="$2"
MODE="$3"
if [ "$EXP_A" = "$EXP_B" ]; then
  echo "不允许重复提交同一个实验。" >&2
  exit 2
fi

SUBMIT_SLEEP_SECONDS=${SUBMIT_SLEEP_SECONDS:-30}

bash scripts/slurm/submit_one.sh "$EXP_A" "$MODE"
if [ "$SUBMIT_SLEEP_SECONDS" -gt 0 ]; then
  echo "[submit_pair] sleep ${SUBMIT_SLEEP_SECONDS}s before submitting $EXP_B"
  sleep "$SUBMIT_SLEEP_SECONDS"
fi
bash scripts/slurm/submit_one.sh "$EXP_B" "$MODE"
""",
        "submit_wave.sh": """#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "用法: bash scripts/slurm/submit_wave.sh W0 dry_run|run" >&2
  exit 2
fi

WAVE="$1"
MODE="$2"
case "$WAVE" in
  W0) exps=(R0 M0) ;;
  W1) exps=(R1 M1) ;;
  W2) exps=(M2 M5) ;;
  W3) exps=(M3 M4) ;;
  W4) exps=(M6 M7) ;;
  WA) exps=(A0 A1) ;;
  *) echo "未知 wave: $WAVE" >&2; exit 2 ;;
esac

if [ "$MODE" = "run" ]; then
  case "$WAVE" in
    W3)
      python scripts/experiments/find_latest_patch.py --source_exp_id M1 --require_existing_patch >/dev/null
      python scripts/experiments/find_latest_patch.py --source_exp_id M2 --require_existing_patch >/dev/null
      ;;
    W4)
      python scripts/experiments/find_latest_patch.py --source_exp_id M5 --require_existing_patch >/dev/null
      ;;
    WA)
      python scripts/experiments/find_latest_patch.py --source_exp_id M1 --require_existing_patch >/dev/null
      ;;
  esac
fi

SUBMIT_SLEEP_SECONDS=${SUBMIT_SLEEP_SECONDS:-30}

for i in "${!exps[@]}"; do
  exp="${exps[$i]}"
  bash scripts/slurm/submit_one.sh "$exp" "$MODE"
  if [ "$i" -lt "$((${#exps[@]} - 1))" ] && [ "$SUBMIT_SLEEP_SECONDS" -gt 0 ]; then
    echo "[submit_wave] sleep ${SUBMIT_SLEEP_SECONDS}s before next submit"
    sleep "$SUBMIT_SLEEP_SECONDS"
  fi
done
""",
        "status.sh": """#!/bin/bash
set -euo pipefail

OUTPUT_ROOT=${1:-outputs/multiview_attack}
find "$OUTPUT_ROOT" -maxdepth 2 -name status.json -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -20
""",
    }


README = """# R/M/A Slurm 脚本

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
- `loss_weight_mode=gradient_calibrated_fixed`

M1/M2/M5 默认启用 stage-1 clean-reference hybrid loss：clean forward 只作为
detached reference，优化目标为 clean-action opposite boundary、clean-adv action
divergence 和 projected visual-token feature deviation；训练时使用以 top-left 为锚点的
mixed EOT，rollout 评估固定左上角；PAD/PSM 和 robust inner 默认关闭。
"""


def main() -> None:
    for exp in EXPERIMENTS:
        write_text(CONFIG_DIR / f"{exp['file_stem']}.yaml", yaml_text(exp))
        write_text(SLURM_DIR / f"submit_{exp['file_stem']}.sbatch", sbatch_text(exp), executable=True)
    for filename, content in helper_scripts().items():
        write_text(SLURM_DIR / filename, content, executable=True)
    write_text(SLURM_DIR / "README.md", README)
    (PROJECT_ROOT / "outputs" / "multiview_attack").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs" / "slurm").mkdir(parents=True, exist_ok=True)
    print(f"generated_RM_configs={CONFIG_DIR}")
    print(f"generated_RM_slurm={SLURM_DIR}")


if __name__ == "__main__":
    main()
