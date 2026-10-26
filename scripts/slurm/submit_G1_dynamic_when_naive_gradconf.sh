#!/bin/bash
#SBATCH --job-name=G1_gradconf_dyn
#SBATCH --partition=a800
#SBATCH --qos=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[gradconf-slurm][error] CUDA_VISIBLE_DEVICES is empty; Slurm did not expose the assigned GPU." >&2
  exit 1
fi

PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON=${PYTHON:-/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python}
RUN_MODE=${RUN_MODE:-run}
MODEL_PATH=${MODEL_PATH:-/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-10}
DATA_ROOT=${DATA_ROOT:-/home/caijiasheng25/Desktop/roboticAttack}
SEARCH_ROOTS=${SEARCH_ROOTS:-"/home/caijiasheng25/Desktop/roboticAttack /home/caijiasheng25/Desktop/roboticAttack/openvla-main/dataset /home/caijiasheng25/data /home/caijiasheng25/robopatch_defense /home/caijiasheng25/UPA-RFAS /home/caijiasheng25 /mnt/beegfs/home/caijiasheng25/Desktop/roboticAttack /mnt/beegfs/home/caijiasheng25/Desktop/roboticAttack/openvla-main/dataset /mnt/beegfs/home/caijiasheng25/data /mnt/beegfs/home/caijiasheng25/robopatch_defense /mnt/beegfs/home/caijiasheng25"}
CONFIG_PATH=${CONFIG_PATH:-configs/experiments/oft_libero10_gradconf.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs/when_oft_gradconf"}
OUTPUT_STAGE=${OUTPUT_STAGE:-gradconf_oft_multiview}
SUITE_NAME=${SUITE_NAME:-libero_10}
RUN_NAME=${RUN_NAME:-G1_dynamic_when_naive}
NUM_BATCHES=${NUM_BATCHES:-2000}
BATCH_SIZE=${BATCH_SIZE:-2}

cd "${PROJECT_ROOT}"
mkdir -p logs/slurm "${OUTPUT_ROOT}/${OUTPUT_STAGE}/${SUITE_NAME}/${RUN_NAME}"
read -r -a SEARCH_ROOT_ARRAY <<< "${SEARCH_ROOTS}"

EXTRA_ARGS=()
if [[ "${RUN_MODE}" == "dry_run" ]]; then
  EXTRA_ARGS+=(--dry_run)
elif [[ "${RUN_MODE}" != "run" ]]; then
  echo "[gradconf-slurm][error] RUN_MODE must be run or dry_run, got ${RUN_MODE}" >&2
  exit 2
fi

echo "[gradconf-slurm] job=${SLURM_JOB_ID:-} node=${SLURMD_NODENAME:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "[gradconf-slurm] G1 dynamic run_name=${RUN_NAME} output_dir=${OUTPUT_ROOT}/${OUTPUT_STAGE}/${SUITE_NAME}/${RUN_NAME}"
nvidia-smi || true
"${PYTHON}" scripts/when/diagnose_oft_multiview_gradients.py \
  --config "${CONFIG_PATH}" \
  --model_path "${MODEL_PATH}" \
  --data_root "${DATA_ROOT}" \
  --search_roots "${SEARCH_ROOT_ARRAY[@]}" \
  --output_root "${OUTPUT_ROOT}" \
  --output_stage "${OUTPUT_STAGE}" \
  --suite_name "${SUITE_NAME}" \
  --run_name "${RUN_NAME}" \
  --mode dynamic \
  --num_batches "${NUM_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"
