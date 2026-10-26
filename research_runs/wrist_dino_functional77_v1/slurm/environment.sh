#!/usr/bin/env bash
FUNCTIONAL77_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd -- "${FUNCTIONAL77_ROOT}/../.." && pwd)"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${FUNCTIONAL77_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:/home/caijiasheng25/Desktop/tex3D/openvla-oft:/home/caijiasheng25/Desktop/LIBERO:${PYTHONPATH:-}"
export _CHECK_PEFT=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TF_FORCE_GPU_ALLOW_GROWTH=true LIBERO_CONFIG_PATH=/home/caijiasheng25/.libero
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "${PROJECT_ROOT}"
if [[ ${1:-} == gpu ]]; then
  if [[ -z ${SLURM_JOB_ID:-} || -z ${CUDA_VISIBLE_DEVICES:-} || ${CUDA_VISIBLE_DEVICES} == *,* || ! ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]]; then
    echo "One explicit numeric Slurm GPU allocation required; no implicit GPU or CPU fallback" >&2
    exit 11
  fi
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl REQUIRE_CUDA=1 DISALLOW_CPU_EXPERIMENTS=1
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}" OFT_STAGE_GPU=4090
fi
