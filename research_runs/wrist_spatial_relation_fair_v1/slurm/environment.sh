#!/usr/bin/env bash
FC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd -- "${FC_ROOT}/../.." && pwd)"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${FC_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:/home/caijiasheng25/Desktop/tex3D/openvla-oft:/home/caijiasheng25/Desktop/LIBERO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 _CHECK_PEFT=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/home/caijiasheng25/.libero
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-2}" TF_NUM_INTEROP_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLCONFIGDIR="${FC_ROOT}/reports/matplotlib" TZ=Asia/Shanghai
cd "${PROJECT_ROOT}"
if [[ ${1:-} == gpu ]]; then
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]] || { echo 'One allocated GPU mapping required' >&2; exit 11; }
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}" OFT_STAGE_GPU=4090
  export REQUIRE_CUDA=1 DISALLOW_CPU_EXPERIMENTS=1
fi
