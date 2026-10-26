#!/usr/bin/env bash
PROJECT_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
SD_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_standoff_mechanism_v1"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${SD_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:/home/caijiasheng25/Desktop/tex3D/openvla-oft:/home/caijiasheng25/Desktop/LIBERO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 _CHECK_PEFT=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TF_FORCE_GPU_ALLOW_GROWTH=true
export LIBERO_CONFIG_PATH=/home/caijiasheng25/.libero
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -c 0
cd "${PROJECT_ROOT}"
if [[ ${1:-} == gpu ]]; then
  [[ -n ${SLURM_JOB_ID:-} && ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]] || { echo 'One numeric Slurm GPU allocation required' >&2; exit 11; }
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OFT_STAGE_GPU=4090 MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES=""
fi
