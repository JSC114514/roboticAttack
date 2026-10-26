#!/usr/bin/env bash
PROJECT_ROOT=/mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
EXPERIMENT_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_relation_bias_v1"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${EXPERIMENT_ROOT}/code:${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:/home/caijiasheng25/Desktop/tex3D/openvla-oft:/home/caijiasheng25/Desktop/LIBERO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 _CHECK_PEFT=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false TF_FORCE_GPU_ALLOW_GROWTH=true
export LIBERO_CONFIG_PATH=/home/caijiasheng25/.libero
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${PROJECT_ROOT}"
if [[ ${1:-} == gpu ]]; then
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]] || { echo 'Exactly one numeric Slurm GPU mapping required' >&2; exit 11; }
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl REQUIRE_CUDA=1 DISALLOW_CPU_EXPERIMENTS=1
  OFT_STAGE_GPU=$("${PYTHON_BIN}" -B -c 'from common import cfg; print(cfg()["resources"]["cache"]["gpu_type"])')
  OFT_STAGE_GPU=${OFT_STAGE_GPU^^}
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}" OFT_STAGE_GPU
fi
