#!/usr/bin/env bash
set -euo pipefail
DIAG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export _CHECK_PEFT=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export OFT_STAGE_GPU=4090
if [[ -n ${SLURM_JOB_ID:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  [[ ${CUDA_VISIBLE_DEVICES} =~ ^[0-9]+$ ]] || { echo 'Exactly one allocated GPU mapping required' >&2; exit 11; }
  export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES}"
fi
exec /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B "${DIAG_ROOT}/code/diag_run.py" "$@"
