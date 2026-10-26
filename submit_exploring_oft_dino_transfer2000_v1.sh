#!/usr/bin/env bash
# Initial user-run chain only. Completed-prefix runs must use the resume script.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
RUN_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_transfer_v1"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export OFT_DINO_TRANSFER_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_transfer2000_v1.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"
if [[ -f ${RUN_ROOT}/training/warmup/status.json ]]; then
  echo 'Existing warmup detected; use submit_exploring_oft_dino_transfer2000_resume_v1.sh instead.' >&2
  exit 2
fi

# Read-only checks run before the first sbatch call.
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_DINO_TRANSFER_wrapper.py --dry-run
"${PYTHON_BIN}" -B VLAAttacker/UADA_OFT_ORIGINAL2000_TRANSFER_wrapper.py --dry-run
"${PYTHON_BIN}" -B scripts/collect_exploring_oft_dino_transfer.py --preflight
"${PYTHON_BIN}" -B scripts/build_exploring_oft_dino_transfer_bank.py --preflight
for job_script in slurm/exploring_oft_dino/{95,96,97,98,99,100,101}_transfer2000_*.sbatch; do
  bash -n "${job_script}"
done
if [[ ${1:-} == --preflight ]]; then
  echo 'PREFLIGHT_PASS; no Slurm jobs submitted'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash submit_exploring_oft_dino_transfer2000_v1.sh [--preflight]' >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}/logs"
sbatch_id() { local receipt; receipt=$(sbatch --parsable "$@"); echo "${receipt%%;*}"; }
prefix=$(sbatch_id slurm/exploring_oft_dino/95_transfer2000_prefix_a800.sbatch)
collect_smoke=$(sbatch_id --dependency="afterok:${prefix}" --array=0 slurm/exploring_oft_dino/96_transfer2000_collect_3090.sbatch)
collect_rest=$(sbatch_id --dependency="afterok:${collect_smoke}" --array=1-9%5 slurm/exploring_oft_dino/96_transfer2000_collect_3090.sbatch)
bank=$(sbatch_id --dependency="afterok:${collect_rest}" slurm/exploring_oft_dino/97_transfer2000_bank_cpu.sbatch)
train=$(sbatch_id --dependency="afterok:${bank}" slurm/exploring_oft_dino/98_transfer2000_train_4090.sbatch)
validate=$(sbatch_id --dependency="afterok:${train}" slurm/exploring_oft_dino/99_transfer2000_validate_cpu.sbatch)
eval_job=$(sbatch_id --dependency="afterok:${validate}" slurm/exploring_oft_dino/100_transfer2000_eval_3090.sbatch)
stats=$(sbatch_id --dependency="afterok:${eval_job}" slurm/exploring_oft_dino/101_transfer2000_stats_cpu.sbatch)
echo "SUBMITTED prefix=${prefix} collect_smoke=${collect_smoke} collect_rest=${collect_rest} bank=${bank} train=${train} validate=${validate} eval=${eval_job} stats=${stats}"
echo "Outputs: ${RUN_ROOT}"
