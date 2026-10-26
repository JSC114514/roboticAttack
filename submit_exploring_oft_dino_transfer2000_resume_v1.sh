#!/usr/bin/env bash
# User-run only. Reuse the completed 250-outer prefix; submit no A800 job.
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
RUN_ROOT="${PROJECT_ROOT}/research_runs/wrist_dino_transfer_v1"
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export OFT_DINO_TRANSFER_CONFIG="${PROJECT_ROOT}/configs/exploring_oft_dino/object_dino_transfer2000_v1.json"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -B -c 'from dino_transfer_common import config, prefix_state; cfg=config(require_prefix=True); saved,digest,patch=prefix_state(cfg); print("PREFIX_OK outer={} sha256={} patch={}".format(saved["outer"],digest,patch))'
bash -n slurm/exploring_oft_dino/96_transfer2000_collect_3090.sbatch
bash -n slurm/exploring_oft_dino/98_transfer2000_train_4090.sbatch
bash -n slurm/exploring_oft_dino/100_transfer2000_eval_3090.sbatch
if [[ ${1:-} == --preflight ]]; then
  echo 'RESUME_PREFLIGHT_PASS; no Slurm jobs submitted'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash submit_exploring_oft_dino_transfer2000_resume_v1.sh [--preflight]' >&2
  exit 2
fi
if [[ -n $(squeue -h -u "$(id -un)" -n oft_transfer_collect,oft_transfer_bank,oft_transfer_train,oft_transfer_validate,oft_transfer_eval,oft_transfer_stats) ]]; then
  echo 'An earlier transfer chain is still queued; refusing a duplicate submission.' >&2
  exit 3
fi
mkdir -p "${RUN_ROOT}/logs"
sbatch_id() { local receipt; receipt=$(sbatch --parsable "$@"); echo "${receipt%%;*}"; }
collect_smoke=$(sbatch_id --array=0 slurm/exploring_oft_dino/96_transfer2000_collect_3090.sbatch)
collect_rest=$(sbatch_id --dependency="afterok:${collect_smoke}" --array=1-9%5 slurm/exploring_oft_dino/96_transfer2000_collect_3090.sbatch)
bank=$(sbatch_id --dependency="afterok:${collect_rest}" slurm/exploring_oft_dino/97_transfer2000_bank_cpu.sbatch)
train=$(sbatch_id --dependency="afterok:${bank}" slurm/exploring_oft_dino/98_transfer2000_train_4090.sbatch)
validate=$(sbatch_id --dependency="afterok:${train}" slurm/exploring_oft_dino/99_transfer2000_validate_cpu.sbatch)
eval_job=$(sbatch_id --dependency="afterok:${validate}" slurm/exploring_oft_dino/100_transfer2000_eval_3090.sbatch)
stats=$(sbatch_id --dependency="afterok:${eval_job}" slurm/exploring_oft_dino/101_transfer2000_stats_cpu.sbatch)
echo "SUBMITTED collect_smoke=${collect_smoke} collect_rest=${collect_rest} bank=${bank} train=${train} validate=${validate} eval=${eval_job} stats=${stats}"
echo "Outputs: ${RUN_ROOT}"
