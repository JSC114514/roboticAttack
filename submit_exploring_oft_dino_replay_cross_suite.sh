#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/VLAAttacker:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -B -m pytest -q tests/test_exploring_oft_dino_replay.py
bash -n slurm/exploring_oft_dino/67_dino_replay_cross_suite_3090.sbatch
bash -n slurm/exploring_oft_dino/68_dino_replay_cross_suite_stats_cpu.sbatch
"${PYTHON_BIN}" -B scripts/run_exploring_oft_dino_replay_cross_suite.py --preflight
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo 'DINO_REPLAY_EVAL_CHECK_ONLY_DONE; no sbatch submitted'
  exit 0
fi
"${PYTHON_BIN}" -B scripts/check_exploring_oft_dino_replay_eval_ready.py
mkdir -p research_runs/wrist_dino_replay_v1/logs research_runs/wrist_dino_replay_v1/evaluation/four_suite
eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/67_dino_replay_cross_suite_3090.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/68_dino_replay_cross_suite_stats_cpu.sbatch)
stats_id=${stats_raw%%;*}
echo "DINO_REPLAY_EVAL=${eval_id} STATS=${stats_id}"
echo 'Four suites, five patches, at most eight RTX3090 evaluation tasks in parallel.'
