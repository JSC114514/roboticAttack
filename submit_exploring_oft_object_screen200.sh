#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/caijiasheng25/robopatch_defense/roboticAttack-main
PYTHON_BIN=/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest -q tests/test_exploring_oft_dino_restart.py
"${PYTHON_BIN}" -m py_compile scripts/run_exploring_oft_object_screen.py scripts/analyze_exploring_oft_object_screen.py
"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import pandas as pd
import torch

cfg = json.load(open("configs/exploring_oft_dino/object_screen200.json", encoding="utf-8"))
assert cfg["manual_visual_review"].startswith("PASS_USER_")
assert len(cfg["state_indices"]) == 10 and len(cfg["conditions"]) == 2
old = pd.read_parquet(cfg["paths"]["existing_final100_rollouts"])
clean = old[old.condition == "clean"]
assert len(clean) == 100 and int(clean.success.sum()) == 99
for condition, name in cfg["patches"].items():
    path = Path(name)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert path.is_file() and tuple(payload["patch"].shape) == (3, 50, 50)
    assert payload["experiment_id"] == "EXPLORING-OFT-DINO-RESTART-TIMEBOXED-01"
print("EXPLORING_OFT_OBJECT_SCREEN200_PREFLIGHT_PASS new_episodes=200 eligible_expected=198")
PY

if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  echo "EXPLORING_OFT_OBJECT_SCREEN200_CHECK_ONLY_PASS; no sbatch submitted"
  exit 0
fi

eval_raw=$(sbatch --parsable slurm/exploring_oft_dino/03_object_screen200.sbatch)
eval_id=${eval_raw%%;*}
stats_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" slurm/exploring_oft_dino/04_object_screen200_stats.sbatch)
stats_id=${stats_raw%%;*}
echo "EXPLORING_OFT_OBJECT_SCREEN200=${eval_id}"
echo "EXPLORING_OFT_OBJECT_SCREEN200_STATS=${stats_id}"
echo "Adds 200 paired attack episodes; reuses 100 frozen clean episodes; at most 5 RTX3090 GPUs."
