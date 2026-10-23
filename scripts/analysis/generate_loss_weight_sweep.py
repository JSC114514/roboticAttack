#!/usr/bin/env python3
"""Generate short loss-weight sweep configs and Slurm scripts without submitting jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.view_batch_utils import read_simple_yaml, write_simple_yaml


DEFAULT_BASE_CONFIGS = [
    "configs/experiments/M1_oft_train_front_attack_front.yaml",
    "configs/experiments/M2_oft_train_wrist_attack_wrist.yaml",
    "configs/experiments/M5_oft_train_multiview_attack_front.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OFT hybrid loss-weight sweep configs.")
    parser.add_argument("--recommended_weights", default="outputs/loss_weight_calibration/recommended_weights.yaml")
    parser.add_argument("--base_configs", nargs="+", default=DEFAULT_BASE_CONFIGS)
    parser.add_argument("--output_config_dir", default="configs/loss_weight_sweep")
    parser.add_argument("--output_slurm_dir", default="scripts/slurm/loss_weight_sweep")
    parser.add_argument("--max_train_iters", default="300")
    parser.add_argument("--eval_tasks", default="2")
    parser.add_argument("--eval_episodes_per_task", default="5")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def factor_label(value: float) -> str:
    return str(value).replace(".", "p")


def sbatch_text(config_rel: str, job_name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a800
#SBATCH --qos=a800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

RUN_MODE=${{RUN_MODE:-dry_run}}
if [ "$RUN_MODE" = "dry_run" ]; then
  DRY_RUN_FLAG="--dry_run"
elif [ "$RUN_MODE" = "run" ]; then
  DRY_RUN_FLAG=""
else
  echo "ERROR: RUN_MODE must be dry_run or run" >&2
  exit 2
fi

PROJECT_ROOT=${{PROJECT_ROOT:-${{SLURM_SUBMIT_DIR:-/home/caijiasheng25/robopatch_defense/roboticAttack-main}}}}
cd "$PROJECT_ROOT"
mkdir -p logs/slurm outputs/loss_weight_sweep

CONFIG_PATH="{config_rel}"
CONDA_ENV_NAME="$(awk -F': *' '$1 == "conda_env" {{print $2; exit}}' "$CONFIG_PATH")"
if [ -n "$CONDA_ENV_NAME" ]; then
  if [ -f "${{CONDA_BASE:-}}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
  conda activate "$CONDA_ENV_NAME"
fi

GPU_ID=${{GPU_ID:-0}}
export GPU_ID
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-$GPU_ID}}
export CUDA_DEVICE_ORDER=${{CUDA_DEVICE_ORDER:-PCI_BUS_ID}}

python scripts/experiments/run_experiment.py \\
  --config "$CONFIG_PATH" \\
  --gpu_id "$GPU_ID" \\
  $DRY_RUN_FLAG
"""


def main() -> None:
    args = parse_args()
    weights_path = resolve(args.recommended_weights)
    weights = read_simple_yaml(weights_path) if weights_path.exists() else {}
    base_divergence = float(weights.get("action_divergence_weight", "0.25"))
    base_feature = float(weights.get("feature_deviation_weight", "0.1"))

    config_dir = resolve(args.output_config_dir)
    slurm_dir = resolve(args.output_slurm_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    for div_factor in [0.5, 1.0, 2.0]:
        for feat_factor in [0.5, 1.0, 2.0]:
            sweep_id = f"div{factor_label(div_factor)}_feat{factor_label(feat_factor)}"
            for base_config in args.base_configs:
                base_path = resolve(base_config)
                config = read_simple_yaml(base_path)
                config["sweep_id"] = sweep_id
                config["output_root"] = f"outputs/loss_weight_sweep/{sweep_id}"
                config["max_train_iters"] = args.max_train_iters
                config["eval_tasks"] = args.eval_tasks
                config["eval_episodes_per_task"] = args.eval_episodes_per_task
                config["num_episodes"] = str(int(args.eval_tasks) * int(args.eval_episodes_per_task))
                config["patch_seed"] = config.get("patch_seed", config.get("seed", "0"))
                config["eot_seed"] = config.get("eot_seed", config.get("seed", "0"))
                config["weight_source"] = "sweep_selected"
                config["action_boundary_weight"] = "1.0"
                config["action_divergence_weight"] = f"{base_divergence * div_factor:.10g}"
                config["feature_deviation_weight"] = f"{base_feature * feat_factor:.10g}"
                stem = Path(config.get("file_stem", base_path.stem)).stem
                out_name = f"{stem}_{sweep_id}.yaml"
                out_path = config_dir / out_name
                write_simple_yaml(out_path, config)
                try:
                    rel_config = str(out_path.relative_to(PROJECT_ROOT))
                except ValueError:
                    rel_config = str(out_path)
                slurm_path = slurm_dir / f"submit_{stem}_{sweep_id}.sbatch"
                slurm_path.write_text(
                    sbatch_text(rel_config, f"lw_{config.get('exp_id', 'M')}_{sweep_id}") + "\n",
                    encoding="utf-8",
                )
                slurm_path.chmod(0o755)
                generated.append(rel_config)

    readme = [
        "# Loss Weight Sweep",
        "",
        "Generated only; no jobs were submitted.",
        "",
        f"- recommended_weights: {weights_path}",
        f"- generated_configs: {len(generated)}",
        f"- short_train_iters: {args.max_train_iters}",
        f"- validation episodes per config: {int(args.eval_tasks) * int(args.eval_episodes_per_task)}",
    ]
    (slurm_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"generated_configs={len(generated)}")
    print(f"config_dir={config_dir}")
    print(f"slurm_dir={slurm_dir}")


if __name__ == "__main__":
    main()
