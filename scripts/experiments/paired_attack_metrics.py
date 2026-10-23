#!/usr/bin/env python3
"""Compute paired clean-vs-attack success metrics.

The primary metric is conditional attack success rate (CASR): among episodes
that succeeded in the clean run, how many fail in the attacked run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def latest_run(output_root: Path, exp_id: str) -> Path:
    candidates = [
        path for path in output_root.glob(f"*_{exp_id}")
        if path.is_dir() and (path / "metrics_per_episode.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run with metrics_per_episode.csv for {exp_id}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "metrics_per_episode.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[int, int]:
    try:
        return int(row["task_id"]), int(row["local_episode_id"])
    except KeyError as exc:
        raise KeyError(
            f"{exc} missing from metrics_per_episode.csv; paired metrics require "
            "task_id and local_episode_id columns."
        ) from exc


def success(row: dict[str, str]) -> int:
    return int(str(row.get("success", "0")).strip() or "0")


def rollout_metric(run_dir: Path, key: str) -> Any:
    path = run_dir / "rollout_metrics.json"
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key, "")
    except json.JSONDecodeError:
        return ""


def compute_pair(clean_dir: Path, attack_dir: Path) -> dict[str, Any]:
    clean_rows = {row_key(row): row for row in read_rows(clean_dir)}
    attack_rows = {row_key(row): row for row in read_rows(attack_dir)}
    common_keys = sorted(set(clean_rows) & set(attack_rows))
    if not common_keys:
        raise ValueError(f"No paired episodes between {clean_dir} and {attack_dir}")

    clean_success = sum(success(clean_rows[key]) for key in common_keys)
    attacked_success = sum(success(attack_rows[key]) for key in common_keys)
    clean_success_attack_failure = sum(
        1 for key in common_keys
        if success(clean_rows[key]) == 1 and success(attack_rows[key]) == 0
    )
    clean_failure_attack_success = sum(
        1 for key in common_keys
        if success(clean_rows[key]) == 0 and success(attack_rows[key]) == 1
    )
    clean_success_rate = clean_success / len(common_keys)
    attacked_success_rate = attacked_success / len(common_keys)
    return {
        "clean_run": str(clean_dir),
        "attack_run": str(attack_dir),
        "paired_episodes": len(common_keys),
        "clean_success_count": clean_success,
        "attacked_success_count": attacked_success,
        "clean_success_rate": clean_success_rate,
        "attacked_success_rate": attacked_success_rate,
        "success_rate_drop": clean_success_rate - attacked_success_rate,
        "clean_success_attack_failure_count": clean_success_attack_failure,
        "clean_failure_attack_success_count": clean_failure_attack_success,
        "conditional_attack_success_rate": (
            clean_success_attack_failure / clean_success if clean_success else 0.0
        ),
        "attack_views": rollout_metric(attack_dir, "attack_views"),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "pair",
        "clean_success_rate",
        "attacked_success_rate",
        "success_rate_drop",
        "clean_success_attack_failure_count",
        "clean_failure_attack_success_count",
        "conditional_attack_success_rate",
    ]
    print(",".join(headers))
    for row in rows:
        pair_name = f"{Path(row['clean_run']).name}_vs_{Path(row['attack_run']).name}"
        values = [pair_name]
        for key in headers[1:]:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print(",".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired clean-vs-attack metrics.")
    parser.add_argument("--output_root", default="outputs/multiview_attack")
    parser.add_argument("--clean_exp_id", default="")
    parser.add_argument("--attack_exp_id", default="")
    parser.add_argument("--clean_run", default="")
    parser.add_argument("--attack_run", default="")
    parser.add_argument("--default_table", action="store_true", help="Print M0/M1 and A0/A1 comparisons.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of CSV table.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    pairs: list[tuple[Path, Path]] = []
    if args.default_table:
        for clean_exp_id, attack_exp_id in [("M0", "M1"), ("A0", "A1")]:
            try:
                pairs.append((latest_run(output_root, clean_exp_id), latest_run(output_root, attack_exp_id)))
            except FileNotFoundError:
                continue
    else:
        if args.clean_run:
            clean = Path(args.clean_run)
            if not clean.is_absolute():
                clean = PROJECT_ROOT / clean
        elif args.clean_exp_id:
            clean = latest_run(output_root, args.clean_exp_id)
        else:
            raise SystemExit("--clean_run or --clean_exp_id is required unless --default_table is used")

        if args.attack_run:
            attack = Path(args.attack_run)
            if not attack.is_absolute():
                attack = PROJECT_ROOT / attack
        elif args.attack_exp_id:
            attack = latest_run(output_root, args.attack_exp_id)
        else:
            raise SystemExit("--attack_run or --attack_exp_id is required unless --default_table is used")
        pairs.append((clean, attack))

    rows: list[dict[str, object]] = []
    for clean, attack in pairs:
        try:
            rows.append(compute_pair(clean, attack))
        except ValueError:
            if args.default_table:
                continue
            raise
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
