#!/usr/bin/env python3
"""Check suite-specific OpenVLA-OFT resources before running evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.view_batch_utils import is_todo, parse_views, read_simple_yaml


SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DEFAULT_UNNORM_KEYS = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
FAMILIES = ["oft", "oft_w"]
MODEL_WEIGHT_PATTERNS = [
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model*.bin",
]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def has_any(path: Path, patterns: list[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)


def missing_indexed_shards(path: Path) -> list[str]:
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    shards = sorted(set(str(value) for value in data.get("weight_map", {}).values()))
    return [shard for shard in shards if not (path / shard).exists()]


def has_stats_key(stats_path: Path, unnorm_key: str) -> tuple[bool, str]:
    if not stats_path.exists():
        return False, "missing dataset_statistics.json"
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"invalid dataset_statistics.json: {exc}"
    if unnorm_key not in data:
        keys = ", ".join(sorted(str(key) for key in data.keys())[:8])
        return False, f"stats missing {unnorm_key}; keys=[{keys}]"
    return True, "ok"


def model_dir_from_registry(registry: dict[str, str], suite: str, family: str) -> tuple[Path | None, str, list[str]]:
    checkpoint = registry.get(f"{suite}_{family}_model_checkpoint", "")
    subfolder = registry.get(f"{suite}_{family}_model_subfolder", "")
    issues: list[str] = []
    if is_todo(checkpoint):
        return None, subfolder, [f"{suite}_{family}_model_checkpoint is TODO"]
    base = resolve_path(checkpoint)
    model_dir = base / subfolder.strip("/") if subfolder.strip() else base
    return model_dir, subfolder, issues


def check_model(registry: dict[str, str], suite: str, family: str) -> dict[str, Any]:
    unnorm_key = registry.get(f"{suite}_unnorm_key", DEFAULT_UNNORM_KEYS[suite])
    model_dir, subfolder, issues = model_dir_from_registry(registry, suite, family)
    data_root = resolve_path(registry.get("data_root", "/home/caijiasheng25/Desktop/roboticAttack"))
    dataset_dir = data_root / "openvla-main" / "dataset" / unnorm_key / "1.0.0"
    if not dataset_dir.exists():
        issues.append(f"dataset missing: {dataset_dir}")

    if model_dir is None:
        return {
            "suite": suite,
            "family": family,
            "unnorm_key": unnorm_key,
            "subfolder": subfolder,
            "model_dir": "",
            "dataset_dir": str(dataset_dir),
            "ok": False,
            "issues": issues,
        }

    if not model_dir.exists():
        issues.append(f"model_dir missing: {model_dir}")
    else:
        required_files = [
            "config.json",
            "dataset_statistics.json",
        ]
        for rel_path in required_files:
            if not (model_dir / rel_path).exists():
                issues.append(f"missing {rel_path}")
        if not has_any(model_dir, MODEL_WEIGHT_PATTERNS):
            issues.append("missing model weights")
        missing_shards = missing_indexed_shards(model_dir)
        if missing_shards:
            issues.append(f"missing indexed shards: {','.join(missing_shards)}")
        if not has_any(model_dir, ["action_head--*_checkpoint.pt"]):
            issues.append("missing action_head checkpoint")
        if not has_any(model_dir, ["proprio_projector--*_checkpoint.pt"]):
            issues.append("missing proprio_projector checkpoint")
        stats_ok, stats_msg = has_stats_key(model_dir / "dataset_statistics.json", unnorm_key)
        if not stats_ok:
            issues.append(stats_msg)

    return {
        "suite": suite,
        "family": family,
        "unnorm_key": unnorm_key,
        "subfolder": subfolder,
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "ok": not issues,
        "issues": issues,
    }


def parse_list(value: str, allowed: list[str]) -> list[str]:
    if value.strip().lower() == "all":
        return allowed
    selected = parse_views(value)
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise SystemExit(f"Unknown values {unknown}; allowed={allowed}")
    return selected


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["suite", "family", "status", "unnorm_key", "model_dir", "issues"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        display = {
            "suite": row["suite"],
            "family": row["family"],
            "status": "OK" if row["ok"] else "MISSING",
            "unnorm_key": row["unnorm_key"],
            "model_dir": row["model_dir"] or "-",
            "issues": "; ".join(row["issues"]) if row["issues"] else "-",
        }
        for key, value in display.items():
            widths[key] = min(max(widths[key], len(value)), 90)

    def trim(value: str, width: int) -> str:
        if len(value) <= width:
            return value
        return value[: width - 3] + "..."

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-|-".join("-" * widths[header] for header in headers))
    for row in rows:
        values = {
            "suite": row["suite"],
            "family": row["family"],
            "status": "OK" if row["ok"] else "MISSING",
            "unnorm_key": row["unnorm_key"],
            "model_dir": row["model_dir"] or "-",
            "issues": "; ".join(row["issues"]) if row["issues"] else "-",
        }
        print(" | ".join(trim(values[header], widths[header]).ljust(widths[header]) for header in headers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/experiments/suite_model_paths.yaml",
        help="Flat YAML registry with suite-specific model paths.",
    )
    parser.add_argument("--suites", default="all", help="Comma-separated suites or all.")
    parser.add_argument("--families", default="all", help="Comma-separated families: oft,oft_w or all.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit nonzero if any selected resource is missing.")
    parser.add_argument("--json-output", default="", help="Optional path to write machine-readable check results.")
    args = parser.parse_args()

    registry_path = resolve_path(args.registry)
    registry = read_simple_yaml(registry_path)
    suites = parse_list(args.suites, SUITES)
    families = parse_list(args.families, FAMILIES)
    rows = [check_model(registry, suite, family) for suite in suites for family in families]
    print_table(rows)

    ready = [f"{row['suite']}:{row['family']}" for row in rows if row["ok"]]
    missing = [f"{row['suite']}:{row['family']}" for row in rows if not row["ok"]]
    print("")
    print(f"ready: {', '.join(ready) if ready else 'none'}")
    print(f"missing: {', '.join(missing) if missing else 'none'}")

    if args.json_output:
        output_path = resolve_path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")

    if args.fail_on_missing and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
