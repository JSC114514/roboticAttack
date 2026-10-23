#!/usr/bin/env python3
"""Discover and sanity-check libero_10 / LIBERO-Long data for OpenVLA-OFT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAMES = (
    "libero_10",
    "LIBERO-10",
    "libero10",
    "libero-10",
    "libero_long",
    "LIBERO-Long",
    "libero-long",
    "LIBERO_LONG",
)
ENV_ROOTS = ("DATA_ROOT", "LIBERO_DATA_ROOT", "RLDS_DATA_ROOT", "OXE_DATA_ROOT")
COMMON_ROOTS = (
    "/home/caijiasheng25/data",
    "/home/caijiasheng25/robopatch_defense",
    "/home/caijiasheng25/UPA-RFAS",
    "/home/caijiasheng25",
    "/mnt/beegfs/home/caijiasheng25/data",
    "/mnt/beegfs/home/caijiasheng25/robopatch_defense",
    "/mnt/beegfs/home/caijiasheng25",
    "/mnt/beegfs/data",
)
DEFAULT_OUTPUT_JSON = "outputs/when_oft_gradconf/data_checks/libero10_candidates.json"
MAX_WALK_DEPTH = 6
PRUNE_DIR_NAMES = {
    ".cache",
    ".conda",
    ".git",
    ".local",
    "__pycache__",
    "adversarial_patches",
    "logs",
    "models",
    "node_modules",
    "outputs",
    "wandb",
}


@dataclass
class Candidate:
    path: str
    match_name: str
    origin: str
    priority: int
    exists: bool
    is_dir: bool
    looks_like_dataset: bool


@dataclass
class FieldCheck:
    status: str
    method: str
    candidate_fields: list[str]
    has_front_image: bool
    has_wrist_image: bool
    has_instruction: bool
    has_action: bool
    has_proprio: bool
    notes: list[str]


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dedupe_paths(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(os.path.expandvars(text)).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _dataset_name_variants() -> list[str]:
    names: list[str] = []
    for name in DATASET_NAMES:
        names.append(name)
        names.append(f"{name}_no_noops")
    return list(dict.fromkeys(names))


def is_libero10_or_long_name(value: str) -> bool:
    compact = _normalize_name(Path(value).name)
    aliases = {_normalize_name(name) for name in _dataset_name_variants()}
    if compact in aliases:
        return True
    return any(compact.startswith(alias + "nonoops") for alias in aliases)


def _candidate_dataset_root(path: Path) -> Path:
    if path.name == "1.0.0":
        return path.parent
    if (path / "1.0.0").exists():
        return path
    return path


def _looks_like_dataset(path: Path) -> bool:
    root = _candidate_dataset_root(path)
    if not root.exists() or not root.is_dir():
        return False
    if (root / "dataset_info.json").exists() or (root / "1.0.0" / "dataset_info.json").exists():
        return True
    for probe in [root, root / "1.0.0"]:
        try:
            if any(probe.glob("*.tfrecord*")) or any(probe.glob("*.array_record*")):
                return True
        except OSError:
            pass
    return root.parent.name == "dataset" and is_libero10_or_long_name(root.name)


def _add_candidate(
    candidates: list[Candidate],
    path: Path,
    *,
    match_name: str,
    origin: str,
    priority: int,
) -> None:
    root = _candidate_dataset_root(path)
    key = str(root)
    if any(item.path == key for item in candidates):
        return
    candidates.append(
        Candidate(
            path=key,
            match_name=match_name,
            origin=origin,
            priority=priority,
            exists=root.exists(),
            is_dir=root.is_dir(),
            looks_like_dataset=_looks_like_dataset(root),
        )
    )


def _direct_dataset_locations(root: Path) -> Iterable[tuple[Path, str]]:
    for name in _dataset_name_variants():
        yield root / name, name
        yield root / name / "1.0.0", name
        yield root / "dataset" / name, name
        yield root / "dataset" / name / "1.0.0", name
        yield root / "openvla-main" / "dataset" / name, name
        yield root / "openvla-main" / "dataset" / name / "1.0.0", name


def _depth(root: Path, path: Path) -> int:
    try:
        return len(path.relative_to(root).parts)
    except ValueError:
        return MAX_WALK_DEPTH + 1


def _walk_for_names(root: Path, *, priority: int, origin: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    if not root.exists():
        return candidates

    for direct, match_name in _direct_dataset_locations(root):
        if direct.exists():
            _add_candidate(candidates, direct, match_name=match_name, origin=f"{origin}:direct", priority=priority)

    try:
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            if _depth(root, current_path) > MAX_WALK_DEPTH:
                dirs[:] = []
                continue
            dirs[:] = [
                item
                for item in dirs
                if item not in PRUNE_DIR_NAMES and not item.startswith(".")
            ]
            for dirname in list(dirs):
                if is_libero10_or_long_name(dirname):
                    _add_candidate(
                        candidates,
                        current_path / dirname,
                        match_name=dirname,
                        origin=f"{origin}:walk",
                        priority=priority,
                    )
            for filename in files:
                if is_libero10_or_long_name(filename):
                    _add_candidate(
                        candidates,
                        current_path,
                        match_name=filename,
                        origin=f"{origin}:file",
                        priority=priority,
                    )
    except PermissionError:
        pass
    return candidates


def _scan_repo_config_references() -> tuple[list[Candidate], list[dict[str, str]]]:
    candidates: list[Candidate] = []
    references: list[dict[str, str]] = []
    roots = [
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "VLAAttacker",
        PROJECT_ROOT / "README.md",
    ]
    suffixes = {".yaml", ".yml", ".json", ".md", ".py", ".sh", ".sbatch", ".txt"}
    path_pattern = re.compile(r"(/[^\s\"'`>,;:]+)")

    for root in roots:
        files = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in files:
            if not path.is_file() or path.suffix not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not any(name.lower() in text.lower() for name in DATASET_NAMES):
                continue
            references.append({"file": str(path), "reason": "mentions libero_10 / LIBERO-Long alias naming"})
            for match in path_pattern.findall(text):
                clean = match.rstrip(").]")
                if not is_libero10_or_long_name(clean):
                    continue
                candidate_path = Path(clean).expanduser()
                if candidate_path.exists():
                    _add_candidate(
                        candidates,
                        candidate_path,
                        match_name=Path(clean).name,
                        origin=f"repo_config:{path.relative_to(PROJECT_ROOT)}",
                        priority=350,
                    )
    return candidates, references


def _collect_roots(data_root: str, search_roots: list[str] | None) -> list[Path]:
    roots: list[str | Path] = []
    if data_root:
        roots.append(data_root)
    for env_name in ENV_ROOTS:
        value = os.environ.get(env_name)
        if value:
            roots.append(value)
    roots.extend(COMMON_ROOTS)
    roots.extend(search_roots or [])
    return _dedupe_paths(roots)


def _flatten_json_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            keys.extend(_flatten_json_keys(child, name))
    elif isinstance(value, list):
        for index, child in enumerate(value[:8]):
            keys.extend(_flatten_json_keys(child, f"{prefix}[{index}]"))
    return keys


def _metadata_field_keys(dataset_path: Path) -> tuple[list[str], str]:
    search_paths = [
        dataset_path / "dataset_info.json",
        dataset_path / "features.json",
        dataset_path / "1.0.0" / "dataset_info.json",
        dataset_path / "1.0.0" / "features.json",
    ]
    search_paths.extend(dataset_path.glob("*/dataset_info.json"))
    search_paths.extend(dataset_path.glob("*/features.json"))
    search_paths = list(dict.fromkeys(search_paths))
    all_fields: list[str] = []
    methods: list[str] = []
    for info_path in search_paths:
        if not info_path.exists():
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        all_fields.extend(_flatten_json_keys(info))
        methods.append(str(info_path))
    return sorted(set(all_fields)), ",".join(methods)


def _flatten_tensor_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            keys.extend(_flatten_tensor_keys(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value[:8]):
            keys.extend(_flatten_tensor_keys(child, f"{prefix}[{index}]"))
    return keys


def _sample_field_keys_with_tfds(dataset_path: Path) -> tuple[list[str], str, list[str]]:
    notes: list[str] = []
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as exc:  # pragma: no cover
        return [], "tfds_unavailable", [f"tensorflow_datasets import failed: {exc}"]

    builder_dirs = [dataset_path / "1.0.0", dataset_path]
    builder_dirs.extend(dataset_path.glob("*/"))
    for builder_dir in builder_dirs:
        if not (builder_dir / "dataset_info.json").exists():
            continue
        try:
            builder = tfds.builder_from_directory(str(builder_dir))
            split = "train" if "train" in builder.info.splits else next(iter(builder.info.splits))
            ds = builder.as_dataset(split=split)
            sample = next(iter(ds.take(1)))
        except Exception as exc:  # pragma: no cover
            notes.append(f"TFDS sample read failed for {builder_dir}: {exc}")
            continue
        return sorted(set(_flatten_tensor_keys(sample))), f"tfds:{builder_dir}", notes
    return [], "tfds_no_builder_dir", notes


def _has_any(fields: list[str], patterns: tuple[str, ...]) -> bool:
    lowered = [field.lower() for field in fields]
    return any(any(pattern in field for pattern in patterns) for field in lowered)


def check_candidate_fields(dataset_path: str | Path, *, dry_run: bool) -> FieldCheck:
    path = Path(dataset_path)
    notes: list[str] = []
    fields, method = _metadata_field_keys(path)
    if not fields and not dry_run:
        fields, method, sample_notes = _sample_field_keys_with_tfds(path)
        notes.extend(sample_notes)
    elif not fields:
        method = "metadata_not_found"
        notes.append("TODO: Could not determine sample fields from metadata in dry-run mode.")

    has_front = _has_any(
        fields,
        ("image_primary", "front", "agentview", "full_image", "primary", "observation.featuresdict.features.image"),
    )
    has_wrist = _has_any(fields, ("image_wrist", "wrist", "hand"))
    has_instruction = _has_any(fields, ("language_instruction", "instruction", "task_description", "language"))
    has_action = _has_any(fields, ("action",))
    has_proprio = _has_any(fields, ("proprio", "state", "robot0_eef", "robot"))

    required_known = bool(fields)
    required_ok = has_front and has_wrist and has_instruction and (has_action or has_proprio)
    if not required_known:
        status = "unknown"
    elif required_ok:
        status = "ok"
    else:
        status = "missing_required_fields"
        notes.append(
            "TODO: Metadata was found, but required OpenVLA-OFT fields were not all identified. "
            "Inspect candidate_fields before using this dataset."
        )

    return FieldCheck(
        status=status,
        method=method,
        candidate_fields=fields[:300],
        has_front_image=has_front,
        has_wrist_image=has_wrist,
        has_instruction=has_instruction,
        has_action=has_action,
        has_proprio=has_proprio,
        notes=notes,
    )


def discover_libero10(
    *,
    data_root: str = "",
    dataset_name: str = "libero_10",
    search_roots: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    roots = _collect_roots(data_root, search_roots)
    candidates: list[Candidate] = []
    for index, root in enumerate(roots):
        candidates.extend(_walk_for_names(root, priority=index, origin=str(root)))

    config_candidates, config_references = _scan_repo_config_references()
    candidates.extend(config_candidates)

    unique_candidates = {item.path: item for item in candidates if item.exists}
    sorted_candidates = sorted(unique_candidates.values(), key=lambda item: (item.priority, item.path))
    selectable_candidates = [item for item in sorted_candidates if item.looks_like_dataset]
    selected = selectable_candidates[0] if selectable_candidates else None
    field_check = check_candidate_fields(selected.path, dry_run=dry_run) if selected else None
    errors: list[str] = []
    if not selected and sorted_candidates:
        errors.append(
            "libero_10 / LIBERO-Long-like names were found, but none looked like an OpenVLA/RLDS dataset directory."
        )
    if not selected:
        errors.append("libero_10 / LIBERO-Long is missing after searching all candidate roots.")
        errors.append(
            "Please prepare/download the LIBERO-10/LIBERO-Long dataset using the repository's existing LIBERO data preparation workflow."
        )

    return {
        "dataset_name": dataset_name,
        "dataset_semantic_name": "LIBERO-Long",
        "selected": asdict(selected) if selected else None,
        "candidates": [asdict(item) for item in sorted_candidates],
        "config_references": config_references,
        "searched_roots": [str(root) for root in roots],
        "field_check": asdict(field_check) if field_check else None,
        "errors": errors,
    }


def _find_prepare_workflows() -> list[str]:
    patterns = [
        "scripts/data/*libero*",
        "scripts/*libero*",
        "scripts/when/*libero*",
        "README.md",
        "docs/**/*.md",
    ]
    hits: list[str] = []
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(set(hits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/home/caijiasheng25/data")
    parser.add_argument("--dataset_name", default="libero_10")
    parser.add_argument("--search_roots", nargs="*", default=[])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prepare_if_missing", action="store_true", default=False)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = discover_libero10(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        search_roots=args.search_roots,
        dry_run=args.dry_run,
    )
    output_path = _resolve_output_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[libero10-check] wrote {output_path}")
    if result["selected"]:
        print(f"[libero10-check] selected {result['selected']['path']}")
        field_check = result.get("field_check") or {}
        print(f"[libero10-check] field_check={field_check.get('status')} method={field_check.get('method')}")
        for note in field_check.get("notes", []):
            print(f"[libero10-check][warn] {note}")
        if field_check.get("status") == "missing_required_fields":
            return 3
        return 0

    for error in result["errors"]:
        print(f"[libero10-check][error] {error}", file=sys.stderr)
    if args.prepare_if_missing:
        workflows = _find_prepare_workflows()
        if workflows:
            print("[libero10-check] Existing LIBERO-related setup/workflow files:")
            for item in workflows:
                print(f"  - {item}")
            print("[libero10-check] Review these files and run the repository's existing LIBERO preparation workflow manually.")
        else:
            print("[libero10-check] TODO: No existing LIBERO preparation script was found; prepare LIBERO-10/LIBERO-Long manually.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
