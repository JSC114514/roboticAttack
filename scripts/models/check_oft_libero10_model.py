#!/usr/bin/env python3
"""Check for the OpenVLA-OFT-libero10 checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = "outputs/when_oft_gradconf/model_checks/oft_libero10_candidates.json"
DEFAULT_CANDIDATE_PATHS = (
    "/home/caijiasheng25/models/openvla-7b-oft-finetuned-libero-10",
    "/home/caijiasheng25/models/openvla-oft-libero-10",
    "/home/caijiasheng25/models/moojink/openvla-7b-oft-finetuned-libero-10",
    "/home/caijiasheng25/models/openvla-oft",
)


@dataclass
class ModelCandidate:
    path: str
    origin: str
    exists: bool
    is_dir: bool
    is_oft: bool
    is_plain_openvla_libero10: bool
    explicit_libero10: bool
    looks_like_model: bool
    priority: int


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _dedupe(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _looks_like_model(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    markers = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_config.json",
    ]
    if any((path / marker).exists() for marker in markers):
        return True
    try:
        if any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")):
            return True
    except OSError:
        return False
    return False


def _candidate_for(path: Path, *, origin: str, priority: int) -> ModelCandidate:
    name = path.name
    compact = _normalize(name)
    is_oft = "oft" in compact
    explicit_libero10 = "libero10" in compact or "liberoten" in compact
    is_plain = "openvla7bfinetunedlibero10" in compact and "oft" not in compact
    return ModelCandidate(
        path=str(path),
        origin=origin,
        exists=path.exists(),
        is_dir=path.is_dir(),
        is_oft=is_oft,
        is_plain_openvla_libero10=is_plain,
        explicit_libero10=explicit_libero10,
        looks_like_model=_looks_like_model(path),
        priority=priority,
    )


def _discover_from_roots(search_roots: list[str]) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []
    for root_index, root_text in enumerate(search_roots):
        root = Path(root_text).expanduser()
        if not root.exists():
            continue
        try:
            for path in root.rglob("*openvla*"):
                if not path.is_dir():
                    continue
                compact = _normalize(path.name)
                if "oft" in compact and ("libero10" in compact or "libero" in compact):
                    candidates.append(_candidate_for(path, origin=f"search_root:{root}", priority=100 + root_index))
        except PermissionError:
            continue
    return candidates


def discover_model(model_path: str = "", search_roots: list[str] | None = None) -> dict[str, object]:
    paths: list[str | Path] = []
    if model_path:
        paths.append(model_path)
    paths.extend(DEFAULT_CANDIDATE_PATHS)
    candidates = [
        _candidate_for(path, origin="default", priority=index)
        for index, path in enumerate(_dedupe(paths))
    ]
    candidates.extend(_discover_from_roots(search_roots or []))

    unique = {item.path: item for item in candidates}
    sorted_candidates = sorted(
        unique.values(),
        key=lambda item: (
            not item.exists,
            not item.explicit_libero10,
            not item.is_oft,
            not item.looks_like_model,
            item.priority,
            item.path,
        ),
    )
    selectable = [
        item
        for item in sorted_candidates
        if item.exists and item.is_dir and item.is_oft and not item.is_plain_openvla_libero10
    ]
    selected = selectable[0] if selectable else None
    errors: list[str] = []
    if not selected:
        errors.append("openvla-7b-oft-finetuned-libero-10 is missing.")
        errors.append("Please prepare the OFT-libero10 checkpoint before running gradient diagnostics.")
    return {
        "selected": asdict(selected) if selected else None,
        "candidates": [asdict(item) for item in sorted_candidates],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--search_roots", nargs="*", default=[])
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = discover_model(model_path=args.model_path, search_roots=args.search_roots)
    output_path = _resolve_output_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[oft-libero10-model-check] wrote {output_path}")
    if result["selected"]:
        print(f"[oft-libero10-model-check] selected {result['selected']['path']}")
        return 0
    for error in result["errors"]:
        print(f"[oft-libero10-model-check][error] {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
