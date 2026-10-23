"""Patch manifest and checksum helpers for WHEN experiments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def patch_sha256(path: str | Path) -> str:
    """Compute SHA256 for a patch/checkpoint file."""
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_patch_manifest(
    manifest_path: str | Path,
    *,
    patch_path: str | Path,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a WHEN patch manifest with a patch SHA256."""
    patch_file = Path(patch_path)
    payload: dict[str, Any] = {
        "manifest_schema_version": "when-1",
        "patch_name": config.get("patch_name", "P_WHEN"),
        "patch_path": str(patch_file.resolve()),
        "patch_sha256": patch_sha256(patch_file),
        "patch_space": "raw_rgb",
        "source_model": config.get("source_model_checkpoint", ""),
        "source_dataset": config.get("dataset", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def read_patch_manifest(path: str | Path) -> dict[str, Any]:
    """Read a patch manifest and validate the recorded SHA256 if the file exists."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch_path = Path(payload.get("patch_path", ""))
    if not patch_path.is_absolute():
        patch_path = manifest_path.parent / patch_path
    if not patch_path.exists():
        raise FileNotFoundError(f"Manifest patch_path does not exist: {patch_path}")
    expected = payload.get("patch_sha256", "")
    actual = patch_sha256(patch_path)
    if expected and expected != actual:
        raise ValueError(f"Patch SHA256 mismatch for {patch_path}: manifest={expected} actual={actual}")
    payload["patch_path"] = str(patch_path.resolve())
    payload["patch_sha256_verified"] = actual
    return payload


def require_patch_manifest(value: str | Path | None = None) -> Path:
    """Require an explicit manifest path from argument or PATCH_MANIFEST env var."""
    raw = str(value or os.environ.get("PATCH_MANIFEST", "")).strip()
    if not raw:
        raise ValueError("WHEN eval requires explicit PATCH_MANIFEST=/absolute/path/to/patch_manifest.json")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"PATCH_MANIFEST must be absolute, got {raw!r}")
    if not path.exists():
        raise FileNotFoundError(f"PATCH_MANIFEST does not exist: {path}")
    return path
