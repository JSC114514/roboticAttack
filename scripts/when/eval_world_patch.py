#!/usr/bin/env python3
"""Spatial world-coordinate patch experiment entry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.when.config import load_when_config, merge_when_defaults, resolve_output_path
from VLAAttacker.when.patch_io import read_patch_manifest, require_patch_manifest
from VLAAttacker.when.trainer import assert_cuda_available
from VLAAttacker.when.visibility import VISIBILITY_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--patch-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = merge_when_defaults(load_when_config(args.config))
    manifest_path = require_patch_manifest(args.patch_manifest or None)
    manifest = read_patch_manifest(manifest_path)
    payload = {
        "config": config,
        "manifest": manifest,
        "output_dir": str(resolve_output_path(args.output_dir)),
        "visibility_fields": VISIBILITY_FIELDS,
        "status": "scene-analysis-required",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    assert_cuda_available()
    raise SystemExit("world_anchored_patch requires LIBERO MuJoCo scene injection; 2D overlay is intentionally refused.")


if __name__ == "__main__":
    main()
