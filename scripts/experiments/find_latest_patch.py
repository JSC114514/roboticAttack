#!/usr/bin/env python3
"""按 source_exp_id 或 patch_family 查找最新 patch manifest。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.multiview.patch_manifest import (
    find_latest_manifest,
    find_latest_patch,
    manifest_patch_exists,
    official_m_dependency_error,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find latest patch manifest")
    parser.add_argument("--source_exp_id", default="", type=str)
    parser.add_argument("--patch_family", default="", type=str)
    parser.add_argument("--output_root", default="outputs/multiview_attack", type=str)
    parser.add_argument("--require_existing_patch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    if args.source_exp_id:
        found_by_exp = find_latest_patch(args.source_exp_id, output_root)
        if not found_by_exp:
            if args.source_exp_id in {"M1", "M2", "M5"}:
                raise SystemExit(official_m_dependency_error("dependent experiment", args.source_exp_id))
            raise SystemExit(f"未找到 source_exp_id={args.source_exp_id} 的 manifest")
        manifest_path = Path(found_by_exp["manifest_path"])
        manifest = found_by_exp
        if args.require_existing_patch and not manifest_patch_exists(manifest_path, manifest):
            raise SystemExit(f"manifest 找到了但 patch 文件不存在: {manifest_path}")
    else:
        if not args.patch_family:
            raise SystemExit("必须提供 --source_exp_id 或 --patch_family")
        found = find_latest_manifest(
            output_root,
            args.patch_family,
            require_existing_patch=args.require_existing_patch,
        )
        if not found:
            raise SystemExit(f"未找到 patch_family={args.patch_family} 的 manifest")
        manifest_path, manifest = found
    if args.json:
        payload = {"manifest_path": str(manifest_path), "manifest": manifest}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(manifest_path)


if __name__ == "__main__":
    main()
