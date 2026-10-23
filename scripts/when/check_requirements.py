#!/usr/bin/env python3
"""Lightweight WHEN requirement checks.

This script does not import torch/transformers and is safe on login nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VLAAttacker.when.data import collect_requirement_status, missing_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print raw JSON status.")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args()

    status = collect_requirement_status()
    missing = missing_requirements(status)
    if args.json:
        print(json.dumps({"status": status, "missing": missing}, indent=2, ensure_ascii=False))
    else:
        print("[when-check] requirement summary")
        for item in missing:
            print(f"[missing] {item}")
        if not missing:
            print("[ok] no missing lightweight path requirements")

    if args.fail_on_missing and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
