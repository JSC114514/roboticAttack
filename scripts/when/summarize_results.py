#!/usr/bin/env python3
"""Summarize WHEN result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/when")
    parser.add_argument("--summary-dir", default="outputs/when/summary")
    args = parser.parse_args()

    root = Path(args.output_root)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob("*.json")):
        if "_validation" in path.parts or "summary" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({"path": str(path), "payload": payload})
    (summary_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    lines = ["# WHEN Summary", "", f"Collected JSON files: {len(rows)}", ""]
    for row in rows:
        lines.append(f"- `{row['path']}`")
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {summary_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
