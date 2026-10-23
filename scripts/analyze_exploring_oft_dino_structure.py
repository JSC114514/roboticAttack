#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "exploring_oft_dino" / "object_timeboxed.json"
METHODS = ("exploring_oft_uada", "exploring_oft_dino")


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_root = Path(cfg["output_root"])
    rows = []
    for method in METHODS:
        directory = output_root / method
        status_path = directory / "status.json"
        if not status_path.is_file():
            rows.append({"method": method, "status": "MISSING"})
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rows.append({"method": method, **status})
    frame = pd.DataFrame(rows)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_dir / "STRUCTURE_SUMMARY.csv", index=False)

    steps = (0, 25, 50, 100, 200)
    tile = 180
    canvas = Image.new("RGB", (len(steps) * tile, len(METHODS) * (tile + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    missing = []
    for row, method in enumerate(METHODS):
        for column, step in enumerate(steps):
            path = output_root / method / f"outer_{step:04d}.png"
            if not path.is_file():
                missing.append(str(path))
                continue
            image = Image.open(path).convert("RGB").resize((tile - 10, tile - 10), Image.Resampling.NEAREST)
            x, y = column * tile + 5, row * (tile + 28) + 5
            canvas.paste(image, (x, y))
            draw.text((x, y + tile - 7), f"{method} outer={step}", fill="black")
    canvas.save(report_dir / "PATCH_EVOLUTION.png")
    overall = "READY_FOR_MANUAL_VISUAL_REVIEW"
    if missing or any(row.get("status") != "COMPLETE" for row in rows):
        overall = "NOT_READY_OR_STRUCTURE_GATE_FAILED"
    (report_dir / "STATUS.json").write_text(json.dumps({
        "status": overall,
        "methods": rows,
        "missing_images": missing,
        "manual_visual_review_required": True,
        "casr_evaluation_authorized": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "report_dir": str(report_dir), "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
