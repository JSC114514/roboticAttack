#!/usr/bin/env python3
"""Summarize lambda_con sweep patches and training traces."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs/when_oft_gradconf/gradconf_oft_multiview/libero_10/lambda_con_sweep"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--values", nargs="+", default=["1", "4", "7"])
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("patch", "delta", "adv_patch"):
            if key in obj:
                obj = obj[key]
                break
    if not torch.is_tensor(obj):
        raise TypeError(f"{path} did not contain a tensor patch")
    tensor = obj.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return tensor.clamp(0, 1)


def tensor_to_image(tensor: torch.Tensor, scale: int = 8) -> Image.Image:
    arr = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    image = Image.fromarray(arr)
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def patch_stats(tensor: torch.Tensor) -> dict[str, float]:
    tv = (tensor[:, :, 1:] - tensor[:, :, :-1]).abs().mean()
    tv += (tensor[:, 1:, :] - tensor[:, :-1, :]).abs().mean()
    sat = ((tensor < 1 / 255) | (tensor > 254 / 255)).float().mean()
    q = torch.quantile(tensor.flatten(), torch.tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "tv": float(tv),
        "saturation": float(sat),
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
    }


def load_pickle_list(path: Path):
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return pickle.load(handle)


def loss_stats(path: Path, inner_loop: int = 50) -> dict[str, float | int | str]:
    if not path.exists():
        return {"loss_count": 0, "loss_mean": "", "loss_last1000_mean": "", "outer_last100_mean": ""}
    loss = torch.load(path, map_location="cpu")
    arr = np.asarray(loss, dtype=np.float64)
    usable = (len(arr) // inner_loop) * inner_loop
    outer = arr[:usable].reshape(-1, inner_loop).mean(axis=1) if usable else np.asarray([])
    return {
        "loss_count": int(len(arr)),
        "loss_mean": float(arr.mean()) if len(arr) else "",
        "loss_last1000_mean": float(arr[-1000:].mean()) if len(arr) else "",
        "outer_last100_mean": float(outer[-100:].mean()) if len(outer) else "",
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    thumbs = []
    for value in args.values:
        run_dir = args.root / f"lcon_{value}"
        patch_path = run_dir / "last" / "patch.pt"
        row: dict[str, str | float | int] = {"lambda_con": value, "run_dir": str(run_dir)}
        if patch_path.exists():
            tensor = load_tensor(patch_path)
            row.update(patch_stats(tensor))
            image = tensor_to_image(tensor)
            image.save(out_dir / f"patch_lcon_{value}_x8.png")
            thumbs.append((f"lcon={value}", image))
        else:
            row.update({"missing_patch": 1})
        row.update(loss_stats(run_dir / "loss"))
        for name in ("val_MSE_Distance", "val_UAD", "val_CE_loss"):
            values = load_pickle_list(run_dir / f"{name}.pkl")
            row[f"{name}_last"] = values[-1] if values else ""
            row[f"{name}_min"] = min(values) if values else ""
        rows.append(row)

    csv_path = out_dir / "lambda_con_sweep_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if thumbs:
        w = max(image.width for _, image in thumbs)
        h = max(image.height for _, image in thumbs) + 24
        sheet = Image.new("RGB", (w * len(thumbs), h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, (label, image) in enumerate(thumbs):
            x = idx * w
            sheet.paste(image, (x, 0))
            draw.text((x + 5, image.height + 5), label, fill=(0, 0, 0))
        sheet.save(out_dir / "patch_contact_sheet.png")

    print(f"summary_csv={csv_path}")
    print(f"preview_dir={out_dir}")


if __name__ == "__main__":
    main()
