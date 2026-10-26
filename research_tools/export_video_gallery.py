#!/usr/bin/env python3
"""Export completed experiment videos without modifying sealed experiment runs.

Reads explicit run roots and their COMPLETE receipts, never arbitrary attempts.
Only the optional compilations require imageio-ffmpeg; copies need stdlib only.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote


LABELS = {
    "comparisons": "00_多条件对比",
    "clean": "01_干净输入",
    "initial_s42": "02_初始补丁_seed42",
    "initial_s43": "02_初始补丁_seed43",
    "uada500": "03_UADA500",
    "radial_reverse_s42": "04_关系补丁_seed42",
    "radial_reverse_s43": "04_关系补丁_seed43",
    "translation_far_boundary_s42": "05_远边界补丁_seed42",
    "translation_far_boundary_s43": "05_远边界补丁_seed43",
    "matched_relation_s42": "06_匹配关系_seed42",
    "matched_relation_s43": "06_匹配关系_seed43",
    "shuffled_relation_s42": "07_置乱关系_seed42",
    "shuffled_relation_s43": "07_置乱关系_seed43",
    "reference_replay": "01_原关系补丁_动作回放",
    "restore_radial": "02_径向恢复",
    "restore_xyz": "03_XYZ平移恢复",
    "restore_pose_grip": "04_姿态和夹爪恢复",
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value)
    tmp.replace(path)


def checked_receipt(root, path, signature=None):
    receipt = read(path)
    if receipt.get("status") != "COMPLETE":
        raise ValueError(f"Incomplete receipt: {path}")
    if signature is not None and receipt.get("signature") != signature:
        raise ValueError(f"Revision differs from final report: {path}")
    for name, expected in receipt["artifacts"].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing/outside artifact: {source}")
        if digest(source) != expected:
            raise ValueError(f"Artifact checksum differs: {source}")
    return receipt


def copy_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Refusing to replace different exported file: {target}")
    else:
        temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(source, temp)
        if digest(temp) != expected:
            temp.unlink()
            raise ValueError(f"Copy checksum mismatch: {source}")
        temp.replace(target)


def compile_videos(files, target, ffmpeg):
    """Stream copy concatenation, followed by CPU decoding to validate output."""
    inputs = [{"file": str(p), "sha256": digest(p)} for p in files]
    receipt = target.with_suffix(".json")
    if target.exists():
        saved = read(receipt)
        if saved["inputs"] != inputs or digest(target) != saved["sha256"]:
            raise ValueError(f"Compilation changed; choose a new output: {target}")
        return saved
    with tempfile.TemporaryDirectory(prefix="video-gallery-") as folder:
        listing = Path(folder) / "concat.txt"
        listing.write_text("".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in files))
        temp = target.with_name(f".{target.stem}.{os.getpid()}.mp4")
        try:
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "0",
                            "-i", str(listing), "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
                            str(temp)], check=True, timeout=180)
            subprocess.run([ffmpeg, "-v", "error", "-xerror", "-nostdin", "-threads", "2",
                            "-i", str(temp), "-f", "null", "-"], check=True, timeout=300)
            saved = {"inputs": inputs, "sha256": digest(temp), "decoded_without_errors": True,
                     "method": "ffmpeg concat stream copy; CPU decode check; no GPU"}
            write(receipt, json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    return saved


def export(runs, output, compilations=False):
    started = time.monotonic()
    output = output.resolve()
    for _, root in runs:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Export directory must be separate from source run roots")
    output.mkdir(parents=True, exist_ok=True)
    records, summaries, groups = [], [], {}
    ffmpeg = None
    if compilations:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for index, (label, root) in enumerate(runs):
        final = checked_receipt(root, root / "results/COMPLETE.json")
        manifest = read(root / "data/MANIFEST.json")
        tasks = {entry["start"]: entry.get("task_name", "")
                 for entry in manifest.get("cases", manifest.get("entries", []))}
        receipts = [("comparisons", p) for p in sorted((root / "comparisons").glob("*/COMPLETE.json"))]
        for part in ("episodes", "replays"):
            receipts.extend((part, p) for p in sorted((root / part).glob("*/*/COMPLETE.json")))
        comparison_files = []
        group = f"{index:02d}_{label}"
        for kind, receipt_path in receipts:
            receipt = checked_receipt(root, receipt_path, final["signature"])
            relative = receipt_path.relative_to(root)
            case = relative.parts[1]
            condition = "comparisons" if kind == "comparisons" else relative.parts[2]
            category = f"{group}/{LABELS.get(condition, condition)}"
            exported = []
            for name, checksum in receipt["artifacts"].items():
                source = root / name
                if source.suffix != ".mp4" and not (kind == "comparisons" and source.name == "curves.png"):
                    continue
                target = output / category / (case + source.suffix)
                copy_verified(source, target, checksum)
                record = {"source": str(source), "target": str(target.relative_to(output)),
                          "sha256": checksum, "receipt": str(receipt_path), "signature": final["signature"],
                          "case": case, "task": tasks.get(case, ""), "category": category}
                records.append(record)
                exported.append(record)
                if source.suffix == ".mp4":
                    groups.setdefault(category, []).append(record)
                    if kind == "comparisons":
                        comparison_files.append(target)
            if sum(r["target"].endswith(".mp4") for r in exported) != 1:
                raise ValueError(f"Expected exactly one video: {receipt_path}")
        if compilations and comparison_files:
            name = f"{index:02d}_{label}_全部{len(comparison_files)}起点.mp4"
            result = compile_videos(comparison_files, output / name, ffmpeg)
            summaries.append({"target": name, "cases": len(comparison_files), **result})
        print(f"Exported {group}: {len(receipts)} completed videos", flush=True)
    cards = []
    for category, rows in groups.items():
        items = []
        for row in rows:
            target = html.escape(quote(row["target"]))
            title = html.escape(row["case"] + " · " + row["task"])
            curve = output / Path(row["target"]).with_suffix(".png")
            curve_url = html.escape(quote(str(curve.relative_to(output))))
            curve_link = (f'<a href="{curve_url}">控制曲线</a>'
                          if curve.exists() else "")
            items.append(f'<article data-search="{title.lower()}"><h3>{title}</h3>'
                         f'<video controls preload="none" src="{target}"></video>'
                         f'<p><a href="{target}">打开 / 下载视频</a> {curve_link}</p></article>')
        cards.append(f'<section data-category="{html.escape(category)}"><h2>{html.escape(category)}</h2>'
                     f'<div class="grid">{"".join(items)}</div></section>')
    options = ''.join(f'<option>{html.escape(c)}</option>' for c in groups)
    compiled = ''.join(f'<li><a href="{html.escape(quote(r["target"]))}">{html.escape(r["target"])}</a></li>'
                       for r in summaries)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>空间关系视频总览</title>
<style>body{font:16px system-ui;margin:24px;background:#f6f7fa}video{width:100%;max-height:360px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}
article{background:white;padding:14px;border:1px solid #ddd}h3{font-size:14px;overflow-wrap:anywhere}
select,input{padding:8px;margin:8px}header{position:sticky;top:0;background:#f6f7fa;padding:8px;z-index:1}</style>
<h1>空间关系补丁：视频总览</h1><p>先看下方两个完整合集。原生补丁与特权动作恢复分开；恢复成功不是攻击成绩。
单段视频按任务与初态命名。源实验保持只读，来源与校验值见 MANIFEST.json。</p><ul>''' + compiled + '''</ul>
<header><label>类别 <select id="category"><option value="">全部</option>''' + options + '''</select></label>
<label>任务 / 起点 <input id="search" placeholder="例如 t00 或 drawer"></label></header>''' + ''.join(cards) + '''
<script>function filter(){const c=document.getElementById('category').value;
const q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('section').forEach(s=>{
let any=false;s.querySelectorAll('article').forEach(a=>{a.hidden=!!q&&!a.dataset.search.includes(q);any=any||!a.hidden});
s.hidden=(!!c&&s.dataset.category!==c)||!any;});}
document.getElementById('category').addEventListener('change',filter);
document.getElementById('search').addEventListener('input',filter);</script></html>'''
    write(output / "INDEX.html", page)
    video_count = sum(r["target"].endswith(".mp4") for r in records)
    report = {"status": "COMPLETE", "source_runs": [str(root) for _, root in runs],
              "videos": video_count, "compilations": summaries, "files": records,
              "elapsed_s": time.monotonic() - started,
              "parent_rss_hwm_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "ffmpeg_child_rss_hwm_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
              "gpu_used": False}
    write(output / "MANIFEST.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write(output / "README.md", "# 视频总入口\n\n打开 INDEX.html 按类别、任务筛选，或直接播放根目录的完整合集。\n\n"
          f"共 {video_count} 个原始视频，另有 {len(summaries)} 个连续对比合集；不需要进入 attempt 目录。\n\n"
          "动作恢复使用干净动作和仿真信息，只是机制诊断，不能当作新补丁成绩。\n"
          "复制文件均核验源 COMPLETE 收据与 SHA256；原实验未修改。MANIFEST.json 保存全部映射。\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compilations", action="store_true")
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        label, name = spec.split("=", 1)
        if not label or Path(label).name != label or label in (".", ".."):
            parser.error("Run label must be one directory name")
        runs.append((label, Path(name).resolve()))
    if len({label for label, _ in runs}) != len(runs):
        parser.error("Run labels must be unique")
    result = export(runs, args.output, args.compilations)
    print(json.dumps({k: result[k] for k in ("status", "videos", "elapsed_s", "gpu_used")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
