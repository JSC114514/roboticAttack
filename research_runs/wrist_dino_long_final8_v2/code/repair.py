"""CPU-only, audited bridge of the SAME batch's completed UADA250 prefix.

Never initialize a model/renderer or reset optimizer/scheduler/RNG. The parent
is immutable; only version/signature and the missing evaluation seed wiring
change. All seven other students still start from their original random seed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, PROJECT, atomic_json, atomic_torch, config, digest, load_json,
                    preflight, require, sha256_file)
from train import validate_resume

PARENT = ROOT.parent / "wrist_dino_long_final8_v1"
PARENT_SIGNATURE = "77a2f95026027e8e74e8e2d5fcd02f64df79bd5c2e483c00d2ce540e816b0295"
UNCHANGED_CODE = ("train.py", "objectives.py", "engine.py", "preprocessing.py", "source_transform.py",
                  "UADA_SOURCE_COPY.py", "teacher.py", "bank.py", "evaluate.py", "analyze.py")
PREFIX_FILES = ("outer_0000.pt", "outer_0000.png", "outer_0250.pt", "outer_0250.png", "resume.pt",
                "training.parquet", "RUN_MANIFEST.json", "GPU_INTERFACE.json", "status.json")


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(same_state(a, b) for a, b in zip(left, right))
    return left == right


def reidentify(value, cfg, signature):
    result = dict(value, signature=signature, config=cfg)
    require(same_state({k: v for k, v in value.items() if k not in {"signature", "config"}},
                       {k: v for k, v in result.items() if k not in {"signature", "config"}}),
            "Prefix optimizer/pixels/RNG/history must not change")
    return result


def check_parent(report=None):
    report = preflight() if report is None else report
    cfg, old_cfg = config(), load_json(PARENT / "config.json")
    compatibility = copy.deepcopy(cfg)
    compatibility["evaluation"]["seed_sources"] = old_cfg["evaluation"]["seed_sources"]
    require(compatibility == old_cfg, "Repair may only wire missing evaluation seeds; training design changed")
    paths = list((PARENT / "code").glob("*.py")) + list((PARENT / "source").glob("*.py"))
    paths += list((PARENT / "slurm").glob("*.sbatch")) + [PARENT / "slurm/environment.sh", PARENT / "config.json"]
    paths += [PROJECT / "submit_exploring_oft_long_final8_2000_v1.sh"]
    paths += [PROJECT / name for name in old_cfg["original_sources"]]
    paths += [PROJECT / "VLAAttacker/multiview/openvla_oft_wrapper.py",
              PROJECT / "VLAAttacker/multiview/openvla_oft_experiments.py"]
    old_signature = digest({"config": old_cfg, "sources": {str(p.resolve()): sha256_file(p) for p in sorted(paths)}})
    prep = load_json(PARENT / "reports/PREPARATION.json")
    receipt = load_json(PARENT / "reports/SUBMISSION_RECEIPT.json")
    require(old_signature == prep["signature"] == receipt["signature"] == PARENT_SIGNATURE,
            "Frozen failed-batch sources/identity changed; do not guess prefix compatibility")
    require(prep["model_jsons"] == report["model_jsons"], "Prefix/model normalization changed")
    for name in UNCHANGED_CODE:
        require((PARENT / "code" / name).read_bytes() == (ROOT / "code" / name).read_bytes(),
                f"Scientific code changed during renderer repair: {name}")
    old_runtime = ast.parse((PARENT / "code/model_runtime.py").read_text())
    new_runtime = ast.parse((ROOT / "code/model_runtime.py").read_text())
    def functions(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    old_functions, new_functions = functions(old_runtime), functions(new_runtime)
    for name in ("observation", "load_wrapper", "raw_views", "build_inputs", "_normalized_proprio",
                 "unnormalize_action", "differentiable_action_and_features", "bind_environment_context",
                 "close_environment", "reset_environment", "reset_wait", "validate_camera_frames"):
        require(old_functions[name] == new_functions[name], f"Non-renderer experiment logic changed: {name}")
    folder = PARENT / "training/uada_original"
    status = load_json(folder / "status.json")
    require(status["status"] == "PREFIX_COMPLETE" and status["outer"] == 250
            and status["signature"] == old_signature and status["optimizer_updates"] == 12500,
            "Parent prefix is not complete250")
    saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    validate_resume(saved, "uada_original", old_cfg, old_signature)
    require(saved["outer"] == 250 and saved["elapsed_s"] == status["elapsed_s"], "Prefix time/outer mismatch")
    require(saved["weights"] == [] and saved["frequency_scale"] is None and saved["distill_coefficient"] is None,
            "Original-UADA prefix must not contain DINO/replay/frequency/distillation state")
    require(tuple(saved["patch"].shape) == (3, 50, 50) and torch.isfinite(saved["patch"]).all()
            and saved["patch"].min() >= 0 and saved["patch"].max() <= 1, "Invalid prefix RGB")
    require(torch.equal(saved["initial"], torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))),
            "Prefix is not this batch's random seed42 start")
    require(saved["torch_rng"].dtype == torch.uint8 and saved["torch_rng"].numel() > 0
            and len(saved["cuda_rng"]) == 1 and saved["cuda_rng"][0].dtype == torch.uint8,
            "Prefix RNG state incomplete")
    pd.testing.assert_frame_equal(pd.read_parquet(folder / "training.parquet"), pd.DataFrame(saved["rows"]))
    for outer in (0, 250):
        path = folder / f"outer_{outer:04d}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        require(value["signature"] == old_signature and value["config"] == old_cfg
                and value["method"] == "uada_original" and value["outer_iterations"] == outer
                and value["optimizer_updates"] == outer * 50, "Prefix snapshot identity changed")
        expected = saved["initial"] if outer == 0 else saved["patch"]
        require(torch.equal(value["patch"], expected) and torch.equal(value["initial_patch"], saved["initial"]),
                "Prefix snapshot and resume RGB disagree")
        with Image.open(path.with_suffix(".png")) as image:
            require(image.mode == "RGB" and image.size == (50, 50)
                    and np.array_equal(np.asarray(image), expected.permute(1, 2, 0).mul(255).round().byte().numpy()),
                    "Prefix PNG/PT quantization mismatch")
    return report, saved, {name: sha256_file(folder / name) for name in PREFIX_FILES}


def prepare_repair():
    report, saved, hashes = check_parent()
    from bank import prepare
    prepare()
    cfg, signature = config(), report["signature"]
    folder = ROOT / "training/uada_original"
    receipt_path = ROOT / "reports/PREFIX_ADOPTION.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["signature"] == signature and receipt["parent_files_sha256"] == hashes,
                "Prefix adoption identity changed")
        require(all(sha256_file(folder / name) == checksum for name, checksum in receipt["static_snapshot_sha256"].items()),
                "Adopted snapshots changed")
        print("PREFIX_ALREADY_ADOPTED; current training progress retained", flush=True)
        return
    if (folder / "resume.pt").exists():
        partial = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_resume(partial, "uada_original", cfg, signature)
        require(partial["outer"] == 250, "Do not overwrite training beyond adopted prefix")
    folder.mkdir(parents=True, exist_ok=True)
    parent_folder = PARENT / "training/uada_original"
    for outer in (0, 250):
        name = f"outer_{outer:04d}"
        snapshot = torch.load(parent_folder / (name + ".pt"), map_location="cpu", weights_only=False)
        atomic_torch(reidentify(snapshot, cfg, signature), folder / (name + ".pt"))
        shutil.copyfile(parent_folder / (name + ".png"), folder / (name + ".png"))
    atomic_torch(reidentify(saved, cfg, signature), folder / "resume.pt")
    for name in ("training.parquet", "GPU_INTERFACE.json"):
        shutil.copyfile(parent_folder / name, folder / name)
    atomic_json(reidentify(load_json(parent_folder / "RUN_MANIFEST.json"), cfg, signature), folder / "RUN_MANIFEST.json")
    atomic_json(dict(load_json(parent_folder / "status.json"), signature=signature), folder / "status.json")
    adopted = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    validate_resume(adopted, "uada_original", cfg, signature)
    require(same_state(adopted, reidentify(saved, cfg, signature)), "Serialized prefix state changed")
    new_hashes = {name: sha256_file(folder / name) for name in PREFIX_FILES}
    atomic_json({"status": "COMPLETE", "signature": signature, "parent_signature": PARENT_SIGNATURE,
                 "parent_root": str(PARENT), "parent_files_sha256": hashes, "adopted_files_sha256": new_hashes,
                 "static_snapshot_sha256": {k: v for k, v in new_hashes.items() if k.startswith("outer_")},
                 "outer": 250, "optimizer_updates_reused": 12500, "gpu_elapsed_s_reused": saved["elapsed_s"],
                 "metadata_only_changes": ["signature", "config.evaluation.seed_sources"],
                 "optimizer_scheduler_rng_history_pixels_unchanged": True,
                 "other_seven_students": "fresh random0; do not adopt prefix", "gpu_started": False}, receipt_path)
    print("PREFIX_ADOPTED: 250 outer,12500 updates; Adam/cosine/RNG/RGB/history unchanged; no GPU", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    if parser.parse_args().prepare:
        prepare_repair()
    else:
        report, saved, _ = check_parent()
        print(f"REPAIR_CHECK_PASS prefix250 elapsed_s={saved['elapsed_s']:.3f}; signature={report['signature']}; NO_GPU")
