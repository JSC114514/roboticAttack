"""Fix initialization order only; execute the unchanged original training loop."""
from __future__ import annotations
import argparse
import torch
from repair_common import (REPAIR_ROOT, BASE_ROOT, BASE_SIGNATURE, GENERATOR,
                           base, identity, preflight, verify_gpu_completion)
import train as original_train
import analyze as original_analyze


def execute_generator():
    # load_wrapper already disables cuDNN, but was called AFTER G's first
    # ConvTranspose2d. Apply the same existing backend policy before any G use.
    torch.backends.cudnn.enabled = False
    original_train.train(GENERATOR)


def run_generator():
    preflight()
    base.require_gpu("training")
    manifest = identity()
    folder = BASE_ROOT / "training" / GENERATOR
    path = folder / "RUNTIME_REPAIR.json"
    if path.exists():
        base.require(base.load_json(path) == manifest, "Generator runtime changed after first repair attempt")
    else:
        base.require(not (folder / "resume.pt").exists(), "Unknown existing generator checkpoint; do not overwrite")
        base.atomic_json(manifest, path)
    status_path = REPAIR_ROOT / "reports/GENERATOR_STATUS.json"
    record = {"base_signature": BASE_SIGNATURE, "repair_signature": manifest["repair_signature"],
              "method": GENERATOR, "status": "RUNNING", "phase": "original_training_loop",
              "progress_file": str(folder / "status.json")}
    base.atomic_json(record, status_path)
    try:
        execute_generator()
        trained = base.load_json(folder / "status.json")
        base.require(trained["status"] == "COMPLETE" and trained["signature"] == BASE_SIGNATURE
                     and trained["outer"] == 250, "Generator did not complete the original budget")
    except Exception as error:
        record.update(status="FAILED", error=repr(error))
        if (folder / "status.json").exists():
            original = base.load_json(folder / "status.json")
            record.update(outer=original.get("outer", 0), phase=original.get("phase", "original_training_loop"))
        else:
            record.update(outer=0, phase="generator_initialization")
        base.atomic_json(record, status_path)
        # The original loop initialized G outside its try block. This wrapper
        # now records early failures too, without changing v1 source/signature.
        if not (folder / "status.json").exists():
            base.atomic_json({"status": "FAILED", "signature": BASE_SIGNATURE, "method": GENERATOR,
                              "outer": 0, "phase": "generator_initialization", "error": repr(error)}, folder / "status.json")
        raise
    record.update(status="COMPLETE", phase="training_complete", outer=250)
    base.atomic_json(record, status_path)


def validate():
    preflight()
    manifest = verify_gpu_completion()
    original_analyze.validate_training()
    base.atomic_json({**manifest, "status": "COMPLETE", "validation_sha256":
                      base.sha256_file(BASE_ROOT / "reports/AUTO_VALIDATION.json")}, REPAIR_ROOT / "reports/VALIDATION_LINEAGE.json")


def stats():
    verify_gpu_completion()
    original_analyze.analyze()
    base.atomic_json({**identity(), "status": "COMPLETE", "result_status_sha256":
                      base.sha256_file(BASE_ROOT / "results/STATUS.json"), "results": str(BASE_ROOT / "results/RESULTS.md")},
                     REPAIR_ROOT / "reports/RESULT_LINEAGE.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generator", "validate", "stats"), required=True)
    {"generator": run_generator, "validate": validate, "stats": stats}[parser.parse_args().stage]()
