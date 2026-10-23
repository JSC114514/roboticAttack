#!/usr/bin/env python3
"""Resume only the audited 1998/1999 commits with the original trainer."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "VLAAttacker")]

import torch
import dino_sampling_clear_common as common
import dino_sampling_clear_recovery_common as recovery
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file


def run(method, cfg):
    require(method in recovery.PENDING, "Recovery permits B/C/D only; completed A is excluded")
    report = recovery.preflight(cfg)
    ledger = recovery.verify_preserved_artifacts(cfg)
    require(os.environ.get("SLURM_JOB_ID"), "A user-submitted Slurm recovery job is required")
    folder = Path(cfg["output_root"]) / "training" / method
    path = recovery.recovery_root(cfg) / f"{method}_RUNTIME.json"
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = json.loads((folder / "status.json").read_text())
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        if status["status"] == "COMPLETE":
            require(path.is_file(), "Completed recovered arm lacks actual execution evidence")
            receipt = json.loads(path.read_text())
            require(receipt["ledger_signature"] == ledger["signature"] and saved["outer"] == 2000,
                    "Completed recovery evidence changed")
            receipt.update(status="COMPLETE", to_outer=2000,
                           final_resume_sha256=sha256_file(folder / "resume.pt"))
            atomic_json(receipt, path)
            recovery.verify_preserved_artifacts(cfg, ledger)
            return {"status": "ALREADY_COMPLETE", "method": method}
        require(recovery.ORIGINAL_OUTERS[method] <= saved["outer"] <= 2000,
                "Recovery must continue the committed tail, with no budget extension")
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "Exactly one Slurm-mapped CUDA GPU required")
        from UADA_OFT_DINO_wrapper import load_wrapper
        from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import ExploringOFTDINOSamplingClearAttacker, source_manifest
        clean = common.source_records(cfg)
        replay, _ = common.replay_records(cfg)
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        started = time.monotonic()
        wrapper = load_wrapper(cfg)
        torch.cuda.synchronize()
        load_s = time.monotonic() - started
        require(not any(p.requires_grad for p in wrapper.vla.parameters()), "VLA must remain frozen")
        actual = source_manifest(cfg, wrapper)
        expected = dict(manifest["source"])
        expected.update(ledger["identity"]["actual_cpu_source"])
        require(actual == expected, "Actually loaded recovery runtime differs from audited original/current source")
        snapshot = recovery.recovery_root(cfg) / f"{method}_actual_source_snapshot.zip"
        if not snapshot.exists():
            temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}.tmp")
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, digest in actual.items():
                    require(sha256_file(name) == digest, "Recovery source changed before actual snapshot")
                    archive.write(name, arcname=name.lstrip("/"))
            os.replace(temporary, snapshot)
        receipt = {"status": "RUNNING", "method": method, "recovery_id": recovery.RECOVERY_ID,
                   "ledger_signature": ledger["signature"], "original_signature": manifest["signature"],
                   "from_outer": recovery.ORIGINAL_OUTERS[method], "resumed_from_outer": saved["outer"],
                   "to_outer": saved["outer"], "input_resume_sha256": sha256_file(folder / "resume.pt"),
                   "actual_runtime_source": actual, "actual_source_snapshot_sha256": sha256_file(snapshot),
                   "original_source_snapshot_sha256": sha256_file(folder / "source_snapshot.zip"),
                   "training_algorithm_changed": False, "evaluation_hooks": False}
        if path.exists():
            previous = json.loads(path.read_text())
            require(previous["ledger_signature"] == receipt["ledger_signature"]
                    and previous["actual_runtime_source"] == actual, "Previous recovery execution source changed")
            receipt["previous_attempt"] = previous
        atomic_json(receipt, path)
        attacker = ExploringOFTDINOSamplingClearAttacker(wrapper, cfg["output_root"], cfg["patch_xy"])
        try:
            # The old manifest/signature is retained solely for the original
            # committed training lineage.  Actual new code is recorded above.
            attacker.train_sampling_clear(clean, replay, cfg, method, manifest["source"], manifest["data"], load_s)
            final = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
            status = json.loads((folder / "status.json").read_text())
            require(final["outer"] == status["outer"] == 2000 and status["status"] == "COMPLETE"
                    and final["rows"][:recovery.ORIGINAL_OUTERS[method]] ==
                        torch.load(recovery.recovery_root(cfg) / "evidence" / method / "resume.pt",
                                   map_location="cpu", weights_only=False)["rows"],
                    "Recovery did not complete precisely the original training budget")
            receipt.update(status="COMPLETE", to_outer=2000,
                           final_resume_sha256=sha256_file(folder / "resume.pt"))
            atomic_json(receipt, path)
            recovery.verify_preserved_artifacts(cfg, ledger)
            return {"status": "COMPLETE", "method": method, "to_outer": 2000}
        except BaseException as error:
            receipt.update(status="FAILED", error=repr(error))
            atomic_json(receipt, path)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=recovery.PENDING)
    parser.add_argument("--preflight", "--dry-run", dest="preflight", action="store_true")
    args = parser.parse_args()
    cfg = common.config()
    require(args.preflight or args.method, "--method is required for a recovery GPU job")
    result = recovery.compact_preflight(recovery.preflight(cfg)) if args.preflight else run(args.method, cfg)
    print(json.dumps(result, indent=2))
