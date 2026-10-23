"""Audited, manual-only recovery of the committed sampling-clear tail.

Old manifests and signatures describe the original training prefix.  The
separate ledger records the code actually executing the recovered tail.
"""
from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import torch

import dino_sampling_clear_common as common
from white_patch.UADA_OFT_CORRESPONDENCE import run_signature
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file
from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import validate_state

ROOT = common.ROOT
RECOVERY_ID = "OFT-DINO-SAMPLING-CLEAR-TAIL-RECOVERY-V1"
ORIGINAL_OUTERS = {"dynamic_soft": 2000, "dynamic_hard": 1998,
                   "stable_hard": 1999, "shuffled_hard": 1998}
PENDING = tuple(method for method in common.METHODS if method != "dynamic_soft")
TRAINER = str(ROOT / "VLAAttacker/white_patch/UADA_OFT_DINO_SAMPLING_CLEAR.py")
COMMON = str(ROOT / "scripts/dino_sampling_clear_common.py")
# These pairs are the exact reviewed changes, not a path-based waiver.  Their
# unified diffs are also preserved in the ledger against the original ZIP.
APPROVED_CHANGES = {
    TRAINER: {"before": "e99dfb25a0c0d17bb6446b1c115d87a57909c6779c6fb9266266a4c3247add4f",
              "after": "311e348951af40df05eaa4e8e4141c880c4d1effe83a0d176da3a09076723eb8"},
    COMMON: {"before": "bdb65f35042760e9ee108d1e3133f68e9a0933626af23a4ecbb79e932aadd7cd",
             "after": "9ba38366335ae365c3086045f1474edcbc526afb89bdef7bc275878b583fce9b"},
}
ADDED_FILES = (
    "scripts/dino_sampling_clear_recovery_common.py",
    "scripts/run_exploring_oft_dino_sampling_clear_recovery.py",
    "submit_exploring_oft_dino_sampling_clear_recovery_v1.sh",
    "slurm/exploring_oft_dino/117_clear_sampling_recovery_prepare_cpu.sbatch",
    "slurm/exploring_oft_dino/118_clear_sampling_recovery_train_4090.sbatch",
    "slurm/exploring_oft_dino/119_clear_sampling_recovery_validate_cpu.sbatch",
    "slurm/exploring_oft_dino/120_clear_sampling_recovery_eval_3090.sbatch",
    "slurm/exploring_oft_dino/121_clear_sampling_recovery_stats_cpu.sbatch",
)
MUTABLE_ARTIFACTS = ("status.json", "resume.pt", "training.parquet", "RUNTIME_ALIGNMENT.json")
STAGES = (
    ("prepare", "117_clear_sampling_recovery_prepare_cpu.sbatch", None),
    ("train", "118_clear_sampling_recovery_train_4090.sbatch", "1,2,3%3"),
    ("validate", "119_clear_sampling_recovery_validate_cpu.sbatch", None),
    ("eval", "120_clear_sampling_recovery_eval_3090.sbatch", "0-9%8"),
    ("stats", "121_clear_sampling_recovery_stats_cpu.sbatch", None),
)


def recovery_root(cfg):
    return Path(cfg["output_root"]) / "recovery_v1"


def ledger_path(cfg):
    return recovery_root(cfg) / "RECOVERY_LEDGER.json"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return common.digest_json(value)


def _artifact_hashes(folder):
    return {p.name: sha256_file(p) for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.suffix in {".json", ".pt", ".png", ".parquet", ".zip"}}


def verify_source_transition(baseline, current):
    """Accept only two exact reviewed replacements and the named new files."""
    additions = {str((ROOT / name).resolve()) for name in ADDED_FILES}
    require(set(baseline) <= set(current), "Recovery removed an original source")
    require(set(current) - set(baseline) == additions,
            "Recovery has missing or unreviewed added sources")
    changed = {name for name in baseline if baseline[name] != current[name]}
    require(changed == set(APPROVED_CHANGES), "Unreviewed original-source drift")
    for name, pair in APPROVED_CHANGES.items():
        require(baseline[name] == pair["before"] and current[name] == pair["after"],
                f"Reviewed source bytes differ: {name}")
    require(all(Path(name).is_file() and sha256_file(name) == digest
                for name, digest in current.items()), "Current recovery source bytes changed")


def _original_preparation(cfg, report):
    path = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    saved = _read_json(path)
    require(saved["status"] == "COMPLETE" and saved["identity"]["config"] == cfg
            and saved["identity"]["data"] == report["data"]
            and saved["signature"] == _digest(saved["identity"]),
            "Original preparation config/data/signature changed")
    verify_source_transition(saved["identity"]["source"], report["source"])
    return saved, sha256_file(path)


def _verify_original_sources(sources):
    for name, digest in sources.items():
        actual = sha256_file(name) if Path(name).is_file() else None
        pair = APPROVED_CHANGES.get(name)
        require(actual == digest or (pair and digest == pair["before"] and actual == pair["after"]),
                f"Original runtime source has unreviewed drift: {name}")


def _reviewed_diffs(folder):
    result = {}
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        for name, pair in APPROVED_CHANGES.items():
            old = archive.read(str(Path(name).resolve()).lstrip("/"))
            new = Path(name).read_bytes()
            require(hashlib.sha256(old).hexdigest() == pair["before"]
                    and hashlib.sha256(new).hexdigest() == pair["after"],
                    "Reviewed diff does not match original snapshot/current source")
            diff = "".join(difflib.unified_diff(old.decode().splitlines(True),
                          new.decode().splitlines(True), fromfile=name + ".original", tofile=name))
            result[name] = {**pair, "unified_diff": diff,
                            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}
    return result


def _audit_arm(cfg, method, preparation, *, original=True, _retry=0):
    folder = Path(cfg["output_root"]) / "training" / method
    manifest, status = [_read_json(folder / name) for name in ("RUN_MANIFEST.json", "status.json")]
    saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    validate_state(saved, cfg)
    expected = ORIGINAL_OUTERS[method]
    require(saved["config"] == manifest["config"] == cfg
            and saved["data"] == manifest["data"] == preparation["identity"]["data"]
            and saved["source"] == manifest["source"]
            and all(manifest["source"].get(n) == h for n, h in preparation["identity"]["source"].items())
            and saved["method"] == status["method"] == manifest["method"] == method
            and saved["signature"] == status["signature"] == manifest["signature"] ==
                run_signature(cfg, method, manifest["data"], manifest["source"])
            and status["optimizer_updates"] == status["outer"] * 50,
            f"Original committed arm identity changed: {method}")
    require((saved["outer"] == expected if original else expected <= saved["outer"] <= 2000),
            f"Recovery committed boundary changed: {method}")
    require(saved["outer"] == status["outer"] if original or method == "dynamic_soft"
            else expected <= status["outer"] <= saved["outer"],
            "Status exceeds the atomic resume boundary")
    if method == "dynamic_soft":
        require(status["status"] == "COMPLETE" and status["outer"] == 2000,
                "Completed A may not be retrained")
        for name, key in (("outer_2000.pt", "final_patch_sha256"),
                          ("outer_2000.png", "final_png_sha256"), ("resume.pt", "resume_sha256")):
            require(sha256_file(folder / name) == status[key], f"Completed A bytes changed: {name}")
    elif original:
        require(status["status"] == "FAILED" and status["attempted_outer"] == expected + 1
                and status["error"] == "RuntimeError('Nonzero LR but unchanged complete outer RGB')",
                f"Not the audited tail-roundoff failure: {method}")
    _verify_original_sources(manifest["source"])
    require(sha256_file(folder / "source_snapshot.zip") == status["source_snapshot_sha256"],
            f"Original source snapshot changed: {method}")
    with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
        require(json.loads(archive.read("RUN_MANIFEST.json")) == manifest, "Snapshot manifest changed")
        for name, digest in manifest["source"].items():
            require(hashlib.sha256(archive.read(str(Path(name).resolve()).lstrip("/"))).hexdigest() == digest,
                    "Original snapshot contents changed")
    rows = pd.read_parquet(folder / "training.parquet")
    if not original and method in PENDING and len(rows) > saved["outer"]:
        # Other recovery array workers may have advanced between these reads.
        # At most two outers remain; reread the atomic authority rather than
        # rejecting a newer derived file or weakening its prefix check.
        require(_retry < 3, "Concurrent recovery artifact reads did not stabilize")
        return _audit_arm(cfg, method, preparation, original=False, _retry=_retry + 1)
    if original or method == "dynamic_soft":
        require(rows.to_dict("records") == saved["rows"], "Committed parquet differs from atomic resume")
    else:
        require(expected <= len(rows) <= saved["outer"]
                and rows.to_dict("records") == saved["rows"][:len(rows)],
                "Derived parquet is not a valid committed prefix")
        preserved = torch.load(recovery_root(cfg) / "evidence" / method / "resume.pt",
                               map_location="cpu", weights_only=False)
        require(saved["rows"][:expected] == preserved["rows"]
                and saved["score_refresh_history"] == preserved["score_refresh_history"]
                and saved["replay_scoring"] == preserved["replay_scoring"],
                "Retry changed the original committed rows/scores")
    for cp in cfg["checkpoint_outer_iterations"]:
        if cp > saved["outer"]:
            continue
        path = folder / f"outer_{cp:04d}.pt"
        patch = common.check_rgb(path, path.with_suffix(".png"))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["signature"] == manifest["signature"] and payload["method"] == method
                and payload["total_outer"] == cp and payload["total_optimizer_updates"] == cp * 50,
                "Original checkpoint budget/signature changed")
        if cp == saved["outer"]:
            require(torch.equal(patch, saved["patch"].float()), "Final checkpoint/resume patch mismatch")
    require(len(saved["replay_scoring"]) == 258 and saved["score_after"] == 1500
            and len(saved["cuda_rng"]) == 1, "Original scores/CUDA RNG coverage changed")
    return {"outer": saved["outer"], "status": status["status"], "signature": saved["signature"],
            "artifacts": _artifact_hashes(folder), "source": manifest["source"],
            "committed_rows_digest": _digest(saved["rows"]),
            "remaining_outer": 2000 - saved["outer"],
            "remaining_optimizer_updates": (2000 - saved["outer"]) * 50}


def _load_ledger(cfg):
    path = ledger_path(cfg)
    require(path.is_file(), "Audited recovery ledger missing; run the recovery prepare stage")
    ledger = _read_json(path)
    require(ledger["recovery_id"] == RECOVERY_ID and ledger["status"] == "PREPARED"
            and ledger["signature"] == _digest(ledger["identity"])
            and ledger["identity"]["config_sha256"] == sha256_file(common.CONFIG)
            and ledger["identity"]["reviewed_changes"] == APPROVED_CHANGES,
            "Recovery ledger identity/signature/review changed")
    return ledger


def verify_preserved_artifacts(cfg, ledger=None):
    ledger = ledger or _load_ledger(cfg)
    identity = ledger["identity"]
    prep = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    require(sha256_file(prep) == identity["preparation_sha256"], "Original preparation bytes changed")
    for method, arm in identity["arms"].items():
        folder = Path(cfg["output_root"]) / "training" / method
        for name, digest in arm["artifacts"].items():
            if method == "dynamic_soft" or name not in MUTABLE_ARTIFACTS:
                require(sha256_file(folder / name) == digest, f"Preserved artifact changed: {method}/{name}")
            if method in PENDING and name in MUTABLE_ARTIFACTS:
                preserved = recovery_root(cfg) / "evidence" / method / name
                require(sha256_file(preserved) == digest, f"Archived failure evidence changed: {method}/{name}")
    return ledger


def verify_preparation_migration(cfg, preparation, report):
    ledger = verify_preserved_artifacts(cfg)
    identity = ledger["identity"]
    require(preparation["identity"]["config"] == cfg
            and preparation["identity"]["data"] == report["data"] == identity["data"]
            and preparation["signature"] == identity["preparation_signature"]
            and report["source"] == identity["actual_cpu_source"],
            "Recovery preparation actual source/config/data changed")
    verify_source_transition(preparation["identity"]["source"], report["source"])
    return ledger


def verify_historical_source(cfg, path, digest):
    ledger = _load_ledger(cfg)
    pair = APPROVED_CHANGES.get(str(path))
    require(pair and digest == pair["before"] and sha256_file(path) == pair["after"]
            and ledger["identity"]["actual_cpu_source"].get(str(path)) == pair["after"],
            f"Training runtime source changed outside audited recovery: {path}")


def preflight(cfg):
    """Read only: no model, CUDA availability call, queue query, or submission."""
    require(not torch.cuda.is_initialized(), "CPU recovery preflight started with initialized CUDA")
    report = common.pipeline_preflight(cfg)
    preparation, prep_hash = _original_preparation(cfg, report)
    if ledger_path(cfg).exists():
        ledger = verify_preparation_migration(cfg, preparation, report)
        arms = {m: _audit_arm(cfg, m, preparation, original=False) for m in common.METHODS}
        identity = ledger["identity"]
    else:
        arms = {m: _audit_arm(cfg, m, preparation) for m in common.METHODS}
        identity = {"config_sha256": sha256_file(common.CONFIG), "data": report["data"],
                    "preparation_sha256": prep_hash, "preparation_signature": preparation["signature"],
                    "original_cpu_source": preparation["identity"]["source"],
                    "actual_cpu_source": report["source"], "reviewed_changes": APPROVED_CHANGES,
                    "reviewed_diffs": _reviewed_diffs(Path(cfg["output_root"]) / "training/dynamic_soft"),
                    "arms": arms, "resume_budget": {m: 2000 - ORIGINAL_OUTERS[m] for m in PENDING},
                    "training_gpu_limit": 3, "evaluation_gpu_limit": 8,
                    "scheduler_total_outer": 2000, "evaluation_hooks": False}
    require(not torch.cuda.is_initialized(), "CPU recovery preflight unexpectedly initialized CUDA")
    pending = []
    for method in PENDING:
        runtime = recovery_root(cfg) / f"{method}_RUNTIME.json"
        final_evidence = runtime.exists() and _read_json(runtime).get("status") == "COMPLETE"
        if arms[method]["outer"] < 2000 or arms[method]["status"] != "COMPLETE" or not final_evidence:
            pending.append(method)
    return {"status": "PASS", "recovery_id": RECOVERY_ID, "identity": identity,
            "pending_methods": pending,
            "remaining_optimizer_updates": sum(arms[m]["remaining_optimizer_updates"] for m in PENDING),
            "completed_arm_unchanged": "dynamic_soft", "model_loaded": False,
            "cuda_initialized": False, "queue_queried": False, "submitted": False}


def compact_preflight(report):
    identity = report["identity"]
    return {k: v for k, v in report.items() if k != "identity"} | {
        "original_outer": ORIGINAL_OUTERS, "recovery_signature": _digest(identity),
        "preparation_signature": identity["preparation_signature"],
        "reviewed_changed_sources": list(APPROVED_CHANGES),
        "added_sources": list(ADDED_FILES), "training_gpu_limit": 3, "evaluation_gpu_limit": 8,
        "ledger": str(ledger_path(common.config()))}


def prepare(cfg):
    root = recovery_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "PREPARE.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = preflight(cfg)
        if ledger_path(cfg).exists():
            common.validate_training(cfg, ("dynamic_soft",), write_report=False)
            return {"status": "ALREADY_PREPARED", "ledger": str(ledger_path(cfg)),
                    "signature": _load_ledger(cfg)["signature"]}
        identity = report["identity"]
        for method in PENDING:
            evidence = root / "evidence" / method
            evidence.mkdir(parents=True, exist_ok=True)
            original = Path(cfg["output_root"]) / "training" / method
            for name in MUTABLE_ARTIFACTS:
                destination = evidence / name
                digest = identity["arms"][method]["artifacts"][name]
                if destination.exists():
                    require(sha256_file(destination) == digest, "Existing preserved failure evidence changed")
                else:
                    shutil.copy2(original / name, destination)
                require(sha256_file(destination) == digest, "Failure evidence copy changed")
        ledger = {"status": "PREPARED", "recovery_id": RECOVERY_ID,
                  "identity": identity, "signature": _digest(identity)}
        atomic_json(ledger, ledger_path(cfg))
        verify_preserved_artifacts(cfg, ledger)
        # Run the unchanged scientific checks for the completed arm once the
        # explicit migration ledger exists.  No A files are written.
        common.validate_training(cfg, ("dynamic_soft",), write_report=False)
        return {"status": "PREPARED", "ledger": str(ledger_path(cfg)), "signature": ledger["signature"]}


def validate_recovered_training(cfg):
    ledger = verify_preserved_artifacts(cfg)
    for method in PENDING:
        receipt = _read_json(recovery_root(cfg) / f"{method}_RUNTIME.json")
        require(receipt["status"] == "COMPLETE" and receipt["ledger_signature"] == ledger["signature"]
                and receipt["from_outer"] == ORIGINAL_OUTERS[method] and receipt["to_outer"] == 2000
                and receipt["original_signature"] == ledger["identity"]["arms"][method]["signature"],
                f"Recovery runtime receipt missing/changed: {method}")
        for name, digest in receipt["actual_runtime_source"].items():
            require(sha256_file(name) == digest, f"Actual recovery runtime source changed: {name}")
        snapshot = recovery_root(cfg) / f"{method}_actual_source_snapshot.zip"
        require(sha256_file(snapshot) == receipt["actual_source_snapshot_sha256"],
                "Recovery actual source snapshot changed")
        with zipfile.ZipFile(snapshot) as archive:
            for name, digest in receipt["actual_runtime_source"].items():
                require(hashlib.sha256(archive.read(str(Path(name).resolve()).lstrip("/"))).hexdigest() == digest,
                        "Recovery actual source snapshot contents changed")
        original = torch.load(recovery_root(cfg) / "evidence" / method / "resume.pt",
                              map_location="cpu", weights_only=False)
        final = torch.load(Path(cfg["output_root"]) / "training" / method / "resume.pt",
                           map_location="cpu", weights_only=False)
        validate_state(final, cfg)
        require(sha256_file(Path(cfg["output_root"]) / "training" / method / "resume.pt") ==
                receipt["final_resume_sha256"], "Recovery receipt does not bind the final atomic resume")
        require(final["rows"][:ORIGINAL_OUTERS[method]] == original["rows"]
                and final["score_refresh_history"] == original["score_refresh_history"]
                and final["replay_scoring"] == original["replay_scoring"],
                "Recovery altered committed rows or resampled old scoring")
    result = common.validate_training(cfg)
    verify_preserved_artifacts(cfg, ledger)
    atomic_json({"status": "COMPLETE", "ledger_signature": ledger["signature"],
                 "validation_sha256": sha256_file(Path(cfg["output_root"]) / "AUTO_VALIDATION.json"),
                 "completed_arm_bytes_unchanged": True, "evaluation_hooks": False},
                recovery_root(cfg) / "VALIDATION.json")
    return result


def check_evidence(cfg):
    ledger = verify_preserved_artifacts(cfg)
    common.require_preparation(cfg)
    return {"status": "PASS", "ledger_signature": ledger["signature"],
            "completed_arm_bytes_unchanged": True, "evaluation_hooks": False}


def submit_chain(cfg):
    # This function is reached only from the user's explicit unflagged helper.
    report = preflight(cfg)
    root = recovery_root(cfg)
    completed = Path(cfg["output_root"]) / "evaluation/four_suite/STATUS.json"
    if completed.exists() and _read_json(completed).get("status") == "COMPLETE":
        final = _read_json(completed)
        ledger = verify_preserved_artifacts(cfg)
        validation = _read_json(root / "VALIDATION.json")
        require(not report["pending_methods"] and final["experiment_id"] == cfg["experiment_id"]
                and final["config_sha256"] == sha256_file(common.CONFIG)
                and final["preparation_signature"] == ledger["identity"]["preparation_signature"]
                and validation["status"] == "COMPLETE" and validation["ledger_signature"] == ledger["signature"]
                and validation["validation_sha256"] == sha256_file(Path(cfg["output_root"]) / "AUTO_VALIDATION.json"),
                "Completed recovery chain identity changed")
        for name, digest in final["raw_shard_hashes"].items():
            require(sha256_file(name) == digest, "Completed recovery evaluation bytes changed")
        return {"status": "ALREADY_COMPLETE", "recovery_id": RECOVERY_ID, "jobs": []}
    root.mkdir(parents=True, exist_ok=True)
    (Path(cfg["output_root"]) / "logs").mkdir(exist_ok=True)
    with (root / "SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ("oft_clear_sampling_prepare,oft_clear_sampling_train,oft_clear_sampling_validate,"
                 "oft_clear_sampling_eval,oft_clear_sampling_stats,oft_clear_recovery_prepare,"
                 "oft_clear_recovery_train,oft_clear_recovery_validate,oft_clear_recovery_eval,oft_clear_recovery_stats")
        queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", names],
                               check=False, capture_output=True, text=True)
        require(queue.returncode == 0, "Recovery duplicate-chain queue query failed: " + queue.stderr.strip())
        require(not queue.stdout.strip(), "An original/recovery chain is still queued; inspect saved jobs")
        path = root / "SUBMISSION_RECEIPT.json"
        prior = _read_json(path) if path.exists() else None
        receipt = {"status": "SUBMITTING", "recovery_id": RECOVERY_ID,
                   "config_sha256": sha256_file(common.CONFIG),
                   "expected_ledger_signature": _digest(report["identity"]), "jobs": []}
        if prior:
            receipt["previous_receipt"] = prior
        atomic_json(receipt, path)
        dependency = None
        for stage, script, array in STAGES:
            if stage == "train" and not report["pending_methods"]:
                continue
            if stage == "train":
                indices = [str(common.METHODS.index(m)) for m in report["pending_methods"]]
                array = ",".join(indices) + "%3"
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
            if array:
                command.append(f"--array={array}")
            command.append(str(ROOT / "slurm/exploring_oft_dino" / script))
            response = subprocess.run(command, check=False, capture_output=True, text=True)
            job = response.stdout.strip().split(";")[0]
            if response.returncode or not re.fullmatch(r"[0-9]+", job):
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                               error=response.stderr.strip() or response.stdout.strip())
                atomic_json(receipt, path)
                raise RuntimeError(f"Recovery submission failed at {stage}; successful job IDs preserved in {path}")
            receipt["jobs"].append({"stage": stage, "job_id": job, "dependency": dependency, "command": command})
            atomic_json(receipt, path)
            dependency = job
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, path)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", "--dry-run", dest="preflight", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--check-evidence", action="store_true")
    args = parser.parse_args()
    cfg = common.config()
    function = check_evidence if args.check_evidence else submit_chain if args.submit else validate_recovered_training if args.validate else prepare if args.prepare else preflight
    result = function(cfg)
    print(json.dumps(compact_preflight(result) if args.preflight else result, indent=2))
