"""Independent, frozen contracts for six fresh 500-outer action-counter arms.

Historical replay and the sampler trace are common source data only. No old
patch, optimizer, calibration, or repair output participates in initialization.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file

CONFIG = Path(os.environ.get("OFT_DINO_ACTION_COUNTER_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_action_counter500_v1.json"))
METHODS = ("none_original", "none_clear", "random_original", "random_clear",
           "dino_original", "dino_clear")
TRAIN_ORDER = ("random_original", "random_clear", "dino_original", "dino_clear",
               "none_original", "none_clear")
SOURCE_SHA = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
BANK_SHA = "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8"
REPLAY_SHA = "5bb8e2f12efc21227b7e8c371cb4d694e34a0d0f057d85a6313ec3bdcd8afd52"
SCHEDULE_SHA = "665908d315d698f6a29cc3a8faecdf17421c06515d0bc0c33c9ae811818bbbd8"
SCHEDULE_COLUMNS = ("outer_iteration", "origin", "record_index", "base_record_index",
                    "task_id", "state_index")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def frozen_digest(cfg, path):
    target = Path(path).resolve()
    matches = [v for table in ("frozen_inputs", "frozen_eval_assets")
               for k, v in cfg.get(table, {}).items() if Path(k).resolve() == target]
    require(matches and len(set(matches)) == 1 and target.is_file() and sha256_file(target) == matches[0],
            f"Frozen input missing/changed: {path}")
    return matches[0]


def config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    require(cfg["experiment_id"] == "OFT-DINO-ACTION-COUNTER500-V1"
            and tuple(cfg["methods"]) == METHODS
            and cfg["seed"] == 42 and cfg["data_seed"] == cfg["rollout_seed"] == 20260929,
            "Action-counter identity/seed changed")
    require(cfg["base_outer_iterations"] == 0 and cfg["bootstrap_outer"] == 250
            and cfg["diverse_after_outer"] == 250 and cfg["outer_iterations"] == 500
            and cfg["scheduler_total_outer"] == 2000 and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500]
            and cfg["refresh_after_outer"] == [] and cfg["resume_every_outer_iterations"] == 1,
            "Fresh-start/budget/schedule drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0] and cfg["batch_size"] == 1
            and cfg["clip_length"] == 3 and cfg["centers_per_trajectory"] == 6
            and cfg["learning_rate"] == .002 and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20 and cfg["geometry"] is True
            and cfg["resize_patch"] is False and cfg["dtype"] == "bfloat16"
            and cfg["num_open_loop_steps"] == 8, "Threat/optimizer/EOT drift")
    require(cfg["tv_thresholds"] == {"original": .4, "clear": .3}
            and cfg["tv_scale"] == .4 and cfg["tv_weight"] == 1
            and cfg["counter_steps"] == 2 and cfg["counter_step_size"] == 2/255
            and cfg["counter_epsilon"] == 4/255, "Counter/appearance drift")
    require(cfg["sampling_schedule_rows"] == 500
            and cfg["sampling_policy"] == "frozen_old_diverse_first500_no_dynamic_refresh"
            and cfg["sampling_schedule_sha256"] == SCHEDULE_SHA
            and cfg["training_bank_sha256"] == BANK_SHA and cfg["replay_bank_sha256"] == REPLAY_SHA,
            "Common source schedule drift")
    require(Path(cfg["output_root"]).resolve() ==
            (ROOT / "research_runs/wrist_dino_action_counter_v1").resolve()
            and cfg["auto_eval_after_training"] is True and cfg["eval_hooks"] is False
            and cfg["eval_quantization"] == "round(clamp(P,0,1)*255).uint8"
            and cfg["bootstrap_draws"] == 10000 and cfg["bootstrap_seed"] == 20260907,
            "Output/evaluation/statistical policy drift")
    require(not any(k in cfg for k in ("baseline_patch", "baseline_patch_sha256", "prefix_config",
        "prefix_patch", "prefix_resume", "calibration", "appearance_grad_ratio", "repair_steps")),
        "Trained warm start/repair calibration prohibited")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(source.stat().st_mode & 0o222 == 0 and sha256_file(source) == SOURCE_SHA,
            "Read-only Exploring source changed")
    for key in ("training_bank", "replay_bank", "sampling_schedule", "eval_reference"):
        require(frozen_digest(cfg, cfg[key]) == cfg[f"{key}_sha256"], f"{key} identity drift")
    return cfg


def source_records(cfg):
    frozen_digest(cfg, cfg["training_bank"])
    records = torch.load(cfg["training_bank"], map_location="cpu", weights_only=False)["records"]
    from white_patch.UADA_OFT_CORRESPONDENCE import record_key
    require(len(records) == len({record_key(r) for r in records}) == 300,
            "Expected 300 distinct Object clean records")
    for task in range(10):
        block = [r for r in records if int(r["task_id"]) == task]
        states = {int(r["state_index"]) for r in block}
        require(len(block) == 30 and len(states) == 5 and all(10 <= s < 40 for s in states),
                f"Object clean train/test split changed: task={task}")
    return records


def replay_records(cfg):
    path = Path(cfg["replay_bank"])
    digest = frozen_digest(cfg, path)
    frozen_digest(cfg, path.with_name("STATUS.json"))
    status = json.loads(path.with_name("STATUS.json").read_text())
    require(status["status"] == "COMPLETE" and status["bank_sha256"] == digest
            and status["source_bank_sha256"] == BANK_SHA and status["records"] == 258
            and status["experiment_id"] == "OFT-DINO-TRANSFER2000-V1",
            "Frozen Object replay lineage changed")
    records = torch.load(path, map_location="cpu", weights_only=False)["records"]
    require(len(records) == 258 and {int(r["task_id"]) for r in records} == set(range(10))
            and {r["phase_proxy"] for r in records} == {"early", "middle", "late"}
            and all(10 <= int(r["state_index"]) < 40 for r in records),
            "Object replay coverage/split changed")
    # The historic collecting-P digest remains in STATUS for audit. It is never loaded.
    return records, digest


@lru_cache(maxsize=4)
def _schedule(path, digest):
    require(sha256_file(path) == digest, "Frozen sampler trace changed")
    table = pd.read_parquet(path, columns=list(SCHEDULE_COLUMNS)).iloc[:500].copy()
    require(len(table) == 500 and table.outer_iteration.tolist() == list(range(1, 501)),
            "Frozen sampler requires unique, ordered outer 1..500")
    require(table.origin.isin(["clean", "replay"]).all()
            and np.isfinite(table.drop(columns="origin").to_numpy(float)).all(),
            "Invalid sampler metadata")
    for name in SCHEDULE_COLUMNS:
        if name != "origin":
            values = table[name].to_numpy(float)
            require(np.equal(values, np.floor(values)).all(), "Noninteger sampler index")
    rows = table.to_dict("records")
    for row in rows:
        row.update({k: int(row[k]) for k in SCHEDULE_COLUMNS if k != "origin"})
        outer = row["outer_iteration"]
        expected = "clean" if outer <= 250 or outer % 2 else "replay"
        require(row["origin"] == expected and 0 <= row["base_record_index"] < 300
                and 0 <= row["record_index"] < (300 if expected == "clean" else 258)
                and 0 <= row["task_id"] < 10 and 10 <= row["state_index"] < 40,
                "Frozen sampler phase/index/split changed")
        require(expected != "clean" or row["record_index"] == row["base_record_index"],
                "Clean sampler base index drift")
    return rows


def schedule_records(cfg):
    """Decode once; all arms use this same immutable historical 500-row ordering."""
    return _schedule(str(Path(cfg["sampling_schedule"]).resolve()), cfg["sampling_schedule_sha256"])


def draw_record(clean, replay, method, outer, cfg):
    require(method in METHODS and 1 <= outer <= 500, "Invalid counter method/outer")
    row = schedule_records(cfg)[outer - 1]
    index, base_index, origin = row["record_index"], row["base_record_index"], row["origin"]
    require(len(clean) == 300 and len(replay) == 258, "Common bank size drift")
    record = (clean if origin == "clean" else replay)[index]
    require(int(record["task_id"]) == row["task_id"]
            and int(record["state_index"]) == row["state_index"]
            and int(clean[base_index]["task_id"]) == row["task_id"],
            "Frozen sampler record/task/state pairing changed")
    if origin == "replay":
        expected_phase = ("early", "middle", "late")[((outer - 250) // 2 - 1) % 3]
        require(record["phase_proxy"] == expected_phase, "Frozen replay phase pairing changed")
    return record, origin, index, base_index


def data_identity(cfg):
    return {"clean_bank": frozen_digest(cfg, cfg["training_bank"]),
            "config": sha256_file(CONFIG), "window_bank": frozen_digest(cfg, cfg["replay_bank"]),
            "sampling_schedule": frozen_digest(cfg, cfg["sampling_schedule"])}


def source_preflight(cfg):
    """Fingerprint actual current dependencies without loading a model or historical P."""
    for table in ("frozen_inputs", "frozen_eval_assets"):
        for name in cfg[table]:
            frozen_digest(cfg, name)
    from white_patch.UADA_OFT_DINO_ACTION_COUNTER import source_manifest
    sources = source_manifest(cfg)
    require(sources and all(Path(p).is_file() and sha256_file(p) == h for p, h in sources.items()),
            "Actual source fingerprint failed")
    return {"status": "PASS", "source": sources, "data": data_identity(cfg),
            "model_loaded": False, "submitted": False, "original_uada_sha256": SOURCE_SHA}


def check_rgb(path, png, *, expected_shape=(3, 50, 50)):
    value = torch.load(path, map_location="cpu", weights_only=False)["patch"].float()
    require(tuple(value.shape) == expected_shape and bool(torch.isfinite(value).all())
            and bool(((value >= 0) & (value <= 1)).all()), f"Invalid RGB tensor: {path}")
    with Image.open(png) as picture:
        require(picture.mode == "RGB" and picture.size == (50, 50)
                and np.array_equal(np.asarray(picture),
                    value.permute(1, 2, 0).mul(255).round().byte().numpy()),
                f"PNG/deployment quantization mismatch: {png}")
    return value


def external_baselines(cfg):
    require(set(cfg["external_baselines"]) == {"original_uada_500", "diverse_reference_500"},
            "Matched 500 baseline coverage changed")
    result = {}
    for condition, item in cfg["external_baselines"].items():
        path = Path(item["pt"])
        require(path.name == "outer_0500.pt" and item["checkpoint_outer"] == 500,
                "External reference must be the pinned 500 checkpoint")
        for file, key in ((path, "pt_sha256"), (path.with_suffix(".png"), "png_sha256"),
                          (Path(item["manifest"]), "manifest_sha256"),
                          (Path(item["history"]), "history_sha256")):
            require(file.is_file() and sha256_file(file) == item[key],
                    f"External 500 reference changed: {file}")
        manifest = json.loads(Path(item["manifest"]).read_text())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["method"] == item["method"] and manifest["config"]["seed"] == 42
                and manifest["config"]["suite"] == "libero_object"
                and manifest["config"]["patch_shape"] == [3, 50, 50],
                "External Object reference identity changed")
        if condition == "diverse_reference_500":
            require(payload["total_outer"] == 500 and payload["total_optimizer_updates"] == 25000
                    and payload["signature"] == manifest["signature"],
                    "Diverse reference checkpoint budget/signature changed")
        else:
            require(payload["source_uada_sha256"] == SOURCE_SHA and manifest["source_uada"] == SOURCE_SHA,
                    "Original UADA reference lineage changed")
        history = pd.read_parquet(item["history"])
        checkpoint = history.loc[history.outer_iteration.eq(500)]
        require(len(checkpoint) == 1 and int(checkpoint.iloc[0].optimizer_updates) == 25000,
                "Historical matched-500 cost/budget row missing")
        result[condition] = {"patch": check_rgb(path, path.with_suffix(".png")),
                             "pt": item["pt_sha256"], "png": item["png_sha256"],
                             "recorded_training_elapsed_s": float(checkpoint.iloc[0].elapsed_s)}
    return result


def pipeline_preflight(cfg):
    source = source_preflight(cfg)
    clean, (replay, replay_digest) = source_records(cfg), replay_records(cfg)
    schedule = schedule_records(cfg)
    for outer in range(1, 501):
        draw_record(clean, replay, METHODS[0], outer, cfg)
    from run_exploring_oft_dino_action_counter_cross_suite import load_config, frozen_clean, DENOMINATORS
    evaluation = load_config(cfg)
    denominators = {s: int(frozen_clean(evaluation, s).success.sum()) for s in evaluation["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators changed")
    refs = external_baselines(cfg)
    return {**source, "clean_records": len(clean), "replay_records": len(replay),
            "schedule_records": len(schedule), "replay_sha256": replay_digest,
            "fresh_start_outer": 0, "denominators": denominators,
            "new_training_methods": list(METHODS), "external_500_references":
                {k: {n: v for n, v in r.items() if n != "patch"} for k, r in refs.items()},
            "maximum_attack_episodes": 3200, "clean_reference_episodes": 400,
            "eligible_rows_expected": 8 * sum(DENOMINATORS.values()),
            "training_gpu_limit": 5, "evaluation_gpu_limit": 8,
            "stop_after_outer": 500, "bootstrap_draws": 10000, "bootstrap_seed": 20260907}


def compact_preflight(report):
    """Keep user-facing preflight short; the preparation JSON retains exact sources."""
    result = dict(report)
    sources = result.pop("source", {})
    result.update(source_files=len(sources), source_manifest_digest=digest_json(sources))
    return result


def preparation_identity(cfg, report):
    return {"config": cfg, "source": report["source"], "data": report["data"]}


def require_preparation(cfg, report=None):
    report = report or source_preflight(cfg)
    path = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    saved = json.loads(path.read_text())
    require(saved["status"] == "COMPLETE" and saved["identity"] == preparation_identity(cfg, report)
            and saved["signature"] == digest_json(saved["identity"]),
            "CPU preparation signature/source/data changed")
    return saved


def prepare(cfg):
    report = pipeline_preflight(cfg)
    identity = preparation_identity(cfg, report)
    result = {"status": "COMPLETE", "identity": identity, "signature": digest_json(identity),
              "experiment_id": cfg["experiment_id"], "preflight": report}
    path = Path(cfg["output_root"]) / "reports/PREPARATION.json"
    if path.exists():
        require(json.loads(path.read_text()) == result, "Existing preparation source/config changed")
    else:
        atomic_json(result, path)
    return {"status": "COMPLETE", "signature": result["signature"], "model_loaded": False,
            "submitted": False, "preparation": str(path)}


def validate_training(cfg, methods=METHODS, *, write_report=True):
    """CPU, fail-closed checkpoint/Adam/scheduler/source and six-start validation."""
    require_preparation(cfg)
    expected_data = data_identity(cfg)
    hashes, starts, movement = {}, {}, {}
    schedule = schedule_records(cfg)
    require(methods and set(methods) <= set(METHODS), "Invalid validation method subset")
    for method in methods:
        folder = Path(cfg["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == 500 and status["optimizer_updates"] == 25000
                and manifest["config"] == cfg and manifest["data"] == expected_data
                and manifest["signature"] == status["signature"]
                and manifest["parameter_count"] == 7500 and manifest["fresh_start_outer"] == 0
                and manifest["initial_optimizer_state_empty"] is True
                and manifest["scheduler_total_outer"] == 2000 and alignment["status"] == "PASS",
                f"Training freshness/lineage/runtime alignment failed: {method}")
        for path, digest in manifest["source"].items():
            require(Path(path).is_file() and sha256_file(path) == digest,
                    f"Training runtime source changed: {path}")
        for file, key in (("outer_0500.pt", "final_patch_sha256"), ("outer_0500.png", "final_png_sha256"),
                          ("resume.pt", "resume_sha256"), ("source_snapshot.zip", "source_snapshot_sha256")):
            require(sha256_file(folder / file) == status[key], f"Completed training bytes changed: {method}/{file}")
        initial = check_rgb(folder / "outer_0000.pt", folder / "outer_0000.png")
        value = check_rgb(folder / "outer_0500.pt", folder / "outer_0500.png")
        starts[method] = hashlib.sha256(initial.contiguous().numpy().tobytes()).hexdigest()
        require(starts[method] == manifest["initial_patch_sha256"] == status["initial_patch_sha256"],
                "Fresh initial RGB digest changed")
        expected = torch.rand(tuple(cfg["patch_shape"]), generator=torch.Generator().manual_seed(42))
        require(torch.equal(initial, expected), "Fresh RGB is not the frozen seed42 initialization")
        movement[method] = float((value - initial).abs().mean())
        require(movement[method] > 1e-7 and not torch.equal(
            value.mul(255).round(), initial.mul(255).round()), f"No deployed RGB update: {method}")
        payload = torch.load(folder / "outer_0500.pt", map_location="cpu", weights_only=False)
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        require(payload["method"] == method and payload["signature"] == status["signature"]
                and payload["total_outer"] == 500 and payload["total_optimizer_updates"] == 25000
                and saved["config"] == cfg and saved["method"] == method
                and saved["signature"] == status["signature"] and saved["outer"] == 500
                and saved["scheduler"]["last_epoch"] == 500 and len(saved["rows"]) == 500
                and torch.equal(saved["patch"].float(), value)
                and saved.keys() >= {"torch_rng", "cuda_rng", "numpy_rng", "python_rng"},
                f"Full optimizer/scheduler/RNG resume failed: {method}")
        states = saved["optimizer"]["state"]
        require(len(states) == 1 and all(int(s["step"]) == 25000 and
                {"exp_avg", "exp_avg_sq"} <= s.keys() and
                bool(torch.isfinite(s["exp_avg"]).all()) and bool(torch.isfinite(s["exp_avg_sq"]).all())
                for s in states.values()), "Adam step/moments missing or reset")
        rows = pd.read_parquet(folder / "training.parquet")
        require(len(rows) == 500 and not rows.outer_iteration.duplicated().any()
                and np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Training record count/finite values failed")
        for actual, frozen in zip(rows.to_dict("records"), schedule, strict=True):
            require(all(actual[k] == frozen[k] for k in SCHEDULE_COLUMNS)
                    and actual["optimizer_updates"] == frozen["outer_iteration"] * 50,
                    "One arm changed the common frozen sampler/update schedule")
        hashes[method] = {"pt": status["final_patch_sha256"], "png": status["final_png_sha256"]}
    require(len(set(starts.values())) == 1, "Six fresh starts are not identical")
    result = {"status": "COMPLETE", "experiment_id": cfg["experiment_id"],
              "config_sha256": sha256_file(CONFIG), "preparation_signature": require_preparation(cfg)["signature"],
              "checkpoints": hashes, "initial_rgb_sha256": starts, "movement_from_initial": movement,
              "fresh_start_outer": 0, "optimizer_updates": 25000, "scheduler_total_outer": 2000,
              "evaluation_ready": True, "evaluation_hooks": False}
    if write_report:
        require(tuple(methods) == METHODS, "Final validation requires all six arms")
        atomic_json(result, Path(cfg["output_root"]) / "AUTO_VALIDATION.json")
    return result


def submit_chain(cfg):
    """Only the explicit user helper invokes this; every receipt precedes the next submit."""
    root = Path(cfg["output_root"])
    final = root / "evaluation/four_suite/STATUS.json"
    if final.exists():
        final_status = json.loads(final.read_text())
        if final_status.get("status") == "COMPLETE":
            require(final_status.get("experiment_id") == cfg["experiment_id"]
                    and final_status.get("config_sha256") == sha256_file(CONFIG)
                    and final_status.get("preparation_signature") == require_preparation(cfg)["signature"],
                    "Completed-chain identity drift")
            return {"status": "ALREADY_COMPLETE", "jobs": []}
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "reports/SUBMISSION.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another action-counter helper is submitting this chain")
        names = "oft_counter_prepare,oft_counter_train,oft_counter_validate,oft_counter_eval,oft_counter_stats"
        queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", names],
                               check=False, capture_output=True, text=True)
        require(queue.returncode == 0, "squeue query failed; duplicate-chain state unknown: " + queue.stderr.strip())
        require(not queue.stdout.strip(), "An action-counter chain remains queued; inspect its receipt/logs before resubmitting")
        completed = [m for m in METHODS if (root / "training" / m / "status.json").exists()
                     and json.loads((root / "training" / m / "status.json").read_text()).get("status") == "COMPLETE"]
        if completed:
            validate_training(cfg, tuple(completed), write_report=False)
        pending = [i for i, m in enumerate(TRAIN_ORDER) if m not in completed]
        receipt_path = root / "reports/SUBMISSION_RECEIPT.json"
        receipt = {"status": "SUBMITTING", "experiment_id": cfg["experiment_id"],
                   "config_sha256": sha256_file(CONFIG), "jobs": [],
                   "dependency_failure_help": "afterok isolates GPU stages; --kill-on-invalid-dep=yes cancels "
                   "an impossible downstream dependency. Inspect failed upstream logs and the saved job IDs."}
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            require(prior["config_sha256"] == receipt["config_sha256"], "Existing submission config changed")
            receipt["previous_receipt"] = prior
        atomic_json(receipt, receipt_path)
        dependency = None
        stages = [("prepare", "107_counter500_prepare_cpu.sbatch", None)]
        if pending:
            array = "0-5%5" if len(pending) == 6 else ",".join(map(str, pending)) + "%5"
            stages.append(("train", "108_counter500_train_4090.sbatch", array))
        stages += [("validate", "109_counter500_validate_cpu.sbatch", None),
                   ("eval", "110_counter500_eval_3090.sbatch", "0-9%8"),
                   ("stats", "111_counter500_stats_cpu.sbatch", None)]
        for stage, script, array in stages:
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
            if dependency:
                command.append(f"--dependency=afterok:{dependency}")
            if array:
                command.append(f"--array={array}")
            command.append(str(ROOT / "slurm/exploring_oft_dino" / script))
            response = subprocess.run(command, check=False, capture_output=True, text=True)
            job_id = response.stdout.strip().split(";")[0]
            if response.returncode or not re.fullmatch(r"[0-9]+", job_id):
                receipt.update(status="SUBMISSION_FAILED", failed_stage=stage,
                               error=response.stderr.strip() or f"Invalid receipt: {response.stdout.strip()}")
                atomic_json(receipt, receipt_path)
                raise RuntimeError(f"Submission failed at {stage}; no downstream job submitted. "
                                   f"Successful IDs saved in {receipt_path}: {receipt['error']}")
            receipt["jobs"].append({"stage": stage, "job_id": job_id, "dependency": dependency,
                                    "command": command})
            atomic_json(receipt, receipt_path)
            dependency = job_id
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt, receipt_path)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    cfg = config()
    action = submit_chain if args.submit else validate_training if args.validate else prepare if args.prepare else pipeline_preflight
    result = action(cfg)
    print(json.dumps(compact_preflight(result) if args.preflight else result, indent=2))
