"""Frozen independent contracts for four live-sampled, fresh 2000-outer arms."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VLAAttacker"))
from white_patch.UADA_OFT_DINO import atomic_json, require, sha256_file

CONFIG = Path(os.environ.get("OFT_DINO_SAMPLING_CLEAR_CONFIG",
    ROOT / "configs/exploring_oft_dino/object_dino_sampling_clear2000_v1.json"))
METHODS = TRAIN_ORDER = ("dynamic_soft", "dynamic_hard", "stable_hard", "shuffled_hard")
CHECKPOINTS = (1000, 2000)
SOURCE_SHA = "808b08ec686dd5bb85d006963cf79a67c0a8e778b9ba7464c30236cc345fca2e"
BANK_SHA = "7822da38434ef65058a10a1844615583903122520fb660b6522f174ea48a58f8"
REPLAY_SHA = "5bb8e2f12efc21227b7e8c371cb4d694e34a0d0f057d85a6313ec3bdcd8afd52"


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
    require(cfg["experiment_id"] == "OFT-DINO-SAMPLING-CLEAR2000-V1"
            and tuple(cfg["methods"]) == METHODS
            and cfg["seed"] == 42 and cfg["data_seed"] == cfg["rollout_seed"] == 20260929,
            "Sampling-clear identity/seed changed")
    require(cfg["base_outer_iterations"] == 0 and cfg["bootstrap_outer"] == 250
            and cfg["diverse_after_outer"] == 250 and cfg["outer_iterations"] == 2000
            and cfg["scheduler_total_outer"] == 2000 and cfg["inner_loop"] == 50
            and cfg["checkpoint_outer_iterations"] == [0, 250, 500, 1000, 1500, 2000]
            and cfg["refresh_after_outer"] == [250, 500, 1000, 1500]
            and cfg["eval_checkpoint_outer_iterations"] == [1000, 2000]
            and cfg["resume_every_outer_iterations"] == 1,
            "Fresh-start/budget/schedule drift")
    require(cfg["patch_shape"] == [3, 50, 50] and cfg["patch_xy"] == [26, 137]
            and cfg["maskidx"] == [0] and cfg["batch_size"] == 1
            and cfg["clip_length"] == 3 and cfg["centers_per_trajectory"] == 6
            and cfg["learning_rate"] == .002 and cfg["weight_decay"] == 0
            and cfg["warmup_outer_iterations"] == 20 and cfg["geometry"] is True
            and cfg["resize_patch"] is False and cfg["dtype"] == "bfloat16"
            and cfg["num_open_loop_steps"] == 8 and cfg["gradient_clip"] is None,
            "Threat/optimizer/EOT drift")
    require(cfg["tv_threshold"] == .3 and cfg["tv_scale"] == .4 and cfg["tv_weight"] == 1
            and cfg["gaussian_sigma"] == .5 and cfg["gaussian_kernel_size"] == 3
            and cfg["feasibility_passes"] == 3 and cfg["hf_epsilon"] == 16/255
            and cfg["tv_quantization_margin"] == 2/255
            and cfg["hf_quantization_margin"] == 1/255,
            "Fixed appearance/quantization budget drift")
    require(cfg["sampling_policy"] == "live_own_patch_task_phase_dino_weights"
            and cfg["training_bank_sha256"] == BANK_SHA and cfg["replay_bank_sha256"] == REPLAY_SHA,
            "Common source/live sampling drift")
    reference = json.loads(Path(cfg["eval_reference"]).read_text())
    require(cfg["suite"] == "libero_object" and cfg["unnorm_key"] == "libero_object_no_noops"
            and Path(cfg["checkpoint"]).resolve() == Path(reference["checkpoints"]["libero_object"]).resolve()
            and cfg["action_head_q01"] == [-.9453125,-.34181639552116394,-.9336718916893005,
                -.828125,-.7187890410423279,-.5859765410423279,-.0034490968100726604]
            and cfg["action_head_q99"] == [.4338674545288086,.9609375,1.0234375,.6953516006469727,
                .8516016006469727,.84765625,1.0078125], "Object-only model/action boundary drift")
    require(Path(cfg["output_root"]).resolve() ==
            (ROOT / "research_runs/wrist_dino_sampling_clear_v1").resolve()
            and cfg["auto_eval_after_training"] is True and cfg["eval_hooks"] is False
            and cfg["eval_quantization"] == "round(clamp(P,0,1)*255).uint8"
            and cfg["bootstrap_draws"] == 10000 and cfg["bootstrap_seed"] == 20260907
            and cfg["training_gpu_limit"] == 4 and cfg["evaluation_gpu_limit"] == 8,
            "Output/evaluation/statistical policy drift")
    require(not any(k in cfg for k in ("baseline_patch", "baseline_patch_sha256", "prefix_config",
        "prefix_patch", "prefix_resume", "calibration", "appearance_grad_ratio", "repair_steps",
        "sampling_schedule", "sampling_schedule_sha256", "counter_steps", "counter_epsilon")),
        "Trained warm start/frozen schedule/counter/calibration prohibited")
    source = ROOT / "VLAAttacker/white_patch/UADA.py"
    require(source.stat().st_mode & 0o222 == 0 and sha256_file(source) == SOURCE_SHA,
            "Read-only Exploring source changed")
    for key in ("training_bank", "replay_bank", "eval_reference"):
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
    return records, digest


def draw_record(clean, replay, method, outer, cfg):
    """Legacy live base-index/task/phase quota with each arm's current score weights."""
    from white_patch.UADA_OFT_DINO import ExploringOFTOpenVLAAttacker
    require(method in METHODS and 1 <= outer <= cfg["outer_iterations"], "Invalid sampling method/outer")
    require(len(clean) == 300 and len(replay) == 258, "Common bank size drift")
    seed = cfg["data_seed"] + cfg["seed"]
    base_index = ExploringOFTOpenVLAAttacker._training_index(outer, len(clean), seed)
    base = clean[base_index]
    local = outer - cfg["bootstrap_outer"]
    if local <= 0 or local % 2:
        return base, "clean", base_index, base_index
    phase = ("early", "middle", "late")[(local // 2 - 1) % 3]
    candidates = [i for i, record in enumerate(replay)
                  if int(record["task_id"]) == int(base["task_id"]) and record["phase_proxy"] == phase]
    require(candidates, f"Missing task/phase: {base['task_id']}/{phase}")
    field = "shuffled_weight" if method == "shuffled_hard" else "dino_weight"
    weights = [float(replay[i][field]) for i in candidates]
    require(all(w in (1., 2.) for w in weights), "Live replay weight outside frozen 1/2 policy")
    index = random.Random(seed * 1_000_000 + local).choices(candidates, weights=weights, k=1)[0]
    return replay[index], "replay", index, base_index


def data_identity(cfg):
    return {"clean_bank": frozen_digest(cfg, cfg["training_bank"]),
            "config": sha256_file(CONFIG), "window_bank": frozen_digest(cfg, cfg["replay_bank"])}


def source_preflight(cfg):
    """Fingerprint current dependencies without model loading, CUDA, or old trained P."""
    for table in ("frozen_inputs", "frozen_eval_assets"):
        for name in cfg[table]:
            frozen_digest(cfg, name)
    from white_patch.UADA_OFT_DINO_SAMPLING_CLEAR import source_manifest
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
    expected = {f"{method}_{cp}" for cp in CHECKPOINTS
                for method in ("original_uada", "diverse_reference")}
    require(set(cfg["external_baselines"]) == expected, "Matched 1000/2000 baseline coverage changed")
    result = {}
    for condition, item in cfg["external_baselines"].items():
        path, cp = Path(item["pt"]), int(item["checkpoint_outer"])
        require(cp in CHECKPOINTS and path.name == f"outer_{cp:04d}.pt"
                and condition.endswith(f"_{cp}"), "External reference checkpoint identity changed")
        for file, key in ((path, "pt_sha256"), (path.with_suffix(".png"), "png_sha256"),
                          (Path(item["manifest"]), "manifest_sha256"),
                          (Path(item["history"]), "history_sha256")):
            require(file.is_file() and sha256_file(file) == item[key], f"External reference changed: {file}")
        manifest = json.loads(Path(item["manifest"]).read_text())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(payload["method"] == item["method"] and manifest["config"]["seed"] == 42
                and manifest["config"]["suite"] == "libero_object"
                and manifest["config"]["patch_shape"] == [3, 50, 50],
                "External Object reference identity changed")
        if condition.startswith("diverse_reference_"):
            require(payload["total_outer"] == cp and payload["total_optimizer_updates"] == cp*50
                    and payload["signature"] == manifest["signature"], "Diverse reference budget/signature changed")
        else:
            require(payload["source_uada_sha256"] == SOURCE_SHA and manifest["source_uada"] == SOURCE_SHA
                    and payload["outer_iterations"] == 2000 and payload["inner_loop"] == 50,
                    "Original UADA reference lineage changed")
        history = pd.read_parquet(item["history"])
        checkpoint = history.loc[history.outer_iteration.eq(cp)]
        require(len(checkpoint) == 1 and int(checkpoint.iloc[0].optimizer_updates) == cp*50,
                "Historical matched-checkpoint cost/budget row missing")
        result[condition] = {"patch": check_rgb(path, path.with_suffix(".png")),
            "pt": item["pt_sha256"], "png": item["png_sha256"], "checkpoint_outer": cp,
            "recorded_training_elapsed_s": float(checkpoint.iloc[0].elapsed_s)}
    return result


def pipeline_preflight(cfg):
    source = source_preflight(cfg)
    clean, (replay, replay_digest) = source_records(cfg), replay_records(cfg)
    # Discard the collecting patch's score fields: exercise only fresh equal weights.
    live = [dict(r, dino_weight=1., shuffled_weight=1.) for r in replay]
    for outer in range(1, 2001):
        for method in METHODS:
            draw_record(clean, live, method, outer, cfg)
    from run_exploring_oft_dino_sampling_clear_cross_suite import load_config, frozen_clean, DENOMINATORS
    evaluation = load_config(cfg)
    denominators = {s: int(frozen_clean(evaluation, s).success.sum()) for s in evaluation["suites"]}
    require(denominators == DENOMINATORS, "Frozen clean denominators changed")
    refs = external_baselines(cfg)
    return {**source, "clean_records": len(clean), "replay_records": len(replay),
            "live_sampler_outer_iterations": 2000, "replay_sha256": replay_digest,
            "fresh_start_outer": 0, "denominators": denominators,
            "new_training_methods": list(METHODS), "external_references":
                {k: {n: v for n, v in r.items() if n != "patch"} for k, r in refs.items()},
            "maximum_attack_episodes": 4800, "clean_reference_episodes": 400,
            "eligible_rows_expected": 12 * sum(DENOMINATORS.values()),
            "training_gpu_limit": 4, "evaluation_gpu_limit": 8,
            "stop_after_outer": 2000, "bootstrap_draws": 10000, "bootstrap_seed": 20260907}


def compact_preflight(report):
    result = dict(report)
    sources = result.pop("source", {})
    result.update(source_files=len(sources), source_manifest_digest=digest_json(sources))
    return result


def preparation_identity(cfg, report):
    return {"config": cfg, "source": report["source"], "data": report["data"]}


def require_preparation(cfg, report=None):
    report = report or source_preflight(cfg)
    saved = json.loads((Path(cfg["output_root"]) / "reports/PREPARATION.json").read_text())
    require(saved["status"] == "COMPLETE"
            and saved["signature"] == digest_json(saved["identity"]),
            "CPU preparation signature/source/data changed")
    if saved["identity"] != preparation_identity(cfg, report):
        from dino_sampling_clear_recovery_common import verify_preparation_migration
        verify_preparation_migration(cfg, saved, report)
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


def _tensor_digest(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_sampling_history(cfg, method, folder, saved, rows, clean, replay):
    """Reconstruct every live draw from the arm's committed score intervals."""
    from white_patch.oft_dino_sampling_clear_objectives import apply_score_weights, stable_score
    intervals = {}
    forwards = 3 if method in ("dynamic_soft", "dynamic_hard") else 5
    stable = method in ("stable_hard", "shuffled_hard")
    for history in saved["score_refresh_history"]:
        cp = int(history["outer"])
        score = json.loads((folder / f"scores_after_{cp:04d}.json").read_text())
        require(score["method"] == method and score["outer"] == cp
                and len(score["records"]) == history["records"] == 258
                and score["full_forward_calls"] == history["full_forward_calls"] == 258*forwards,
                "Scoring checkpoint/count/forward-cost identity changed")
        patch = torch.load(folder / f"outer_{cp:04d}.pt",map_location="cpu",weights_only=False)["patch"]
        require(score["patch_tensor_sha256"] == history["patch_tensor_sha256"] == _tensor_digest(patch)
                and score["quantized_patch_sha256"] == history["quantized_patch_sha256"] ==
                    _tensor_digest(patch.mul(255).round().div(255)),
                "Scoring did not use the own checkpoint patch")
        merged = [dict(record, **item) for record,item in zip(replay,score["records"],strict=True)]
        for record in merged:
            require(record["score_full_forward_calls"] == forwards
                    and record["score_view_count"] == (2 if stable else 1)
                    and np.isfinite([record[k] for k in ("dino_score","patched_target_mse","restored_target_mse",
                                                         "dino_delta_rms")]).all(),
                    "Scoring view/finite/full-forward count changed")
            if stable:
                views = record["score_views"]
                require(len(views) == 2 and all(v["patched_siglip_retained"] is True for v in views)
                        and abs(record["dino_score"]-stable_score(views[0]["raw_delta"],views[1]["raw_delta"])) < 1e-12,
                        "Stable score did not use both raw view deltas/current SigLIP")
        expected = apply_score_weights(merged,cfg["data_seed"],cp,stable=stable)
        require(all(actual["dino_weight"] == wanted["dino_weight"]
                    and actual["shuffled_weight"] == wanted["shuffled_weight"]
                    for actual,wanted in zip(merged,expected,strict=True)),
                "Current task/phase top-half weights or own shuffled correspondence changed")
        intervals[cp] = merged
    require(score["records"] == saved["replay_scoring"], "Final scoring JSON differs from atomic resume")
    for actual in rows.to_dict("records"):
        outer = int(actual["outer_iteration"])
        score_after = max([0]+[cp for cp in cfg["refresh_after_outer"] if cp < outer])
        active = intervals.get(score_after,replay)
        record,origin,index,base = draw_record(clean,active,method,outer,cfg)
        require(actual["origin"] == origin and int(actual["record_index"]) == index
                and int(actual["base_record_index"]) == base
                and int(actual["task_id"]) == int(record["task_id"])
                and int(actual["state_index"]) == int(record["state_index"])
                and int(actual["score_after_outer"]) == score_after
                and float(actual["dino_score"]) == float(record.get("dino_score",0.)),
                "Exact live own-weight draw/score/refresh sequence changed")


def validate_training(cfg, methods=METHODS, *, write_report=True):
    """CPU fail-closed validation of full RGB, all checkpoints and continuous resume."""
    from white_patch.oft_dino_sampling_clear_objectives import enforce_constraints, constraint_metrics
    preparation = require_preparation(cfg)
    expected_data = data_identity(cfg)
    hashes, starts, raw_starts, movement, structures = {}, {}, {}, {}, {}
    require(methods and set(methods) <= set(METHODS), "Invalid validation method subset")
    raw_expected = torch.rand(tuple(cfg["patch_shape"]), generator=torch.Generator().manual_seed(42))
    initial_expected, _ = enforce_constraints(raw_expected, cfg)
    clean, (replay, _) = source_records(cfg), replay_records(cfg)
    for method in methods:
        folder = Path(cfg["output_root"]) / "training" / method
        status = json.loads((folder / "status.json").read_text())
        manifest = json.loads((folder / "RUN_MANIFEST.json").read_text())
        alignment = json.loads((folder / "RUNTIME_ALIGNMENT.json").read_text())
        require(status["status"] == "COMPLETE" and status["method"] == method
                and status["outer"] == 2000 and status["optimizer_updates"] == 100000
                and manifest["config"] == cfg and manifest["data"] == expected_data
                and manifest["signature"] == status["signature"]
                and manifest["parameter_count"] == 7500 and manifest["fresh_start_outer"] == 0
                and manifest["initial_optimizer_state_empty"] is True
                and manifest["scheduler_total_outer"] == 2000 and alignment["status"] == "PASS"
                and manifest["feature_loss"] is False
                and manifest["feature_replacement_hook_during_training"] is False
                and manifest["active_gradient_clipping"] is False
                and status["score_refresh_outer"] == [250, 500, 1000, 1500],
                f"Training freshness/lineage/runtime alignment failed: {method}")
        for path, digest in manifest["source"].items():
            if not Path(path).is_file() or sha256_file(path) != digest:
                from dino_sampling_clear_recovery_common import verify_historical_source
                verify_historical_source(cfg, path, digest)
        for file, key in (("outer_2000.pt", "final_patch_sha256"), ("outer_2000.png", "final_png_sha256"),
                          ("resume.pt", "resume_sha256"), ("source_snapshot.zip", "source_snapshot_sha256")):
            require(sha256_file(folder / file) == status[key], f"Completed training bytes changed: {method}/{file}")
        with zipfile.ZipFile(folder / "source_snapshot.zip") as archive:
            for path, digest in manifest["source"].items():
                require(hashlib.sha256(archive.read(str(Path(path).resolve()).lstrip("/"))).hexdigest() == digest,
                        "Source snapshot archive contents differ from manifest")
        raw = check_rgb(folder / "initial_raw.pt", folder / "initial_raw.png")
        initial = check_rgb(folder / "outer_0000.pt", folder / "outer_0000.png")
        require(torch.equal(raw, raw_expected) and torch.equal(initial, initial_expected),
                "Fresh RGB is not seed42 followed by shared feasible initialization")
        starts[method], raw_starts[method] = _tensor_digest(initial), _tensor_digest(raw)
        require(starts[method] == manifest["initial_patch_sha256"] == status["initial_patch_sha256"]
                and raw_starts[method] == manifest["raw_random_patch_sha256"] == status["raw_random_patch_sha256"],
                "Initial raw/feasible RGB digest changed")
        hashes[method], structures[method] = {}, {}
        previous = initial
        movement[method] = {}
        for cp in cfg["checkpoint_outer_iterations"]:
            path = folder / f"outer_{cp:04d}.pt"
            value = check_rgb(path, path.with_suffix(".png"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["method"] == method and payload["signature"] == status["signature"]
                    and payload["total_outer"] == cp and payload["total_optimizer_updates"] == cp*50,
                    "Checkpoint identity/update budget changed")
            continuous = constraint_metrics(value, cfg)
            quantized = constraint_metrics(value.mul(255).round().div(255), cfg)
            if cp == 0 or method != "dynamic_soft":
                require(continuous["feasible_internal"] and quantized["feasible_scientific"],
                        f"Actual continuous/PNG constraints exceeded: {method}/{cp}")
            structures[method][str(cp)] = {"continuous": continuous, "png": quantized}
            hashes[method][str(cp)] = {"pt": sha256_file(path), "png": sha256_file(path.with_suffix(".png"))}
            movement[method][str(cp)] = {"mae_from_initial": float((value-initial).abs().mean()),
                "mae_from_previous_checkpoint": float((value-previous).abs().mean()),
                "changed_fraction_from_initial": float(value.mul(255).round().ne(initial.mul(255).round()).float().mean())}
            previous = value
        value = previous
        require(movement[method]["2000"]["mae_from_initial"] > 1e-7
                and movement[method]["2000"]["changed_fraction_from_initial"] > 0,
                f"No deployed RGB update: {method}")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        require(saved["config"] == cfg and saved["method"] == method
                and saved["signature"] == status["signature"] and saved["outer"] == 2000
                and saved["source"] == manifest["source"] and saved["data"] == expected_data
                and saved["scheduler"]["last_epoch"] == 2000 and len(saved["rows"]) == 2000
                and torch.equal(saved["patch"].float(), value)
                and saved.keys() >= {"torch_rng", "cuda_rng", "numpy_rng", "python_rng"}
                and saved["score_after"] == 1500 and len(saved["replay_scoring"]) == 258
                and [int(item["outer"]) for item in saved["score_refresh_history"]] == [250,500,1000,1500],
                f"Full optimizer/scheduler/scoring/RNG resume failed: {method}")
        states = saved["optimizer"]["state"]
        require(len(states) == 1 and all(int(s["step"]) == 100000 and
                {"exp_avg", "exp_avg_sq"} <= s.keys() and
                bool(torch.isfinite(s["exp_avg"]).all()) and bool(torch.isfinite(s["exp_avg_sq"]).all())
                for s in states.values()), "Adam step/moments missing or reset")
        rows = pd.read_parquet(folder / "training.parquet")
        require(len(rows) == 2000 and rows.outer_iteration.tolist() == list(range(1,2001))
                and rows.optimizer_updates.tolist() == list(range(50,100001,50))
                and rows.to_dict("records") == saved["rows"]
                and np.isfinite(rows.select_dtypes(include=[np.number]).to_numpy()).all(),
                "Training record count/finite values failed")
        validate_sampling_history(cfg,method,folder,saved,rows,clean,replay)
    require(len(set(starts.values())) == len(set(raw_starts.values())) == 1, "Four fresh starts are not identical")
    result = {"status": "COMPLETE", "experiment_id": cfg["experiment_id"],
        "config_sha256": sha256_file(CONFIG), "preparation_signature": preparation["signature"],
        "checkpoints": hashes, "initial_rgb_sha256": starts, "raw_random_rgb_sha256": raw_starts,
        "movement_from_initial": movement, "constraint_metrics": structures,
        "fresh_start_outer": 0, "optimizer_updates": 100000, "scheduler_total_outer": 2000,
        "evaluation_ready": True, "evaluation_hooks": False}
    if write_report:
        require(tuple(methods) == METHODS, "Final validation requires all four arms")
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
            raise RuntimeError("Another sampling-clear helper is submitting this chain")
        names = "oft_clear_sampling_prepare,oft_clear_sampling_train,oft_clear_sampling_validate,oft_clear_sampling_eval,oft_clear_sampling_stats"
        queue = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", names],
                               check=False, capture_output=True, text=True)
        require(queue.returncode == 0, "squeue query failed; duplicate-chain state unknown: " + queue.stderr.strip())
        require(not queue.stdout.strip(), "A sampling-clear chain remains queued; inspect its receipt/logs before resubmitting")
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
        stages = [("prepare", "112_clear_sampling_prepare_cpu.sbatch", None)]
        if pending:
            array = "0-3%4" if len(pending) == 4 else ",".join(map(str, pending)) + "%4"
            stages.append(("train", "113_clear_sampling_train_4090.sbatch", array))
        stages += [("validate", "114_clear_sampling_validate_cpu.sbatch", None),
                   ("eval", "115_clear_sampling_eval_3090.sbatch", "0-9%8"),
                   ("stats", "116_clear_sampling_stats_cpu.sbatch", None)]
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
