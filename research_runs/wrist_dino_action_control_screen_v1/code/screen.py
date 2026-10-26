"""Run all restartable internal-site/action-endpoint arms on one GPU."""
from __future__ import annotations
import gc
import time
from pathlib import Path
import pandas as pd
import torch

from common import (ROOT, atomic_json, atomic_parquet, config, load_json, load_records,
                    require, require_gpu, run_signature, runtime_config, sha256_file, verify_provenance)
from engine import ControlEngine
from resources import snapshot
from screen_math import (action_metrics, clean_scale, consensus, cosine, endpoint_target,
                         group_rms, matched_random, normalized_descent_direction, primary_loss,
                         project_relative_rms_, stable_seed)
from sites import site_registry


def arm_paths(site, endpoint):
    stem = f"{site}__{endpoint}"
    return ROOT / "raw" / f"{stem}.parquet", ROOT / "raw" / f"{stem}.json"


def arm_trace_path(site, endpoint):
    return ROOT / "raw" / f"{site}__{endpoint}__steps.parquet"


def arm_complete(site, endpoint, signature):
    table, report = arm_paths(site, endpoint)
    trace = arm_trace_path(site, endpoint)
    if not table.exists() or not trace.exists() or not report.exists():
        return False
    value = load_json(report)
    require(value["signature"] == signature, f"Completed arm identity changed: {site}/{endpoint}")
    require(value["status"] == "COMPLETE" and value["rows"] == 40
            and value["trace_rows"] == 40 * 13
            and sha256_file(table) == value["parquet_sha256"]
            and sha256_file(trace) == value["trace_parquet_sha256"], f"Completed arm changed: {site}/{endpoint}")
    return True


def _add_gradient(accumulator, split, gradient):
    gradient = gradient.detach().float().cpu()
    state = accumulator[split]
    if state["sum"] is None:
        state["sum"] = torch.zeros_like(gradient)
        state["abs"] = torch.zeros_like(gradient)
    state["sum"].add_(gradient)
    state["abs"].add_(gradient.abs())
    state["norm_sum"] += float(gradient.norm())
    state["count"] += 1


def optimize_record(engine, record, site, endpoint, cfg):
    settings = cfg["intervention"]
    with torch.no_grad():
        clean_action, clean_value = engine.forward(record, site)
    clean_action = clean_action.detach()
    clean_value = clean_value.detach().float()
    # Source actions were captured through this exact native interface. Threshold
    # parity is the scientific invariant; a small BF16 numeric tolerance is logged.
    stored = record["clean_action"].to(clean_action)
    require(torch.equal(stored[..., 6] < .5, clean_action[..., 6] < .5), "Stored clean gripper classification changed")
    target = endpoint_target(record, endpoint, clean_action.device)
    scale = clean_scale(clean_value, site.headwise)
    z = torch.zeros_like(clean_value, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=settings["learning_rate"])
    best_loss, best_z, best_metrics, best_step = float("inf"), None, None, 0
    best_actual_rms_max = 0.
    initial_gradient = None
    trace = []
    for step in range(settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        replacement = clean_value + scale * z
        action, actual = engine.forward(record, site, replacement)
        primary = primary_loss(action, record, endpoint, settings["open_margin"], target)
        normalized_actual = (actual.float() - clean_value) / scale
        actual_rms = group_rms(normalized_actual, site.headwise)
        primary_value = float(primary.detach())
        metrics = action_metrics(action, clean_action, record, endpoint, settings["open_margin"], target)
        trace.append({"step": step, "primary_loss": primary_value,
                      "relative_rms_max": float(actual_rms.max()), **metrics})
        current_norm = float(group_rms(z, site.headwise).max())
        best_norm = float("inf") if best_z is None else float(group_rms(best_z, site.headwise).max())
        better = (best_z is None or primary_value < best_loss - 1e-12
                  or (abs(primary_value - best_loss) <= 1e-12 and current_norm < best_norm))
        if better:
            best_loss = primary_value
            best_z = z.detach().clone()
            best_metrics = metrics
            best_step = step
            best_actual_rms_max = float(actual_rms.max())
        if step == settings["steps"]:
            break
        total = primary + settings["regularizer"] * z.square().mean()
        total.backward()
        require(z.grad is not None and torch.isfinite(z.grad).all(), f"Invalid gradient at {site.name}/{endpoint}")
        if step == 0:
            initial_gradient = z.grad.detach().clone()
        optimizer.step()
        project_relative_rms_(z, settings["relative_rms_limit"], site.headwise)
    require(initial_gradient is not None and best_z is not None, "Optimization did not execute")
    initial_loss = trace[0]["primary_loss"]
    # The final iteration is evaluation-only, so release its downstream action
    # graph before the matched-random forward. This keeps the 4090 peak bounded.
    del action, actual, primary, replacement, normalized_actual, actual_rms
    random_z = matched_random(best_z, stable_seed(cfg["seed"], site.name, endpoint, record["uid"]), site.headwise)
    with torch.no_grad():
        random_action, random_actual = engine.forward(record, site, clean_value + scale * random_z)
        random_loss = float(primary_loss(random_action, record, endpoint, settings["open_margin"], target))
        random_metrics = action_metrics(random_action, clean_action, record, endpoint, settings["open_margin"], target)
        random_actual_rms = float(group_rms((random_actual.float() - clean_value) / scale, site.headwise).max())
    row = {"uid": record["uid"], "episode_id": record["episode_id"], "task_id": int(record["task_id"]),
           "split": record["pilot_split"], "site": site.name, "site_kind": site.kind,
           "layer": -1 if site.layer is None else site.layer, "endpoint": endpoint,
           "close_slot_count": int(record["close_slots"].sum()),
           "stored_clean_action_max_abs_error": float((stored - clean_action).abs().max()),
           "initial_loss": initial_loss, "best_loss": best_loss, "loss_improvement": initial_loss - best_loss,
           "best_step": best_step, "intended_relative_rms_max": float(group_rms(best_z, site.headwise).max()),
           "actual_relative_rms_max": best_actual_rms_max,
           "random_loss": random_loss, "optimized_loss_advantage_over_random": random_loss - best_loss,
           "random_actual_relative_rms_max": random_actual_rms,
           "random_action_delta_l2": random_metrics["action_delta_l2"],
           "optimized_action_delta_advantage_over_random": best_metrics["action_delta_l2"] - random_metrics["action_delta_l2"],
           "initial_gradient_l2": float(initial_gradient.norm()), "gradient_zero": bool(initial_gradient.norm() == 0),
           **best_metrics}
    for key, value in random_metrics.items():
        if key.startswith("wrong_") or key.startswith("any_") or key.startswith("all_") or key in ("minimum_target_gripper", "mean_target_gripper"):
            row["random_" + key] = value
    return row, initial_gradient, trace


def run_arm(engine, records, site, endpoint, cfg, signature):
    started = time.monotonic()
    rows, traces, trace_rows = [], {}, []
    gradients = {split: {"sum": None, "abs": None, "norm_sum": 0., "count": 0}
                 for split in ("train", "validation")}
    for index, record in enumerate(records):
        row, gradient, trace = optimize_record(engine, record, site, endpoint, cfg)
        rows.append(row)
        traces[record["uid"]] = trace
        trace_rows.extend({"uid": record["uid"], "task_id": int(record["task_id"]),
                           "split": record["pilot_split"], "site": site.name,
                           "endpoint": endpoint, **step} for step in trace)
        _add_gradient(gradients, record["pilot_split"], gradient)
        if (index + 1) % 10 == 0:
            print(f"ARM {site.name}/{endpoint} {index+1}/40 elapsed_s={time.monotonic()-started:.1f}", flush=True)
    split_consensus = {split: consensus(value["sum"], value["abs"], value["norm_sum"])
                       | {"records": value["count"]} for split, value in gradients.items()}
    train_validation_cosine = cosine(gradients["train"]["sum"], gradients["validation"]["sum"])
    settings = cfg["intervention"]
    shared_unit = normalized_descent_direction(gradients["train"]["sum"], 1., site.headwise)
    shared_unit = shared_unit.to(engine.wrapper.device)
    train_amplitude_losses = {amplitude: [] for amplitude in settings["shared_rms_candidates"]}
    for record in records:
        if record["pilot_split"] != "train":
            continue
        with torch.no_grad():
            clean_action, clean_value = engine.forward(record, site)
            clean_value = clean_value.detach().float()
            scale = clean_scale(clean_value, site.headwise)
            target = endpoint_target(record, endpoint, clean_action.device)
            for amplitude in settings["shared_rms_candidates"]:
                action, _actual = engine.forward(record, site, clean_value + scale * shared_unit * amplitude)
                train_amplitude_losses[amplitude].append(
                    float(primary_loss(action, record, endpoint, settings["open_margin"], target)))
    train_amplitude_means = {amplitude: sum(losses) / len(losses)
                             for amplitude, losses in train_amplitude_losses.items()}
    selected_amplitude = min(settings["shared_rms_candidates"],
                             key=lambda amplitude: (train_amplitude_means[amplitude], amplitude))
    shared_z = shared_unit * selected_amplitude
    shared_random_z = matched_random(shared_z, stable_seed(cfg["seed"], site.name, endpoint, "shared"),
                                     site.headwise)
    shared_random_z = shared_random_z.to(engine.wrapper.device)
    row_by_uid = {row["uid"]: row for row in rows}
    for record in records:
        if record["pilot_split"] != "validation":
            continue
        with torch.no_grad():
            clean_action, clean_value = engine.forward(record, site)
            clean_value = clean_value.detach().float()
            scale = clean_scale(clean_value, site.headwise)
            target = endpoint_target(record, endpoint, clean_action.device)
            initial = float(primary_loss(clean_action, record, endpoint, settings["open_margin"], target))
            shared_action, shared_actual = engine.forward(record, site, clean_value + scale * shared_z)
            random_action, random_actual = engine.forward(record, site, clean_value + scale * shared_random_z)
            shared_loss = float(primary_loss(shared_action, record, endpoint, settings["open_margin"], target))
            random_loss = float(primary_loss(random_action, record, endpoint, settings["open_margin"], target))
            shared_metrics = action_metrics(shared_action, clean_action, record, endpoint,
                                            settings["open_margin"], target)
            random_metrics = action_metrics(random_action, clean_action, record, endpoint,
                                            settings["open_margin"], target)
            shared_rms = float(group_rms((shared_actual.float() - clean_value) / scale, site.headwise).max())
            random_rms = float(group_rms((random_actual.float() - clean_value) / scale, site.headwise).max())
        row = row_by_uid[record["uid"]]
        row.update(shared_loss=shared_loss, shared_loss_improvement=initial - shared_loss,
                   shared_random_loss=random_loss,
                   shared_loss_advantage_over_random=random_loss - shared_loss,
                   shared_actual_relative_rms_max=shared_rms,
                   shared_random_actual_relative_rms_max=random_rms,
                   **{"shared_" + key: value for key, value in shared_metrics.items()},
                   **{"shared_random_" + key: value for key, value in random_metrics.items()})
    table, report = arm_paths(site.name, endpoint)
    trace_path = arm_trace_path(site.name, endpoint)
    atomic_parquet(pd.DataFrame(rows), table)
    atomic_parquet(pd.DataFrame(trace_rows), trace_path)
    value = {"status": "COMPLETE", "signature": signature, "site": site.name, "endpoint": endpoint,
             "rows": len(rows), "parquet": str(table.relative_to(ROOT)), "parquet_sha256": sha256_file(table),
             "trace_rows": len(trace_rows), "trace_parquet": str(trace_path.relative_to(ROOT)),
             "trace_parquet_sha256": sha256_file(trace_path),
             "elapsed_s": time.monotonic() - started, "gradient_consensus": split_consensus,
             "train_validation_gradient_cosine": train_validation_cosine,
             "shared_direction_intended_rms_max": float(group_rms(shared_z, site.headwise).max()),
             "shared_direction_source": "negative summed train initial gradient",
             "shared_direction_selected_rms": selected_amplitude,
             "shared_direction_train_amplitude_mean_losses": {
                 str(amplitude): loss for amplitude, loss in train_amplitude_means.items()},
             "shared_direction_validation_records": 20,
             "trace_summary": {uid: {"steps": len(trace), "initial_loss": trace[0]["primary_loss"],
                                      "final_loss": trace[-1]["primary_loss"]} for uid, trace in traces.items()},
             **snapshot()}
    atomic_json(value, report)
    return value


def screen():
    cfg = config()
    verify_provenance()
    records = load_records()
    signature = run_signature()
    gpu = require_gpu()
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    from model_runtime import load_wrapper
    wrapper = load_wrapper(runtime_config(), "libero_spatial")
    engine = ControlEngine(wrapper, cfg)
    sites = site_registry(cfg["candidates"])
    interface = engine.interface_check(records[0], sites, cfg["intervention"]["open_margin"])
    atomic_json({"status": "GPU_INTERFACE_PASS", "signature": signature, "gpu": gpu,
                 "sites": interface, "model_parameters_updated": False, "simulator_episodes": 0,
                 "deployment_png_produced": False, **snapshot()}, ROOT / "reports/GPU_INTERFACE.json")
    completed = 0
    atomic_json({"status": "RUNNING", "signature": signature, "completed_arms": completed,
                 "planned_arms": len(sites) * len(cfg["endpoints"]), **snapshot()}, ROOT / "reports/SCREEN_STATUS.json")
    for site_name in cfg["candidates"]:
        site = sites[site_name]
        for endpoint in cfg["endpoints"]:
            if arm_complete(site_name, endpoint, signature):
                completed += 1
                continue
            report = run_arm(engine, records, site, endpoint, cfg, signature)
            completed += 1
            atomic_json({"status": "RUNNING", "signature": signature, "completed_arms": completed,
                         "planned_arms": len(sites) * len(cfg["endpoints"]),
                         "latest": f"{site_name}/{endpoint}", "latest_elapsed_s": report["elapsed_s"],
                         **snapshot()}, ROOT / "reports/SCREEN_STATUS.json")
        gc.collect()
    artifacts = {}
    for site_name in cfg["candidates"]:
        for endpoint in cfg["endpoints"]:
            table, report = arm_paths(site_name, endpoint)
            trace = arm_trace_path(site_name, endpoint)
            require(arm_complete(site_name, endpoint, signature), "Arm disappeared during completion")
            artifacts[str(table.relative_to(ROOT))] = sha256_file(table)
            artifacts[str(trace.relative_to(ROOT))] = sha256_file(trace)
            artifacts[str(report.relative_to(ROOT))] = sha256_file(report)
    atomic_json({"status": "COMPLETE", "signature": signature, "completed_arms": completed,
                 "planned_arms": len(sites) * len(cfg["endpoints"]), "artifacts": artifacts,
                 "simulator_episodes": 0, "deployment_png_produced": False, **snapshot()},
                ROOT / "reports/SCREEN_STATUS.json")
    print("ACTION_CONTROL_SCREEN_COMPLETE", completed, flush=True)
