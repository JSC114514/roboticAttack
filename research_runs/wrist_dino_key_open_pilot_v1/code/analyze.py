"""CPU validation and paired four-suite results. Never count engineering failures as attack wins."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHODS, SUITES, config, require, load_json, atomic_json, atomic_parquet,
                    sha256_file, run_signature, prepared, validate_grid)

def collect_steps(folder, completed):
    rows = {}
    for path in sorted((folder / "attempts").glob("*/INNER_STEPS.jsonl")):
        with path.open() as stream:
            for line in stream:
                try: row = json.loads(line)
                except json.JSONDecodeError:
                    require(not line.endswith("\n") and not stream.read(), "Corrupt completed JSON log")
                    break
                if row["index"] < completed: rows[row["index"]] = row
    require(set(rows) == set(range(completed)), "Missing completed candidate records")
    return [rows[i] for i in range(completed)]

def validate_training():
    prepared(); signature = run_signature(); final = load_json(ROOT / "training/STATUS.json")
    require(final["signature"] == signature, "Training identity changed")
    if final["status"] in ("NO_TARGET", "NO_PIXEL_CHANGE", "NO_DUAL_CLOSE"):
        atomic_json({"status": "COMPLETE", "signature": signature, "evaluation_allowed": False,
                     "scientific_stop": final["status"], "patches": {}}, ROOT / "reports/AUTO_VALIDATION.json")
        print("PILOT_STOP", final["status"], "No evaluation should be submitted"); return
    require(final["status"] == "COMPLETE", "Training incomplete")
    qa = load_json(ROOT / "reports/GPU_INTERFACE.json")
    require(qa["post_k_norm_identity"] and qa["direct_native_key_parity"] and qa["png_fixed_parity"], "Native Key interface not validated")
    patches = {}; canonical = {}
    for method in METHODS:
        folder = ROOT / "training/universal" / method; s = load_json(folder / "status.json")
        saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        require(s["status"] == "COMPLETE" and s["signature"] == saved["signature"] == signature
                and s["completed_candidates"] == saved["completed"] == 2000, "Candidate budget/identity mismatch")
        require(sum(s["counts"].values()) == 2000, "Accept/reject/no-target counts do not close")
        rows = collect_steps(folder, 2000)
        for row in rows:
            if row["outcome"] == "ACCEPTED":
                last = row["trials"][-1]
                require(last["key_loss"] < row["key_before"] and last["open_loss"] <= row["open_before"]
                        and row["quantized_pixels_changed"] > 0, "Native acceptance condition violated")
            else: require(row["rgb_step_l2"] == 0 and row["quantized_pixels_changed"] == 0, "Rejected/no-target RGB changed")
        counts = {k: sum(r["outcome"] == k.upper() for r in rows) for k in ("accepted", "rejected", "no_target")}
        require(all(counts[k] == s["counts"].get(k, 0) for k in counts), "Counters disagree with actual steps")
        for state in saved["optimizer"]["state"].values():
            require(int(state["step"]) == counts["accepted"], "Rejected candidate advanced Adam state")
        with Image.open(folder / "final.png") as im:
            require(im.mode == "RGB" and im.size == (50, 50)
                    and np.array_equal(np.asarray(im), saved["patch"].permute(1, 2, 0).mul(255).round().byte().numpy()), "PNG differs from trained RGB")
        checksum = sha256_file(folder / "final.png")
        require(checksum == s["png_sha256"], "Final PNG drift")
        atomic_parquet(pd.DataFrame(rows), folder / "CANDIDATE_STEPS.parquet")
        canonical.setdefault(checksum, method)
        patches[method] = {"path": str(folder / "final.png"), "png": checksum, "canonical": canonical[checksum], "counts": counts}
    atomic_json({"status": "COMPLETE", "signature": signature, "evaluation_allowed": True, "patches": patches,
                 "model_parameters_updated": False, "real_rgb_gradient": "DINO Key loss only",
                 "native_acceptance_forward": True, "scientific_result": "PENDING_PNG_CLOSED_LOOP"}, ROOT / "reports/AUTO_VALIDATION.json")
    print("PILOT_ASSETS_VALIDATED; final two PNGs only; historical baselines remain read-only")

def paired_summary(raw, draws=10000):
    conditions = ["spatial_uada500", "long_uada1000", *METHODS]
    keys = ["suite", "task_id", "state_index"]
    clean = raw[raw.condition.eq("clean")][keys+["success"]].rename(columns={"success": "clean_success"})
    pairs = raw[raw.condition.isin(conditions)].merge(clean, on=keys, validate="many_to_one")
    rng = np.random.default_rng(20260907); summaries = []; contrasts = []; per_task = []
    boots = {}; points = {}
    comparisons = [(m, b) for m in METHODS for b in ["spatial_uada500", "long_uada1000"]]+[(METHODS[1], METHODS[0])]
    for suite in SUITES:
        sub = pairs[pairs.suite.eq(suite) & pairs.clean_success.eq(1)]
        wide = sub.pivot(index=["task_id", "state_index"], columns="condition", values="failure")[conditions]
        groups = [g.to_numpy(float) for _, g in wide.groupby(level=0)]
        require(groups, "Suite has no clean-success qualified tasks")
        point = np.stack([g.mean(0) for g in groups]).mean(0)
        selected = rng.integers(0, len(groups), size=(draws, len(groups)))
        samples = np.zeros((draws, len(groups), len(conditions)))
        for i, group in enumerate(groups):
            rows, cols = np.nonzero(selected == i)
            idx = rng.integers(0, len(group), size=(len(rows), len(group)))
            samples[rows, cols] = group[idx].mean(1)
        boot = samples.mean(1); boots[suite] = boot; points[suite] = point
        for j, c in enumerate(conditions):
            low, high = np.quantile(boot[:, j], [.025, .975])
            summaries.append({"suite": suite, "condition": c, "task_equal_casr": point[j], "ci_low": low, "ci_high": high,
                              "qualified_states": len(wide), "qualified_tasks": len(groups)})
        for task_id, task in wide.groupby(level=0):
            for c in conditions: per_task.append({"suite": suite, "task_id": int(task_id), "condition": c,
                                                  "casr": float(task[c].mean()), "qualified_states": len(task)})
        for a, b in comparisons:
            i, j = conditions.index(a), conditions.index(b); low, high = np.quantile(boot[:, i]-boot[:, j], [.025, .975])
            contrasts.append({"suite": suite, "left": a, "right": b, "delta_casr": point[i]-point[j],
                              "ci_low": low, "ci_high": high, "new_failures": int(((wide[a]==1)&(wide[b]==0)).sum()),
                              "lost_failures": int(((wide[a]==0)&(wide[b]==1)).sum())})
    mean_point = np.stack(list(points.values())).mean(0); mean_boot = np.stack(list(boots.values())).mean(0)
    for j, c in enumerate(conditions):
        low, high = np.quantile(mean_boot[:, j], [.025, .975])
        summaries.append({"suite": "equal_suite_mean", "condition": c, "task_equal_casr": mean_point[j], "ci_low": low, "ci_high": high})
    for a, b in comparisons:
        i, j = conditions.index(a), conditions.index(b); low, high = np.quantile(mean_boot[:, i]-mean_boot[:, j], [.025, .975])
        contrasts.append({"suite": "equal_suite_mean", "left": a, "right": b, "delta_casr": mean_point[i]-mean_point[j], "ci_low": low, "ci_high": high})
    return pd.DataFrame(summaries), pd.DataFrame(contrasts), pd.DataFrame(per_task)

def stats():
    prepared(); signature = run_signature(); v = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(v["evaluation_allowed"] and v["signature"] == signature, "No evaluated learned PNGs")
    from evaluate import task_complete
    frames = [pd.read_parquet(ROOT / "data/HISTORICAL_ROLLOUTS.parquet")]
    for method in METHODS:
        info = v["patches"][method]; canonical = info["canonical"]
        for suite in SUITES:
            folder = ROOT / "evaluation" / canonical / suite
            for task in range(10):
                require(task_complete(folder, task, signature, info["png"]), "Incomplete formal evaluation")
                f = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
                require(f.signature.eq(signature).all() and not f.evaluation_training_hooks.any(), "Unexpected deployment intervention")
                f["condition"] = method; f["within_run_png_alias"] = method != canonical
                frames.append(f)
    raw = pd.concat(frames, ignore_index=True)
    validate_grid(raw, ("clean", "spatial_uada500", "long_uada1000", *METHODS))
    summary, contrasts, tasks = paired_summary(raw)
    atomic_parquet(raw, ROOT / "results/ROLLOUTS.parquet")
    for name, frame in [("SUMMARY", summary), ("CONTRASTS", contrasts), ("PER_TASK", tasks)]: frame.to_csv(ROOT / f"results/{name}.csv", index=False)
    costs = [load_json(p) for p in sorted((ROOT / "reports/WORKER_COSTS").glob("*.json"))]
    atomic_json({"attempts": costs, "gpu_worker_hours_recorded_including_failures": sum(c["elapsed_s"] for c in costs if c["stage"] in ("train", "eval"))/3600,
                 "note": "Process HWM and CUDA peaks; not whole-job memory. SIGKILL may prevent Python receipts; inspect Slurm and time -v logs for censored attempts."}, ROOT / "results/COSTS.json")
    probes = []
    for method in METHODS:
        for row in load_json(ROOT / "training/universal" / method / "probe_final.json"): probes.append({"condition": method, **row})
    atomic_json(probes, ROOT / "results/PNG_ACTION_PROBES.json")
    table = summary.pivot(index="condition", columns="suite", values="task_equal_casr")
    text = ["# Spatial Key-open pilot: true PNG closed-loop results", "", "Single seed42; short training, development states40/41. No claim of full500 equivalence or seed stability.", "",
            "|Method|Object|Spatial|Goal|Long|Equal mean|", "|---|---:|---:|---:|---:|---:|"]
    for c in ["spatial_uada500", "long_uada1000", *METHODS]:
        text.append("|"+c+"|"+"|".join(f"{table.loc[c,s]*100:.2f}%" for s in (*SUITES,"equal_suite_mean"))+"|")
    text += ["", "Primary reference: Spatial UADA500. Long UADA1000 is a matched stronger historical reference.",
             "Internal Key interventions are target-feasibility diagnostics only. Formal actions use the native policy and static RGB PNG.",
             "See TARGET_SUMMARY, single fits and PNG_ACTION_PROBES to separate Key feasibility, pixel realization and universal/generalization failures.",
             "Contact/lift/drop measurements are simulation proxies; no causal drop claim. No automatic follow-up."]
    (ROOT / "results/RESULTS.md").write_text("\n".join(text)+"\n")
    artifacts = ["ROLLOUTS.parquet", "SUMMARY.csv", "CONTRASTS.csv", "PER_TASK.csv", "COSTS.json", "PNG_ACTION_PROBES.json", "RESULTS.md"]
    atomic_json({"status": "COMPLETE", "signature": signature, "episodes": len(raw), "new_condition_rows": 160,
                 "unique_new_episodes": 80*len({i["canonical"] for i in v["patches"].values()}),
                 "reused_historical_rows": 240, "automatic_follow_up": False,
                 "artifacts": {n: sha256_file(ROOT / "results" / n) for n in artifacts}}, ROOT / "results/STATUS.json")
