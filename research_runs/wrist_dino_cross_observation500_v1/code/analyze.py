"""Validate the executed algorithm; report direct paired comparisons with UADA."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from PIL import Image
from common import (ROOT, METHOD, CONDITIONS, SUITES, config, lineage, prepared, run_signature,
                    load_json, atomic_json, atomic_parquet, require, sha256_file, validate_grid)
from monitor import collect_steps

def validate_training():
    from train import validate_resume
    cfg, signature = config(), run_signature(); prepared()
    folder = ROOT / "training" / METHOD; status = load_json(folder / "status.json")
    require(status["status"] == "COMPLETE" and status["signature"] == signature
            and status["outer"] == 500 and status["optimizer_updates"] == 25000, "Training incomplete")
    saved = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    validate_resume(saved, cfg, signature)
    initial = torch.rand((3, 50, 50), generator=torch.Generator().manual_seed(42))
    require(torch.equal(saved["initial"], initial) and not torch.equal(saved["patch"], initial), "Fresh initialization/learning invalid")
    frame = pd.read_parquet(folder / "training.parquet")
    require(frame.outer_iteration.tolist() == list(range(1, 501)) and frame.optimizer_updates.tolist() == list(range(50, 25001, 50)), "Training clock incomplete")
    expected_lr = [0.002*(step/20 if step < 20 else .5*(1+math.cos(math.pi*(step-20)/(2000-20)))) for step in range(500)]
    require(np.allclose(frame.learning_rate, expected_lr, atol=1e-12, rtol=1e-10), "Cosine horizon/warmup differs from parent")
    for outer in (0, 250, 500):
        path = folder / f"outer_{outer:04d}.pt"; asset = torch.load(path, map_location="cpu", weights_only=False)
        p = asset["patch"]
        require(asset["signature"] == signature and asset["config"] == cfg and asset["method"] == METHOD
                and asset["outer_iterations"] == outer and asset["optimizer_updates"] == outer*50
                and torch.equal(asset["initial_patch"], initial) and p.shape == (3, 50, 50)
                and torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Snapshot provenance/RGB invalid")
        with Image.open(path.with_suffix(".png")) as im:
            require(im.mode == "RGB" and im.size == (50, 50)
                    and np.array_equal(np.asarray(im), p.permute(1, 2, 0).mul(255).round().byte().numpy()), "PNG differs from trained RGB")
        if outer == 500: require(torch.equal(p, saved["patch"]), "Final snapshot and optimizer state differ")
    steps = collect_steps(folder)
    require(len(steps) == 25000 and list(zip(steps.outer, steps.inner)) == [(o, i) for o in range(1, 501) for i in range(50)]
            and steps.signature.eq(signature).all(), "Actual monitoring incomplete")
    require(np.isfinite(steps.select_dtypes(include=["number"]).to_numpy()[~steps.select_dtypes(include=["number"]).isna().to_numpy()]).all(), "Nonfinite monitored update")
    pair_index = load_json(ROOT / "data/PAIR_INDEX.json")["pairs"]
    for outer, group in steps.groupby("outer"):
        pair = pair_index[int(outer)-1]
        require(group.uid.eq(pair["uid"]).all(), "Training used another source sample")
        actual_queries = group.query_uid.dropna().unique().tolist()
        require(actual_queries == ([] if pair["query_uid"] is None else [pair["query_uid"]]), "Training successor changed")
        if pair["query_uid"] is None: require(not group.used_query.any(), "Boundary window incorrectly paired")
    matched = steps.loc[steps.virtual_status.eq("MATCHED")]
    require(np.allclose(matched.virtual_rgb_l2, matched.virtual_target_l2, atol=5e-8, rtol=2e-5), "Virtual step norm matching failed")
    require(steps.used_query.any(), "No cross-observation updates actually executed")
    atomic_parquet(steps, folder / "MONITORING_STEPS.parquet")
    probes = {}
    for path in sorted((folder / "attempts").glob("*/probe_*.json")):
        v = load_json(path); require(v["signature"] == signature and path.with_suffix(".pt").exists(), "Missing/stale probe tensors")
        probes[(v["outer"], v["inner"])] = v
    require(set(probes) == {(o, i) for o in cfg["diagnostic_outers"] for i in cfg["diagnostic_inners"]}, "Fixed probes incomplete")
    interface = [load_json(p) for p in folder.glob("GPU_INTERFACE_*.json")]
    require(interface and all(v["status"] == "GPU_INTERFACE_PASS" for v in interface), "Native GPU QA incomplete")
    parent_png = Path(lineage()["parent"]["root"]) / "training/uada_original/outer_0500.png"
    patches = {METHOD: {"path": str(folder / "outer_0500.png"), "png": sha256_file(folder / "outer_0500.png")},
               "uada500": {"path": str(parent_png), "png": sha256_file(parent_png)}}
    report = {"status": "COMPLETE", "signature": signature, "patches": patches,
              "real_updates": 25000, "virtual_optimizer_steps": 0,
              "query_updates": int(steps.used_query.sum()), "native_fallback_updates": int((~steps.used_query).sum()),
              "virtual_status_counts": {str(k): int(v) for k, v in steps.virtual_status.value_counts().items()},
              "virtual_quantized_zero_fraction": float(steps.loc[steps.used_query, "virtual_quantized_changed_fraction"].eq(0).mean()),
              "current_query_conflict_fraction": float(steps.loc[steps.used_query, "current_query_cosine"].lt(0).mean()),
              "native_current_loss_increase_probe_fraction": float(np.mean([v["evidence"]["current"]["native_loss_change"] > 0 for v in probes.values()])),
              "probe_count": len(probes), "model_parameters_updated": False, "scientific_result": "PENDING_CLOSED_LOOP"}
    atomic_json(report, ROOT / "reports/AUTO_VALIDATION.json")
    print("CROSS_OBS_VALIDATED:25000 real updates; native PNG; successor and gradient logs checked", flush=True)

def paired_tasks(raw, suite):
    sub = raw.loc[raw.suite.eq(suite)]
    wide = sub.pivot(index=["task_id", "state_index"], columns="condition", values="success")
    eligible = wide.loc[wide.clean.eq(1)]
    return eligible, {int(t): g for t, g in eligible.groupby(level="task_id")}

def summarize(raw, draws=2000, seed=20261017):
    require(draws > 0, "Positive descriptive bootstrap count required")
    rng = np.random.default_rng(seed); rows = []; contrasts = []; suite_draws = []
    methods = ("uada500", "uada1000", METHOD)
    for suite in SUITES:
        eligible, groups = paired_tasks(raw, suite)
        require(groups, f"No clean-eligible tasks in {suite}")
        task_ids = sorted(groups)
        estimates = {m: float(np.mean([1-g[m].mean() for g in groups.values()])) for m in methods}
        sampled = np.empty((draws, len(methods)))
        for b in range(draws):
            values = []
            for t in rng.choice(task_ids, len(task_ids), replace=True):
                g = groups[int(t)]; idx = rng.integers(0, len(g), len(g))
                values.append([float(1-g.iloc[idx][m].mean()) for m in methods])
            sampled[b] = np.mean(values, axis=0)
        suite_draws.append(sampled)
        for m in methods:
            rows.append({"suite": suite, "condition": m, "casr": estimates[m], "eligible_episodes": len(eligible),
                         "eligible_tasks": len(groups), "excluded_tasks": 10-len(groups),
                         "unconditional_success_rate": float(raw.loc[raw.suite.eq(suite) & raw.condition.eq(m), "success"].mean())})
        for j, baseline in enumerate(methods[:2]):
            delta = sampled[:, 2]-sampled[:, j]
            contrasts.append({"suite": suite, "comparison": f"{METHOD}-{baseline}", "delta_casr": estimates[METHOD]-estimates[baseline],
                "descriptive_ci_low": float(np.quantile(delta, .025)), "descriptive_ci_high": float(np.quantile(delta, .975)),
                "new_failures": int((eligible[METHOD].eq(0) & eligible[baseline].eq(1)).sum()),
                "lost_failures": int((eligible[METHOD].eq(1) & eligible[baseline].eq(0)).sum())})
    means = np.mean(suite_draws, axis=0)
    for j, baseline in enumerate(methods[:2]):
        delta = means[:, 2]-means[:, j]
        contrasts.append({"suite": "equal_suite_mean", "comparison": f"{METHOD}-{baseline}",
            "delta_casr": float(np.mean([r["delta_casr"] for r in contrasts if r["suite"] in SUITES and r["comparison"].endswith("-"+baseline)])),
            "descriptive_ci_low": float(np.quantile(delta, .025)), "descriptive_ci_high": float(np.quantile(delta, .975)),
            "new_failures": None, "lost_failures": None})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)

def stats():
    cfg, signature = config(), run_signature(); prepared()
    validation = load_json(ROOT / "reports/AUTO_VALIDATION.json")
    require(validation["signature"] == signature and validation["status"] == "COMPLETE", "Training validation not complete")
    frames = [pd.read_parquet(ROOT / "data/HISTORICAL_ROLLOUTS.parquet")]
    from evaluate import task_complete
    for condition in CONDITIONS:
        for suite in SUITES:
            folder = ROOT / "evaluation" / condition / suite
            checksum = validation["patches"][condition]["png"]
            for task in range(10):
                require(task_complete(folder, task, signature, checksum), "Evaluation incomplete")
                frame = pd.read_parquet(folder / f"task_{task:02d}_rollouts.parquet")
                require(frame.signature.eq(signature).all() and frame.deployment_png_sha256.eq(checksum).all()
                        and not frame.evaluation_training_hooks.any() and not frame.reused.any()
                        and frame.condition.eq(condition).all() and frame.suite.eq(suite).all(), "New inference provenance changed")
                frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    validate_grid(raw, ("clean", "uada500", "uada1000", METHOD))
    require(raw.reused.sum() == 160 and (~raw.reused).sum() == 160, "Reused/new outcomes mixed")
    summary, contrasts = summarize(raw, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    atomic_parquet(raw, ROOT / "results/ROLLOUTS.parquet")
    (ROOT / "results").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "results/SUMMARY.csv", index=False); contrasts.to_csv(ROOT / "results/CONTRASTS.csv", index=False)
    primary = contrasts.loc[contrasts.suite.eq("equal_suite_mean") & contrasts.comparison.eq(f"{METHOD}-uada1000")].iloc[0]
    decision = {"status": "COMPLETE", "signature": signature, "primary_delta_casr": float(primary.delta_casr),
        "outcome": "PILOT_EXCEEDS_UADA1000" if primary.delta_casr > 0 else "NO_IMPROVEMENT_OVER_UADA1000",
        "new_episodes": 160, "reused_episodes": 160, "training_seeds": 1, "automatic_follow_up": False,
        "interpretation": "fixed500 candidate against strong1000 baseline; single-seed pilot, no DINO-specific ablation claim"}
    atomic_json(decision, ROOT / "results/DECISION.json")
    costs = [load_json(p) for p in (ROOT / "reports/WORKER_COSTS").glob("*.json")]
    atomic_json({"attempts": costs, "gpu_worker_hours_including_failures": sum(c["elapsed_s"] for c in costs if c["stage"] in ("train", "eval"))/3600}, ROOT / "results/COSTS.json")
    lines = ["# Cross-observation pilot results", "", "|Suite|UADA500|UADA1000|DINO cross observation|", "|---|---:|---:|---:|"]
    for suite in SUITES:
        v = summary.loc[summary.suite.eq(suite)].set_index("condition").casr
        lines.append(f"|{suite}|{v['uada500']:.2%}|{v['uada1000']:.2%}|{v[METHOD]:.2%}|")
    lines += ["", f"Primary equal-suite delta versus UADA1000: {primary.delta_casr:+.2%}.", decision["outcome"], "",
              "CASR is conditional on paired clean success, task equal; empty eligible tasks are excluded and counted in SUMMARY.csv.",
              "See AUTO_VALIDATION for actual gradient/quantization/fallback evidence and COSTS for measured resource peaks.",
              "The candidate uses expert successor observations, not differentiable attacked simulator futures. No automatic follow-up."]
    (ROOT / "results/RESULTS.md").write_text("\n".join(lines)+"\n")
    artifacts = {n: sha256_file(ROOT / "results" / n) for n in ("ROLLOUTS.parquet", "SUMMARY.csv", "CONTRASTS.csv", "DECISION.json", "COSTS.json", "RESULTS.md")}
    atomic_json({"status": "COMPLETE", "signature": signature, "artifacts": artifacts}, ROOT / "results/STATUS.json")
    print(decision["outcome"], f"delta={primary.delta_casr:+.2%}", flush=True)
