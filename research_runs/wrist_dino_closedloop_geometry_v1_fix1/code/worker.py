"""GPU QA / four-condition evaluation / CPU summary. Never submits resources."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import traceback
import numpy as np
from loop_core import (BASE, CONDITIONS, ROOT, SUITES, cfg, finish, locked, native, prepared,
                       read_json, require, require_gpu, save_frame, signature, stage_done, write_json)
from qa_reuse import qa_done


def seed_for(suite, task):
    import pandas as pd
    rows = pd.read_parquet(ROOT / "data/HISTORICAL_REFERENCE.parquet")
    rows = rows.loc[rows.condition.eq("clean") & rows.suite.eq(suite) & rows.task_id.eq(task) & rows.state_index.eq(40)]
    require(len(rows) == 1, "Missing exact historical initial-state seed")
    return int(rows.iloc[0].rollout_seed)


def load(suite):
    from model_runtime import load_wrapper, prepare_graphics_backend
    runtime = native().runtime_config(suite)
    prepare_graphics_backend()
    wrapper = load_wrapper(runtime, suite)
    # The historical model loader prepends several legacy roots. Keep this
    # experiment's modules ahead of generic names such as geometry/summarize.
    own_code = str(ROOT / "code")
    sys.path[:] = [own_code] + [p for p in sys.path if p != own_code]
    wrapper.freeze_model()
    require(all(not p.requires_grad for module in (wrapper.vla, wrapper.action_head, wrapper.proprio_projector)
                if module is not None for p in module.parameters()), "The victim must remain frozen")
    return runtime, wrapper


def qa(suite):
    prepared(); require_gpu()
    folder = ROOT / "qa" / suite
    with locked(folder / "qa.lock"):
        if qa_done(suite):
            print("QA ALREADY_COMPLETE_OR_REUSED", suite, flush=True); return
        runtime, wrapper = load(suite)
        from model_runtime import make_env, reset_wait, close_environment, renderer_metadata
        from branch_state import BranchState, state_difference
        from closedloop import check_commit, image_hash, plan, window
        from native_predictor import NativePredictor
        from task_progress import TaskProgress
        env = None
        started = time.monotonic()
        try:
            env, task, initial = make_env(runtime, suite, 0)
            renderer, vendor = renderer_metadata(env)
            raw = reset_wait(env, initial[40], seed_for(suite, 0), 10)
            progress = TaskProgress(env, cfg()["progress_distance_scale_m"])
            snapshot = BranchState(env, raw)
            with NativePredictor(wrapper, task.language) as predictor:
                first_raw, first = window(env, raw, predictor, progress, runtime, cfg()["candidates"][0], 16)
                restored, error = snapshot.restore(env)
                second_raw, second = window(env, restored, predictor, progress, runtime, cfg()["candidates"][0], 16)
                parity = check_commit(first, second)
                require(image_hash(first_raw) == image_hash(second_raw), "Rollback replay changed actual rendered observations")
                raw, _ = snapshot.restore(env)
                _, committed, receipt, candidates = plan(env, raw, predictor, progress, runtime, 16,
                                                         "loop_geometry", 0, folder / "REAL_INPUTS.pt")
                elapsed = time.monotonic() - started
                result = {"suite": suite, "renderer": renderer, "renderer_vendor": vendor,
                          "restore_error": error, "identity_replay": parity,
                          "all_candidates_exercised": len(candidates), "candidate_receipts": candidates,
                          "plan_receipt": receipt, "policy_calls": predictor.calls,
                          "native_policy_seconds": predictor.seconds, "elapsed_s": elapsed,
                          "task_goal_schema": progress.goals, "no_effect_threshold": True,
                          "gradients": "none: finite-candidate simulator search; model weights frozen"}
                write_json(result, folder / "ENGINEERING_QA.json")
        finally:
            if env is not None:
                close_environment(env)
        finish(folder / "STATUS.json", [folder / "ENGINEERING_QA.json", folder / "REAL_INPUTS.pt"], suite=suite)
        print("GPU_ENGINEERING_QA_PASS", suite, flush=True)


def evaluate(suite):
    import pandas as pd
    from PIL import Image
    prepared(); require_gpu()
    require(all(qa_done(s) for s in SUITES), "GPU engineering QA is incomplete")
    folder = ROOT / "evaluation" / suite
    with locked(folder / "suite.lock"):
        if all(stage_done(folder / f"task_{t:02d}" / c / "STATUS.json") for t in cfg()["tasks"] for c in CONDITIONS):
            print("EVALUATION ALREADY_COMPLETE", suite, flush=True); return
        runtime, wrapper = load(suite)
        from model_runtime import make_env, reset_wait, close_environment, renderer_metadata
        from closedloop import adaptive_episode, simple_episode
        from native_predictor import NativePredictor
        limit = native().config()["evaluation"]["max_steps_by_suite"][suite]
        with Image.open(BASE / "training/m01_uada_original_b1/outer_0500.png") as image:
            png = np.asarray(image).copy()
        for task_id in cfg()["tasks"]:
            task_folder = folder / f"task_{task_id:02d}"
            if all(stage_done(task_folder / c / "STATUS.json") for c in CONDITIONS):
                continue
            env = None
            try:
                env, task, initial = make_env(runtime, suite, task_id)
                renderer, vendor = renderer_metadata(env)
                seed = seed_for(suite, task_id)
                order = list(CONDITIONS[1:])
                random.Random(cfg()["seed"] + task_id * 17 + SUITES.index(suite)).shuffle(order)
                for condition in ("clean", *order):
                    status = task_folder / condition / "STATUS.json"
                    if stage_done(status):
                        continue
                    started = time.monotonic()
                    attempt = status.parent / f"attempt_{time.time_ns()}"
                    attempt.mkdir(parents=True)
                    write_json({"status": "RUNNING", "signature": signature(), "suite": suite,
                                "task_id": task_id, "condition": condition, "attempt": str(attempt.relative_to(ROOT)),
                                "job_id": os.environ.get("SLURM_JOB_ID")}, status)
                    try:
                        import torch
                        random.seed(seed); np.random.seed(seed % (2 ** 32 - 1)); torch.manual_seed(seed)
                        raw = reset_wait(env, initial[40], seed, 10)
                        with NativePredictor(wrapper, task.language) as predictor:
                            if condition in ("clean", "uada_static"):
                                result, traces, plans, candidates = simple_episode(env, raw, predictor, runtime, condition, png, limit)
                            else:
                                result, traces, plans, candidates = adaptive_episode(env, raw, predictor, runtime, condition, limit, attempt)
                            meta = {"suite": suite, "task_id": task_id, "state_index": 40, "rollout_seed": seed,
                                    "condition": condition, "signature": signature(), "renderer": renderer,
                                    "renderer_vendor": vendor, "baseline_png_sha256": cfg()["baseline_png_sha256"],
                                    "privileged_simulator_planning": condition.endswith("geometry"),
                                    "policy_calls": predictor.calls, "native_policy_seconds": predictor.seconds,
                                    "elapsed_s": time.monotonic() - started}
                        files = []
                        for name, rows in (("ROLLOUT", [{**meta, **result}]), ("TRACES", traces),
                                           ("PLANS", plans), ("CANDIDATES", candidates)):
                            path = attempt / (name + ".parquet")
                            save_frame(pd.DataFrame(rows), path); files.append(path)
                        for name in ("FIRST_PLAN_INPUTS.pt", "LIVE_PLANS.jsonl"):
                            if (attempt / name).exists():
                                files.append(attempt / name)
                        finish(status, files, **meta, result=result, attempt=str(attempt.relative_to(ROOT)))
                        print(f"EPISODE_COMPLETE {suite} task={task_id} {condition} success={result['success']} "
                              f"steps={result['executed_steps']} calls={meta['policy_calls']}", flush=True)
                    except BaseException as error:
                        write_json({"status": "FAILED", "signature": signature(), "suite": suite, "task_id": task_id,
                                    "condition": condition, "attempt": str(attempt.relative_to(ROOT)),
                                    "elapsed_s": time.monotonic() - started, "error": repr(error),
                                    "traceback": traceback.format_exc()}, status)
                        raise
            finally:
                if env is not None:
                    close_environment(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("qa", "eval", "stats"))
    args = parser.parse_args()
    suite = SUITES[int(os.environ["SLURM_ARRAY_TASK_ID"])] if args.stage != "stats" else None
    started, complete = time.monotonic(), False
    try:
        if args.stage == "qa":
            qa(suite)
        elif args.stage == "eval":
            evaluate(suite)
        else:
            from summarize import summarize
            summarize()
        complete = True
    except BaseException as error:
        if args.stage == "qa" and suite is not None:
            write_json({"status": "FAILED", "signature": signature(), "suite": suite,
                        "error": repr(error), "traceback": traceback.format_exc(),
                        "job_id": os.environ.get("SLURM_JOB_ID")}, ROOT / "qa" / suite / "STATUS.json")
        write_json({"status": "FAILED", "signature": signature(), "stage": args.stage, "suite": suite,
                    "error": repr(error), "traceback": traceback.format_exc(), "job_id": os.environ.get("SLURM_JOB_ID")},
                   ROOT / "reports/FAILURES" / f"{args.stage}_{suite}_{time.time_ns()}.json")
        raise
    finally:
        write_json({"stage": args.stage, "suite": suite, "signature": signature(), "complete": complete,
                    "gpu_worker": args.stage != "stats", "elapsed_s": time.monotonic() - started,
                    "job_id": os.environ.get("SLURM_JOB_ID")},
                   ROOT / "reports/WORKER_COSTS" / f"{args.stage}_{suite}_{time.time_ns()}.json")


if __name__ == "__main__":
    main()
