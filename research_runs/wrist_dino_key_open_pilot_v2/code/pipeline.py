"""User-run entry submits the full dependency chain; workers never submit new jobs."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import re
import subprocess
import time
from common import (ROOT, METHODS, SUITES, config, preflight, prepared, run_signature, require,
                    load_json, atomic_json, sha256_file)

FILES = {"prepare": "01_prepare_cpu.sbatch", "train": "02_train_4090.sbatch", "validate": "03_validate_cpu.sbatch",
         "eval": "04_eval_4090.sbatch", "stats": "05_stats_cpu.sbatch"}

def gpu_dependencies(value):
    result = set()
    for row in value.splitlines():
        fields = row.split("|"); require(len(fields) == 4, "Unrecognized Slurm queue output")
        job, state, tres, reason = fields
        if "gpu" not in tres.lower() or state in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"): continue
        require("DependencyNeverSatisfied" not in reason, "An existing GPU chain has failed dependencies; inspect it before new submission")
        m = re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?", job.strip())
        require(m is not None, "Unknown Slurm array identifier")
        result.add(m.group(1))
    return sorted(result, key=int)

def command_for(stage, signature, dependency=None, existing=(), indices=None):
    cmd = ["sbatch", "--parsable", "--job-name=kop2_"+signature[:8]+"_"+stage]
    deps = []
    if dependency: deps.append("afterok:"+dependency)
    if stage in ("train", "eval") and existing: deps.append("afterany:"+":".join(existing))
    if deps: cmd.append("--dependency="+",".join(deps))
    if stage == "eval":
        require(indices, "No evaluation indices")
        cmd.append("--array="+",".join(map(str, indices))+"%2")
    return cmd + [str(ROOT / "slurm" / FILES[stage])]

def evaluation_indices(validation):
    from evaluate import task_complete
    unique = [m for m in METHODS if validation["patches"][m]["canonical"] == m]
    return [i for i, s in enumerate(SUITES) if any(not task_complete(ROOT / "evaluation" / m / s, t,
            validation["signature"], validation["patches"][m]["png"]) for m in unique for t in range(10))]

def submit(evaluation=False):
    preflight(bank=not evaluation); signature = run_signature()
    group = "EVAL" if evaluation else "FULL"
    final_result = ROOT / "results/STATUS.json"
    if final_result.exists():
        f = load_json(final_result)
        if f["status"] in ("COMPLETE", "STOPPED"):
            require(f["signature"] == signature and all(sha256_file(ROOT / "results" / p) == h for p, h in f["artifacts"].items()), "Completed result drift")
            print("ALREADY_COMPLETE_OR_STOPPED; no submission"); return
    if evaluation:
        prepared(); v = load_json(ROOT / "reports/AUTO_VALIDATION.json")
        require(v["signature"] == signature and v["status"] == "COMPLETE" and v["evaluation_allowed"],
                "Training has no validated changed PNG; do not submit evaluation")
        for d in v["patches"].values(): require(sha256_file(d["path"]) == d["png"], "Validated PNG changed")
        final = ROOT / "results/STATUS.json"
        if final.exists():
            f = load_json(final)
            if f["status"] == "COMPLETE":
                require(f["signature"] == signature and all(sha256_file(ROOT / "results" / p) == h for p, h in f["artifacts"].items()), "Completed result drift")
                print("ALREADY_COMPLETE; no submission"); return
        indices = evaluation_indices(v); stages = (["eval"] if indices else [])+["stats"]
    else:
        indices = [0, 1, 2, 3]; final = ROOT / "training/STATUS.json"
        finished = final.exists() and load_json(final).get("status") in ("COMPLETE", "NO_TARGET", "NO_PIXEL_CHANGE", "NO_DUAL_CLOSE")
        validation = ROOT / "reports/AUTO_VALIDATION.json"
        if finished:
            require(load_json(final)["signature"] == signature, "Finished training identity changed")
            if validation.exists():
                v = load_json(validation)
                require(v["signature"] == signature and v["status"] == "COMPLETE", "Validation identity drift")
                if not v["evaluation_allowed"]:
                    print("SCIENTIFIC_STOP; no valid changed PNG; no new GPU submission"); return
                return submit(evaluation=True)
        if (ROOT / "reports/PREPARATION.json").exists(): prepared()
        stages = ([] if (ROOT / "reports/PREPARATION.json").exists() else ["prepare"])+([] if finished else ["train"])+["validate", "eval", "stats"]
    path = ROOT / f"reports/{group}_SUBMISSION.json"; path.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = load_json(path) if path.exists() else None
        if old:
            require(old["signature"] == signature and old["status"] not in ("SUBMITTING", "AMBIGUOUS_SBATCH_RESPONSE"),
                    "Prior submission identity/response unresolved; no duplicate allowed")
        q = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%b|%R"], capture_output=True, text=True, timeout=30)
        require(q.returncode == 0, "Cannot read queue; no jobs submitted: "+q.stderr)
        active = {l.split("|", 1)[0].split("_", 1)[0] for l in q.stdout.splitlines()}
        for prior in (ROOT / "reports").glob("*_SUBMISSION.json"):
            r = load_json(prior)
            require(not active.intersection(j["job_id"] for j in r["jobs"]), "This pilot already has active jobs; do not resubmit")
        existing = gpu_dependencies(q.stdout)
        receipt = {"status": "SUBMITTING", "signature": signature, "group": group, "jobs": [],
                   "attempt": time.time_ns(), "existing_gpu_dependencies": existing, "resources": config()["resources"]}
        if old: atomic_json(old, ROOT / "reports/SUBMISSION_ARCHIVE" / f"{receipt['attempt']}_{group}.json")
        atomic_json(receipt, path); dependency = None
        for stage in stages:
            command = command_for(stage, signature, dependency, existing, indices)
            receipt["pending_command"] = command; atomic_json(receipt, path)
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                job = result.stdout.strip().split(";")[0]
                require(result.returncode == 0 and re.fullmatch(r"\d+", job), result.stderr or result.stdout or "Empty sbatch response")
            except BaseException as error:
                receipt.update(status="AMBIGUOUS_SBATCH_RESPONSE", failed_stage=stage, error=repr(error)); atomic_json(receipt, path)
                raise RuntimeError("Inspect submission receipt before retrying; no automatic cancellation or resubmission") from error
            receipt["jobs"].append({"stage": stage, "job_id": job, "command": command}); dependency = job
            receipt.pop("pending_command", None); atomic_json(receipt, path); print(stage, job, flush=True)
        receipt["status"] = "SUBMITTED"; atomic_json(receipt, path)
        print("Full chain submitted by the user-run entry; afterok validation gates PNG evaluation and statistics. Use --status; no second evaluation command required.")

def status():
    for rel in ["reports/FULL_SUBMISSION.json", "reports/EVAL_SUBMISSION.json", "reports/PREPARATION.json", "reports/DUAL_CLOSE.json", "reports/TARGET_SUMMARY.json", "training/STATUS.json", "reports/AUTO_VALIDATION.json", "reports/SCIENTIFIC_STOP.json", "results/STATUS.json"]:
        p = ROOT / rel
        if p.exists():
            d = load_json(p)
            print(rel, {k: v for k, v in d.items() if k not in ("sources", "native_rows", "prepared_artifacts", "artifacts", "resources")})
        else: print(rel, "NOT_STARTED")
    for m in METHODS:
        p = ROOT / "training/universal" / m / "status.json"
        if p.exists(): print(m, load_json(p))
    failures = sorted((ROOT / "reports/FAILURES").glob("*.json"))
    if failures: print("LATEST_FAILURE", load_json(failures[-1]))

def main():
    p = argparse.ArgumentParser(); p.add_argument("--action", choices=("preflight", "dry-run", "status", "submit", "evaluate"), required=True); a = p.parse_args()
    if a.action in ("submit", "evaluate"): submit(a.action == "evaluate")
    elif a.action == "preflight":
        d = preflight(); print("CPU_PREFLIGHT_PASS", d["signature"], "no model/GPU/scheduler/submission", config()["resources"])
    elif a.action == "status": status()
    else:
        print("USER_SUBMITS_FULL_CHAIN", [command_for(s, run_signature(), "PREVIOUS_JOB" if i else None,
              indices=[0,1,2,3] if s == "eval" else None) for i, s in enumerate(("prepare", "train", "validate", "eval", "stats"))])
        print("DRY_RUN_ONLY", config()["resources"], "No query or submission. Up to5600 RGB candidates, not500 formal outers.")

if __name__ == "__main__": main()
