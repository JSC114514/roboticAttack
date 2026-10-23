"""Manual Slurm submission, duplicate protection, resume, and status."""
from __future__ import annotations
import argparse, fcntl, getpass, re, subprocess, time
from common import ROOT, atomic_json, cfg, complete, preflight, read_json, require, signature

STAGES=("prepare","cache","train","validate","select","heldout","analyze")
RECEIPTS={"prepare":"PREPARE_COMPLETE","cache":"CACHE_COMPLETE","train":"TRAIN_COMPLETE","validate":"VALIDATION_COMPLETE",
          "select":"SELECT_COMPLETE","heldout":"HELDOUT_COMPLETE","analyze":"../results/COMPLETE"}

def done(stage):
    path=ROOT/("results/COMPLETE.json" if stage=="analyze" else f"reports/{RECEIPTS[stage]}.json")
    return complete(path)

def active_queue():
    response=subprocess.run(["squeue","-h","-u",getpass.getuser(),"-o","%i|%T|%b|%R|%j"],capture_output=True,text=True,timeout=30)
    require(response.returncode==0,"Slurm queue unavailable: "+response.stderr.strip())
    return response.stdout

def gpu_dependencies(text,prefix):
    jobs=[]
    for line in text.splitlines():
        job,state,tres,reason,name=line.split("|")
        if prefix in name or "gpu" not in tres.lower(): continue
        require("DependencyNeverSatisfied" not in reason,"An existing GPU job has a failed dependency; inspect it before serializing this run")
        match=re.fullmatch(r"(\d+)(?:_[\d,\[\]%\-]+)?",job.strip()); require(match,"Cannot parse active GPU job")
        jobs.append(match.group(1))
    return sorted(set(jobs),key=int)

def command(stage,previous=None,external=()):
    cmd=["sbatch","--parsable","--job-name="+f"rel1_{signature()[:8]}_{stage}",
         "--export=ALL,RELATION_EXPECTED_SIGNATURE="+signature()]
    dependencies=[]
    if previous: dependencies.append("afterok:"+previous)
    if external: dependencies.append("afterany:"+":".join(external))
    if dependencies: cmd.append("--dependency="+",".join(dependencies))
    return cmd+[str(ROOT/f"slurm/{STAGES.index(stage)+1:02d}_{stage}.sbatch")]

def submit():
    (ROOT/"logs").mkdir(exist_ok=True); (ROOT/"reports/locks").mkdir(parents=True,exist_ok=True)
    with (ROOT/"reports/locks/submission.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); queue=active_queue(); prefix=f"rel1_{signature()[:8]}_"
        require(not any(line.rsplit("|",1)[-1].strip().startswith("rel1_") for line in queue.splitlines()),
                "An active relation-bias chain (possibly an older source revision) already exists; inspect --status --slurm; nothing submitted")
        preflight(); remaining=[s for s in STAGES if not done(s)]
        if not remaining: print("ALREADY_COMPLETE; no submission"); return
        external=gpu_dependencies(queue,prefix); receipt={"status":"SUBMITTING","signature":signature(),"jobs":[],
                 "serialized_after_existing_gpu_jobs":external,"resources":cfg()["resources"]}
        path=ROOT/"reports/SUBMISSION.json"
        if path.exists():
            archived=ROOT/f"reports/SUBMISSIONS/{time.time_ns()}.json"
            atomic_json(read_json(path),archived)
        atomic_json(receipt,path); previous=None; first_gpu=True
        for stage in remaining:
            ext=external if stage in ("cache","train","validate","heldout") and first_gpu else ()
            if stage in ("cache","train","validate","heldout"): first_gpu=False
            cmd=command(stage,previous,ext); receipt["pending_command"]=cmd; atomic_json(receipt,path)
            response=subprocess.run(cmd,capture_output=True,text=True,timeout=30); job=response.stdout.strip().split(";")[0]
            if response.returncode!=0 or not re.fullmatch(r"\d+",job):
                receipt.update(status="AMBIGUOUS_SBATCH_RESPONSE",error=response.stderr.strip()); atomic_json(receipt,path)
                raise RuntimeError("Submission not retried or cancelled; inspect Slurm and SUBMISSION.json")
            receipt["jobs"].append({"stage":stage,"job_id":job,"command":cmd}); receipt.pop("pending_command",None)
            atomic_json(receipt,path); previous=job; print(stage,job,flush=True)
        receipt["status"]="SUBMITTED"; atomic_json(receipt,path)

def status(slurm=False):
    for stage in STAGES:
        try: value="COMPLETE" if done(stage) else "PENDING"
        except Exception as error: value="STALE: "+str(error)
        print(stage,value)
    for name in ("reports/PREFLIGHT.json","reports/SUBMISSION.json","reports/PROGRESS.json","results/WINNER.json","results/STATUS.json"):
        path=ROOT/name
        if not path.exists(): print(name,"NOT_STARTED"); continue
        value=read_json(path)
        if name=="reports/PREFLIGHT.json":
            value={k:value[k] for k in ("status","signature","frames","trajectories","source_clean_decisions","source_clean_trajectories")}
        elif name=="reports/SUBMISSION.json":
            value={"status":value.get("status"),"signature":value.get("signature"),"jobs":value.get("jobs",[]),
                   "matches_current_source":value.get("signature")==signature(),
                   "serialized_after_existing_gpu_jobs":value.get("serialized_after_existing_gpu_jobs",[])}
        print(name,value)
    print("completed_cache_shards",sum(not p.name.endswith(".cost.json")
                                      for p in (ROOT/"data/cache").glob("*/*/task_*.json")))
    if (ROOT/"reports/CACHE_REUSE.json").is_file():
        reuse=read_json(ROOT/"reports/CACHE_REUSE.json")
        print("audited_reused_cache_shards",reuse.get("reused_shards"),
              "observations",reuse.get("reused_observations"),
              "matches_current_source",reuse.get("target_signature")==signature())
    print("completed_training",len(list((ROOT/"training").glob("*/COMPLETE.json"))))
    if slurm and (ROOT/"reports/SUBMISSION.json").exists():
        jobs=",".join(j["job_id"] for j in read_json(ROOT/"reports/SUBMISSION.json").get("jobs",[]))
        if jobs:
            for cmd in (["squeue","-h","-j",jobs,"-o","%i|%T|%M|%R|%j"],
                        ["sacct","-j",jobs,"--format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode","-P"]):
                response=subprocess.run(cmd,capture_output=True,text=True,timeout=30); print(response.stdout or response.stderr)
            print("Slurm MaxRSS is sampled separately from process RSS HWM in reports/COSTS.")

def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    for mode in ("preflight","dry-run","submit","status"): group.add_argument("--"+mode,action="store_true")
    parser.add_argument("--slurm",action="store_true"); args=parser.parse_args()
    if args.preflight: print(preflight())
    elif args.dry_run:
        remaining=[stage for stage in STAGES if not done(stage)]
        previous=None
        first_gpu=True
        for stage in remaining:
            external=["EXISTING_GPU_JOB"] if first_gpu and stage in ("cache","train","validate","heldout") else []
            print(" ".join(command(stage,previous,external)))
            if stage in ("cache","train","validate","heldout"): first_gpu=False
            previous=stage.upper()+"_JOB"
        print(cfg()["resources"])
    elif args.submit: submit()
    else: status(args.slurm)
if __name__=="__main__":main()
