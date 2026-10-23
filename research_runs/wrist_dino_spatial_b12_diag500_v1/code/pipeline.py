"""Explicit submission, immutable receipt, no duplicate jobs or cancellation."""
from __future__ import annotations
import argparse,fcntl,getpass,re,subprocess
from common import ROOT,atomic_json,load_json,preflight,require,run_signature

STAGES=(("prepare","01_prepare_cpu.sbatch"),("train","02_train_4090.sbatch"),
        ("validate","03_validate_cpu.sbatch"),("eval","04_eval_4090.sbatch"),("stats","05_stats_cpu.sbatch"))

def submit():
    report=preflight()
    if (ROOT/"results/STATUS.json").exists():
        old=load_json(ROOT/"results/STATUS.json")
        require(old["signature"]==run_signature(),"Changed completed run")
        if old["status"]=="COMPLETE":print("ALREADY_COMPLETE; no submission");return
    with (ROOT/"reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=subprocess.run(["squeue","-h","-u",getpass.getuser(),"-o","%i|%T|%b|%R"],capture_output=True,text=True,timeout=40)
        require(q.returncode==0,"Queue unavailable; no submission: "+q.stderr.strip())
        path=ROOT/"reports/SUBMISSION_RECEIPT.json"
        active={line.split("|",1)[0].split("_",1)[0] for line in q.stdout.splitlines()}
        if path.exists():
            previous=load_json(path)
            require(previous["signature"]==report["signature"],"Previously submitted sources changed")
            require(not (active & {j["job_id"] for j in previous["jobs"]}),"This run already has active jobs")
        # Do not quietly overlap the approved caps with another own GPU batch.
        own_gpu=[line for line in q.stdout.splitlines() if "gpu" in line.split("|")[2].lower()]
        require(not own_gpu,"Another own GPU batch is active; no duplicate or overlapping submission")
        receipt={"status":"SUBMITTING","signature":report["signature"],"jobs":[],"training_gpu_cap":6,
                 "evaluation_gpu_cap":8,"evaluation_episodes":1360,"deadline_local":config_deadline()}
        atomic_json(receipt,path)
        dependency=None
        for stage,file in STAGES:
            cmd=["sbatch","--parsable","--kill-on-invalid-dep=yes"]
            if dependency:cmd.append("--dependency=afterok:"+dependency)
            cmd.append(str(ROOT/"slurm"/file))
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
            job=r.stdout.strip().split(";")[0]
            if r.returncode or not re.fullmatch(r"\d+",job):
                receipt.update(status="SUBMISSION_FAILED",failed_stage=stage,error=r.stderr or r.stdout)
                atomic_json(receipt,path);raise RuntimeError(receipt["error"])
            receipt["jobs"].append({"stage":stage,"job_id":job,"command":cmd});dependency=job
            atomic_json(receipt,path);print(stage,job,flush=True)
        receipt["status"]="SUBMITTED";atomic_json(receipt,path)

def config_deadline():
    from common import config
    return config()["deadline_local"]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--submit",action="store_true");a=p.parse_args()
    if a.submit:submit()
    else:
        x=preflight();atomic_json(x,ROOT/"reports/CPU_PREFLIGHT.json")
        print("CPU_PREFLIGHT_PASS",x["signature"],"16 arms;1360 episodes;no models loaded or jobs submitted")
