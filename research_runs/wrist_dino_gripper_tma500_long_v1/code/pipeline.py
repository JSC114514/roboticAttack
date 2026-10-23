#!/usr/bin/env python3
"""User-invoked submission only; --preflight never queries/submits Slurm."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import re
import subprocess

from gripper_common import ROOT, METHODS, atomic_json, config, load_json, preflight, require, sha256_file, CONFIG


def parse_queue(output):
    wait = set()
    for line in output.splitlines():
        fields = line.strip().split("|")
        require(len(fields)==4,"Unexpected squeue response; no submission")
        job,name,resource,reason = fields
        require(not name.startswith("oft_grip_"),"This experiment already has queued jobs; do not duplicate")
        if "gpu" in resource or name.startswith("oft_"):
            require("DependencyNeverSatisfied" not in reason,"Existing invalid GPU/OFT dependency: resolve before submission")
            match = re.match(r"^(\d+)",job)
            require(match is not None,"Unable to resolve existing job identity")
            wait.add(match.group(1))
    return sorted(wait,key=int)


def is_complete(path):
    return path.is_file() and load_json(path).get("status")=="COMPLETE"


def stages_to_submit():
    resources = config()["resources"]
    labels = [i for i in range(10) if not is_complete(ROOT/"data/labels"/f"task_{i:02d}.json")]
    train = [i for i,m in enumerate(METHODS) if not is_complete(ROOT/"training"/m/"status.json")]
    evaluation = [i for i in range(10) if not all(is_complete(ROOT/"evaluation"/s/f"task_{i:02d}_STATUS.json")
                                                 for s in config()["evaluation"]["suites"])]
    stages = [("prepare","147_prepare_cpu.sbatch",None)]
    if labels:
        stages.append(("labels","148_labels_a800.sbatch",",".join(map(str,labels))+f"%{resources['labels']['max_concurrent']}"))
    stages.append(("pool","149_pool_cpu.sbatch",None))
    if train:
        stages.append(("train","150_train_a800.sbatch",",".join(map(str,train))+f"%{resources['training']['max_concurrent']}"))
    stages.append(("validate","151_validate_cpu.sbatch",None))
    if evaluation:
        stages.append(("eval","152_eval_4090.sbatch",",".join(map(str,evaluation))+f"%{resources['evaluation']['max_concurrent']}"))
    stages.append(("stats","153_stats_cpu.sbatch",None))
    return stages


def submit():
    """The assistant must NOT invoke this. It is the command given to the user."""
    preflight()
    if is_complete(ROOT/"evaluation/STATUS.json"):
        print("ALREADY_COMPLETE; no new jobs")
        return
    (ROOT/"logs").mkdir(exist_ok=True)
    (ROOT/"reports").mkdir(exist_ok=True)
    with (ROOT/"reports/SUBMISSION.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        response = subprocess.run(["squeue","-h","-u",getpass.getuser(),"-o","%i|%j|%b|%R"],
                                  capture_output=True,text=True,check=False,timeout=30)
        require(response.returncode==0,"squeue failed; no jobs submitted because occupancy is unknown: "+response.stderr.strip())
        waits = parse_queue(response.stdout)
        path = ROOT/"reports/SUBMISSION_RECEIPT.json"
        resources = config()["resources"]
        receipt = {"status":"SUBMITTING","config_sha256":sha256_file(CONFIG),"jobs":[],"external_wait_jobs":waits,
                   "experiment_id":config()["experiment_id"],"training_suite":config()["training_suite"],
                   "gpu_resources_by_serial_phase":resources,
                   "gpu_limits_by_serial_phase":{f"{stage}_{spec['gpu_type']}":spec["max_concurrent"]
                                                 for stage,spec in resources.items()}}
        if path.exists():
            receipt["previous_receipt"] = load_json(path)
        atomic_json(receipt,path)
        dependency = None
        for stage,script,array in stages_to_submit():
            command = ["sbatch","--parsable","--kill-on-invalid-dep=yes"]
            gpu_stage = {"labels":"labels","train":"training","eval":"evaluation"}.get(stage)
            if gpu_stage is not None:
                spec = resources[gpu_stage]
                # Explicit CLI options also override stale SBATCH_* environment defaults.
                command.extend([f"--partition={spec['partition']}",f"--gres=gpu:{spec['gpu_type']}:1",
                                "--nodes=1","--ntasks=1"])
                if spec["qos"] is not None:
                    command.append(f"--qos={spec['qos']}")
            if dependency:
                command.append("--dependency=afterok:"+dependency)
            elif waits:
                command.append("--dependency=afterany:"+":".join(waits))
            if array:
                command.append("--array="+array)
            command.append(str(ROOT/"slurm"/script))
            result = subprocess.run(command,capture_output=True,text=True,check=False,timeout=30)
            job = result.stdout.strip().split(";")[0]
            if result.returncode or not re.fullmatch(r"\d+",job):
                receipt.update(status="SUBMISSION_FAILED",failed_stage=stage,error=result.stderr.strip() or result.stdout.strip())
                atomic_json(receipt,path)
                raise RuntimeError(f"Submission failed at {stage}; submitted stages retained: {receipt['error']}")
            receipt["jobs"].append({"stage":stage,"job_id":job,"command":command})
            dependency = job
            atomic_json(receipt,path)
            print(f"{stage}: {job}",flush=True)
        receipt["status"] = "SUBMITTED"
        atomic_json(receipt,path)
        summary = " -> ".join(f"{stage} <= {spec['max_concurrent']}x{spec['device_name']}"
                              for stage,spec in resources.items())
        print("Training source=LIBERO-10 (Long), OFT-Long; "+summary+
              "; evaluation is automatic after training; phases do not overlap")
        if waits:
            print("Waiting for your existing jobs: "+",".join(waits))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--submit",action="store_true")
    flags.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit()
    else:
        report = preflight()
        print(f"CPU_PREFLIGHT_PASS; source=libero_10; new Long bank will be collected, "
              f"3 methods x 25000 updates, 2000 eval episodes; "
              f"gpu_limits={report['maximum_gpu_concurrency_by_phase']}; stages={stages_to_submit()}")
