from __future__ import annotations
import argparse,subprocess,sys
from common import ROOT,atomic_json,worker_schedule,run_signature,EVAL_TASKS

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["train","eval"]);p.add_argument("index",type=int);a=p.parse_args()
    if a.stage=="eval":
        suite,task=EVAL_TASKS[a.index]
        raise SystemExit(subprocess.call([sys.executable,"-B",str(ROOT/"code/evaluate.py"),"--suite",suite,"--task-id",str(task)]))
    rows=[]
    for method in worker_schedule()[a.index]:
        rc=subprocess.call([sys.executable,"-B",str(ROOT/"code/train.py"),"--method",method])
        rows.append({"method":method,"returncode":rc})
        atomic_json({"signature":run_signature(),"worker":a.index,"results":rows,
                     "status":"FAILED" if any(r["returncode"] for r in rows) else "RUNNING"},ROOT/f"reports/WORKER_{a.index}.json")
    failed=any(r["returncode"] for r in rows)
    atomic_json({"signature":run_signature(),"worker":a.index,"results":rows,
                 "status":"FAILED" if failed else "COMPLETE"},ROOT/f"reports/WORKER_{a.index}.json")
    raise SystemExit(int(failed))
