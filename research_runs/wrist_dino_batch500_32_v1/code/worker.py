"""Six fixed training workers; independent child releases GPU between patches."""
from __future__ import annotations
import argparse
import subprocess
import sys
from common import ROOT, atomic_json, prepared, require_gpu, run_signature, worker_schedule


def run_worker(index):
    prepared()
    require_gpu("training")
    methods, results = worker_schedule()[index], []
    print(f"BATCH500_WORKER {index}: " + ", ".join(methods), flush=True)
    for method in methods:
        response = subprocess.run([sys.executable, "-B", str(ROOT / "code/train.py"), "--method", method], check=False)
        results.append({"method": method, "returncode": response.returncode})
        atomic_json({"signature": run_signature(), "worker": index, "results": results,
                     "status": "RUNNING"}, ROOT / "reports" / f"WORKER_{index}.json")
        print(f"BATCH500_WORKER {index} method={method} returncode={response.returncode}", flush=True)
    success = all(row["returncode"] == 0 for row in results)
    atomic_json({"signature": run_signature(), "worker": index, "results": results,
                 "status": "COMPLETE" if success else "FAILED"}, ROOT / "reports" / f"WORKER_{index}.json")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=range(6), required=True)
    run_worker(parser.parse_args().worker)
