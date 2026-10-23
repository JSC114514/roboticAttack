"""One explicit stage per user-submitted Slurm job; never submits resources."""
from __future__ import annotations
import argparse, os, time, traceback
from common import ROOT, atomic_json, cfg, preflight, require, resources, signature, cost_attempt

def run(stage):
    if stage=="prepare": preflight()
    elif stage=="cache":
        from cache import run_cache; run_cache()
    elif stage=="train":
        from train import run_train; run_train()
    elif stage=="validate":
        from evaluate import run_split; run_split("validation")
    elif stage=="select":
        from analyze import select_winner; select_winner()
    elif stage=="heldout":
        from evaluate import run_split; run_split("heldout")
    elif stage=="analyze":
        from analyze import analyze; analyze()
    else: raise ValueError(stage)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=("prepare","cache","train","validate","select","heldout","analyze")); args=parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"),"Workers run only inside a user-submitted Slurm job")
    with cost_attempt(args.stage):
        expected=os.environ.get("RELATION_EXPECTED_SIGNATURE")
        require(expected==signature() if expected else
                os.environ.get("SLURM_JOB_NAME","").startswith(f"rel1_{signature()[:8]}_"),
                "Submitted job belongs to a different source revision; refusing mixed-version execution")
        run(args.stage)
if __name__=="__main__":main()
