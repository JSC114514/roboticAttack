"""Allocated workers only; atomic failure/cost receipts and nonblocking locks."""
from __future__ import annotations
import argparse
import fcntl
import os
import signal
import time
from pilot_common import (ROOT,atomic_json,complete,directories,load_json,prepared,require,resources)


def prerequisites(stage,sid):
    required={'prepare':[], 'mechanism':['PREPARE_COMPLETE'], 'pilot':['MECHANISM_COMPLETE'],
              'analyze':[]}[stage]
    for name in required: require(complete(ROOT/'reports'/(name+'.json'),sid),'Upstream stage incomplete: '+name)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['prepare','mechanism','pilot','analyze'],required=True)
    args=parser.parse_args(); require(os.environ.get('SLURM_JOB_ID'),'Use the user-submitted Slurm entrypoint')
    directories()
    def stop(signum,_): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGUSR1,stop)
    with (ROOT/'reports/locks'/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic(); status='FAILED'
        name=f'{args.stage}_{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
        cost=ROOT/'reports/COSTS'/(name+'.json')
        atomic_json({'stage':args.stage,'status':'STARTED','job_id':os.environ['SLURM_JOB_ID'],
                     'started_unix':time.time(),**resources()},cost)
        try:
            prep=prepared(verify_inputs=True)
            # Run this BEFORE importing torch/loading weights in a dependent GPU job.
            prerequisites(args.stage,prep['signature'])
            if args.stage in ('mechanism','pilot'):
                import torch
                require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one allocated GPU required')
                require('4090' in torch.cuda.get_device_name(0),'Expected 4090; no automatic fallback')
                torch.cuda.reset_peak_memory_stats()
            if args.stage=='prepare':
                from pilot_readouts import prepare_readouts
                prepare_readouts(prep)
            elif args.stage=='mechanism':
                from pilot_mechanism import mechanism
                mechanism(prep)
            elif args.stage=='pilot':
                from pilot_train import pilot
                pilot(prep)
            else:
                from pilot_analysis import analyze
                analyze(prep)
            status='COMPLETE'
        except BaseException as error:
            progress_path=ROOT/'reports/PROGRESS.json'
            atomic_json({'stage':args.stage,'status':'FAILED','job_id':os.environ['SLURM_JOB_ID'],
                         'error':repr(error),'elapsed_s':time.monotonic()-started,
                         'last_progress':load_json(progress_path) if progress_path.exists() else None,**resources()},
                        ROOT/'reports/FAILURES'/(name+'.json'))
            raise
        finally:
            atomic_json({'stage':args.stage,'status':status,'job_id':os.environ['SLURM_JOB_ID'],
                         'elapsed_s':time.monotonic()-started,**resources()},cost)


if __name__=='__main__': main()
