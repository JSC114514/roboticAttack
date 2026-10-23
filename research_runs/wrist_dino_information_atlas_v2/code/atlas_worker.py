"""Slurm-only stage wrapper with cost/failure receipts and duplicate-worker lock."""
import argparse
import fcntl
import os
import signal
import time
from atlas_common import ROOT,atomic_json,prepared,require,resources


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['capture','analyze'],required=True)
    args=p.parse_args(); require(os.environ.get('SLURM_JOB_ID'),'Worker requires a manual Slurm submission')
    prepared(verify_inputs=True); folder=ROOT/'reports'; (folder/'locks').mkdir(exist_ok=True)
    def stop(signum,_): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    with (folder/'locks'/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic(); status='FAILED'; name=f'{args.stage}_{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
        atomic_json({'stage':args.stage,'status':'STARTED','job_id':os.environ['SLURM_JOB_ID'],**resources()},
                    folder/'COSTS'/(name+'.json'))
        try:
            if args.stage=='capture':
                from atlas_capture import capture
                capture()
            else:
                from atlas_analysis import analyze
                analyze()
            status='COMPLETE'
        except BaseException as error:
            atomic_json({'stage':args.stage,'status':'FAILED','error':repr(error),
                         'job_id':os.environ['SLURM_JOB_ID'],'elapsed_s':time.monotonic()-started,**resources()},
                        folder/'FAILURES'/(name+'.json'))
            raise
        finally:
            atomic_json({'stage':args.stage,'status':status,'job_id':os.environ['SLURM_JOB_ID'],
                         'elapsed_s':time.monotonic()-started,**resources()},folder/'COSTS'/(name+'.json'))


if __name__=='__main__': main()
