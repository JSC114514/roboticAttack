"""One explicitly submitted Slurm worker; never submits/cancels another job."""
import argparse,os,signal
from common import ROOT,signature,require,complete,Monitor,cost_attempt
# Import local modules before the historical runtime modifies sys.path.
import data,train,evaluate,report

PILOT_TRAIN=('individual','projection','full')
NIGHT_TRAIN=('full_s42','full_s43','projection_s43')

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=('pilot_train','oracle','pilot_check','pilot_report',
                                                             'night_train','night_eval','night_report'))
    args=p.parse_args();index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
    detail=PILOT_TRAIN[index] if args.task=='pilot_train' else NIGHT_TRAIN[index] if args.task=='night_train' else data.cfg()['suites'][index] if args.task=='night_eval' else ''
    key=args.task+('_'+detail if detail else '')
    def stopped(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped)
    with Monitor(key) as monitor,cost_attempt(key):
        require(os.environ.get('GEOM_EXPECTED_SIGNATURE')==signature(),'Submitted source signature changed; nothing executed')
        require(complete(ROOT/'reports/PREPARE_COMPLETE.json'),'CPU preparation missing or stale')
        if args.task=='pilot_train':train.run_pilot(detail,monitor)
        elif args.task=='oracle':evaluate.run_oracle(monitor)
        elif args.task=='pilot_check':evaluate.run_pilot_check(monitor)
        elif args.task=='pilot_report':report.pilot_report(monitor)
        elif args.task=='night_train':train.run_night(detail,monitor)
        elif args.task=='night_eval':evaluate.run_night_eval(detail,monitor)
        else:report.night_report(monitor)

if __name__=='__main__':main()
