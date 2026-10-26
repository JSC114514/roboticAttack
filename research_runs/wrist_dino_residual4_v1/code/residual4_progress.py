from pathlib import Path
import json, os, threading, time
from residual4_common import atomic_json

class Progress:
    def __init__(self, path, **values):
        self.path, self.lock = Path(path), threading.Lock()
        self.values = {**values, 'status': 'RUNNING', 'started_unix': time.time(), 'pid': os.getpid(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'phase': 'bootstrap'}
        self.stop = threading.Event()
        self.update('bootstrap')
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _write(self):
        self.values['heartbeat_unix'] = time.time()
        atomic_json(dict(self.values), self.path)

    def _heartbeat(self):
        while not self.stop.wait(15):
            with self.lock:
                self._write()

    def update(self, phase, **values):
        with self.lock:
            self.values.update(phase=phase, phase_started_unix=time.time(), **values)
            self._write()
        print('RESIDUAL4_PHASE ' + json.dumps({'phase': phase, **values}, ensure_ascii=False), flush=True)

    def finish(self, status, **values):
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            self.values.update(status=status, finished_unix=time.time(), **values)
            self._write()
            archive = self.path.parent / 'worker_attempts' / (str(self.values['started_unix']) + '_' + str(os.getpid()) + '.json')
            atomic_json(dict(self.values), archive)

