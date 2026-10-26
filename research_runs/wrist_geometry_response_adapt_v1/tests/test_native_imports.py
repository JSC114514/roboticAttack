"""Exercise native preprocessing imports before requesting GPU work."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from wr_common import PYTHON


class NativeImportTests(unittest.TestCase):
    def test_native_preprocessing_and_l1_head_on_cpu(self):
        # Use this run's real shell setup, even when tests are invoked directly.
        setup = (ROOT/'run.sh').read_text().split('\nexec ', 1)[0]
        probe = '''
import importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from wr_common import PROJECT, config, runtime_config
path = PROJECT/config()['runtime_run']/'code'
sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location('wr_cpu_native_probe', path/'model_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
cfg = runtime_config()
runtime.add_paths(cfg)
raw = dict(agentview_image=np.zeros((256,256,3), np.uint8),
    robot0_eye_in_hand_image=np.full((256,256,3), 127, np.uint8),
    robot0_eef_pos=np.zeros(3), robot0_eef_quat=np.array([0.,0.,0.,1.]),
    robot0_gripper_qpos=np.zeros(2))
obs = runtime.observation(raw, cfg)
from prismatic.models.action_heads import L1RegressionActionHead
import torch
head = L1RegressionActionHead(input_dim=4, hidden_dim=8, action_dim=7)
with torch.no_grad():
    actions = head.predict_action(torch.zeros(1,56,4))
assert torch.isfinite(actions).all()
print(json.dumps(dict(shapes={k:list(v.shape) for k,v in obs.items()},
    actions=list(actions.shape), cuda_initialized=torch.cuda.is_initialized())))
'''
        with tempfile.TemporaryDirectory() as folder:
            shell = Path(folder)/'cpu_probe.sh'
            shell.write_text(setup+'\nexec '+shlex.join([PYTHON, '-B', '-c', probe, str(ROOT/'code')])+'\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_NUM_INTRAOP_THREADS='2',
                       TF_NUM_INTEROP_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            result = subprocess.run(['bash', str(shell)], env=env, capture_output=True,
                                    text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-6000:])
        observed = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(observed['shapes'], dict(full_image=[224,224,3],
                         wrist_image=[224,224,3], state=[8]))
        self.assertEqual(observed['actions'], [1,8,7])
        self.assertFalse(observed['cuda_initialized'])


if __name__ == '__main__':
    unittest.main()
