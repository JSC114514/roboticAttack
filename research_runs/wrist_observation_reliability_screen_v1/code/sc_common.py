"""Isolated in-process reuse of audited infrastructure; historical files stay read-only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = PROJECT / 'research_runs/wrist_joint_checkpoint_patch_retest_v1'
sys.path.insert(1, str(BASE / 'code'))
import jr_common as common
common.ROOT = ROOT
_dependencies = common.dependencies
def dependencies():
    return _dependencies() + [BASE/'code'/n for n in ('jr_common.py','jr_cli.py','jr_math.py','jr_loading.py','jr_rollout.py')]
common.dependencies = dependencies
from jr_common import (config, read, sha, digest, atomic, require, signature, complete, finish,
    lock, Monitor, runtime, runtime_config, module, verify, manifest, episode_receipt, episode_files, resources, PYTHON)


def native():
    # Imported only by the user-submitted GPU workers.
    require(__import__('os').environ.get('SLURM_JOB_ID'), 'Native policy work requires Slurm allocation')
    import jr_rollout
    return jr_rollout


def geometry():
    import ast
    import itertools
    import numpy as np
    p=PROJECT/'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py'
    names={'spec','vertices','anchor','truth'}
    nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(nodes)==len(names),'Fixed target geometry helper changed')
    ns=dict(np=np,itertools=itertools,require=require)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns


_prior_dependencies=common.dependencies
common.dependencies=lambda: _prior_dependencies()+[PROJECT/'research_runs/wrist_spatial_approach_pilot_v1/code/sr_geometry.py']
